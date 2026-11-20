"""Render-only path: inpaint and typeset regions the client already knows about.

Backs the /render/* endpoints. The client supplies text boxes plus the text to
put in them (its own vision model or translation, or the array returned by
/translate/json with translations filled in). The server skips detection, OCR
and translation and runs only the tail of the normal pipeline: mask
refinement -> inpainting -> text rendering.

Region field names intentionally mirror the Translation objects from
/translate/json so a client can post them straight back.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from manga_translator.config import Config
from manga_translator.utils import Context, TextBlock, dump_image, load_image, sort_regions

from .to_json import TextColor


class RenderRegion(BaseModel):
    """One text region to inpaint and typeset, in original-image coordinates."""

    minX: int
    minY: int
    maxX: int
    maxY: int
    angle: float = 0
    prob: float = 1.0
    text: Dict[str, str] = Field(default_factory=dict)
    """Source texts keyed by language code, same shape as /translate/json `text`."""
    translation: Optional[str] = None
    """Text to render. Falls back to text[config.translator.target_lang]."""
    text_color: Optional[TextColor] = None
    """Omit to let the server estimate fg/bg from the region pixels."""
    is_bulleted_list: bool = False
    """Accepted for round-tripping; the pipeline recomputes it from the text."""
    background: Optional[str] = None
    """Accepted for round-tripping and ignored; inpainting regenerates it."""

    @model_validator(mode="after")
    def _check_box(self):
        if self.maxX <= self.minX or self.maxY <= self.minY:
            raise ValueError(
                f"degenerate region ({self.minX}, {self.minY}, {self.maxX}, {self.maxY}): "
                "needs minX < maxX and minY < maxY"
            )
        return self

    def resolve_translation(self, target_lang: str) -> str:
        if self.translation is not None and self.translation.strip():
            return self.translation
        if target_lang and self.text.get(target_lang, "").strip():
            return self.text[target_lang]
        raise ValueError(
            f"no text to render: provide `translation` or `text[{target_lang!r}]` "
            f"for region at ({self.minX}, {self.minY}, {self.maxX}, {self.maxY})"
        )

    def resolve_source_text(self, target_lang: str, translation: str) -> str:
        """Text fed to mask fitting; prefer the source language the client OCR'd."""
        for lang, value in self.text.items():
            if lang != target_lang and value.strip():
                return value
        return translation


def _estimate_colors(crop: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Crude 2-cluster fg/bg guess from a region crop, used only when the client omits colors."""
    pixels = crop.reshape(-1, 3).astype(np.float32)
    if pixels.shape[0] < 8:
        return (0, 0, 0), (0, 0, 0)  # adjust_bg_color in the renderer fixes contrast
    pixels = pixels[:: max(1, pixels.shape[0] // 2048)]  # keep kmeans cheap on big boxes
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=2)
    bg_idx, fg_idx = (0, 1) if counts[0] >= counts[1] else (1, 0)
    # The fg cluster is mostly anti-aliased edge pixels whose mean sits well
    # inside the background; the glyph cores are the pixels farthest from the
    # bg centre in luminance, so bias the fg pick toward that tail.
    fg_pixels = pixels[labels.flatten() == fg_idx]
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    deltas = np.abs(fg_pixels @ weights - centers[bg_idx] @ weights)
    tail = fg_pixels[np.argsort(deltas)[::-1][: max(1, len(fg_pixels) // 3)]]
    fg = tuple(int(c) for c in tail.mean(axis=0))
    bg = tuple(int(c) for c in centers[bg_idx])
    return fg, bg


def build_blocks(
    regions: List[RenderRegion], config: Config, img_rgb: np.ndarray
) -> Tuple[List[TextBlock], np.ndarray]:
    """Turn client regions into pipeline TextBlocks plus a raw inpaint mask.

    Each region becomes one single-line block: an axis-aligned rectangle in
    image bounds, with the box angle carried over exactly like /translate/json
    round-trips it. Raises ValueError (-> 422) on unusable regions.
    """
    height, width = img_rgb.shape[:2]
    target_lang = config.translator.target_lang
    blocks: List[TextBlock] = []
    mask_raw = np.zeros((height, width), dtype=np.uint8)

    for i, region in enumerate(regions):
        translation = region.resolve_translation(target_lang)
        source_text = region.resolve_source_text(target_lang, translation)

        x1 = max(0, min(width - 1, region.minX))
        y1 = max(0, min(height - 1, region.minY))
        x2 = max(0, min(width - 1, region.maxX))
        y2 = max(0, min(height - 1, region.maxY))
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError(
                f"regions[{i}] at ({region.minX}, {region.minY}, {region.maxX}, {region.maxY}) "
                f"has no usable overlap with the {width}x{height} image"
            )

        if region.text_color is not None:
            fg, bg = tuple(region.text_color.fg), tuple(region.text_color.bg)
        else:
            fg, bg = _estimate_colors(img_rgb[y1:y2, x1:x2])

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        blocks.append(
            TextBlock(
                [pts],
                [source_text],
                # Glyph size fits the box's short side: box height for horizontal
                # lines, box width for vertical columns (the renderer picks the
                # direction from the box). Using height alone made a tall narrow
                # column request a font taller than the column is wide, blowing
                # the aspect-matched render canvas up to gigabytes.
                font_size=min(x2 - x1, y2 - y1),
                angle=region.angle,
                prob=region.prob,
                translation=translation,
                fg_color=fg,
                bg_color=bg,
                target_lang=target_lang,
            )
        )
        cv2.rectangle(mask_raw, (x1, y1), (x2, y2), 255, -1)

    return blocks, mask_raw


async def render_regions(manga, image: Image.Image, config: Config, regions: List[RenderRegion]) -> Context:
    """Inpaint + typeset `regions` onto `image`; returns a Context with .result set.

    Reuses the MangaTranslator stage methods exactly as the full pipeline calls
    them (manga_translator.py tail), just without detection/OCR/translation.
    Raises ValueError for client-side problems (-> 422).
    """
    ctx = Context()
    ctx.input = image
    ctx.img_colorized = image
    ctx.upscaled = image
    ctx.img_rgb, ctx.img_alpha = load_image(image)

    blocks, mask_raw = build_blocks(regions, config, ctx.img_rgb)
    blocks = sort_regions(
        blocks,
        right_to_left=config.render.rtl,
        img=ctx.img_rgb,
        force_simple_sort=config.force_simple_sort,
    )

    ctx.text_regions = blocks
    ctx.mask_raw = mask_raw
    ctx.mask = None

    ctx.mask = await manga._run_mask_refinement(config, ctx)
    ctx.img_inpainted = await manga._run_inpainting(config, ctx)
    ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))
    ctx.img_rendered = await manga._run_text_rendering(config, ctx)

    ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)
    return ctx
