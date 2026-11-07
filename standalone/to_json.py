"""JSON response models: the text-region structure inspired by the Ichigo extension."""

import base64
from typing import Annotated, Dict, List

import cv2
import numpy as np
from pydantic import BaseModel, Field, WithJsonSchema

from manga_translator import Context
from manga_translator.utils import TextBlock

NumpyNdarray = Annotated[
    np.ndarray,
    WithJsonSchema({
        'type': 'string',
        'format': 'base64',
        'examples': ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."],
    }),
]


class TextColor(BaseModel):
    fg: tuple[int, int, int]
    bg: tuple[int, int, int]


class Translation(BaseModel):
    minX: int
    minY: int
    maxX: int
    maxY: int
    is_bulleted_list: bool
    angle: float | int
    prob: float
    text_color: TextColor
    text: dict[str, str]
    background: NumpyNdarray = Field(
        ...,
        description="Background image encoded as a base64 string",
        examples=["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."],
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda array: Translation.encode_background(array)
        }

    @staticmethod
    def encode_background(array: np.ndarray) -> str:
        _, buffer = cv2.imencode('.png', array)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


class TranslationResponse(BaseModel):
    translations: List[Translation]


def to_translation(ctx: Context) -> TranslationResponse:
    # On skip paths (e.g. no regions found) these stay None; treat as empty.
    text_regions: List[TextBlock] = ctx.text_regions or []
    inpaint = ctx.img_inpainted
    translations: Dict[str, List[str]] = ctx.translations or {}
    results = []
    for i, blk in enumerate(text_regions):
        minX, minY, maxX, maxY = blk.xyxy
        text_region = text_regions[i]
        if 'translations' in ctx:
            trans = {key: value[i] for key, value in translations.items()}
        else:
            trans = {}
        trans[text_region.source_lang] = text_regions[i].text
        text_region.adjust_bg_color = False
        color1, color2 = text_region.get_font_colors()
        results.append(Translation(
            text=trans,
            minX=int(minX), minY=int(minY), maxX=int(maxX), maxY=int(maxY),
            background=inpaint[minY:maxY, minX:maxX],
            is_bulleted_list=text_region.is_bulleted_list,
            text_color=TextColor(fg=color1.tolist(), bg=color2.tolist()),
            prob=text_region.prob,
            angle=text_region.angle,
        ))

    return TranslationResponse(translations=results)
