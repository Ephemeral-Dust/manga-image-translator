import os
import sys
import cv2
import pytest
import numpy as np

from manga_translator.rendering import (
    MAX_RENDER_CANVAS_PIXELS,
    clamp_quad_to_canvas,
    dispatch as dispatch_rendering,
    dispatch_eng_render,
    quad_is_degenerate,
    render as render_single,
    resize_regions_to_font_size,
    text_render,
)
from manga_translator.utils import (
    TextBlock,
    visualize_textblocks,
)


RENDER_IMAGE_FOLDER = 'test/testdata/render'
os.makedirs(RENDER_IMAGE_FOLDER, exist_ok=True)

text_render.set_font('')

def save_result(path, img, regions):
    path = os.path.join(RENDER_IMAGE_FOLDER, path)
    cv2.imwrite(path, visualize_textblocks(img, regions))


@pytest.mark.asyncio
async def test_default_renderer():
    width, height = 1000, 1000
    img = np.zeros((height, width, 3))
    regions = [
        TextBlock(
            [[[10, 10], [200, 10], [10, 400], [200, 400]]],
            texts=['a', 'b','c', 'd', 'e', 'f'],
            translation='aaaaaa bbbbbbbbbbbb cccc ddddddddddd eeeeeeeeeeeeee fff'
        ),
        TextBlock(
            [[[410, 10], [900, 10], [410, 800], [900, 800]]],
            texts=['eng', 'pne'],
            translation=#'aaaaaa bbbbbbbbbbbb cccc' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd eeeeeeeeeeeeee fff' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd fff' \
                # 'dddddddddddddddddddddddddddddddddddddddddddddddddddd ' \
                'normal english sentences can be hyphenated! ' \
                'Pneumonoultramicroscopicsilicovolcanoconiosis'
        ),
    ]
    for region in regions:
        region.target_lang = 'ENG'
        region.set_font_colors([255, 255, 255], [200, 200, 200])
        region.font_size = 100

    img_rendered = await dispatch_rendering(img, regions, hyphenate=False)
    save_result('default1.png', img_rendered, regions)


def test_resize_regions_stays_within_image():
    """A longer translation expands the text box, but the expanded box must stay
    on the canvas, otherwise the text warp drops everything past the image edge."""
    width, height = 500, 325
    img = np.zeros((height, width, 3), dtype=np.uint8)
    region = TextBlock(
        [[[85, 61], [384, 61], [85, 135], [384, 135]]],
        texts=['自転車を除く'],
        translation='Except for the bike.',
        font_size=50,
    )
    region.target_lang = 'ENG'

    dst_points_list = resize_regions_to_font_size(img, [region], None, 0, -1)
    dst_points = dst_points_list[0]
    assert dst_points is not None
    assert dst_points[..., 0].max() <= width - 1
    assert dst_points[..., 1].max() <= height - 1


def test_quad_is_degenerate_flags_flat_boxes():
    """Flat, collapsed and NaN boxes are the ones cv2.findHomography cannot map."""
    assert quad_is_degenerate(np.array([[[10, 50], [100, 50], [200, 50], [50, 50]]], dtype=np.int64))
    assert quad_is_degenerate(np.array([[[10, 50]] * 4], dtype=np.int64))
    assert quad_is_degenerate(np.array([[[10, 50], [np.nan, 10], [200, 60], [10, 60]]], dtype=np.float64))
    assert not quad_is_degenerate(np.array([[[10, 10], [200, 10], [200, 60], [10, 60]]], dtype=np.int64))


def test_clamp_quad_to_canvas_keeps_boxes_renderable():
    """Clipping every corner on its own folds an angled box flat against one canvas
    edge, and a flat box aborts the text warp. Overflow must not cost the box its area."""
    canvas = (300, 400)  # (height, width)
    below_canvas = np.array([[[10, 310], [200, 310], [200, 420], [10, 420]]], dtype=np.int64)
    kept = clamp_quad_to_canvas(below_canvas, canvas)
    assert not quad_is_degenerate(kept)
    assert np.array_equal(kept, below_canvas)

    # Ordinary overflow is still pulled back onto the canvas.
    over_right = np.array([[[350, 10], [500, 10], [500, 60], [350, 60]]], dtype=np.int64)
    clamped = clamp_quad_to_canvas(over_right, canvas)
    assert clamped[..., 0].max() == canvas[1] - 1
    assert not quad_is_degenerate(clamped)


def make_flat_test_region(translation='Hello world', font_size=30):
    region = TextBlock([[[10, 10], [200, 10], [10, 60], [200, 60]]], texts=['source'],
                       translation=translation, font_size=font_size)
    region.target_lang = 'ENG'
    region.set_font_colors([0, 0, 0], [255, 255, 255])
    return region


def test_render_single_skips_boxes_without_area():
    """A text box flattened onto a line used to abort the whole page render with an
    OpenCV error; it must be skipped instead."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    flat = np.array([[[10, 50], [100, 50], [200, 50], [50, 50]]], dtype=np.int64)
    np.testing.assert_array_equal(render_single(img.copy(), make_flat_test_region(), flat, True, None, False), img)

    collapsed = np.array([[[10, 50], [200, 50], [200, 50], [10, 50]]], dtype=np.int64)
    np.testing.assert_array_equal(render_single(img.copy(), make_flat_test_region(), collapsed, True, None, False), img)

    intact = np.array([[[10, 10], [200, 10], [200, 60], [10, 60]]], dtype=np.int64)
    assert render_single(img.copy(), make_flat_test_region(), intact, True, None, False).sum() > img.sum()


def test_render_single_caps_the_warp_canvas(monkeypatch):
    """Padding stretches the typeset strip to the box's aspect ratio, and a huge
    fixed font size on a tall narrow box used to inflate that padded canvas to
    gigabytes before the warp. The strip must shrink to the budget instead."""
    monkeypatch.setattr('manga_translator.rendering.MAX_RENDER_CANVAS_PIXELS', 100_000)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    tall_narrow = np.array([[[10, 10], [40, 10], [40, 290], [10, 290]]], dtype=np.int64)
    region = make_flat_test_region(translation='word ' * 60, font_size=200)
    out = render_single(img.copy(), region, tall_narrow, True, None, False)
    assert not np.array_equal(out, img)  # text still painted
    assert int(np.count_nonzero(out)) > 0
