import os
import sys
import cv2
import pytest
import numpy as np

from manga_translator.rendering import (
    dispatch as dispatch_rendering,
    dispatch_eng_render,
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
