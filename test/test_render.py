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
    sample_canvas_bg,
    shrink_font_size_to_fit,
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


def test_render_single_preshrinks_canvas_before_downscale(monkeypatch):
    """A strip padded past the destination box's size used to be warped down in
    one INTER_LINEAR step, aliasing thin strokes. It must be area-resized close
    to the destination size first so the warp itself only upscales."""
    resizes = []
    real_resize = cv2.resize

    def spy(img, dsize, *args, **kwargs):
        resizes.append((img.shape[:2], dsize, kwargs.get('interpolation')))
        return real_resize(img, dsize, *args, **kwargs)

    monkeypatch.setattr(cv2, 'resize', spy)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # 100x30 box; the wrapped 60px strip padded to this ratio is far larger.
    small_box = np.array([[[10, 10], [110, 10], [110, 40], [10, 40]]], dtype=np.int64)
    region = make_flat_test_region(translation='word ' * 40, font_size=60)
    out = render_single(img.copy(), region, small_box, True, None, False)
    assert int(np.count_nonzero(out)) > 0  # text still painted

    area_resizes = [(src, dst) for src, dst, interp in resizes if interp == cv2.INTER_AREA]
    assert area_resizes, 'expected an INTER_AREA pre-shrink of the text canvas'
    src_shape, (new_w, new_h) = area_resizes[-1]
    dst_area = 100 * 30
    assert new_w * new_h < src_shape[0] * src_shape[1]  # actually shrank
    assert dst_area * 0.5 <= new_w * new_h <= dst_area * 1.5  # landed near box size


def test_render_single_maps_quads_with_exact_perspective_transform(monkeypatch):
    """Four point correspondences need the closed-form 4-point transform;
    findHomography's RANSAC adds random sampling without earning anything."""
    calls = []
    real_get_perspective = cv2.getPerspectiveTransform

    def spy(src, dst):
        calls.append((src.shape, dst.shape, dst.dtype))
        return real_get_perspective(src, dst)

    monkeypatch.setattr(cv2, 'getPerspectiveTransform', spy)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    box = np.array([[[10, 10], [210, 10], [210, 70], [10, 70]]], dtype=np.int64)
    out = render_single(img.copy(), make_flat_test_region(), box, True, None, False)
    assert int(np.count_nonzero(out)) > 0
    assert calls == [((4, 2), (4, 2), np.dtype('float32'))]


def make_stroke_test_region(fg=(255, 255, 255), bg=(255, 0, 0), translation='Hello world', font_size=40):
    region = TextBlock([[[10, 10], [200, 10], [10, 60], [200, 60]]], texts=['source'],
                       translation=translation, font_size=font_size)
    region.target_lang = 'ENG'
    region.set_font_colors(list(fg), list(bg))
    return region


STROKE_TEST_BOX = np.array([[[10, 10], [210, 10], [210, 70], [10, 70]]], dtype=np.int64)


def test_sample_canvas_bg_reads_the_box_interior():
    img = np.full((100, 100, 3), 77, dtype=np.uint8)
    quad = np.array([[[10, 10], [60, 10], [60, 60], [10, 60]]], dtype=np.int32)
    assert sample_canvas_bg(img, quad) == pytest.approx((77.0, 77.0, 77.0))
    # A box smaller than the 5px erosion margin has no trustworthy interior.
    tiny = np.array([[[10, 10], [12, 10], [12, 12], [10, 12]]], dtype=np.int32)
    assert sample_canvas_bg(img, tiny) is None


def test_render_single_stroke_width_controls_the_outline():
    """render.stroke_width scales the outline (default 0.07); 0 renders the
    glyphs alone instead of dropping the text, as boundingRect of the empty
    border map used to do."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    region = make_stroke_test_region()  # white glyphs, red outline

    def red_pixels(out):
        return int(np.count_nonzero((out[..., 0] > 200) & (out[..., 1] < 60) & (out[..., 2] < 60)))

    default = render_single(img.copy(), region, STROKE_TEST_BOX, True, None, False)
    thick = render_single(img.copy(), region, STROKE_TEST_BOX, True, None, False, stroke_width=0.3)
    none = render_single(img.copy(), region, STROKE_TEST_BOX, True, None, False, stroke_width=0)

    assert int(np.count_nonzero(default)) > 0
    assert red_pixels(thick) > red_pixels(default) > 0
    assert red_pixels(none) == 0
    assert int(np.count_nonzero(none)) > 0  # outline-less text is still painted


def test_render_single_contrast_aware_stroke_repaints_invisible_outline():
    """The outline colour is estimated before inpainting. When it matches the
    inpainted backdrop and the text itself is only marginal on it, the flag
    repaints the outline against the real canvas; without the flag the
    render is untouched."""
    img = np.full((300, 400, 3), 121, dtype=np.uint8)  # inpainted backdrop
    # fg/outline ΔE ≈ 37: survives fg_bg_compare's 30 floor, inside the flag's 40
    region = make_stroke_test_region(fg=(220, 220, 220), bg=(121, 121, 121))

    def white_pixels(out):
        return int(np.count_nonzero((out > 240).all(axis=2)))

    off = render_single(img.copy(), region, STROKE_TEST_BOX, True, None, False)
    on = render_single(img.copy(), region, STROKE_TEST_BOX, True, None, False, contrast_aware_stroke=True)

    assert white_pixels(off) == 0  # default: outline same colour as backdrop
    assert white_pixels(on) > 0
    assert int(np.count_nonzero(on != off)) > 0

    # High-contrast text keeps its designed outline even with the flag on.
    crisp = make_stroke_test_region(fg=(255, 255, 255), bg=(0, 0, 0))
    kept = render_single(img.copy(), crisp, STROKE_TEST_BOX, True, None, False, contrast_aware_stroke=True)
    assert int(np.count_nonzero((kept[..., 0] < 60) & (kept[..., 1] < 60) & (kept[..., 2] < 60))) > 0  # black outline painted


def make_fit_test_region(translation, polygon=None, font_size=40):
    if polygon is None:
        polygon = [[10, 10], [200, 10], [10, 60], [200, 60]]
    region = TextBlock([polygon], texts=['source'], translation=translation, font_size=font_size)
    region.target_lang = 'ENG'
    return region


LONG_TRANSLATION = 'Alpha Beta Gamma Delta Epsilon Zeta Eta Theta Iota Kappa Lambda Mu Nu'
SHORT_TRANSLATION = 'Hi there'


def quad_width(pts):
    pts = np.asarray(pts).reshape(-1, 2)
    return pts[:, 0].max() - pts[:, 0].min()


def test_shrink_font_size_to_fit_binary_searches_the_largest_fitting_size():
    region = make_fit_test_region(LONG_TRANSLATION)  # 190x50 box, 40px font
    fitted = shrink_font_size_to_fit(region, 40, 10)
    assert fitted is not None
    assert 10 <= fitted < 40
    lines, widths = text_render.calc_horizontal(fitted, LONG_TRANSLATION, 190, 50, 'ENG', True)
    assert len(lines) * fitted <= 50 and max(widths) <= 190
    # the next size up must overflow, i.e. the search found the optimum
    lines, widths = text_render.calc_horizontal(fitted + 1, LONG_TRANSLATION, 190, 50, 'ENG', True)
    assert len(lines) * (fitted + 1) > 50 or max(widths) > 190


def test_shrink_font_size_to_fit_passthrough_and_give_up_cases():
    region = make_fit_test_region(SHORT_TRANSLATION)
    assert shrink_font_size_to_fit(region, 40, 10) == 40  # already fits
    region = make_fit_test_region(LONG_TRANSLATION * 4)
    assert shrink_font_size_to_fit(region, 40, 10) is None  # hopeless even at 10px


def test_resize_regions_shrink_to_fit_prefers_font_over_box_growth():
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    region = make_fit_test_region(LONG_TRANSLATION)
    shrunk = resize_regions_to_font_size(img, [region], None, 0, 10, shrink_to_fit=True)[0]
    assert region.font_size < 40
    assert quad_width(shrunk) == quad_width(region.min_rect)  # box untouched

    region = make_fit_test_region(LONG_TRANSLATION)
    grown = resize_regions_to_font_size(img, [region], None, 0, 10)[0]
    assert region.font_size == 40  # default path: font kept, box widened
    assert quad_width(grown) > quad_width(region.min_rect)

    # Hopeless text still takes the old box-growth path with the flag on.
    region = make_fit_test_region(LONG_TRANSLATION * 4)
    grown = resize_regions_to_font_size(img, [region], None, 0, 10, shrink_to_fit=True)[0]
    assert region.font_size == 40
    assert quad_width(grown) > quad_width(region.min_rect)


# --- kinsoku (禁則処理) CJK line breaking ------------------------------------


def test_apply_kinsoku_seats_boundary_characters():
    fs = 20
    # A forbidden line-start character (full stop) is pulled onto the line above.
    assert text_render.apply_kinsoku_line_breaking(['す' * 5, '。' + 'す' * 3], fs, 5 * fs) \
        == ['す' * 5 + '。', 'す' * 3]
    # An opening bracket stranded at a line end is pushed down instead.
    assert text_render.apply_kinsoku_line_breaking(['す' * 5 + '「', 'す' * 2], fs, 5 * fs) \
        == ['す' * 5, '「' + 'す' * 2]
    # Hanging-punctuation budget is one character: the surplus mark stays put,
    # and a line vacated completely is dropped rather than left blank.
    assert text_render.apply_kinsoku_line_breaking(['す' * 5, '。、す' * 1], fs, 5 * fs) \
        == ['す' * 5 + '。', '、す']
    assert text_render.apply_kinsoku_line_breaking(['す' * 5, '。'], fs, 5 * fs) == ['す' * 5 + '。']
    # Legal boundaries are left alone.
    assert text_render.apply_kinsoku_line_breaking(['す' * 5, 'す' * 5], fs, 5 * fs) \
        == ['す' * 5, 'す' * 5]


def test_calc_horizontal_kinsoku_moves_breaks_not_content():
    fs = 20
    text = 'す' * 10 + '。' + 'す' * 10   # 21 chars, 5 fit per 100px line
    kwargs = dict(max_width=100, max_height=10 * fs, language='ja_JP', hyphenate=False)

    plain, _ = text_render.calc_horizontal(fs, text, **kwargs)
    assert any(line[0] == '。' for line in plain[1:])  # default happily breaks before it

    fixed, widths = text_render.calc_horizontal(fs, text, kinsoku=True, **kwargs)
    assert ''.join(fixed).replace('-', '') == text  # same content, only moved
    assert all(line[0] not in text_render.KINSOKU_LINE_START_FORBIDDEN for line in fixed)
    assert all(line[-1] not in text_render.KINSOKU_LINE_END_FORBIDDEN for line in fixed)
    # Hanging punctuation: at most one character past the line limit.
    assert all(w <= 100 + fs for w in widths)
    assert len(fixed) <= len(plain)  # emptying a line may even save one


ARABIC = 'مرحبا يا صديقي، هل أنت مستعد للمغامرة؟'

def test_arabic_wraps_in_reading_order():
    text_render.set_font('')
    lines, widths = text_render.calc_horizontal_rtl(48, ARABIC, 300, 140)
    # Lines break at spaces only, keep the space after the comma,
    # and the sentence starts on the first line
    assert len(lines) > 1
    assert ' '.join(lines) == ARABIC
    assert lines[0].startswith('مرحبا')
    assert widths == [text_render.get_rtl_string_width(48, line) for line in lines]

def test_arabic_translation_is_rendered_as_is():
    # Shaping and right-to-left layout happen in the renderer, so the
    # translation keeps its logical order, Latin words and digits included
    region = TextBlock([[[10, 10], [200, 10], [10, 400], [200, 400]]], texts=['a'], translation='اقرأ manga كل يوم 3 مرات')
    region.target_lang = 'ARA'
    assert region.get_translation_for_rendering() == 'اقرأ manga كل يوم 3 مرات'

@pytest.mark.asyncio
async def test_default_renderer_arabic():
    width, height = 600, 600
    img = np.zeros((height, width, 3))
    region = TextBlock([[[10, 10], [500, 10], [10, 300], [500, 300]]], texts=['a'], translation=ARABIC)
    region.target_lang = 'ARA'
    region.set_font_colors([255, 255, 255], [0, 0, 0])
    region.font_size = 48

    blank = img.copy()
    img_rendered = await dispatch_rendering(img, [region], hyphenate=False)
    assert (img_rendered != blank).any()
    save_result('default_arabic.png', img_rendered, [region])
