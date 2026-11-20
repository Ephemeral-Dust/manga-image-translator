import os
import cv2
import numpy as np
from typing import List, Optional
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from .text_render_pillow_eng import render_textblock_list_eng as render_textblock_list_eng_pillow
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger('render')

# Ceiling on the RGBA strip a single text region is typeset onto before it gets
# warped into the canvas. The strip is padded to the destination box's aspect
# ratio, so a pathological box/text combination can otherwise ask for gigabytes.
MAX_RENDER_CANVAS_PIXELS = 40_000_000

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

def sample_canvas_bg(img: np.ndarray, dst_points) -> tuple | None:
    """Mean colour of the inpainted canvas inside a text box.

    The stored outline colour was estimated before inpainting; this is what the
    text will actually sit on. A 5px erosion keeps box edges (neighbouring text
    halos) from polluting the sample."""
    quad = np.asarray(dst_points, dtype=np.int32).reshape(-1, 2)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad, 1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    inner = img[mask > 0]
    if len(inner) == 0:
        return None
    return tuple(float(c) for c in inner.mean(axis=0))

def count_text_length(text: str) -> float:
    """Calculate text length, treating っッぁぃぅぇぉ as 0.5 characters"""
    half_width_chars = 'っッぁぃぅぇぉ'  
    length = 0.0
    for char in text.strip():
        if char in half_width_chars:
            length += 0.5
        else:
            length += 1.0
    return length

def quad_is_degenerate(pts: np.ndarray) -> bool:
    """Whether a 4-point text box has lost the area a perspective warp needs.

    Covers boxes folded onto a line or a point, self-crossing boxes whose signed
    area cancels, and non-finite coordinates. All of them make cv2.findHomography
    return a singular matrix, which cv2.warpPerspective then throws on.
    """
    quad = np.asarray(pts, dtype=np.float64).reshape(-1, 4, 2)
    if not np.isfinite(quad).all():
        return True
    x, y = quad[..., 0], quad[..., 1]
    area = np.abs(np.sum(x * np.roll(y, -1, axis=-1) - np.roll(x, -1, axis=-1) * y, axis=-1)) / 2
    return bool(area.max() < 1.0)


def clamp_quad_to_canvas(pts: np.ndarray, canvas_shape) -> np.ndarray:
    """Keep a destination text box on the canvas without flattening it.

    Clipping every corner on its own can pull an angled box flat against one
    canvas edge, and a flat box has no homography. When that would happen the
    overflow is kept instead and the warp simply clips what falls outside, the
    behavior from before the canvas clamp existed.
    """
    quad = np.asarray(pts, dtype=np.float64).reshape(-1, 4, 2)
    clipped = quad.copy()
    clipped[..., 0] = clipped[..., 0].clip(0, canvas_shape[1] - 1)
    clipped[..., 1] = clipped[..., 1].clip(0, canvas_shape[0] - 1)
    clipped = clipped.astype(np.int64)
    if quad_is_degenerate(clipped):
        logger.warning('Clamping this text box to the image would collapse it, keeping the overflow')
        return quad.astype(np.int64)
    return clipped


def _text_fits_horizontal(font_size: int, text: str, w: float, h: float, lang: str, hyphenate: bool, kinsoku: bool = False) -> bool:
    lines, widths = text_render.calc_horizontal(font_size, text, w, h, lang, hyphenate, kinsoku=kinsoku)
    if not lines:
        return True
    # calc_horizontal inflates its own width limit on overflow, so validate
    # the wrap against the real box instead of trusting what it was fed.
    return len(lines) * font_size <= h and max(widths) <= w


def _text_fits_vertical(font_size: int, text: str, w: float, h: float) -> bool:
    cols, heights = text_render.calc_vertical(font_size, text, h)
    if not cols:
        return True
    # Columns advance horizontally, glyphs flow vertically.
    return len(cols) * font_size <= w and max(heights) <= h


def shrink_font_size_to_fit(region: 'TextBlock', target_font_size: int, font_size_minimum: int,
                            hyphenate: bool = True, kinsoku: bool = False):
    """Largest font size <= target whose text fits the region's own box.

    Returns target_font_size when it already fits and None when the text does
    not fit even at font_size_minimum (the caller should grow the box instead).
    Never returns a size above target. Fitting is monotonic in font size, so a
    binary search finds the optimum in a handful of calc_* probes.
    """
    w, h = region.unrotated_size
    text = region.translation
    if w <= 0 or h <= 0 or not text:
        return None
    if region.vertical:
        def fits(fs): return _text_fits_vertical(fs, text, w, h)
    else:
        lang = getattr(region, "target_lang", "en_US")
        def fits(fs): return _text_fits_horizontal(fs, text, w, h, lang, hyphenate, kinsoku)
    if fits(target_font_size):
        return target_font_size
    if not fits(font_size_minimum):
        return None  # does not fit even at the smallest usable size
    lo, hi = font_size_minimum, target_font_size  # lo fits, hi does not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def resize_regions_to_font_size(img: np.ndarray, text_regions: List['TextBlock'], font_size_fixed: int, font_size_offset: int, font_size_minimum: int,
                                shrink_to_fit: bool = False, hyphenate: bool = True, kinsoku: bool = False):
    """
    Adjust text region size to accommodate font size and translated text length.
    
    Args:  
        img: Input image
        text_regions: List of text regions to process
        font_size_fixed: Fixed font size (overrides other font parameters)
        font_size_offset: Font size offset
        font_size_minimum: Minimum font size (-1 for auto-calculation)
        shrink_to_fit: Try shrinking the font into the original box before
            growing the box or relying on the warp to downscale the glyphs
        hyphenate: Whether hyphenation may be assumed when probing fit
        kinsoku: Whether CJK line breaks must obey Japanese line-start/line-end
            forbidden-character rules (禁則処理)

    Returns:  
        List of adjusted text region bounding boxes
    """    
    
    # Define minimum font size
    if font_size_minimum == -1:  
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)  
    # logger.debug(f'font_size_minimum {font_size_minimum}')  
    font_size_minimum = max(1, font_size_minimum)  

    dst_points_list = []  
    for region in text_regions: 
    
        # Store and validate original font size
        original_region_font_size = region.font_size  
        if original_region_font_size <= 0:  
            # logger.warning(f"Invalid original font size ({original_region_font_size}) for text '{region.translation}'. Using default value {font_size_minimum}.")  
            original_region_font_size = font_size_minimum

        # Determine target font size
        current_base_font_size = original_region_font_size  
        if font_size_fixed is not None:  
            target_font_size = font_size_fixed  
        else:  
            target_font_size = current_base_font_size + font_size_offset  

        target_font_size = max(target_font_size, font_size_minimum, 1)
        # print("-" * 50)
        # logger.debug(f"Calculated target font size: {target_font_size} for text '{region.translation}'")

        # Shrink-to-fit: prefer a smaller font inside the original box over
        # growing the box (or letting the warp silently downscale the glyphs).
        if shrink_to_fit:
            fitted = shrink_font_size_to_fit(region, target_font_size, font_size_minimum, hyphenate, kinsoku)
            if fitted is not None:
                target_font_size = fitted
                dst_points_list.append(region.min_rect)
                region.font_size = int(target_font_size)
                continue

        # Single-axis text box expansion
        single_axis_expanded = False
        dst_points = None
        
        if region.horizontal: 
            used_rows = len(region.texts)
            # logger.debug(f"Horizontal text - used rows: {used_rows}")
            
            line_text_list, _ = text_render.calc_horizontal(
                region.font_size,
                region.translation,
                max_width=region.unrotated_size[0],
                max_height=region.unrotated_size[1],
                language=getattr(region, "target_lang", "en_US"),
                kinsoku=kinsoku,
            )
            needed_rows = len(line_text_list)
            # logger.debug(f"Needed rows: {needed_rows}")                

            if needed_rows > used_rows:
                scale_x = ((needed_rows - used_rows) / used_rows) * 1 + 1
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    poly = affinity.scale(poly, xfact=scale_x, yfact=1.0, origin=(minx, miny))        
                
                    pts = np.array(poly.exterior.coords[:4])  
                    dst_points = rotate_polygons(  
                        region.center, pts.reshape(1, -1), -region.angle,  
                        to_int=False  
                    ).reshape(-1, 4, 2)  
                    dst_points = clamp_quad_to_canvas(dst_points, img.shape)
                    single_axis_expanded = True
                    # logger.debug(f"Successfully expanded horizontal text width: xfact={scale_x:.2f}")  
                except Exception as e:  
                    # logger.error(f"Failed to expand horizontal text: {e}")  
                    pass
                    
        if region.vertical:
            used_cols = len(region.texts)
            # logger.debug(f"Vertical text - used columns: {used_cols}")
            
            line_text_list, _ = text_render.calc_vertical(
                region.font_size, 
                region.translation, 
                max_height=region.unrotated_size[1],
            )
            needed_cols = len(line_text_list)
            # logger.debug(f"Needed columns: {needed_cols}") 
            if needed_cols > used_cols:
                scale_x = ((needed_cols - used_cols) / used_cols) * 1 + 1
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    poly = affinity.scale(poly, xfact=1.0, yfact=scale_x, origin=(minx, miny))                    
                    
                    pts = np.array(poly.exterior.coords[:4])  
                    dst_points = rotate_polygons(  
                        region.center, pts.reshape(1, -1), -region.angle,  
                        to_int=False  
                    ).reshape(-1, 4, 2)  
                    dst_points = clamp_quad_to_canvas(dst_points, img.shape)
                    single_axis_expanded = True
                    # logger.debug(f"Successfully expanded vertical text width: xfact={scale_x:.2f}")  
                except Exception as e:  
                    # logger.error(f"Failed to expand vertical text: {e}")  
                    pass

        # If single-axis expansion failed, use general scaling
        if not single_axis_expanded:
            # Calculate scaling factor based on text length ratio
            orig_text = getattr(region, "text_raw", region.text)
            char_count_orig = count_text_length(orig_text)
            char_count_trans = count_text_length(region.translation.strip())     
            length_ratio = 1.0

            if char_count_orig > 0 and char_count_trans > char_count_orig:  
                increase_percentage = (char_count_trans - char_count_orig) / char_count_orig
                font_increase_ratio = 1 + (increase_percentage * 0.3)
                font_increase_ratio = min(1.5, max(1.0, font_increase_ratio))
                # logger.debug(f"Translation is {increase_percentage:.2%} longer, font increase ratio: {font_increase_ratio:.2f}")
                target_font_size = int(target_font_size * font_increase_ratio)
                # logger.debug(f"Adjusted target font size: {target_font_size}")
                # Need greater bounding box scaling to accommodate larger font size and longer text
                target_scale = max(1, min(1 + increase_percentage * 0.3, 2))  # Possibly max(1, min(1 + (font_increase_ratio-1), 2))
                # logger.debug(f"Translation is longer than original and font increased, need larger bounding box scaling. Target scale factor: {target_scale:.2f}")
            # Short text box expansion is quite aggressive, in many cases short text boxes don't need expansion
            # elif char_count_orig > 0 and char_count_trans < char_count_orig:
            #     # Translation is shorter, increase font proportionally
            #     decrease_percentage = (char_count_orig - char_count_trans) / char_count_orig
            #     # Font increase ratio equals text reduction ratio
            #     font_increase_ratio = 1 + decrease_percentage
            #     # Limit font increase ratio to reasonable range, e.g., between 1.0 and 1.5
            #     font_increase_ratio = min(1.5, max(1.0, font_increase_ratio))
            #     logger.debug(f"Translation is {decrease_percentage:.2%} shorter than original, font increase ratio: {font_increase_ratio:.2f}")
            #     # Update target font size
            #     target_font_size = int(target_font_size * font_increase_ratio)
            #     logger.debug(f"Adjusted target font size: {target_font_size}")
            #     target_scale = 1.0  # No additional bounding box scaling needed
            #     logger.debug(f"Translation is shorter than original, no bounding box scaling applied, only font increase. Target scale factor: {target_scale:.2f}")            
            else:  
                target_scale = 1              
                # logger.debug(f"No length ratio scaling applied. Target scale factor: {target_scale:.2f}")   

            # Calculate final scaling factor
            font_size_scale = (((target_font_size - original_region_font_size) / original_region_font_size) * 0.4 + 1) if original_region_font_size > 0 else 1.0  
            # logger.debug(f"Font size ratio: ({target_font_size} / {original_region_font_size})")  
            final_scale = max(font_size_scale, target_scale)
            final_scale = max(1, min(final_scale, 1.1))  
            
            # logger.debug(f"Final scaling factor: {final_scale:.2f}")  

            # Scale bounding box if needed
            if final_scale > 1.001:  
                # logger.debug(f"Scaling bounding box: text='{region.translation}', scale={final_scale:.2f}")  
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])  
                     # Scale from the center  
                    poly = affinity.scale(poly, xfact=final_scale, yfact=final_scale, origin='center')  
                    scaled_unrotated_points = np.array(poly.exterior.coords[:4])  

                    dst_points = rotate_polygons(region.center, scaled_unrotated_points.reshape(1, -1), -region.angle, to_int=False).reshape(-1, 4, 2)  
                    dst_points = clamp_quad_to_canvas(dst_points, img.shape)
                    # logger.debug(f"Finished calculating scaled dst_points.")  

                except Exception as e:  
                    # logger.error(f"Error during scaling for text '{region.translation}': {e}. Using original min_rect.")  
                    dst_points = region.min_rect
            else:
                dst_points = region.min_rect

        # Store results and update font size
        dst_points_list.append(dst_points)  
        region.font_size = int(target_font_size)

    return dst_points_list

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False,
    stroke_width: Optional[float] = None,
    contrast_aware_stroke: bool = False,
    shrink_to_fit: bool = False,
    kinsoku: bool = False,
    ) -> np.ndarray:

    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum,
                                                  shrink_to_fit=shrink_to_fit, hyphenate=hyphenate, kinsoku=kinsoku)

    # TODO: Maybe remove intersections

    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            # set render_mask to 1 for the region that is inside dst_points
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border,
                     stroke_width=stroke_width, contrast_aware_stroke=contrast_aware_stroke,
                     kinsoku=kinsoku)
    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border,
    stroke_width=None,
    contrast_aware_stroke=False,
    kinsoku=False,
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border :
        bg = None

    if contrast_aware_stroke and bg is not None:
        # The outline was colour-matched to the pre-inpaint pixels, but the box
        # is painted over first: an outline matching the inpainted backdrop
        # separates nothing, and marginal fg-on-bg contrast then leaves the
        # text unreadable. Repaint the outline against the real backdrop.
        actual_bg = sample_canvas_bg(img, dst_points)
        if (actual_bg is not None
                and color_difference(bg, actual_bg) < 15
                and color_difference(fg, actual_bg) < 40):
            bg = (255, 255, 255) if np.mean(actual_bg) <= 127 else (0, 0, 0)

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    # A box flattened onto a line has no aspect ratio to match and no usable
    # homography, so there is nothing to composite onto the canvas.
    if norm_h[0] <= 0 or norm_v[0] <= 0:
        logger.warning(f'Skipping "{region.translation}": its text box collapsed to a line')
        return img
    r_orig = np.mean(norm_h / norm_v)
    if not np.isfinite(r_orig):
        logger.warning(f'Skipping "{region.translation}": its text box has no usable shape')
        return img

    # If configuration is set to non-automatic mode, use configuration to determine direction directly
    forced_direction = region._direction if hasattr(region, "_direction") else region.direction
    if forced_direction != "auto":
        if forced_direction in ["horizontal", "h"]:
            render_horizontally = True
        elif forced_direction in ["vertical", "v"]:
            render_horizontally = False
        else:
            render_horizontally = region.horizontal
    else:
        render_horizontally = region.horizontal

    #print(f"Region text: {region.text}, forced_direction: {forced_direction}, render_horizontally: {render_horizontally}")

    if render_horizontally:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]),
            round(norm_v[0]),
            region.alignment,
            region.direction == 'hl',
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
            stroke_width=stroke_width,
            kinsoku=kinsoku,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg,
            bg,
            line_spacing,
            stroke_width=stroke_width,
        )
    h, w, _ = temp_box.shape
    if h == 0 or w == 0:
        logger.warning(f'Skipping "{region.translation}": the rendered text has no size')
        return img
    r_temp = w / h

    # Padding below stretches this strip to the destination box's aspect ratio,
    # which inflates its area by up to max(r_orig, 1/r_orig). A tall narrow box
    # with a long translation pushed the padded canvas past 2 GiB and the warp
    # died on the allocation, so shrink the strip while it is still small.
    if r_orig > 0:
        padded_area = max(w * h, w * w / r_orig, h * h * r_orig)
        if padded_area > MAX_RENDER_CANVAS_PIXELS:
            scale = (MAX_RENDER_CANVAS_PIXELS / padded_area) ** 0.5
            temp_box = cv2.resize(temp_box, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            h, w, _ = temp_box.shape
            r_temp = w / h
            logger.debug(f'Shrinking the text canvas for "{region.translation}" by {scale:.3f} to {w}x{h}')

    # Extend temporary box so that it has same ratio as original
    box = None  
    #print("\n" + "="*50)  
    #print(f"Processing text: \"{region.get_translation_for_rendering()}\"")  
    #print(f"Text direction: {'Horizontal' if region.horizontal else 'Vertical'}")  
    #print(f"Font size: {region.font_size}, Alignment: {region.alignment}")  
    #print(f"Target language: {region.target_lang}")      
    #print(f"Region horizontal: {region.horizontal}")  
    #print(f"Starting image adjustment: r_temp={r_temp}, r_orig={r_orig}, h={h}, w={w}")  
    if region.horizontal:  
        #print("Processing HORIZONTAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0  
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:  
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [h_ext:h_ext+h, :w] = [{h_ext}:{h_ext+h}, 0:{w}]")  
                # Columns fully filled, rows centered
                box[h_ext:h_ext+h, 0:w] = temp_box  
            else:  
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()  
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) // 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [:, :w] = [0:{h}, 0:{w}]")  
         
                # The line is full, and there should be no empty columns on the left side of the text. Otherwise, when multiple text boxes are aligned on the left, the translated text cannot be aligned. Common scenarios: borderless comics, comic postscript.  
                # When there are bubbles on the current page, it can be changed to center: box[0:h, w_ext:w_ext+w] = temp_box, requiring more accurate bubble detection. But not changing it doesn't have much impact.
                box[0:h, 0:w] = temp_box  
            else:  
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()  
    else:  
        #print("Processing VERTICAL region")  
        
        if r_temp > r_orig:   
            #print(f"Case: r_temp({r_temp}) > r_orig({r_orig}) - Need vertical padding")  
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0   
            #print(f"Calculated h_ext = {h_ext}")  
            
            if h_ext >= 0:   
                #print(f"Creating new box with dimensions: {h + h_ext * 2}x{w}")  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, 0:w] = [0:{h}, 0:{w}]")  
                # The rows are full, and there should be no empty lines above the text; otherwise, when multiple text boxes have their top edges aligned, the text cannot be aligned. Common scenario: borderless comics, CG. 
                # When there are bubbles on the current page, it can be changed to center: box[h_ext:h_ext+h, 0:w] = temp_box, requiring more accurate bubble detection.
                box[0:h, 0:w] = temp_box  
            else:   
                #print("h_ext < 0, using original temp_box")  
                box = temp_box.copy()   
        else:   
            #print(f"Case: r_temp({r_temp}) <= r_orig({r_orig}) - Need horizontal padding")  
            w_ext = int((h * r_orig - w) / 2)  
            #print(f"Calculated w_ext = {w_ext}")  
            
            if w_ext >= 0:  
                #print(f"Creating new box with dimensions: {h}x{w + w_ext * 2}")  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                #print(f"Placing temp_box at position [0:h, w_ext:w_ext+w] = [0:{h}, {w_ext}:{w_ext+w}]") 
                # Rows are fully filled, columns are centered
                box[0:h, w_ext:w_ext+w] = temp_box  
            else:   
                #print("w_ext < 0, using original temp_box")  
                box = temp_box.copy()   
    #print(f"Final box dimensions: {box.shape if box is not None else 'None'}")  

    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    #src_pts[:, 0] = np.clip(np.round(src_pts[:, 0]), 0, enlarged_w * 2)
    #src_pts[:, 1] = np.clip(np.round(src_pts[:, 1]), 0, enlarged_h * 2)

    # warpPerspective's INTER_LINEAR neither pre-filters shrinks nor keeps thin
    # strokes: a strip padded taller/wider than the destination box is warped
    # straight down, which aliases the glyphs. Pre-shrink with INTER_AREA (area
    # averaging) so the final warp only ever upscales.
    dst_quad = np.asarray(dst_points, dtype=np.float32).reshape(-1, 2)
    canvas_area = float(box.shape[0] * box.shape[1])
    dst_area = cv2.contourArea(dst_quad)
    if 0 < dst_area < canvas_area:
        scale = (dst_area / canvas_area) ** 0.5
        new_w = max(1, int(round(box.shape[1] * scale)))
        new_h = max(1, int(round(box.shape[0] * scale)))
        box = cv2.resize(box, (new_w, new_h), interpolation=cv2.INTER_AREA)
        src_points = np.array([[0, 0], [new_w, 0], [new_w, new_h], [0, new_h]], dtype=np.float32)
        logger.debug(f'Shrinking the text canvas for "{region.translation}" to {new_w}x{new_h} before warping it down to its box')

    if len(dst_quad) == 4 and dst_area > 0:
        # Exactly four correspondences: findHomography's RANSAC only earns its
        # keep with more than four and merely adds random sampling here.
        M = cv2.getPerspectiveTransform(src_points, dst_quad)
    else:
        M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if M is None or not np.isfinite(M).all():
        logger.warning(f'Skipping "{region.translation}": no homography maps its text box onto the image')
        return img
    rgba_region = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, w, h = cv2.boundingRect(dst_points.astype(np.int32))
    canvas_region = rgba_region[y:y+h, x:x+w, :3]
    mask_region = rgba_region[y:y+h, x:x+w, 3:4].astype(np.float32) / 255.0
    img[y:y+h, x:x+w] = np.clip((img[y:y+h, x:x+w].astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 0, 255).astype(np.uint8)
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, line_spacing=line_spacing, size_tol=1.2, original_img=original_img, downscale_constraint=0.8,disable_font_border=disable_font_border)

async def dispatch_eng_render_pillow(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/NotoSansMonoCJK-VF.ttf.ttc')
    text_render.set_font(font_path)

    return render_textblock_list_eng_pillow(font_path, img_canvas, text_regions, original_img=original_img, downscale_constraint=0.95)
