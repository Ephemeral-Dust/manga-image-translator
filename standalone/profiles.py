"""Named config profiles.

A profile supplies default config values for a class of images so callers do not
need to know stage names. Values explicitly set in the request config always win
over the profile.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# The upstream Config defaults are already tuned for manga, so that profile is empty.
PROFILES: Dict[str, Dict[str, Any]] = {
    "manga": {},
    # First guesses for screenshots / UI / game text. The earlier paddle+mocr+nllb
    # preset found 0 regions on real sign photos and merged lines on synthetic
    # ones, so it is replaced by the stock detector with more sensitive
    # thresholds — verified on signage 2026-11-07 (single image per setting,
    # Low confidence; still recovers missed small text but merges multi-line
    # blocks into one region).
    "generic": {
        "detector": {
            "text_threshold": 0.4,
            "box_threshold": 0.5,
            "unclip_ratio": 1.5,
            # Washed-out photos defeat the DBNet-style detector; gamma
            # correction normalises contrast before detection (existing flag,
            # effect on real signage not yet A/B-verified)
            "det_gamma_correct": True,
            # Split neighbouring sign lines into separate regions instead of one
            # merged block. Default keeps lines merged while their pitch (corner
            # distance) is below 1.5x font size — almost any two lines of a sign
            # qualify; 0.2 splits at 1.2x while wrapped lines with tight leading
            # (< ~1.15x pitch) stay one region. Tunable per request via
            # detector.merge_split_gamma / merge_char_gap_tolerance(2).
            "merge_split_gamma": 0.2,
        },
        "mask_dilation_offset": 4,
        "render": {
            "rtl": False,
            # Clients supply colours estimated from the *original* pixels (or
            # none at all); after inpainting the backdrop is surrounding
            # texture, so an outline that matched the old pixels can vanish
            # into it. Re-check against the actual canvas before typesetting.
            "contrast_aware_stroke": True,
            # Sign text that does not fit should shrink its font, not stretch
            # the box into neighbouring artwork (or be silently downscaled by
            # the perspective warp together with the padding).
            "shrink_to_fit": True,
        },
    },
}

# profile="auto" is a server-side cascade, not a preset: try one profile, and
# rerun with the other when the first one detects no text. The starting point
# comes from a millisecond-scale pixel statistic — manga is overwhelmingly
# black-and-white, so a colorful input probably wants the generic preset first.
COLORFULNESS_THRESHOLD = 25.0
AUTO_PROFILE = "auto"
AUTO_ORDER_GRAYSCALE = ("manga", "generic")
AUTO_ORDER_COLOR = ("generic", "manga")


def colorfulness(image: Image.Image) -> float:
    """Hasler & Susstrunk colorfulness metric on a 64px thumbnail.

    Exactly 0 for grayscale; measured 2026-11-07: synthetic B&W renders 0-16,
    real-world color sign photos 36-44.
    """
    thumb = image.convert("RGB").copy()
    thumb.thumbnail((64, 64))
    arr = np.asarray(thumb, dtype=np.float64)
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    return float(np.hypot(rg.var(), yb.var()) + 0.611 * np.hypot(rg.mean(), yb.mean()))


def auto_order(image: Image.Image) -> Tuple[str, str]:
    """Cascade order for profile="auto" on this image."""
    if colorfulness(image) >= COLORFULNESS_THRESHOLD:
        return AUTO_ORDER_COLOR
    return AUTO_ORDER_GRAYSCALE


def apply_profile(config: Dict[str, Any], profile: Optional[str]) -> Dict[str, Any]:
    """Merge the request config on top of the named profile's defaults.

    Raises KeyError for an unknown profile name.
    """
    if not profile:
        return config
    defaults = PROFILES[profile]  # raises KeyError for unknown names
    return _merge(defaults, config)


def _merge(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
