"""Named config profiles.

A profile supplies default config values for a class of images so callers do not
need to know stage names. Values explicitly set in the request config always win
over the profile.
"""

from typing import Any, Dict, Optional

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
        },
        "mask_dilation_offset": 4,
        "render": {"rtl": False},
    },
}


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
