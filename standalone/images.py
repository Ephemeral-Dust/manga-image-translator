"""Image decoding for request payloads: raw bytes, data URIs, bare base64, or URLs."""

import builtins
import io
import re
from base64 import b64decode
from typing import Union

import requests
from PIL import Image
from fastapi import HTTPException

URL_RE = re.compile(r"^https?://", re.IGNORECASE)
DATA_URI_RE = re.compile(r"^data:image/.+;base64,")

# Bound on wall-clock time for remote image fetches, in seconds.
FETCH_TIMEOUT = 30


async def to_pil_image(image: Union[str, bytes]) -> Image.Image:
    """Turn a request image field into a PIL image.

    Accepts raw image bytes, a data:image URI, a bare base64 string, or an
    http(s) URL to fetch. Raises HTTPException(422) when decoding fails.
    """
    try:
        if isinstance(image, builtins.bytes):
            return Image.open(io.BytesIO(image))
        if DATA_URI_RE.match(image):
            payload = b64decode(image.split(",", 1)[1])
            return Image.open(io.BytesIO(payload))
        if URL_RE.match(image):
            response = requests.get(image, timeout=FETCH_TIMEOUT)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        # Not a URL and not prefixed: last resort is bare base64.
        return Image.open(io.BytesIO(b64decode(image)))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not load image: {exc}")
