"""Offline smoke tests for the standalone server.

The pipeline engine is built once per module (CPU, no model preload because
MIMG_MODELS_TTL is set nonzero) and every request uses detector=none, so these
tests never download or load model weights.
"""

import io
import json
import os
import zipfile

# Must be set before the app's lifespan runs; the engine reads them at startup.
os.environ["MIMG_WARMUP"] = "1"
os.environ["MIMG_USE_GPU"] = "0"
os.environ["MIMG_MODELS_TTL"] = "3600"
os.environ.pop("MIMG_API_KEY", None)

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from standalone.server import app

# detector none -> no textlines -> pipeline returns the input unchanged,
# which keeps the whole test run weight-free.
PASSTHROUGH_CONFIG = {"detector": {"detector": "none"}}


def make_png(width=8, height=8, color=(255, 255, 255)):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def data_uri(raw: bytes) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def decode_png(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw))


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert body["gpu"] is False


def test_config_schema(client):
    resp = client.get("/config-schema")
    assert resp.status_code == 200
    schema = resp.json()
    assert "translator" in schema["properties"]
    assert "detector" in schema["properties"]


def test_passthrough_image(client):
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png(12, 7)),
        "config": PASSTHROUGH_CONFIG,
    })
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["x-mimg-profile"] == "manga"  # profile omitted == manga
    result = decode_png(resp.content)
    assert result.size == (12, 7)


def test_passthrough_with_form(client):
    resp = client.post(
        "/translate/with-form/image",
        files={"image": ("page.png", make_png(9, 5), "image/png")},
        data={"config": '{"detector": {"detector": "none"}}'},
    )
    assert resp.status_code == 200, resp.text
    assert decode_png(resp.content).size == (9, 5)


def test_translate_json_empty(client):
    resp = client.post("/translate/json", json={
        "image": data_uri(make_png()),
        "config": PASSTHROUGH_CONFIG,
    })
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"translations": []}


def test_profile_manga_is_a_noop(client):
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png(4, 4)),
        "config": PASSTHROUGH_CONFIG,
        "profile": "manga",
    })
    assert resp.status_code == 200, resp.text
    assert decode_png(resp.content).size == (4, 4)


def test_explicit_config_beats_profile(client):
    # generic tunes the stock detector's thresholds; an explicit detector in
    # the request must still win.
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png(6, 6)),
        "config": PASSTHROUGH_CONFIG,
        "profile": "generic",
    })
    assert resp.status_code == 200, resp.text
    assert decode_png(resp.content).size == (6, 6)


def test_unknown_profile_rejected(client):
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png()),
        "config": PASSTHROUGH_CONFIG,
        "profile": "does-not-exist",
    })
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "does-not-exist" in detail
    assert "manga" in detail and "generic" in detail and "auto" in detail


def test_bad_config_rejected(client):
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png()),
        "config": {"detector": {"detector": "not-a-detector"}},
    })
    assert resp.status_code == 422


def test_bad_image_rejected(client):
    resp = client.post("/translate/image", json={
        "image": "data:image/png;base64,not-base64-at-all!!!",
        "config": PASSTHROUGH_CONFIG,
    })
    assert resp.status_code == 422


def test_api_key_enforced():
    saved_engine = getattr(app.state, "engine", None)
    os.environ["MIMG_API_KEY"] = "sekrit"
    try:
        with TestClient(app) as test_client:
            resp = test_client.get("/health")
            assert resp.status_code == 200  # health stays open

            resp = test_client.post("/translate/image", json={
                "image": data_uri(make_png()),
                "config": PASSTHROUGH_CONFIG,
            })
            assert resp.status_code == 401

            resp = test_client.post(
                "/translate/image",
                json={"image": data_uri(make_png()), "config": PASSTHROUGH_CONFIG},
                headers={"X-API-Key": "sekrit"},
            )
            assert resp.status_code == 200, resp.text
    finally:
        os.environ.pop("MIMG_API_KEY", None)
        # The nested TestClient's lifespan rewrote the shared app state (key
        # and engine); undo it so later tests reusing the module client work.
        app.state.api_key = None
        app.state.engine = saved_engine


# --- auto profile -------------------------------------------------------------

def make_color_png(size=(8, 8)):
    """Bright red square: colorfulness far above the auto threshold."""
    arr = np.zeros((size[1], size[0], 3), np.uint8)
    arr[..., 0] = 255
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_auto_order_grayscale_vs_color():
    from standalone.profiles import auto_order

    assert auto_order(Image.open(io.BytesIO(make_png()))) == ("manga", "generic")
    assert auto_order(Image.fromarray(
        np.full((8, 8, 3), 128, np.uint8), "RGB")) == ("manga", "generic")
    assert auto_order(Image.open(io.BytesIO(make_color_png()))) == ("generic", "manga")


def test_auto_grayscale_cascade_order(client, monkeypatch):
    from types import SimpleNamespace

    from standalone.server import Engine

    seen_rtl = []

    async def never_detects(self, image, config):
        seen_rtl.append(config.render.rtl)
        return SimpleNamespace(text_regions=[], result=Image.new("RGB", image.size))

    monkeypatch.setattr(Engine, "translate", never_detects)
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png()),
        "config": PASSTHROUGH_CONFIG,
        "profile": "auto",
    })
    assert resp.status_code == 200, resp.text
    assert seen_rtl == [True, False]  # manga tried first, then generic
    assert resp.headers["x-mimg-profile"] == "generic"  # last attempt serves when none detect


def test_auto_color_cascade_order(client, monkeypatch):
    from types import SimpleNamespace

    from standalone.server import Engine

    seen_rtl = []

    async def never_detects(self, image, config):
        seen_rtl.append(config.render.rtl)
        return SimpleNamespace(text_regions=[], result=Image.new("RGB", image.size))

    monkeypatch.setattr(Engine, "translate", never_detects)
    resp = client.post("/translate/image", json={
        "image": data_uri(make_color_png()),
        "config": PASSTHROUGH_CONFIG,
        "profile": "auto",
    })
    assert resp.status_code == 200, resp.text
    assert seen_rtl == [False, True]  # generic tried first, then manga
    assert resp.headers["x-mimg-profile"] == "manga"  # last attempt serves when none detect


def test_auto_cascade_stops_at_first_detection(client, monkeypatch):
    from types import SimpleNamespace

    from standalone.server import Engine

    seen_rtl = []

    async def always_detects(self, image, config):
        seen_rtl.append(config.render.rtl)
        return SimpleNamespace(text_regions=[object()], result=Image.new("RGB", image.size))

    monkeypatch.setattr(Engine, "translate", always_detects)
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png()),
        "config": PASSTHROUGH_CONFIG,
        "profile": "auto",
    })
    assert resp.status_code == 200, resp.text
    assert seen_rtl == [True]  # manga detects first; generic never runs
    assert resp.headers["x-mimg-profile"] == "manga"

    seen_rtl.clear()
    resp = client.post("/translate/image", json={
        "image": data_uri(make_color_png()),
        "config": PASSTHROUGH_CONFIG,
        "profile": "auto",
    })
    assert resp.status_code == 200, resp.text
    assert seen_rtl == [False]  # generic detects first; manga never runs
    assert resp.headers["x-mimg-profile"] == "generic"


def test_auto_real_pipeline_passthrough(client):
    # Unpatched pipeline with detector=none: neither preset detects anything,
    # so the last-attempted one serves. On grayscale that is generic.
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png()),
        "config": PASSTHROUGH_CONFIG,
        "profile": "auto",
    })
    assert resp.status_code == 200, resp.text
    assert decode_png(resp.content).size == (8, 8)
    assert resp.headers["x-mimg-profile"] == "generic"


def test_explicit_profile_header_has_no_fallback(client):
    resp = client.post("/translate/image", json={
        "image": data_uri(make_png()),
        "config": PASSTHROUGH_CONFIG,
        "profile": "generic",
    })
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-mimg-profile"] == "generic"


def test_json_endpoint_reports_profile(client):
    resp = client.post("/translate/json", json={
        "image": data_uri(make_png()),
        "config": PASSTHROUGH_CONFIG,
        "profile": "generic",
    })
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"translations": []}
    assert resp.headers["x-mimg-profile"] == "generic"


def test_batch_reports_per_image_profiles(client):
    resp = client.post("/translate/batch/image", json={
        "images": [data_uri(make_color_png()), data_uri(make_png())],
        "config": PASSTHROUGH_CONFIG,
        "profile": "auto",
    })
    assert resp.status_code == 200, resp.text
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        assert sorted(archive.namelist()) == ["result_1.png", "result_2.png"]
    # Passthrough detects nothing, so each image is served by the last preset
    # tried: the color one ends on manga, the grayscale one on generic.
    assert resp.headers["x-mimg-profile"] == "manga,generic"


# --- /render endpoints: client detects + translates, server inpaints + typesets.
# Inpainter "none" keeps these weight-free; the default renderer is PIL-based.

RENDER_CONFIG = {"inpainter": {"inpainter": "none"}}


def render_region(**over):
    region = {
        "minX": 2, "minY": 2, "maxX": 60, "maxY": 26,
        "angle": 0, "prob": 0.9,
        "text": {"JPN": "hello there", "ENG": "Hello"},
    }
    region.update(over)
    return region


def test_render_image_basic(client):
    img = make_png(64, 48)
    resp = client.post("/render/image", json={
        "image": data_uri(img),
        "regions": [render_region(translation="Hello")],
        "config": RENDER_CONFIG,
    })
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["x-mimg-profile"] == "manga"
    result = decode_png(resp.content)
    assert result.size == (64, 48)
    original = np.asarray(Image.open(io.BytesIO(img)).convert("RGB"))
    assert not np.array_equal(np.asarray(result.convert("RGB")), original)  # text was painted


def test_render_falls_back_to_text_map(client):
    # Flow B shape: post the /translate/json array back with only the text map
    # filled (translation lives under the target language key).
    body = {
        "image": data_uri(make_png(64, 48)),
        "regions": [render_region()],
        "config": RENDER_CONFIG,
    }
    resp_fallback = client.post("/render/image", json=body)
    assert resp_fallback.status_code == 200, resp_fallback.text
    resp_explicit = client.post("/render/image", json={
        **body, "regions": [render_region(translation="Hello")],
    })
    assert resp_explicit.status_code == 200, resp_explicit.text
    # The pipeline is deterministic, so the same rendered string must produce
    # identical bytes: proves text["ENG"] served as the translation.
    assert resp_fallback.content == resp_explicit.content


def test_render_missing_translation_422(client):
    resp = client.post("/render/image", json={
        "image": data_uri(make_png(64, 48)),
        "regions": [render_region(text={"JPN": "yatta"})],
        "config": RENDER_CONFIG,
    })
    assert resp.status_code == 422, resp.text
    assert "ENG" in resp.text


def test_render_degenerate_box_422(client):
    resp = client.post("/render/image", json={
        "image": data_uri(make_png(64, 48)),
        "regions": [render_region(maxX=2, translation="x")],
        "config": RENDER_CONFIG,
    })
    assert resp.status_code == 422
    assert "degenerate" in resp.text


def test_render_region_outside_image_422(client):
    resp = client.post("/render/image", json={
        "image": data_uri(make_png(8, 8)),
        "regions": [render_region(minX=100, minY=100, maxX=120, maxY=120, translation="x")],
        "config": RENDER_CONFIG,
    })
    assert resp.status_code == 422
    assert "overlap" in resp.text


def test_render_clamps_partially_outside_box(client):
    resp = client.post("/render/image", json={
        "image": data_uri(make_png(64, 48)),
        "regions": [render_region(minX=-10, maxY=100, translation="Hello")],
        "config": RENDER_CONFIG,
    })
    assert resp.status_code == 200, resp.text
    assert decode_png(resp.content).size == (64, 48)


def test_render_empty_regions_returns_input_unchanged(client):
    img = make_png(10, 6, color=(1, 2, 3))
    resp = client.post("/render/image", json={"image": data_uri(img), "regions": []})
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-mimg-profile"] == "manga"
    result = np.asarray(decode_png(resp.content).convert("RGB"))
    original = np.asarray(Image.open(io.BytesIO(img)).convert("RGB"))
    assert np.array_equal(result, original)


def test_render_accepts_json_roundtrip_fields(client):
    # Everything /translate/json emits must post back cleanly, including the
    # fields the render path ignores (background, is_bulleted_list).
    resp = client.post("/render/image", json={
        "image": data_uri(make_png(64, 48)),
        "regions": [render_region(
            translation="Hello",
            is_bulleted_list=True,
            text_color={"fg": [255, 255, 255], "bg": [0, 0, 0]},
            background="data:image/png;base64,ignored",
        )],
        "config": RENDER_CONFIG,
    })
    assert resp.status_code == 200, resp.text


def test_render_with_form(client):
    resp = client.post(
        "/render/with-form/image",
        files={"image": ("page.png", make_png(64, 48), "image/png")},
        data={
            "regions": json.dumps([render_region(translation="Hello")]),
            "config": json.dumps(RENDER_CONFIG),
        },
    )
    assert resp.status_code == 200, resp.text
    assert decode_png(resp.content).size == (64, 48)


def test_render_form_bad_regions_json_422(client):
    resp = client.post(
        "/render/with-form/image",
        files={"image": ("page.png", make_png(8, 8), "image/png")},
        data={"regions": "not json"},
    )
    assert resp.status_code == 422
    assert "regions is not valid JSON" in resp.text


def test_render_auto_picks_single_profile_by_color(client):
    # No cascade is possible on render (detection already happened client-side),
    # so auto just picks the color-cue-first preset for the render knobs.
    resp = client.post("/render/image", json={
        "image": data_uri(make_color_png((64, 48))),
        "regions": [render_region(translation="Hello")],
        "config": RENDER_CONFIG,
        "profile": "auto",
    })
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-mimg-profile"] == "generic"  # color image -> generic first


def test_render_unknown_profile_422(client):
    resp = client.post("/render/image", json={
        "image": data_uri(make_png(8, 8)),
        "regions": [],
        "profile": "nope",
    })
    assert resp.status_code == 422
    assert "Unknown profile" in resp.text
