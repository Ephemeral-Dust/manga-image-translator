"""Offline smoke tests for the standalone server.

The pipeline engine is built once per module (CPU, no model preload because
MIMG_MODELS_TTL is set nonzero) and every request uses detector=none, so these
tests never download or load model weights.
"""

import io
import os

# Must be set before the app's lifespan runs; the engine reads them at startup.
os.environ["MIMG_WARMUP"] = "1"
os.environ["MIMG_USE_GPU"] = "0"
os.environ["MIMG_MODELS_TTL"] = "3600"
os.environ.pop("MIMG_API_KEY", None)

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
    # generic would pick paddle detection (a model download); an explicit
    # detector in the request must still win.
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
    assert "manga" in resp.json()["detail"]


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
