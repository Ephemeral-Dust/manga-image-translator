"""Standalone "POST an image, get a typeset image back" server.

Runs the manga_translator pipeline in-process (like mode/local.py does for files)
instead of the gateway + subprocess worker + pickle hop of the bundled server/.
Endpoint names and payload shapes mirror server/main.py so existing client code
transfers unchanged.

Environment variables (all optional):
    MIMG_USE_GPU      force GPU on/off (default: auto-detect)
    MIMG_MODEL_DIR    weights directory (default: ./models next to the pipeline)
    MIMG_FONT_PATH    path to a font file used for rendering
    MIMG_MODELS_TTL   seconds models stay loaded; 0 keeps them resident (default: 0)
    MIMG_KERNEL_SIZE  post-process kernel size (default: 3)
    MIMG_WARMUP       0 to skip building the pipeline at startup (default: 1)
    MIMG_API_KEY      if set, translate endpoints require X-API-Key to match
    MIMG_OFFLINE      1 to forbid downloading model weights
    MIMG_CACHE_ONLY   1 to use only already-downloaded weights
    MIMG_VERBOSE      1 for extra debug artifacts under ./result
    MIMG_LOG_LEVEL    python logging level (default: INFO)
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import zipfile
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError

from manga_translator import Config, MangaTranslator

from .images import to_pil_image
from .profiles import PROFILES, apply_profile
from .to_json import TranslationResponse, to_translation


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _gpu_available() -> bool:
    try:
        return bool(torch.cuda.is_available() or torch.xpu.is_available()
                    or torch.backends.mps.is_available())
    except Exception:
        return False


def _gpu_setting() -> bool:
    """MIMG_USE_GPU=1/0 forces it; auto (or unset) detects hardware."""
    raw = os.environ.get("MIMG_USE_GPU")
    if raw is None or raw.strip().lower() == "auto":
        return _gpu_available()
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_params() -> dict:
    """Init params for MangaTranslator from environment overrides."""
    params = {
        "use_gpu": _gpu_setting(),
        "models_ttl": int(os.environ.get("MIMG_MODELS_TTL", "0")),
        # parse_init_params() reads kernel_size unconditionally; CLI default is 3.
        "kernel_size": int(os.environ.get("MIMG_KERNEL_SIZE", "3")),
        "log_level": os.environ.get("MIMG_LOG_LEVEL", "INFO"),
        "verbose": _env_flag("MIMG_VERBOSE", False),
        "offline": _env_flag("MIMG_OFFLINE", False),
        "cache_only": _env_flag("MIMG_CACHE_ONLY", False),
        "batch_size": 1,
    }
    for env_name, param_name in (("MIMG_MODEL_DIR", "model_dir"), ("MIMG_FONT_PATH", "font_path")):
        value = os.environ.get(env_name)
        if value:
            params[param_name] = value
    return params


class _ProgressHook:
    """Usable both as `hook(state, finished)` and `await hook(state, finished)`.

    MangaTranslator calls registered hooks both ways (plain call for
    pre-translation hooks, awaited inside _report_progress), and share.py gets
    away with an async hook only by leaking un-awaited coroutines. This object
    updates state synchronously and is trivially awaitable.
    """

    def __init__(self, engine: "Engine"):
        self._engine = engine

    def __call__(self, state: str, finished: bool):
        # May run on a pipeline worker thread; plain attribute writes are safe
        # under the GIL and only feed /health.
        self._engine.stage = state
        self._engine.finished = bool(finished)
        return self

    def __await__(self):
        return iter(())


class Engine:
    """Holds the pipeline instance and serializes translations through it.

    One pipeline in memory means one translation at a time, the same behavior
    the bundled server had with a single worker. Callers queue instead of
    getting an error.
    """

    def __init__(self, params: dict):
        self.manga = MangaTranslator(params)
        self.lock = asyncio.Lock()
        self.stage = "loading"
        self.finished = False
        self.manga.add_progress_hook(_ProgressHook(self))

    async def translate(self, image, config) -> object:
        async with self.lock:
            self.stage, self.finished = "queued", False
            try:
                return await self.manga.translate(image, config)
            finally:
                self.stage, self.finished = "idle", True

    async def translate_batch(self, images, config) -> list:
        async with self.lock:
            results = []
            for image in images:
                self.stage, self.finished = "queued", False
                try:
                    results.append(await self.manga.translate(image, config))
                finally:
                    self.stage, self.finished = "idle", True
            return results


def build_config(raw, profile: Optional[str]) -> Config:
    """Validate a request-supplied config dict against the pipeline Config model.

    `raw` may be a dict or a JSON string (multipart form fields). The named
    profile contributes defaults; anything the request sets wins.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"config is not valid JSON: {exc}")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise HTTPException(422, "config must be a JSON object")
    try:
        merged = apply_profile(raw, profile)
    except KeyError:
        raise HTTPException(422, f"Unknown profile {profile!r}. Known profiles: {', '.join(sorted(PROFILES))}")
    try:
        return Config.model_validate(merged)
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid config: {exc}")


def png_bytes(ctx) -> bytes:
    if ctx.result is None:
        raise HTTPException(500, "Pipeline finished without a result image")
    buf = io.BytesIO()
    ctx.result.save(buf, format="PNG")
    return buf.getvalue()


def zip_pngs(contexts: list) -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, ctx in enumerate(contexts, 1):
            archive.writestr(f"result_{i}.png", png_bytes(ctx))
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="results.zip"'},
    )


class TranslateRequest(BaseModel):
    image: str | bytes
    """URL, data URI, bare base64, or raw image bytes"""
    config: dict = Field(default_factory=dict)
    """Partial pipeline config; unset fields come from the profile, then upstream defaults"""
    profile: Optional[str] = None
    """Named preset for the config defaults (see profiles.py)"""


class BatchTranslateRequest(BaseModel):
    images: List[str | bytes]
    config: dict = Field(default_factory=dict)
    profile: Optional[str] = None


def get_engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(503, "Server is not ready: pipeline not initialized (MIMG_WARMUP=0?)")
    return engine


async def check_api_key(request: Request, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    expected = getattr(request.app.state, "api_key", None)
    if expected and x_api_key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.api_key = os.environ.get("MIMG_API_KEY") or None
    if _env_flag("MIMG_WARMUP", True):
        app.state.engine = Engine(build_params())
    else:
        app.state.engine = None
    yield
    app.state.engine = None


app = FastAPI(
    title="Manga Image Translator (standalone)",
    description="POST an image, get a translated and typeset image back.",
    lifespan=lifespan,
)

# Browser extensions and web apps call this cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _translate_one(request: Request, image, config_data, profile):
    engine = get_engine(request)
    pil_image = await to_pil_image(image)
    config = build_config(config_data, profile)
    return await engine.translate(pil_image, config)


@app.get("/health")
async def health(request: Request):
    engine = getattr(request.app.state, "engine", None)
    return {
        "status": "ok",
        "ready": engine is not None,
        "stage": engine.stage if engine else "not-initialized",
        "busy": bool(engine and engine.lock.locked()),
        "gpu": engine.manga.using_gpu if engine else None,
    }


@app.get("/config-schema")
async def config_schema():
    """JSON Schema of the accepted config object (upstream Config model)."""
    return Config.model_json_schema()


@app.post("/translate/image", response_class=Response, response_description="translated image, PNG")
async def translate_image(data: TranslateRequest, request: Request, _: None = Depends(check_api_key)):
    ctx = await _translate_one(request, data.image, data.config, data.profile)
    return Response(png_bytes(ctx), media_type="image/png")


@app.post("/translate/with-form/image", response_class=Response, response_description="translated image, PNG")
async def translate_image_form(
    request: Request,
    image: UploadFile = File(...),
    config: str = Form("{}"),
    profile: Optional[str] = Form(None),
    _: None = Depends(check_api_key),
):
    ctx = await _translate_one(request, await image.read(), config, profile)
    return Response(png_bytes(ctx), media_type="image/png")


@app.post("/translate/json", response_model=TranslationResponse,
          response_description="text regions with inpainted backgrounds, Ichigo-inspired structure")
async def translate_json(data: TranslateRequest, request: Request, _: None = Depends(check_api_key)):
    ctx = await _translate_one(request, data.image, data.config, data.profile)
    return to_translation(ctx)


@app.post("/translate/with-form/json", response_model=TranslationResponse)
async def translate_json_form(
    request: Request,
    image: UploadFile = File(...),
    config: str = Form("{}"),
    profile: Optional[str] = Form(None),
    _: None = Depends(check_api_key),
):
    ctx = await _translate_one(request, await image.read(), config, profile)
    return to_translation(ctx)


@app.post("/translate/batch/image", response_class=Response, response_description="zip of PNG results")
async def translate_batch_image(data: BatchTranslateRequest, request: Request, _: None = Depends(check_api_key)):
    engine = get_engine(request)
    images = [await to_pil_image(img) for img in data.images]
    config = build_config(data.config, data.profile)
    contexts = await engine.translate_batch(images, config)
    return zip_pngs(contexts)


@app.post("/translate/batch/with-form/image", response_class=Response, response_description="zip of PNG results")
async def translate_batch_image_form(
    request: Request,
    images: List[UploadFile] = File(...),
    config: str = Form("{}"),
    profile: Optional[str] = Form(None),
    _: None = Depends(check_api_key),
):
    engine = get_engine(request)
    pil_images = [await to_pil_image(await img.read()) for img in images]
    config = build_config(config, profile)
    contexts = await engine.translate_batch(pil_images, config)
    return zip_pngs(contexts)
