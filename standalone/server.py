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
from . import profiles
from .profiles import AUTO_PROFILE, PROFILES, apply_profile
from .render import RenderRegion, render_regions
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

    async def render(self, image, config, regions) -> object:
        async with self.lock:
            self.stage, self.finished = "queued", False
            try:
                return await render_regions(self.manga, image, config, regions)
            finally:
                self.stage, self.finished = "idle", True


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
        merged = apply_profile(raw, None if profile == AUTO_PROFILE else profile)
    except KeyError:
        known = ", ".join(sorted(PROFILES)) + ", auto"
        raise HTTPException(422, f"Unknown profile {profile!r}. Known profiles: {known}")
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


def zip_pngs(contexts: list, profiles_used: Optional[List[str]] = None) -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, ctx in enumerate(contexts, 1):
            archive.writestr(f"result_{i}.png", png_bytes(ctx))
    headers = {"Content-Disposition": 'attachment; filename="results.zip"'}
    if profiles_used is not None:
        headers["X-MIMG-Profile"] = served_profile_header(profiles_used)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers=headers,
    )


class TranslateRequest(BaseModel):
    image: str | bytes
    """URL, data URI, bare base64, or raw image bytes"""
    config: dict = Field(default_factory=dict)
    """Partial pipeline config; unset fields come from the profile, then upstream defaults"""
    profile: Optional[str] = None
    """Config defaults preset: "manga" (default), "generic", or "auto" (cascade; see profiles.py)"""


class BatchTranslateRequest(BaseModel):
    images: List[str | bytes]
    config: dict = Field(default_factory=dict)
    profile: Optional[str] = None


class RenderRequest(BaseModel):
    image: str | bytes
    """URL, data URI, bare base64, or raw image bytes (original scale: region coords must match)"""
    regions: List[RenderRegion]
    """Boxes + text to typeset; same shape as the translations array from /translate/json"""
    config: dict = Field(default_factory=dict)
    """Partial pipeline config; unset fields come from the profile, then upstream defaults"""
    profile: Optional[str] = None
    """Config defaults preset: "manga" (default), "generic", or "auto" (picks by color cue, no cascade)"""


def get_engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(503, "Server is not ready: pipeline not initialized (MIMG_WARMUP=0?)")
    return engine


async def check_api_key(request: Request, x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    expected = getattr(request.app.state, "api_key", None)
    if expected and x_api_key != expected:
        raise HTTPException(401, "Invalid or missing X-API-Key")


def resolve_profiles(profile: Optional[str], image) -> List[str]:
    """Ordered preset names to try for this request's profile."""
    if profile is None:
        return ["manga"]
    if profile == AUTO_PROFILE:
        return list(profiles.auto_order(image))
    if profile in profiles.PROFILES:
        return [profile]
    known = ", ".join(sorted(profiles.PROFILES)) + ", auto"
    raise HTTPException(422, f"Unknown profile {profile!r}. Known profiles: {known}")


def served_profile_header(names) -> str:
    """Comma-joined unique profile names, in order of first use."""
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return ",".join(seen)


async def _translate_cascade(engine, pil_image, raw_config: dict, profile: Optional[str]):
    """Run presets until one detects text; auto starts from the image's color cue.

    Returns (ctx, served_profile). An explicit profile gets exactly one attempt.
    """
    names = resolve_profiles(profile, pil_image)
    for name in names:
        config = build_config(raw_config, name)
        ctx = await engine.translate(pil_image, config)
        if ctx.text_regions:
            return ctx, name
    return ctx, names[-1]  # nothing detected anywhere; last attempt's (unchanged) result serves


def _parse_raw_config(raw) -> dict:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"config is not valid JSON: {exc}")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise HTTPException(422, "config must be a JSON object")
    return raw


def _parse_regions(raw) -> List[RenderRegion]:
    """Validate a regions payload (JSON string from multipart forms)."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"regions is not valid JSON: {exc}")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise HTTPException(422, "regions must be a JSON array of region objects")
    try:
        return [RenderRegion.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise HTTPException(422, f"Invalid regions: {exc}")


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
    # lets browser JS read which profile actually served the request
    expose_headers=["X-MIMG-Profile"],
)


async def _translate_one(request: Request, image, config_data, profile):
    engine = get_engine(request)
    pil_image = await to_pil_image(image)
    raw = _parse_raw_config(config_data)
    return await _translate_cascade(engine, pil_image, raw, profile)


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
    ctx, served = await _translate_one(request, data.image, data.config, data.profile)
    return Response(
        png_bytes(ctx),
        media_type="image/png",
        headers={"X-MIMG-Profile": served},
    )


@app.post("/translate/with-form/image", response_class=Response, response_description="translated image, PNG")
async def translate_image_form(
    request: Request,
    image: UploadFile = File(...),
    config: str = Form("{}"),
    profile: Optional[str] = Form(None),
    _: None = Depends(check_api_key),
):
    ctx, served = await _translate_one(request, await image.read(), config, profile)
    return Response(
        png_bytes(ctx),
        media_type="image/png",
        headers={"X-MIMG-Profile": served},
    )


@app.post("/translate/json", response_model=TranslationResponse,
          response_description="text regions with inpainted backgrounds, Ichigo-inspired structure")
async def translate_json(data: TranslateRequest, request: Request, response: Response, _: None = Depends(check_api_key)):
    ctx, served = await _translate_one(request, data.image, data.config, data.profile)
    response.headers["X-MIMG-Profile"] = served
    return to_translation(ctx)


@app.post("/translate/with-form/json", response_model=TranslationResponse)
async def translate_json_form(
    request: Request,
    response: Response,
    image: UploadFile = File(...),
    config: str = Form("{}"),
    profile: Optional[str] = Form(None),
    _: None = Depends(check_api_key),
):
    ctx, served = await _translate_one(request, await image.read(), config, profile)
    response.headers["X-MIMG-Profile"] = served
    return to_translation(ctx)


async def _translate_batch(engine, pil_images, raw: dict, profile):
    """Per-image cascade so auto can pick a starting preset for each image."""
    contexts, served = [], []
    for pil_image in pil_images:
        ctx, name = await _translate_cascade(engine, pil_image, raw, profile)
        contexts.append(ctx)
        served.append(name)
    return contexts, served


@app.post("/translate/batch/image", response_class=Response, response_description="zip of PNG results")
async def translate_batch_image(data: BatchTranslateRequest, request: Request, _: None = Depends(check_api_key)):
    engine = get_engine(request)
    images = [await to_pil_image(img) for img in data.images]
    raw = _parse_raw_config(data.config)
    contexts, served = await _translate_batch(engine, images, raw, data.profile)
    return zip_pngs(contexts, served)


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
    raw = _parse_raw_config(config)
    contexts, served = await _translate_batch(engine, pil_images, raw, profile)
    return zip_pngs(contexts, served)


async def _render_one(request: Request, image, regions, config_data, profile):
    """Shared body for the two /render endpoints. Returns (pil_result, served_profile)."""
    engine = get_engine(request)
    pil_image = await to_pil_image(image)
    raw = _parse_raw_config(config_data)
    # No cascade here: detection already happened client-side, so "auto" can
    # only pick which preset's render knobs apply (the color-cue-first one).
    served = resolve_profiles(profile, pil_image)[0]
    config = build_config(raw, served)
    if not regions:
        return pil_image, served  # nothing to inpaint or typeset
    try:
        ctx = await engine.render(pil_image, config, regions)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return ctx.result, served


def _png_response(image, served: str) -> Response:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"X-MIMG-Profile": served})


@app.post("/render/image", response_class=Response, response_description="typeset image, PNG")
async def render_image(data: RenderRequest, request: Request, _: None = Depends(check_api_key)):
    result, served = await _render_one(request, data.image, data.regions, data.config, data.profile)
    return _png_response(result, served)


@app.post("/render/with-form/image", response_class=Response, response_description="typeset image, PNG")
async def render_image_form(
    request: Request,
    image: UploadFile = File(...),
    regions: str = Form(...),
    config: str = Form("{}"),
    profile: Optional[str] = Form(None),
    _: None = Depends(check_api_key),
):
    data = await image.read()
    result, served = await _render_one(request, data, _parse_regions(regions), config, profile)
    return _png_response(result, served)
