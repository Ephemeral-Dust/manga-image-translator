# standalone — POST an image, get a typeset image back

A minimal FastAPI server around the `manga_translator` pipeline for external
apps (browser extensions, game translators, anything HTTP). It runs the pipeline
**in-process** — no gateway, no worker subprocess, no queue — so it is one
Python file to embed or run, not a distributed system.

The bundled `server/` package keeps the multi-worker gateway and streaming
progress protocol; this package keeps only what a client integration needs.
Endpoint names and payload shapes match `server/main.py`, so client code written
against the original server works here unchanged.

## Run

```powershell
python -m standalone                     # 127.0.0.1:5003, GPU if present
python -m standalone --port 8000 --api-key secret
```

Model weights download into `./models` on first use (several GB for the default
stages). The first start is slow; later starts reuse the cache.

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| POST | `/translate/image` | JSON `TranslateRequest` | PNG |
| POST | `/translate/with-form/image` | multipart `image` file + `config` form field | PNG |
| POST | `/translate/json` | JSON `TranslateRequest` | text regions + inpainted backgrounds (Ichigo-style) |
| POST | `/translate/with-form/json` | multipart | same JSON |
| POST | `/translate/batch/image` | JSON `BatchTranslateRequest` (`images` list) | zip of PNGs |
| POST | `/translate/batch/with-form/image` | multipart, repeated `images` files | zip of PNGs |
| GET | `/health` | — | `{status, ready, stage, busy, gpu}` |
| GET | `/config-schema` | — | JSON Schema of the `config` object |

`image` accepts a `data:` URI, an `http(s)` URL, bare base64, or raw bytes (in
JSON that means base64-encoded). `config` is any subset of the pipeline config —
see `/config-schema` or `python -m manga_translator config-help`.

Example:

```bash
curl -X POST http://127.0.0.1:5003/translate/image \
  -H 'Content-Type: application/json' \
  -d '{"image": "data:image/png;base64,iVBOR...", "config": {"translator": {"target_lang": "ENG"}}}' \
  -o out.png
```

CORS is open (`*`) so browser extensions can call it directly. If `MIMG_API_KEY`
(env) or `--api-key` (CLI) is set, every translate endpoint requires the
matching `X-API-Key` header; `/health` and `/config-schema` stay open.

## Profiles

`profile` picks a preset of config defaults; anything explicitly in `config`
overrides the preset.

- `manga` — no overrides: the upstream defaults are already manga-tuned.
- `generic` — for screenshots/UI/real-world text: stock detector with more
  sensitive thresholds (picks up small/missed text at the cost of merging
  multi-line blocks), lighter mask dilation, LTR rendering. Based on signage
  tests 2026-11-07; still tune `standalone/profiles.py` against more images.

## Concurrency

One pipeline instance, one translation at a time — the same effective behavior
as the original server with a single worker. Extra requests queue (they do not
get rejected), so a browser extension firing several translations at once is
safe. `/health` reports `busy` and the current pipeline `stage`.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `MIMG_HOST` / `MIMG_PORT` | `127.0.0.1` / `5003` | bind address |
| `MIMG_USE_GPU` | `auto` | `1`/`0` to force; auto detects CUDA/XPU/MPS |
| `MIMG_MODEL_DIR` | `./models` | weights cache |
| `MIMG_FONT_PATH` | built-in | font file for rendering |
| `MIMG_MODELS_TTL` | `0` | `0` keeps models resident; N unloads after N idle seconds |
| `MIMG_API_KEY` | unset | require `X-API-Key` |
| `MIMG_OFFLINE` / `MIMG_CACHE_ONLY` | off | no downloads / cache-only |
| `MIMG_WARMUP` | `1` | `0` starts without building the pipeline (translate returns 503) |
| `MIMG_VERBOSE` | off | debug artifacts under `./result` |
| `MIMG_LOG_LEVEL` | `INFO` | logging level |

## What this server does *not* have (vs `server/`)

- Multi-instance gateway, registration nonce, load-balanced executor pool.
- `/translate/bytes` and the `*/stream` chunked progress protocol (the pipeline
  progress is visible via `/health` instead).
- `debug_folder` in the JSON response.

Embedding: `from standalone.server import app` and mount it in any FastAPI/ASGI
app, or build your own `Engine(params)` if you want the pipeline without HTTP.
