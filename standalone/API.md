# Standalone Image-Translation Server — API Reference

> Written for AI agents and programmatic clients: everything needed to call
> this service without reading the source. Facts verified 2026-11-07 against
> commit `e2ce54d` + auto-profile changes. If this file and `/openapi.json`
> disagree, the live server wins.

## 1. What this service does

One HTTP call in, one HTTP response out. It takes an image containing text in
any language and returns the same image with that text **translated and
re-typeset in place** (original text painted out, translation rendered into
the freed space with matched position and colors).

Internally: detect text regions → OCR → translate → inpaint background →
render. All stages run locally by default — no third-party API keys required
(default translator `sugoi` is a local model).

Capabilities an agent can rely on:

- Input: any image format PIL reads (PNG/JPG/WebP/...), passed as data URI,
  URL, or raw bytes.
- Output: typeset PNG, and/or structured JSON (per-region boxes, source text,
  translation, colors, inpainted crop).
- Batch: many images in one call.
- Behavior presets (`profile`) adapt the manga-tuned pipeline to generic
  images (screenshots, signage, UI), and `auto` picks a preset per image.
- Full pipeline tuning via a partial `config` object in the request.

It does **not**: stream progress, do OCR-only mode without rendering (use
`/translate/json` for text extraction), or manage a task queue — requests are
serialized inside the server.

## 2. Run the server

```powershell
# from the repo root (Windows)
run.bat                          # wrapper: git pull + venv python + GPU default
venv\Scripts\python.exe -m standalone            # equivalent, GPU if present
venv\Scripts\python.exe -m standalone --no-gpu   # force CPU
```

Default bind: `http://127.0.0.1:5003`. Swagger UI at `GET /docs`.

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--host` | `MIMG_HOST` | `127.0.0.1` | bind address |
| `--port` | `MIMG_PORT` | `5003` | bind port |
| `--api-key` | `MIMG_API_KEY` | unset (no auth) | require `X-API-Key` header |
| `--model-dir` | `MIMG_MODEL_DIR` | `./models` | weights cache |
| `--font-path` | `MIMG_FONT_PATH` | bundled font | font used for rendering |
| `--use-gpu` / `--no-gpu` | `MIMG_USE_GPU=1/0` | auto-detect | force GPU/CPU |
| `--offline` | `MIMG_OFFLINE` | off | never download weights |
| `--cache-only` | `MIMG_CACHE_ONLY` | off | only use already-downloaded weights |
| `--verbose` | `MIMG_VERBOSE` | off | write debug artifacts to `./result` |
| — | `MIMG_WARMUP=0` | warmup on | skip model preload at startup (`/health.ready` stays false until first use) |

First start downloads model weights into `./models` (several GB total; the
default detector/OCR/inpaint set is ~700 MB). Startup answers `/health`
immediately with `ready: false` until the pipeline is loaded.

## 3. Endpoints

| Method | Path | Request | 200 response |
|---|---|---|---|
| POST | `/translate/image` | `TranslateRequest` (JSON) | `image/png` |
| POST | `/translate/with-form/image` | multipart: `image` file + `config`, `profile` string fields | `image/png` |
| POST | `/translate/json` | `TranslateRequest` (JSON) | `application/json` (section 6) |
| POST | `/translate/with-form/json` | multipart (same fields) | `application/json` |
| POST | `/translate/batch/image` | `BatchTranslateRequest` (JSON) | `application/zip` (`result_1.png`, `result_2.png`, ... in request order) |
| POST | `/translate/batch/with-form/image` | multipart, repeated `images` file fields (+ `config`, `profile`) | `application/zip` |
| GET | `/health` | — | `{"status":"ok","ready":bool,"stage":str,"busy":bool,"gpu":bool}` |
| GET | `/config-schema` | — | JSON Schema of the full `config` object |
| GET | `/docs`, `/openapi.json` | — | Swagger UI / OpenAPI spec |

`/health` and `/config-schema` never require an API key.

### TranslateRequest (JSON body)

```json
{
  "image":  "data:image/png;base64,iVBORw0KGgo...",   // required
  "config": { ... },                                   // optional, default {}
  "profile": "auto"                                    // optional; see section 5
}
```

- `image` (string): `data:<mime>;base64,...` URI, `http(s)://` URL (fetched
  server-side), bare base64, or raw image bytes (JSON string with unescaped
  bytes is not valid — use a data URI or form upload).
- `config`: any subset of the pipeline config (section 7). Validated against
  the upstream schema; bad keys/values → 422.
- `profile`: preset name, `null`, `"manga"`, `"generic"`, or `"auto"`.
  Omitted or `null` ≡ `"manga"`. Unknown name → 422 listing known names.

### BatchTranslateRequest

```json
{"images": ["data:image/png;base64,...", "https://..."], "config": {}, "profile": "auto"}
```

## 4. The core call (examples)

```bash
# minimal — manga defaults
curl -X POST http://127.0.0.1:5003/translate/image \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/png;base64,iVBORw0..."}' \
  --output result.png
```

```javascript
// browser extension (CORS is wide open; response is PNG)
const b64 = /* base64 of the image */;
const res = await fetch("http://127.0.0.1:5003/translate/image", {
  method: "POST",
  headers: { "Content-Type": "application/json", "X-API-Key": key }, // key only if server has one
  body: JSON.stringify({ image: `data:image/png;base64,${b64}`, profile: "auto" }),
});
const profileUsed = res.headers.get("x-mimg-profile"); // see section 5
const blob = await res.blob();
```

```javascript
// JSON extraction without downloading the whole rendered image
const { translations } = await (await fetch("http://127.0.0.1:5003/translate/json", {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ image: dataUri, profile: "generic" }),
})).json();
```

```python
# Python: upload a file via form
import requests
requests.post(
    "http://127.0.0.1:5003/translate/with-form/image",
    files={"image": open("page.png", "rb")},
    data={"profile": "generic"},
).content  # PNG bytes
```

## 5. Profiles (`profile` field)

A profile supplies config defaults. Precedence: **request `config` > profile >
upstream defaults** — any key you set in `config` always wins.

| Profile | Behavior |
|---|---|
| `"manga"` (default) | Upstream defaults, manga-tuned (B&W panels, RTL panel flow, higher detection thresholds). |
| `"generic"` | For screenshots/UI/real-world text: same models, but detector thresholds `text_threshold 0.4, box_threshold 0.5, unclip_ratio 1.5` (catches small/missed text; merges multi-line blocks), lighter inpaint mask (`mask_dilation_offset 4`), left-to-right rendering (`render.rtl false`). |
| `"auto"` | Not a preset — a per-image cascade: a millisecond colorfulness check (Hasler-Süsstrunk on a 64px thumbnail, threshold 25.0) orders two full attempts: grayscale → `manga` then `generic`; colorful → `generic` then `manga`. The **first preset that detects ≥1 text region serves**; the other never runs. If neither detects anything, the last-attempted preset serves (unchanged image). Cost: a wrong first guess doubles latency for that image. Known weakness: color manga (webtoons) start on `generic` and keep its result even when `manga` would typeset better — fallback triggers only on zero detections. |

- Explicit profiles never cascade — exactly one attempt, no fallback.
- Every translate response carries **`X-MIMG-Profile`** naming the preset that
  actually served (CORS-exposed; readable from browser JS).
- Batch responses list profiles per image, comma-joined, deduplicated in order
  of first use: `"generic"` = all images served by generic; `"manga,generic"`
  = manga served first, at least one image later served by generic. Exact
  per-image pairing is not encoded.

## 6. JSON response (`/translate/json`, `/translate/with-form/json`)

```json
{
  "translations": [
    {
      "minX": 12, "minY": 40, "maxX": 96, "maxY": 61,
      "angle": 0,
      "prob": 0.87,
      "is_bulleted_list": false,
      "text_color": {"fg": [255, 255, 255], "bg": [0, 0, 0]},
      "text": {"JPN": "境内", "ENG": "The precincts"},
      "background": "data:image/png;base64,..."
    }
  ]
}
```

- `minX/minY/maxX/maxY`: region bounding box in image pixels.
- `angle`: detected text angle in degrees. `prob`: detector confidence 0–1.
- `text`: language-code-keyed map — the source-language key holds detected
  text, the target-language key holds the translation.
- `text_color.fg/bg`: estimated foreground/background RGB triplets.
- `background`: the inpainted (text-removed) crop of the region as PNG data
  URI — useful for custom re-rendering clients (e.g. browser extensions).
- No detections → `{"translations": []}` with status 200.

Image endpoints always return a full-size **PNG**. If nothing was detected,
the input image comes back unchanged (200, not an error). To distinguish
"nothing found" from success, use the JSON endpoint or count regions.

## 7. The `config` object

`config` is a partial of the upstream `Config` model
(`manga_translator/config.py`). Full machine-readable schema:
`GET /config-schema`; human-readable with defaults:
`python -m manga_translator config-help`. Unknown keys and bad enum values
→ 422.

Verified upstream defaults: detector `default`, OCR `48px`, translator `sugoi`
(local), target language `ENG`, inpainter `lama_large`, `render.rtl true`.

Knobs a client is most likely to touch:

| Path in config | Default | Notes |
|---|---|---|
| `translator.translator` | `"sugoi"` | Local: `sugoi`, `nllb`, `m2m100`, `mbart50`, `qwen2`, `sakura`, `offline`, `none` (skip translation), `original` (no-op passthrough). Cloud (need keys in server env/`.env`): `deepl`, `papago`, `caiyun`, `youdao`, `baidu`, `chatgpt`, `deepseek`, `gemini`, `groq`, `custom_openai`, ... Full list in `/config-schema`. |
| `translator.target_lang` | `"ENG"` | Free-form language code/name as accepted by the chosen translator. |
| `detector.detector` | `"default"` | Also: `dbconvnext`, `ctd`, `craft`, `paddle`, `none` (treat the whole image as one text region). |
| `ocr.ocr` | `"48px"` | Also: `32px`, `48px_ctc`, `mocr`. |
| `inpainter.inpainter` | `"lama_large"` | Also: `lama_mpe`, `sd`, `none`, `original`. |
| `detector.text_threshold` / `box_threshold` / `unclip_ratio` | `0.5` / `0.7` / `2.3` | Detection sensitivity; `generic` profile lowers these (see section 5). |
| `render.rtl` | `true` | Region ordering; set `false` for LTR content. |
| `colorization.colorizer` | `"none"` | Optional B&W→color stage (`mc2`). |
| `upscale.upscale_ratio` | off | Optional upscale before OCR for small text. |

Override example — translate to Japanese with a different detector:

```json
{
  "image": "data:image/png;base64,...",
  "profile": "generic",
  "config": {
    "translator": {"target_lang": "JPN"},
    "detector": {"detector": "ctd"}
  }
}
```

`config` keys override the profile key-by-key; unmentioned keys keep the
profile's values.

## 8. Errors

| Status | When | Body |
|---|---|---|
| 401 | server has an API key set and `X-API-Key` is missing/wrong | `{"detail":"Invalid or missing X-API-Key"}` |
| 422 | `config` not valid JSON / not an object | `{"detail":"Invalid config: ..."}` |
| 422 | `config` fails the schema | FastAPI validation array (`detail[].loc` names the offending path) |
| 422 | unknown profile | `{"detail":"Unknown profile 'x'. Known profiles: generic, manga, auto"}` |
| 422 | image undecodable / bad data URI | pydantic validation array on `image` |
| 503 | server started with `MIMG_WARMUP=0` and pipeline not built yet | `{"detail":"Server is not ready..."}` |
| 500 | pipeline finished without a result image | `{"detail":"..."}` |

A wrong API key is 401 even when the request body is also invalid (auth runs
first). 422 bodies from FastAPI are arrays of validation errors, not plain
strings — parse accordingly.

## 9. Auth, CORS, limits

- Auth: optional. `--api-key KEY` (or `MIMG_API_KEY`) makes every translate
  endpoint require header `X-API-Key: KEY`. `/health`, `/config-schema`,
  `/docs` stay open. Timing-safe comparison, but there is **no rate limiting**.
- CORS: `*` origins/methods/headers on all endpoints; `X-MIMG-Profile` is in
  `Access-Control-Expose-Headers` so browser JS can read it.
- Concurrency: one pipeline instance; translations are serialized behind an
  async lock. Extra requests **queue** (no timeout, no rejection) — clients
  should set generous HTTP timeouts. `/health` reports `busy` and current
  `stage` (`loading`/`queued`/`idle`).
- Latency (measured 2026-11-07, CPU `--no-gpu`, warm, default stages):
  ~8–19 s/image for 1000–2000 px inputs; `auto` roughly doubles when the
  first preset finds nothing. GPU is markedly faster. First request after
  startup pays model load (tens of seconds on CPU).
- No request size limit is enforced by the server itself; very large images
  cost proportional time.

## 10. Embedding without HTTP

```python
from standalone.server import app          # mount in any ASGI server
from standalone.server import Engine, build_params

engine = Engine(build_params())            # pipeline without HTTP layer
ctx = await engine.translate(pil_image, config)   # ctx.result = PIL image,
                                                  # ctx.text_regions = regions
```

## 11. Verifying / testing the service

```powershell
# offline unit tests (no weights needed)
venv\Scripts\python.exe -m pytest test/test_standalone.py -q

# end-to-end smoke: runs every endpoint once against a live server,
# prints the served profile, saves all outputs next to the image
venv\Scripts\python.exe try_server.py path\to\image.png [--profile auto] [--url http://127.0.0.1:5003]
```

## 12. Where to look in the code (only when this doc isn't enough)

| File | Contents |
|---|---|
| `standalone/server.py` | endpoints, request models, profile resolution + auto cascade (`resolve_profiles`, `_translate_cascade`), engine/lock, CORS, auth |
| `standalone/profiles.py` | preset dicts, `colorfulness()`, `auto_order()`, threshold constant |
| `standalone/to_json.py` | JSON response models (`Translation`, `TranslationResponse`) |
| `standalone/images.py` | input coercion: data URI / URL / base64 / bytes → PIL |
| `standalone/__main__.py` | CLI flags → `MIMG_*` env translation |
| `manga_translator/config.py` | the full `Config` model behind `config` (enums, defaults, descriptions) |
| `test/test_standalone.py` | 19 offline tests encoding the contract above |
| `try_server.py` | reference client |

## 13. Known limitations

- `auto` is a pixel heuristic, not a classifier: colorful manga (webtoons) get
  generic-first treatment (section 5).
- `generic` merges multi-line blocks into one region (measured on signage;
  single-image evidence).
- No progress streaming — only `/health.stage`. For the original streaming
  protocol, see the bundled `server/` package.
- Cloud translators need keys in the server's environment (`.env`);
  the default `sugoi` does not.
- Rendering clamps text to detected boxes (fork fix `d021d66`): very dense
  text may be scaled down rather than overflowing.
