# Standalone Image-Translation Server — API Reference

> Written for AI agents and programmatic clients: everything needed to call
> this service without reading the source. Facts verified 2026-11-07 against
> commit `e2ce54d` + auto-profile + render-endpoint changes. If this file and
> `/openapi.json` disagree, the live server wins.

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
- Render-only mode (`/render/*`, section 14): post back text boxes +
  translations from your own vision model and the server only inpaints and
  typesets them.

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
| POST | `/render/image` | `RenderRequest` (JSON, section 14) | `image/png` |
| POST | `/render/with-form/image` | multipart: `image` file + `regions`, `config`, `profile` string fields | `image/png` |
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
| `"generic"` | For screenshots/UI/real-world text: same models, but detector thresholds `text_threshold 0.4, box_threshold 0.5, unclip_ratio 1.5` (catches small/missed text), gamma-corrected detection input (`det_gamma_correct true`, helps washed-out photos), line-split tolerance `merge_split_gamma 0.2` (neighbouring text lines become separate regions instead of one merged block), lighter inpaint mask (`mask_dilation_offset 4`), left-to-right rendering (`render.rtl false`), contrast-aware outlines (`render.contrast_aware_stroke true`, see section 7 knobs), and shrink-to-fit font sizing (`render.shrink_to_fit true`: overflowing text shrinks its font into the original box instead of stretching it). |
| `"auto"` | Not a preset — a per-image cascade: a millisecond colorfulness check (Hasler-Süsstrunk on a 64px thumbnail, threshold 25.0) orders two full attempts: grayscale → `manga` then `generic`; colorful → `generic` then `manga`. The **first preset that detects ≥1 text region serves**; the other never runs. If neither detects anything, the last-attempted preset serves (unchanged image). Cost: a wrong first guess doubles latency for that image. Known weakness: color manga (webtoons) start on `generic` and keep its result even when `manga` would typeset better — fallback triggers only on zero detections. |

- Explicit profiles never cascade — exactly one attempt, no fallback.
- On `/render/*` endpoints there is nothing to detect, so `auto` cannot
  cascade: it simply takes the preset its color cue would try first, and only
  render-side knobs (e.g. `render.rtl`) apply. See section 14.
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
- `angle`: detected text angle in degrees. `prob`: OCR confidence 0–1 — the
  area-weighted mean of the region's own textlines' confidence. Set
  `config.ocr.prob` (default off) to drop regions below a confidence floor.
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
| `render.stroke_width` | `null` | Outline thickness as a fraction of font size (default renderer; built-in default ≈`0.07`). Set `0` for no outline, `0.15` for a heavier one. |
| `render.contrast_aware_stroke` | `false` (`true` in `generic`) | If the outline colour would blend into the inpainted backdrop and the text itself is only marginally visible on it, the outline is repainted black or white so the text stays readable. |
| `render.shrink_to_fit` | `false` (`true` in `generic`) | When the translation doesn't fit its box, step the font size down until it does instead of growing the box or warping the text smaller. The box is only grown when even the minimum font size wouldn't fit. |
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
| 422 | render request: degenerate box, region outside the image, or no resolvable translation (section 14) | `{"detail":"..."}` plain string |
| 422 | render `regions` not a JSON array / bad region objects | `{"detail":"..."}` (form) or FastAPI validation array (JSON body) |
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

from standalone.render import RenderRegion
ctx = await engine.render(pil_image, config, [RenderRegion(...)])  # render-only path
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
| `standalone/render.py` | render-only path: `RenderRegion` model, region→TextBlock conversion, mask synthesis, `render_regions()` |
| `standalone/profiles.py` | preset dicts, `colorfulness()`, `auto_order()`, threshold constant |
| `standalone/to_json.py` | JSON response models (`Translation`, `TranslationResponse`) |
| `standalone/images.py` | input coercion: data URI / URL / base64 / bytes → PIL |
| `standalone/__main__.py` | CLI flags → `MIMG_*` env translation |
| `manga_translator/config.py` | the full `Config` model behind `config` (enums, defaults, descriptions) |
| `test/test_standalone.py` | 31 offline tests encoding the contract above |
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

## 14. Render endpoints (client-side detection + translation)

For clients that detect/OCR/translate text themselves (e.g. a vision model in
a browser extension) and only need the server's inpainting + typesetting:

| Method | Path | Request | 200 response |
|---|---|---|---|
| POST | `/render/image` | `RenderRequest` (JSON) | `image/png` |
| POST | `/render/with-form/image` | multipart: `image` file + `regions`, `config`, `profile` string fields (`regions` = JSON array as string) | `image/png` |

### RenderRequest (JSON body)

```json
{
  "image":   "data:image/png;base64,iVBORw0KGgo...",   // required, same forms as TranslateRequest
  "regions": [ { ... } ],                              // required, see below
  "config":  { ... },                                  // optional, default {}
  "profile": "generic"                                 // optional; see section 5 note
}
```

### Region objects

Field names intentionally match the `translations[]` objects `/translate/json`
returns (section 6), so a client can fill translations in and post the array
straight back:

| Field | Required | Notes |
|---|---|---|
| `minX/minY/maxX/maxY` | yes | Pixel coords in the **original** image (clamped to bounds; a box with no overlap → 422). If the first pass used `upscale_ratio`, `/translate/json` coords are upscaled — convert back before posting. |
| `translation` | usually | String to typeset. If omitted, `text[config.translator.target_lang]` (default `ENG`) is used. Neither present → 422. |
| `text` | optional | Language-keyed source-text map, same shape as section 6. A non-target-language entry is used for mask fitting (deciding which pixels get painted out). |
| `angle` | optional (0) | Text angle in degrees — carry over from `/translate/json` unchanged. |
| `prob` | optional (1.0) | OCR confidence; unused when rendering. |
| `text_color` | optional | `{"fg":[r,g,b],"bg":[r,g,b]}`. Omit and the server estimates both from the region's pixels. `bg` is the outline color; when `render.contrast_aware_stroke` is on (`generic` profile default) the server re-checks it against the inpainted backdrop and repaints the outline black/white if it would vanish into it. |
| `is_bulleted_list` | optional | Accepted and ignored — recomputed from the text. |
| `background` | optional | Accepted and ignored — inpainting always regenerates it from the input image. |

### What the server does with them

Per region: inpaint (paint whatever was inside the box out, honoring
`inpainter` config) → typeset `translation` into the box using profile/config
render settings (font, size fit, RTL order, hyphenation...). Detection, OCR
and translation never run on these endpoints, so `detector`, `ocr` and
`translator.translator` config keys have no effect here;
`translator.target_lang` still decides the translation fallback lookup and the
render script direction.

### Round-trip example (server detects, your model translates, server renders)

```python
import base64, requests

BASE = "http://127.0.0.1:5003"
raw = open("page.jpg", "rb").read()
data_uri = "data:image/jpeg;base64," + base64.b64encode(raw).decode()

# 1. Server detects + OCRs, no server-side translation
regions = requests.post(f"{BASE}/translate/json", json={
    "image": data_uri,
    "config": {"translator": {"translator": "none"}},
}).json()["translations"]

# 2. Your model translates each region
for r in regions:
    src = next(iter(r["text"]))
    r["translation"] = my_translate(r["text"][src], "ENG")

# 3. Server inpaints + typesets
png = requests.post(f"{BASE}/render/image", json={
    "image": data_uri,
    "regions": regions,
}).content
open("out.png", "wb").write(png)
```

Behavior details:

- Empty `regions` array → 200 with the input image returned unchanged (no
  models touched).
- Degenerate box (`maxX <= minX` or `maxY <= minY`) → 422.
- No resolvable translation (no `translation`, no target-language key in
  `text`) → 422 naming the target language.
- Boxes partially outside the image are clamped; fully outside → 422.
- Typeset glyph size starts at the box's **short** side (box height for
  horizontal lines, box width for vertical columns) and is then fitted by the
  render config (`render.font_size` fixes it outright; `render.font_size_offset`
  / `font_size_minimum` adjust). A box too small for the translation is widened
  to fit rather than shrinking the font, so rendered text can extend past the
  posted box.
- A region whose box collapses to a line, or whose text cannot be mapped back
  onto it, is skipped with a server-side warning — the other regions still
  render and the response is still 200. There is no per-region success flag.
- `profile` selects render-side defaults only; `auto` cannot cascade here
  (section 5). `X-MIMG-Profile` is sent on every render response.
- Output is a full-size PNG, same as the translate image endpoints.
