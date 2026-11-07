#!/usr/bin/env python3
"""Send one image to a running standalone server and save everything it gives back.

Usage:
    python try_server.py path\\to\\image.png [--url URL] [--api-key KEY]
                          [--profile manga] [--no-generic] [--outdir DIR]

With no file argument, the path in IMAGE_PATH below is used.
Runs: /health, /translate/image, /translate/with-form/image,
      /translate/json (chosen profile + generic for comparison),
      /translate/batch/image. Results are saved for review; nothing else
      in this script is a test suite check.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile

# Paste the path to the image you want to test here (a CLI argument wins).
IMAGE_PATH = r""

DEFAULT_URL = "http://127.0.0.1:5003"
TIMEOUT = 600


def http(url, method="GET", data=None, headers=None):
    req = urllib.request.Request(url, method=method, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        sys.exit(f"cannot reach {url} ({exc.reason}) - start the server first (run.bat)")


def call(label, url, method="POST", data=None, headers=None):
    start = time.perf_counter()
    status, body = http(url, method, data, headers)
    elapsed = time.perf_counter() - start
    if status // 100 != 2:
        print(f"[FAIL] {label}: HTTP {status} {body.decode('utf-8', 'replace')[:400]}")
        return None
    print(f"[ OK ] {label}: {elapsed:.1f}s, {len(body)} bytes")
    return body


def save(directory, name, data):
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(data)
    print(f"       -> {path}")
    return path


def multipart(filename, file_bytes, fields):
    boundary = "----tryServer" + uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
        f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def print_regions(body):
    regions = json.loads(body).get("translations", [])
    print(f"       {len(regions)} text region(s) detected")
    for i, region in enumerate(regions):
        texts = " | ".join(f"[{lang}] {text}" for lang, text in region["text"].items())
        box = (region["minX"], region["minY"], region["maxX"], region["maxY"])
        prob = region.get("prob")
        prob = f"{prob:.2f}" if isinstance(prob, (int, float)) else "?"
        print(f"       #{i} box={box} angle={region.get('angle')} prob={prob} {texts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", nargs="?", default=IMAGE_PATH, help="image file to send")
    parser.add_argument("--url", default=os.environ.get("MIMG_URL", DEFAULT_URL))
    parser.add_argument("--api-key", default=os.environ.get("MIMG_API_KEY"))
    parser.add_argument("--profile", default="manga", help="server profile (manga|generic)")
    parser.add_argument("--no-generic", action="store_true", help="skip the generic-profile JSON run")
    parser.add_argument("--outdir", help="where to write results (default: <image>_server_test)")
    args = parser.parse_args()

    if not args.image or not os.path.isfile(args.image):
        sys.exit(f"image not found: {args.image!r} (pass a path or set IMAGE_PATH in this file)")
    stem = os.path.splitext(os.path.basename(args.image))[0]
    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(args.image)), f"{stem}_server_test"
    )
    os.makedirs(outdir, exist_ok=True)

    with open(args.image, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode()
    payload = {"image": f"data:image/png;base64,{b64}", "profile": args.profile}
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    print(f"server: {args.url}  image: {args.image}  profile: {args.profile}")

    status, body = http(f"{args.url}/health", headers={"Accept": "application/json"})
    print(f"[{' OK ' if status == 200 else 'FAIL'}] /health: {body.decode('utf-8', 'replace')}")

    png = call(f"POST /translate/image (profile {args.profile})", f"{args.url}/translate/image",
               data=json.dumps(payload).encode(), headers=headers)
    if png:
        save(outdir, f"{stem}_{args.profile}.png", png)

    form_body, content_type = multipart(os.path.basename(args.image), raw,
                                        {"config": "{}", "profile": args.profile})
    form_headers = {"Content-Type": content_type}
    if args.api_key:
        form_headers["X-API-Key"] = args.api_key
    form_png = call("POST /translate/with-form/image", f"{args.url}/translate/with-form/image",
                    data=form_body, headers=form_headers)
    if form_png:
        save(outdir, f"{stem}_{args.profile}_form.png", form_png)
        if png:
            print(f"       form result {'matches' if form_png == png else 'DIFFERS from'} JSON-endpoint result")

    js = call(f"POST /translate/json (profile {args.profile})", f"{args.url}/translate/json",
              data=json.dumps(payload).encode(), headers=headers)
    if js:
        save(outdir, f"{stem}_{args.profile}.json", js)
        print_regions(js)

    if not args.no_generic and args.profile != "generic":
        generic_payload = dict(payload, profile="generic")
        gj = call("POST /translate/json (profile generic)", f"{args.url}/translate/json",
                  data=json.dumps(generic_payload).encode(), headers=headers)
        if gj:
            save(outdir, f"{stem}_generic.json", gj)
            print_regions(gj)

    batch_payload = {"images": [payload["image"]], "config": {}, "profile": args.profile}
    zip_body = call("POST /translate/batch/image", f"{args.url}/translate/batch/image",
                    data=json.dumps(batch_payload).encode(), headers=headers)
    if zip_body:
        save(outdir, f"{stem}_{args.profile}_batch.zip", zip_body)
        with zipfile.ZipFile(io.BytesIO(zip_body)) as archive:
            for name in archive.namelist():
                save(outdir, f"batch_{name}", archive.read(name))

    print(f"\nall results in: {outdir}")


if __name__ == "__main__":
    main()
