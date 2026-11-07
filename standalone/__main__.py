"""CLI entry point: python -m standalone [--host H --port P ...]"""

import argparse
import os

import uvicorn

from .server import app


def main():
    parser = argparse.ArgumentParser(
        prog="python -m standalone",
        description="Standalone image translation server: POST an image, get a typeset image back.",
    )
    parser.add_argument("--host", default=os.environ.get("MIMG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MIMG_PORT", "5003")))
    parser.add_argument("--api-key", help="require this key in the X-API-Key header (env: MIMG_API_KEY)")
    parser.add_argument("--model-dir", help="weights directory (env: MIMG_MODEL_DIR)")
    parser.add_argument("--font-path", help="font file used for rendering (env: MIMG_FONT_PATH)")
    parser.add_argument("--use-gpu", action="store_true", help="force GPU usage (env: MIMG_USE_GPU=1)")
    parser.add_argument("--no-gpu", action="store_true", help="force CPU usage (env: MIMG_USE_GPU=0)")
    parser.add_argument("--offline", action="store_true", help="never download model weights (env: MIMG_OFFLINE)")
    parser.add_argument("--cache-only", action="store_true", help="use only downloaded weights (env: MIMG_CACHE_ONLY)")
    parser.add_argument("--verbose", action="store_true", help="write debug artifacts to ./result (env: MIMG_VERBOSE)")
    parser.add_argument("--log-level", default=os.environ.get("MIMG_LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    # The pipeline builds itself from the environment at startup (lifespan),
    # so translating CLI flags into env vars before uvicorn.run is enough.
    if args.api_key:
        os.environ["MIMG_API_KEY"] = args.api_key
    if args.model_dir:
        os.environ["MIMG_MODEL_DIR"] = args.model_dir
    if args.font_path:
        os.environ["MIMG_FONT_PATH"] = args.font_path
    if args.use_gpu and args.no_gpu:
        parser.error("--use-gpu and --no-gpu cannot be combined")
    if args.use_gpu:
        os.environ["MIMG_USE_GPU"] = "1"
    elif args.no_gpu:
        os.environ["MIMG_USE_GPU"] = "0"
    if args.offline:
        os.environ["MIMG_OFFLINE"] = "1"
    if args.cache_only:
        os.environ["MIMG_CACHE_ONLY"] = "1"
    if args.verbose:
        os.environ["MIMG_VERBOSE"] = "1"
    if args.log_level:
        os.environ["MIMG_LOG_LEVEL"] = args.log_level

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
