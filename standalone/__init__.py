"""Standalone image-translation HTTP server.

Thin FastAPI wrapper around the manga_translator pipeline that runs in-process,
dropping the gateway/queue/subprocess layer of the bundled server/ package.
"""
