"""Serve SGLang models on Modal.

    from modal_inference_server import SGLangConfig, build_app

    app = build_app(SGLangConfig.for_model("Qwen/Qwen2.5-0.5B-Instruct", gpu="L4"))
    app = build_app(SGLangConfig.load("cookbook:Qwen3.6-35B-A3B", gpu="H200"))

Then `modal serve`, `modal deploy` or `modal run` the file that defines it.
"""

from .client import chat, stream_chat, wait_for_server
from .config import ConfigError, SGLangConfig, UnsupportedGPUError
from .flags import Catalog, Flag, catalog_for_image
from .image import build_image, build_volumes
from .server import build_app, build_server, make_server_class
from .sglang_args import parse_argv, to_argv

__all__ = [
    "SGLangConfig",
    "ConfigError",
    "UnsupportedGPUError",
    "Catalog",
    "Flag",
    "catalog_for_image",
    "parse_argv",
    "to_argv",
    "build_app",
    "build_server",
    "make_server_class",
    "build_image",
    "build_volumes",
    "chat",
    "stream_chat",
    "wait_for_server",
]
