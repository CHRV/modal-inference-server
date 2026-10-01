"""Container image construction for SGLang servers."""

from __future__ import annotations

import modal

from .config import SGLangConfig

__all__ = ["build_image", "build_volumes"]


def build_image(config: SGLangConfig) -> modal.Image:
    """The container image an SGLang server runs in.

    Starts from the image published by the SGLang team, points the Hugging Face
    cache at the mounted Volume, and layers on any extra packages.
    """
    image = (
        modal.Image.from_registry(config.image)
        # The upstream image sets an entrypoint that floods container logs.
        .entrypoint([])
        # The SGLang image ships a populated Hugging Face cache, and Modal
        # refuses to mount a Volume over a non-empty path.
        .run_commands(f"rm -rf {config.hf_cache_path}")
        .env(
            {
                "HF_HUB_CACHE": config.hf_cache_path,
                "HF_XET_HIGH_PERFORMANCE": "1",
                # Parallel Inductor compile workers can make snapshotting fail.
                **({"TORCHINDUCTOR_COMPILE_THREADS": "1"} if config.gpu_snapshot else {}),
                **config.env,
            }
        )
    )
    if config.pip:
        image = image.uv_pip_install(*config.pip)
    return image


def build_volumes(config: SGLangConfig) -> dict[str, modal.Volume]:
    """Mount points for the model cache plus any extra Volumes."""
    volumes: dict[str, modal.Volume] = {
        config.hf_cache_path: modal.Volume.from_name(
            config.hf_cache_volume, create_if_missing=True
        )
    }
    for path, volume_name in config.volumes.items():
        volumes[path] = modal.Volume.from_name(volume_name, create_if_missing=True)
    return volumes
