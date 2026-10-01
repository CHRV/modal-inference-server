"""Presets converted from the SGLang Cookbook (https://docs.sglang.io/cookbook).

`scripts/import_cookbook.py` turns each cookbook model into one YAML file
under `presets/cookbook/`, holding one ready-to-use config per (GPU, variant):

    model: Qwen3.6-35B-A3B
    model_path: Qwen/Qwen3.6-35B-A3B
    configs:
      H100/default:
        attributes: {optimization: balanced, quantization: bf16}
        modal: {gpu: "H100:1"}
        sglang: {model-path: Qwen/Qwen3.6-35B-A3B, tp-size: 1, ...}

and `cookbook:Qwen3.6-35B-A3B@H100/default` picks one of them.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError, split_gpu

__all__ = ["Preset", "all_presets", "find_preset", "resolve_cookbook"]

PRESET_DIR = Path(__file__).parent / "presets" / "cookbook"

# When neither the reference nor the config names a GPU, take the cheapest one
# the recipe supports.
GPU_PREFERENCE = ("L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "A100", "H100", "H200",
                  "B200")


@dataclass(frozen=True)
class Preset:
    model: str
    model_path: str
    configs: dict[str, dict[str, Any]]
    vendor: str | None = None
    family: str | None = None
    description: str | None = None
    source: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def gpus(self) -> list[str]:
        seen = dict.fromkeys(key.split("/")[0] for key in self.configs)
        return sorted(seen, key=lambda g: (GPU_PREFERENCE.index(g)
                                           if g in GPU_PREFERENCE else len(GPU_PREFERENCE), g))

    def variants(self, gpu: str) -> list[str]:
        return [key.split("/", 1)[1] for key in self.configs if key.split("/")[0] == gpu]

    def pick(self, gpu: str | None = None, variant: str | None = None) -> tuple[str, dict]:
        """Choose a config: the requested GPU or the cheapest, then `default`."""
        if gpu is None:
            gpu = self.gpus[0]
        gpu = next((g for g in self.gpus if g.upper() == gpu.upper()), gpu)
        variants = self.variants(gpu)
        if not variants:
            raise ConfigError(
                f"cookbook has no {gpu} recipe for {self.model}; "
                f"available: {', '.join(self.gpus)}"
            )
        if variant is None:
            variant = "default" if "default" in variants else variants[0]
        if variant not in variants:
            raise ConfigError(
                f"cookbook has no {variant!r} variant of {self.model} on {gpu}; "
                f"available: {', '.join(variants)}"
            )
        key = f"{gpu}/{variant}"
        return key, self.configs[key]


@functools.cache
def all_presets() -> tuple[Preset, ...]:
    presets = []
    for path in sorted(PRESET_DIR.glob("*.yaml")):
        presets.append(Preset(**yaml.safe_load(path.read_text())))
    return tuple(presets)


def find_preset(name: str) -> Preset | None:
    """Look a model up by cookbook name or Hugging Face path, ignoring case."""
    wanted = name.lower()
    for preset in all_presets():
        if wanted in (preset.model.lower(), preset.model_path.lower()):
            return preset
    return None


def search(query: str = "") -> list[Preset]:
    words = query.lower().split()
    return [
        p for p in all_presets()
        if all(w in " ".join(filter(None, (p.model, p.model_path, p.vendor, p.family,
                                           p.description))).lower() for w in words)
    ]


def resolve_cookbook(ref: str, gpu_hint: str | None = None) -> tuple[dict[str, Any], str]:
    """`Qwen3.6-35B-A3B@H100/speculative-mtp` -> (config data, canonical label).

    `@GPU` defaults to `gpu_hint` (the GPU the rest of the config asks for),
    then to the cheapest GPU the recipe covers; `/variant` defaults to `default`.
    """
    name, _, rest = ref.partition("@")
    gpu, _, variant = rest.partition("/") if rest else ("", "", "")
    if not rest and "/" in name and find_preset(name) is None:
        # `NAME/variant` without a GPU (a model path also contains "/").
        name, _, variant = name.rpartition("/")
    preset = find_preset(name)
    if preset is None:
        close = [p.model for p in all_presets() if name.lower() in p.model.lower()]
        raise ConfigError(
            f"no cookbook preset named {name!r}"
            + (f"; did you mean {', '.join(close[:5])}?" if close else "")
            + " (list them with `modal-sglang presets`)"
        )
    if not gpu and gpu_hint:
        hinted = split_gpu(gpu_hint)[0]
        gpu = hinted if preset.variants(hinted) else ""
    key, data = preset.pick(gpu or None, variant or None)
    return ({"modal": dict(data.get("modal") or {}), "sglang": dict(data["sglang"])},
            f"cookbook:{preset.model}@{key}")
