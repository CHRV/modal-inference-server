"""Where configs come from, and how several of them stack.

A config is built from layers, weakest first:

    defaults < extends < --from #1 < --from #2 < ... < Modal CLI options < SGLang CLI flags

Each layer is `{"modal": {...}, "sglang": {...}}`, optionally with
`extends: SOURCE` (or a list). A source is any of:

    ./qwen.yaml  qwen.json             local file (relative to the file that extends it)
    https://.../cfg.yaml               URL; github.com/.../blob/... is fetched raw
    cookbook:MODEL[@GPU][/VARIANT]     preset converted from the SGLang Cookbook
    hf:ORG/NAME                        cookbook preset for that model, else just the model
    hf:ORG/NAME/path/to/cfg.yaml       a config file stored in a Hugging Face repo

A file without `modal`/`sglang` keys is read as a plain SGLang `--config`
file, so SGLang's own YAML configs work unchanged.

Merging: `sglang` flags replace earlier values (a list replaces the whole
list) and `null` deletes an inherited flag; `env` and `volumes` merge key by
key; a GPU given without a count keeps the earlier count, so
`--gpu H200` over `H100:2` means `H200:2` (write `H200:1` to force one).
"""

from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import yaml

from .config import MODAL_FIELDS, ConfigError, SGLangConfig, check_modal_keys, split_gpu
from .flags import catalog_for_image
from .sglang_args import normalize, parse_argv

__all__ = ["Composed", "Layer", "compose", "load_source"]

_MERGED_DICTS = ("env", "volumes")
_TOP_LEVEL = {"modal", "sglang", "extends"}


@dataclass
class Layer:
    """One source of settings, with a label used to report provenance."""

    label: str
    modal: dict[str, Any] = field(default_factory=dict)
    sglang: dict[str, Any] = field(default_factory=dict)
    argv: list[str] | None = None
    """Raw SGLang tokens, parsed once the image (hence the catalog) is known."""
    extends: list[str] = field(default_factory=list)
    base: str | None = None
    """Directory or URL that relative `extends` are resolved against."""

    @classmethod
    def from_data(cls, data: Any, label: str, base: str | None = None) -> Layer:
        if not isinstance(data, dict):
            raise ConfigError(f"{label}: expected a mapping, got {type(data).__name__}")
        if not set(data) & _TOP_LEVEL:
            data = {"sglang": data}  # a plain SGLang `--config` file
        unknown = set(data) - _TOP_LEVEL
        if unknown:
            raise ConfigError(
                f"{label}: unknown top-level keys {sorted(unknown)}; expected "
                "`modal:`, `sglang:` and optionally `extends:`"
            )
        modal = dict(data.get("modal") or {})
        try:
            check_modal_keys(modal)
        except ConfigError as error:
            raise ConfigError(f"{label}: {error}") from None
        extends = data.get("extends") or []
        return cls(
            label=label,
            modal=modal,
            sglang=dict(data.get("sglang") or {}),
            extends=[extends] if isinstance(extends, str) else list(extends),
            base=base,
        )

    @classmethod
    def ref(cls, source: str) -> Layer:
        """A layer standing for `source`, fetched when the config is composed.

        Resolution is deferred so a `cookbook:` preset can follow a `--gpu`
        given later on the command line.
        """
        return cls(label=f"--from {source}", extends=[source])

    @classmethod
    def from_source(cls, source: str, *, gpu_hint: str | None = None,
                    base: str | None = None) -> Layer:
        data, label, new_base = load_source(source, gpu_hint=gpu_hint, base=base)
        return cls.from_data(data, label, new_base)

    def expand(self, gpu_hint: str | None = None, _stack: tuple[str, ...] = ()) -> list[Layer]:
        """This layer preceded by everything it extends, recursively."""
        if self.label in _stack:
            raise ConfigError(f"circular extends: {' -> '.join((*_stack, self.label))}")
        hint = self.modal.get("gpu") or gpu_hint
        layers: list[Layer] = []
        for source in self.extends:
            parent = Layer.from_source(source, gpu_hint=hint, base=self.base)
            layers.extend(parent.expand(hint, (*_stack, self.label)))
        layers.append(self)
        return layers


@dataclass
class Composed:
    config: SGLangConfig
    provenance: dict[str, str]
    """`"sglang.tp-size"` / `"modal.gpu"` -> label of the layer that set it."""


def compose(layers: list[Layer], unset: Sequence[str] = (),
            gpu_hint: str | None = None) -> Composed:
    """Merge layers (weakest first) into a config. See the module docstring."""
    if gpu_hint is None:  # cookbook presets follow the GPU the strongest layer asks for
        gpu_hint = next((layer.modal["gpu"] for layer in reversed(layers)
                         if layer.modal.get("gpu")), None)
    expanded = [part for layer in layers for part in layer.expand(gpu_hint)]
    provenance: dict[str, str] = {}

    # Modal first: it decides the image, and the image decides the catalog.
    modal: dict[str, Any] = {}
    for layer in expanded:
        for key, value in layer.modal.items():
            if value is None:
                modal.pop(key, None)
                provenance.pop(f"modal.{key}", None)
                continue
            if key == "gpu":
                value = _inherit_gpu_count(str(value), modal.get("gpu"))
            elif key in _MERGED_DICTS and isinstance(modal.get(key), dict):
                value = {**modal[key], **value}
            modal[key] = value
            provenance[f"modal.{key}"] = layer.label

    image = modal.get("image") or SGLangConfig.image
    catalog = catalog_for_image(image)

    sglang: dict[str, Any] = {}
    for layer in expanded:
        try:
            args = parse_argv(layer.argv, catalog) if layer.argv is not None else {}
        except ValueError as error:
            raise ConfigError(f"{layer.label}: {error}") from None
        args = {**normalize(layer.sglang, catalog), **args}
        for key, value in args.items():
            sglang.pop(key, None)
            provenance.pop(f"sglang.{key}", None)
            if value is not None:
                sglang[key] = value
                provenance[f"sglang.{key}"] = layer.label

    for key in unset:
        _unset(key, modal, sglang, provenance, catalog)

    config = SGLangConfig.from_dict({"modal": modal, "sglang": sglang})
    return Composed(config=config, provenance=provenance)


def _inherit_gpu_count(new: str, old: str | None) -> str:
    kind, count = split_gpu(new)
    if count is None and old:
        old_count = split_gpu(old)[1]
        if old_count:
            return f"{kind}:{old_count}"
    return new


def _unset(key: str, modal: dict, sglang: dict, provenance: dict, catalog) -> None:
    section, _, name = key.rpartition(".")
    name = name.removeprefix("--")
    if section in ("", "sglang"):
        canonical = (catalog.resolve(name) if catalog else None) or name
        if canonical in sglang:
            del sglang[canonical]
            provenance.pop(f"sglang.{canonical}", None)
            return
    if section in ("", "modal"):
        field_name = name.replace("-", "_")
        if field_name in MODAL_FIELDS:
            modal.pop(field_name, None)
            provenance.pop(f"modal.{field_name}", None)
            return
    if section in ("", "sglang"):
        return  # unsetting a flag nobody set is harmless
    raise ConfigError(f"cannot unset {key!r}: expected a SGLang flag or a modal field")


# --- loading ---------------------------------------------------------------

def load_source(source: str, *, gpu_hint: str | None = None,
                base: str | None = None) -> tuple[Any, str, str | None]:
    """Fetch one source: returns (parsed data, label, base for its extends)."""
    from .presets import find_preset, resolve_cookbook

    if source.startswith("cookbook:"):
        data, label = resolve_cookbook(source.removeprefix("cookbook:"), gpu_hint)
        return data, label, None

    if source.startswith("hf:"):
        ref = source.removeprefix("hf:").strip("/")
        parts = ref.split("/")
        if len(parts) == 2:
            if find_preset(ref):
                data, label = resolve_cookbook(ref, gpu_hint)
                return data, label, None
            return {"sglang": {"model-path": ref}}, source, None
        if len(parts) < 2:
            raise ConfigError(f"bad Hugging Face source {source!r}, expected hf:ORG/NAME[/FILE]")
        repo, path = "/".join(parts[:2]), "/".join(parts[2:])
        url = f"https://huggingface.co/{repo}/resolve/main/{path}"
        token = os.environ.get("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return _parse(_fetch(url, headers), source), source, url

    if re.match(r"https?://", source) or (base and re.match(r"https?://", base)):
        url = urllib.parse.urljoin(base or "", source)
        url = _raw_github(url)
        return _parse(_fetch(url), url), url, url

    path = Path(source).expanduser()
    if not path.is_absolute() and base:
        path = Path(base) / path
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    return _parse(path.read_text(), str(path)), str(path), str(path.resolve().parent)


def _raw_github(url: str) -> str:
    match = re.match(r"https://github\.com/([^/]+/[^/]+)/blob/(.+)", url)
    return f"https://raw.githubusercontent.com/{match[1]}/{match[2]}" if match else url


def _fetch(url: str, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "modal-sglang", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode()
    except OSError as error:
        raise ConfigError(f"could not fetch {url}: {error}") from None


def _parse(text: str, label: str) -> Any:
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as error:
        raise ConfigError(f"{label} is not valid YAML/JSON: {error}") from None
