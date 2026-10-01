"""Declarative configuration for an SGLang server running on Modal.

A config has two halves with one owner each:

- the Modal side (GPU, image, secrets, autoscaling...) — the dataclass fields;
- the SGLang side — `sglang`, a `{flag: value}` mapping in SGLang's own
  spelling, passed to `sglang.launch_server` as-is. Every SGLang flag works,
  including ones released after this library; the bundled flag catalog only
  checks them (see `flags.py`).
"""

from __future__ import annotations

import dataclasses
import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from .flags import Catalog, catalog_for_image
from .sglang_args import normalize, to_argv

__all__ = ["ConfigError", "SGLangConfig", "UnsupportedGPUError"]

MINUTES = 60  # seconds

# Pre-Ampere cards (T4 == Turing, sm75) have no bfloat16 and no FlashInfer /
# FlashAttention kernels, so they need a conservative set of backends. These
# only matter alongside `allow_unsupported_gpu` on an SGLang 0.4.x image —
# on 0.5.x no flag rescues a T4, see SGLANG_MIN_COMPUTE_CAPABILITY below.
# Only flags absent from `sglang` are filled in, so an explicit choice wins.
_TURING_DEFAULTS: dict[str, Any] = {
    "dtype": "float16",
    "attention-backend": "triton",
    "sampling-backend": "pytorch",
    "disable-cuda-graph": True,
}

GPU_DEFAULTS: dict[str, dict[str, Any]] = {
    "T4": _TURING_DEFAULTS,
}

# Compute capability per Modal GPU type, used to catch "no kernel image is
# available for execution on the device" before paying for a cloud boot.
GPU_COMPUTE_CAPABILITY: dict[str, float] = {
    "T4": 7.5,
    "L4": 8.9,
    "A10G": 8.6,
    "A100": 8.0,
    "A100-40GB": 8.0,
    "A100-80GB": 8.0,
    "L40S": 8.9,
    "H100": 9.0,
    "H200": 9.0,
    "B200": 10.0,
}

# sgl-kernel ships prebuilt CUDA binaries. It dropped sm75 (Turing) in the 0.5
# series, so a T4 fails at runtime with "no kernel image is available" as soon
# as the first RMSNorm runs — flags cannot work around a missing binary.
SGLANG_MIN_COMPUTE_CAPABILITY = 8.0

# The server must listen where Modal routes traffic; these come from `port`.
_MANAGED_FLAGS = ("host", "port")

# What SGLang needs to hand its GPU memory back before a snapshot
# (`/release_memory_occupation`) and take it again after restore. Applied
# under `gpu_snapshot` unless the flag is set explicitly.
_SNAPSHOT_FLAGS: dict[str, Any] = {
    "enable-memory-saver": True,
    "enable-weights-cpu-backup": True,
}
_SNAPSHOT_DRAFT_FLAGS: dict[str, Any] = {"enable-draft-weights-cpu-backup": True}

# Spellings of the flags this module reads, for images without a catalog
# (with one, the catalog knows every alias).
_FALLBACK_ALIASES: dict[str, tuple[str, ...]] = {
    "model-path": ("model",),
    "tp-size": ("tp", "tensor-parallel-size"),
    "pp-size": ("pipeline-parallel-size",),
    "dp-size": ("dp", "data-parallel-size"),
}


class UnsupportedGPUError(RuntimeError):
    """Raised when the requested GPU has no kernels in the SGLang image."""


class ConfigError(ValueError):
    """Raised for a config that cannot produce a working server."""


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def split_gpu(spec: str) -> tuple[str, int | None]:
    """`"H100:2"` -> `("H100", 2)`, `"L4"` -> `("L4", None)`."""
    kind, sep, count = spec.partition(":")
    if not sep:
        return kind, None
    if not count.isdigit() or int(count) < 1:
        raise ConfigError(f"bad GPU spec {spec!r}, expected TYPE or TYPE:COUNT")
    return kind, int(count)


@dataclass(frozen=True)
class SGLangConfig:
    """Everything needed to turn an SGLang model into a Modal Server.

    Only `sglang["model-path"]` is required:

        SGLangConfig(gpu="H100:2", sglang={"model-path": "Qwen/Qwen3-8B"})
        SGLangConfig.for_model("Qwen/Qwen3-8B", gpu="H100:2")
        SGLangConfig.load("config.yaml", "cookbook:Qwen3-8B", gpu="H200")
    """

    # --- SGLang ------------------------------------------------------------
    sglang: dict[str, Any] = field(default_factory=dict)
    """`sglang serve` flags in SGLang's spelling, e.g. {"model-path": ..., "tp": 2}.
    `True` is a bare switch, `False`/`None` leaves the flag out, lists give
    several values. `host` and `port` are managed for you."""

    # --- infrastructure ----------------------------------------------------
    gpu: str = "L4"
    """Modal GPU spec, `TYPE` or `TYPE:COUNT`. The cheapest GPU current
    SGLang kernels run on. A count > 1 sets `tp-size` unless you do."""
    name: str | None = None
    """Modal App name. Defaults to a slug of the model."""
    port: int = 8000
    region: str | None = None
    routing_region: str = "us-east"
    cloud: str | None = None
    allow_unsupported_gpu: bool = False
    """Skip the compute-capability check, e.g. when pinning an SGLang 0.4.x
    image that still shipped sm75 kernels."""
    validate_flags: bool = True
    """Treat unknown or ill-typed SGLang flags as errors rather than warnings."""
    gpu_snapshot: bool = False
    """Snapshot the booted, warmed-up server (GPU memory included) and restore
    later replicas from it, skipping imports, JIT compilation and CUDA graph
    capture. Single GPU only, and only for deployed Apps — ephemeral `run`
    Apps never snapshot. Alpha on Modal's side; see the README."""

    # --- container image ---------------------------------------------------
    image: str = "lmsysorg/sglang:v0.5.19-cu129"
    """Verified booting on L4. Turing needs a 0.4.x image (see `validate`)."""
    pip: tuple[str, ...] = ()
    """Extra packages layered on top of the SGLang image."""
    hf_cache_volume: str = "huggingface-cache"
    hf_cache_path: str = "/root/.cache/huggingface"
    volumes: dict[str, str] = field(default_factory=dict)
    """Extra Modal Volumes to mount, as {container_path: volume_name}."""
    secrets: tuple[str, ...] = ()
    """Names of Modal Secrets to inject (e.g. `HF_TOKEN` for gated models)."""
    env: dict[str, str] = field(default_factory=dict)

    # --- autoscaling -------------------------------------------------------
    min_containers: int = 0
    max_containers: int | None = None
    target_concurrency: int = 8
    scaledown_window: int | None = None
    startup_timeout: int = 20 * MINUTES
    exit_grace_period: int = 15
    unauthenticated: bool = True

    # --- readiness ---------------------------------------------------------
    warmup_requests: int = 1
    """Requests sent to the server on boot before it is marked ready."""

    # --- construction ------------------------------------------------------
    @classmethod
    def for_model(cls, model: str, **fields: Any) -> SGLangConfig:
        """`for_model("Qwen/Qwen3-8B", gpu="H100", sglang={"tp": 1})`."""
        sglang = {"model-path": model, **fields.pop("sglang", {})}
        return cls(sglang=sglang, **fields)

    @classmethod
    def load(cls, *sources: str, **overrides: Any) -> SGLangConfig:
        """Compose config files, URLs, `cookbook:` and `hf:` presets.

        Later sources win; `overrides` (Modal fields, plus `sglang={...}`)
        win over all of them. See `sources.py` for the accepted forms.
        """
        from .sources import Layer, compose

        sglang = overrides.pop("sglang", None)
        layers = [Layer.ref(source) for source in sources]
        layers.append(Layer("python", modal=overrides, sglang=sglang or {}))
        return compose(layers).config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SGLangConfig:
        """Inverse of `to_dict`: `{"modal": {...}, "sglang": {...}}`."""
        unknown = set(data) - {"modal", "sglang"}
        if unknown:
            raise ConfigError(f"unknown top-level keys {sorted(unknown)}, expected modal/sglang")
        modal = dict(data.get("modal") or {})
        check_modal_keys(modal)
        for key in ("pip", "secrets"):
            if key in modal and modal[key] is not None:
                modal[key] = tuple(modal[key])
        return cls(sglang=dict(data.get("sglang") or {}), **modal)

    def to_dict(self, *, full: bool = False) -> dict[str, Any]:
        """`{"modal": ..., "sglang": ...}`, the shape of a config file.

        Modal fields left at their default are omitted unless `full`, except
        `gpu` and `image`, which are what make a deployment reproducible.
        """
        modal: dict[str, Any] = {}
        for item in dataclasses.fields(self):
            if item.name == "sglang":
                continue
            value = getattr(self, item.name)
            default = (item.default_factory() if item.default_factory
                       is not dataclasses.MISSING else item.default)
            if full or value != default or item.name in ("gpu", "image"):
                modal[item.name] = list(value) if isinstance(value, tuple) else value
        return {"modal": modal, "sglang": dict(self.sglang)}

    def to_yaml(self, *, full: bool = False) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(full=full), sort_keys=False)

    def with_(self, **changes: Any) -> SGLangConfig:
        """Derive a modified copy. `sglang=` is merged rather than replaced,
        and a `None` value in it removes the flag:

            cfg.with_(gpu="H100:2", sglang={"context-length": 32768, "dtype": None})
        """
        if "sglang" in changes:
            merged = {**self.sglang, **changes["sglang"]}
            changes["sglang"] = {k: v for k, v in merged.items() if v is not None}
        return dataclasses.replace(self, **changes)

    # --- derived values ----------------------------------------------------
    @property
    def catalog(self) -> Catalog | None:
        """Flag catalog for the SGLang version in `image`, if bundled."""
        return catalog_for_image(self.image)

    @property
    def args(self) -> dict[str, Any]:
        """`sglang` with every flag under its canonical name (`tp` -> `tp-size`)."""
        return normalize(self.sglang, self.catalog)

    def _key(self, flag: str) -> str:
        """The spelling `flag` goes by in `args`: canonical with a catalog,
        otherwise whichever known alias the user wrote."""
        catalog = self.catalog
        if catalog is not None:
            return catalog.resolve(flag) or flag
        args = self.args
        return next((s for s in (flag, *_FALLBACK_ALIASES.get(flag, ())) if s in args), flag)

    @property
    def model(self) -> str | None:
        return self.args.get(self._key("model-path"))

    @property
    def app_name(self) -> str:
        return self.name or f"sglang-{_slugify(self.model or 'model')}"

    @property
    def model_name(self) -> str | None:
        """Name clients pass as `model` in OpenAI requests."""
        return self.args.get(self._key("served-model-name")) or self.model

    @property
    def gpu_type(self) -> str:
        return split_gpu(self.gpu)[0]

    @property
    def n_gpus(self) -> int:
        return split_gpu(self.gpu)[1] or 1

    @property
    def gpu_spec(self) -> str:
        return self.gpu

    @property
    def compute_capability(self) -> float | None:
        """Compute capability of the requested GPU, if we know it."""
        return GPU_COMPUTE_CAPABILITY.get(self.gpu_type.rstrip("!").upper())

    def resolved_args(self) -> dict[str, Any]:
        """The flags SGLang is actually launched with: `sglang` plus GPU
        defaults, the tensor-parallel default and the managed host/port."""
        args = self.args
        for flag, value in GPU_DEFAULTS.get(self.gpu_type.rstrip("!").upper(), {}).items():
            args.setdefault(self._key(flag), value)
        if self.gpu_snapshot:
            for flag, value in _SNAPSHOT_FLAGS.items():
                args.setdefault(self._key(flag), value)
            if args.get(self._key("speculative-draft-model-path")):
                for flag, value in _SNAPSHOT_DRAFT_FLAGS.items():
                    args.setdefault(self._key(flag), value)
        tp = self._key("tp-size")
        if self.n_gpus > 1 and tp not in args:
            args[tp] = self.n_gpus
        for flag in _MANAGED_FLAGS:
            args.pop(self._key(flag), None)
        args[self._key("host")] = "0.0.0.0"
        args[self._key("port")] = self.port
        return args

    def launch_command(self) -> list[str]:
        """The `python -m sglang.launch_server ...` argv for this config."""
        return ["python", "-m", "sglang.launch_server",
                *to_argv(self.resolved_args(), self.catalog)]

    # --- validation --------------------------------------------------------
    def check(self) -> tuple[list[str], list[str]]:
        """Every problem with this config, as (errors, warnings). Offline."""
        errors: list[str] = []
        warnings: list[str] = []
        try:
            split_gpu(self.gpu)
        except ConfigError as error:
            errors.append(str(error))
            return errors, warnings
        if not self.model:
            errors.append("no model: set sglang `model-path` (CLI: -- --model-path ORG/NAME)")

        for flag in _MANAGED_FLAGS:
            if self._key(flag) in self.args:
                warnings.append(f"--{flag} is managed by modal-sglang and was ignored "
                                "(set the Modal field `port` instead)")

        catalog = self.catalog
        args = self.resolved_args()
        if catalog is None:
            warnings.append(
                f"no flag catalog for {self.image}: SGLang flags are passed through "
                "unchecked (run `modal-sglang sync-flags --image ...` to add one)"
            )
        else:
            flag_errors, flag_warnings = catalog.check(args)
            (errors if self.validate_flags else warnings).extend(flag_errors)
            warnings.extend(flag_warnings)

        expected = self._gpus_needed(args)
        if self.gpu_snapshot and (self.n_gpus > 1 or (expected or 1) > 1):
            errors.append(
                f"gpu_snapshot needs a single GPU, but this config uses "
                f"{max(self.n_gpus, expected or 1)} (gpu={self.gpu!r}); Modal GPU memory "
                "snapshots do not support multi-GPU servers yet"
            )
        elif self.gpu_snapshot and self.warmup_requests < 1:
            warnings.append("gpu_snapshot with warmup_requests=0 snapshots a cold server; "
                            "warm-up requests are what move compilation into the snapshot")
        if expected is not None and expected != self.n_gpus:
            warnings.append(
                f"the parallelism flags use {expected} GPU(s) but gpu={self.gpu!r} "
                f"gives {self.n_gpus}; set gpu to {self.gpu_type}:{expected}?"
            )
        return errors, warnings

    def _gpus_needed(self, args: dict[str, Any]) -> int | None:
        """GPUs implied by tp × pp × dp (dp attention reuses the tp GPUs)."""
        try:
            tp = int(args.get(self._key("tp-size")) or 1)
            pp = int(args.get(self._key("pp-size")) or 1)
            dp = int(args.get(self._key("dp-size")) or 1)
        except (TypeError, ValueError):
            return None  # reported by the catalog check
        if args.get(self._key("enable-dp-attention")):
            dp = 1
        return tp * pp * dp

    def validate(self) -> None:
        """Fail fast on configurations that cannot work. Raises
        `UnsupportedGPUError` for a GPU without kernels and `ConfigError` for
        everything else `check` reports as an error."""
        capability = self.compute_capability
        if (
            not self.allow_unsupported_gpu
            and capability is not None
            and capability < SGLANG_MIN_COMPUTE_CAPABILITY
        ):
            raise UnsupportedGPUError(
                f"{self.gpu_type} has compute capability {capability}, but the prebuilt "
                f"kernels in {self.image} need at least "
                f"{SGLANG_MIN_COMPUTE_CAPABILITY} (sgl-kernel dropped sm75 in the 0.5 "
                "series). It would fail in the container with 'no kernel image is "
                "available for execution on the device'. Use L4, A10G or newer, or "
                "pin an SGLang 0.4.x image and set allow_unsupported_gpu=True."
            )
        errors, _ = self.check()
        if errors:
            raise ConfigError("invalid config:\n  " + "\n  ".join(errors))


MODAL_FIELDS = tuple(f.name for f in dataclasses.fields(SGLangConfig) if f.name != "sglang")


def check_modal_keys(modal: dict[str, Any]) -> None:
    for key in modal:
        if key not in MODAL_FIELDS:
            hint = difflib.get_close_matches(key, MODAL_FIELDS, n=1)
            raise ConfigError(
                f"unknown modal key {key!r}"
                + (f" (did you mean {hint[0]!r}?)" if hint else "")
                + ". SGLang flags go under `sglang:`."
            )
