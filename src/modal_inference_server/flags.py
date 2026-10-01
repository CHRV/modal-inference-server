"""The catalog of `sglang serve` flags, extracted from the SGLang image itself.

SGLang has hundreds of server flags and they change with every release, so
nothing here lists them by hand. `extract_catalog` builds SGLang's own
argparse parser inside the image and records every option; the result is
stored per SGLang version under `flags/` and used locally to normalise
aliases, emit the right argv shape (`nargs`, booleans) and catch typos before
a container ever boots.
"""

from __future__ import annotations

import difflib
import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Catalog", "Flag", "catalog_for_image", "extract_catalog", "sglang_version"]

CATALOG_DIR = Path(__file__).parent / "flags"

# argparse action class name -> how the flag is spelled on the command line.
#   value: `--key VALUE` (or several values when nargs allows it)
#   flag:  bare `--key`, there is no way to say "off" but omitting it
#   bool:  `--key` / `--no-key` (argparse.BooleanOptionalAction)
_ACTION_KINDS = {
    "BooleanOptionalAction": "bool",
    "_StoreTrueAction": "flag",
    "DeprecatedStoreTrueAction": "flag",
    "DeprecatedAction": "flag",
    "DeprecatedStoreConstAction": "flag",
    "_StoreConstAction": "flag",
    "_StoreFalseAction": "flag",
}

# Runs inside the SGLang image: dumps SGLang's own parser as JSON on stdout.
_EXTRACT_SCRIPT = r"""
import argparse, json
import sglang
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(parser)
groups = {id(a): g.title for g in parser._action_groups for a in g._group_actions}
flags = []
for a in parser._actions:
    if not a.option_strings or a.dest == "help":
        continue
    default = a.default
    try:
        json.dumps(default)
    except TypeError:
        default = repr(default)
    flags.append({
        "options": a.option_strings,
        "dest": a.dest,
        "action": type(a).__name__,
        "nargs": a.nargs,
        "type": getattr(a.type, "__name__", None if a.type is None else repr(a.type)),
        "choices": list(a.choices) if a.choices is not None else None,
        "default": default,
        "help": a.help,
        "group": groups.get(id(a)),
    })
print(json.dumps({"version": getattr(sglang, "__version__", None), "flags": flags}))
"""


@dataclass(frozen=True)
class Flag:
    """One `sglang serve` option. `name` is its first spelling, without `--`."""

    name: str
    kind: str = "value"
    aliases: tuple[str, ...] = ()
    nargs: str | int | None = None
    type: str | None = None
    choices: tuple[Any, ...] | None = None
    default: Any = None
    help: str | None = None
    deprecated: bool = False

    @property
    def takes_list(self) -> bool:
        return self.nargs in ("+", "*") or (isinstance(self.nargs, int) and self.nargs > 1)

    def coerce(self, text: str) -> Any:
        """Turn a command-line string into the value this flag expects."""
        cast = {"int": int, "float": float, "human_readable_int": int}.get(self.type or "")
        if cast is None:
            return text
        try:
            return cast(text)
        except ValueError:
            return text  # left for `Catalog.check` to report

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind}
        for key in ("aliases", "nargs", "type", "choices", "default", "help"):
            value = getattr(self, key)
            if value is None or value == ():
                continue
            if key == "default" and self.kind == "flag" and value is False:
                continue  # every bare switch defaults to off
            out[key] = list(value) if isinstance(value, tuple) else value
        if self.deprecated:
            out["deprecated"] = True
        return out


@dataclass
class Catalog:
    """Every flag one SGLang version accepts, plus lookup by any spelling."""

    version: str
    flags: dict[str, Flag]
    image: str | None = None
    _spellings: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for flag in self.flags.values():
            for spelling in (flag.name, *flag.aliases):
                self._spellings[spelling] = flag.name

    # --- lookup ------------------------------------------------------------
    def resolve(self, key: str) -> str | None:
        """Canonical name for any spelling, alias or unambiguous prefix.

        Prefixes follow argparse (`--tp` is `--tp-size`), because that is what
        SGLang accepts and what recipes in the wild rely on
        (`--speculative-algo`).
        """
        key = key.removeprefix("--").replace("_", "-")
        if key in self._spellings:
            return self._spellings[key]
        owners = {name for spelling, name in self._spellings.items()
                  if spelling.startswith(key)}
        return owners.pop() if len(owners) == 1 else None

    def get(self, key: str) -> Flag | None:
        name = self.resolve(key)
        return self.flags[name] if name else None

    def suggest(self, key: str, limit: int = 3) -> list[str]:
        key = key.removeprefix("--").replace("_", "-")
        return difflib.get_close_matches(key, list(self._spellings), n=limit, cutoff=0.6)

    def search(self, query: str) -> list[Flag]:
        """Flags whose name, alias or help mentions every word of `query`."""
        words = query.lower().split()
        return [
            flag for flag in self.flags.values()
            if all(word in " ".join((flag.name, *flag.aliases, flag.help or "")).lower()
                   for word in words)
        ]

    # --- validation --------------------------------------------------------
    def check(self, args: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Return (errors, warnings) for a normalised `{flag: value}` mapping."""
        errors: list[str] = []
        warnings: list[str] = []
        for key, value in args.items():
            flag = self.flags.get(key)
            if flag is None:
                hint = self.suggest(key)
                errors.append(
                    f"unknown SGLang flag --{key}"
                    + (f" (did you mean {', '.join('--' + h for h in hint)}?)" if hint else "")
                )
                continue
            if flag.deprecated:
                warnings.append(f"--{key} is deprecated in SGLang {self.version}: {flag.help}")
            if value is None or (value is False and flag.kind == "value"):
                continue  # not set / switched off with --no-FLAG
            if flag.kind in ("flag", "bool"):
                if not isinstance(value, bool):
                    errors.append(f"--{key} is a switch, expected true/false, got {value!r}")
                continue
            values = value if isinstance(value, list) else [value]
            if isinstance(value, list) and not flag.takes_list:
                errors.append(f"--{key} takes a single value, got a list {value!r}")
            for item in values:
                if flag.type in ("int", "float") and not isinstance(item, (int, float)):
                    errors.append(f"--{key} expects {flag.type}, got {item!r}")
                elif flag.choices and item not in flag.choices:
                    errors.append(
                        f"--{key}={item!r} is not one of {', '.join(map(str, flag.choices))}"
                    )
        return errors, warnings

    # --- (de)serialisation -------------------------------------------------
    @classmethod
    def from_extraction(cls, raw: dict[str, Any], image: str | None = None) -> Catalog:
        """Build a catalog from `_EXTRACT_SCRIPT`'s JSON output."""
        flags: dict[str, Flag] = {}
        for item in raw["flags"]:
            spellings = [o.removeprefix("--") for o in item["options"]]
            kind = _ACTION_KINDS.get(item["action"], "value")
            if kind == "bool":  # drop the `--no-x` spelling, it is `x: false`
                spellings = [s for s in spellings if not s.startswith("no-")]
            name, *aliases = spellings
            flags[name] = Flag(
                name=name,
                kind=kind,
                aliases=tuple(aliases),
                nargs=item["nargs"] if kind == "value" else None,
                type=item["type"] if kind == "value" else None,
                choices=tuple(item["choices"]) if item["choices"] else None,
                default=item["default"],
                help=item["help"],
                deprecated=item["action"].startswith("Deprecated"),
            )
        return cls(version=raw["version"], flags=flags, image=image)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sglang_version": self.version,
            "image": self.image,
            "flags": {name: flag.to_dict() for name, flag in sorted(self.flags.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Catalog:
        flags = {}
        for name, spec in data["flags"].items():
            spec = dict(spec)
            for key in ("aliases", "choices"):
                if key in spec:
                    spec[key] = tuple(spec[key])
            if spec.get("kind") == "flag":
                spec.setdefault("default", False)  # omitted by `to_dict`
            flags[name] = Flag(name=name, **spec)
        return cls(version=str(data["sglang_version"]), flags=flags, image=data.get("image"))

    def save(self, directory: Path = CATALOG_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"sglang-{self.version}.yaml"
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, width=100))
        return path


def sglang_version(image: str) -> str | None:
    """`lmsysorg/sglang:v0.5.19-cu129` -> `0.5.19`."""
    tag = image.rpartition(":")[2] if ":" in image.rpartition("/")[2] else ""
    match = re.search(r"(\d+\.\d+\.\d+(?:\.post\d+|rc\d+)?)", tag)
    return match.group(1) if match else None


@functools.cache
def catalog_for_image(image: str) -> Catalog | None:
    """The bundled catalog for the SGLang version in `image`, if there is one."""
    version = sglang_version(image)
    if version is None:
        return None
    path = CATALOG_DIR / f"sglang-{version}.yaml"
    if not path.exists():
        return None
    return Catalog.from_dict(yaml.safe_load(path.read_text()))


def available_versions() -> list[str]:
    return sorted(p.stem.removeprefix("sglang-") for p in CATALOG_DIR.glob("sglang-*.yaml"))


def extract_catalog(image: str) -> Catalog:
    """Dump the flags of the SGLang in `image`, using a CPU-only Modal Sandbox."""
    import modal

    app = modal.App.lookup("modal-sglang-tools", create_if_missing=True)
    sandbox = modal.Sandbox.create(
        "python", "-c", _EXTRACT_SCRIPT,
        image=modal.Image.from_registry(image).entrypoint([]),
        app=app,
        timeout=15 * 60,
    )
    sandbox.wait()
    if sandbox.returncode != 0:
        raise RuntimeError(
            f"flag extraction failed in {image} (exit {sandbox.returncode}):\n"
            + sandbox.stderr.read()[-4000:]
        )
    return Catalog.from_extraction(json.loads(sandbox.stdout.read()), image=image)
