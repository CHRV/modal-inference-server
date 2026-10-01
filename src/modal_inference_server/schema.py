"""JSON Schema for config files, for editors (yaml-language-server) and LLMs.

The Modal half comes from the SGLangConfig dataclass, the SGLang half from the
flag catalog, so the schema documents every flag of that SGLang version.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from typing import Any

from .config import SGLangConfig
from .flags import Catalog, Flag

__all__ = ["json_schema"]

_SCALARS = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _python_type(hint: Any) -> dict[str, Any]:
    origin = typing.get_origin(hint)
    args = [a for a in typing.get_args(hint) if a is not type(None)]
    if origin in (typing.Union, types.UnionType):
        inner = _python_type(args[0])
        return {**inner, "type": [inner["type"], "null"]} if "type" in inner else inner
    if origin is dict:
        return {"type": "object", "additionalProperties": {"type": "string"}}
    if origin is tuple:
        return {"type": "array", "items": {"type": "string"}}
    return {"type": _SCALARS.get(hint, "string")}


def _flag_schema(flag: Flag) -> dict[str, Any]:
    if flag.kind in ("flag", "bool"):
        item: dict[str, Any] = {"type": ["boolean", "null"]}
    else:
        base: dict[str, Any] = {"type": {"int": "integer", "float": "number"}.get(
            flag.type or "", "string")}
        if flag.choices:
            base = {"enum": list(flag.choices)}
        item = ({"type": ["array", "null"], "items": base} if flag.takes_list
                else {"anyOf": [base, {"type": "null"}]})
    if flag.help:
        item["description"] = flag.help
    if flag.default not in (None, False, []):
        item["default"] = flag.default
    if flag.deprecated:
        item["deprecated"] = True
    return item


def json_schema(catalog: Catalog | None) -> dict[str, Any]:
    hints = typing.get_type_hints(SGLangConfig)
    modal = {
        f.name: _python_type(hints[f.name])
        for f in dataclasses.fields(SGLangConfig) if f.name != "sglang"
    }
    sglang: dict[str, Any] = {"type": "object",
                              "description": "`sglang serve` flags, SGLang spelling without --"}
    if catalog is not None:
        properties: dict[str, Any] = {}
        for flag in catalog.flags.values():
            for spelling in (flag.name, *flag.aliases):
                properties[spelling] = _flag_schema(flag)
        sglang["properties"] = properties
        sglang["additionalProperties"] = False
        sglang["description"] += f" (SGLang {catalog.version})"
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "modal-sglang config",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "extends": {
                "description": "config(s) to start from: path, URL, cookbook:..., hf:...",
                "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}],
            },
            "modal": {"type": "object", "properties": modal, "additionalProperties": False},
            "sglang": sglang,
        },
    }
