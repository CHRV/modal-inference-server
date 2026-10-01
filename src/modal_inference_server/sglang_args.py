"""SGLang flags as data: argv <-> `{flag: value}`.

The mapping uses SGLang's own spelling (kebab-case, no `--`), the same shape
as `sglang serve --config file.yaml`. Values follow one rule everywhere —
config files, the command line, the Python API:

    true          bare switch            --enable-metrics
    false         switch off             (omitted, or --no-x for --x/--no-x flags)
    null          not set                (omitted; also deletes an inherited value)
    [a, b]        several values         --lora-paths a b
    anything else --key value

With a `Catalog` the argv shape comes from SGLang's parser (nargs, switches,
aliases). Without one — an image we have no catalog for — it falls back to
reading the tokens literally.
"""

from __future__ import annotations

from typing import Any

from .flags import Catalog

__all__ = ["normalize", "parse_argv", "to_argv"]

# Words that may precede the flags, so `-- serve --model-path m` and
# `-- python -m sglang.launch_server --model-path m` both work.
_COMMAND_WORDS = {"sglang", "serve", "python", "python3", "-m", "sglang.launch_server",
                  "launch_server"}


def _literal(text: str) -> Any:
    """Best-effort typing for a value we have no catalog entry for."""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _canonical(key: str, catalog: Catalog | None) -> str:
    key = key.removeprefix("--").replace("_", "-")
    if catalog is None:
        return key
    return catalog.resolve(key) or key  # unknown keys survive for `check` to report


def normalize(args: dict[str, Any], catalog: Catalog | None) -> dict[str, Any]:
    """Rewrite keys to their canonical spelling (`tp` -> `tp-size`).

    Later spellings of the same flag win, so `{tp: 1, tp-size: 2}` is `tp-size: 2`.
    """
    out: dict[str, Any] = {}
    for key, value in args.items():
        name = _canonical(str(key), catalog)
        out.pop(name, None)  # keep insertion order meaningful: last writer moves to the end
        out[name] = list(value) if isinstance(value, tuple) else value
    return out


def parse_argv(tokens: list[str], catalog: Catalog | None = None) -> dict[str, Any]:
    """`["serve", "--model-path", "m", "--tp", "2"]` -> `{"model-path": "m", "tp-size": 2}`."""
    tokens = list(tokens)
    while tokens and tokens[0] in _COMMAND_WORDS:
        tokens.pop(0)

    args: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("--"):
            raise ValueError(f"unexpected SGLang argument {token!r} (flags start with --)")
        key, has_inline, inline = token[2:].partition("=")

        # Collect the values that follow, up to the next flag.
        following: list[str] = []
        if has_inline:
            following = [inline]
        else:
            while index < len(tokens) and not tokens[index].startswith("--"):
                following.append(tokens[index])
                index += 1

        flag = catalog.get(key) if catalog else None
        if flag is None and key.startswith("no-") and not following:
            # `--no-x`: switch an inherited `--x` off.
            positive = catalog.get(key[3:]) if catalog else None
            if catalog is None or positive is not None:
                name = positive.name if positive else key[3:]
                args.pop(name, None)
                args[name] = False
                continue

        name = flag.name if flag else key
        args.pop(name, None)
        if flag is not None and flag.kind in ("flag", "bool"):
            if following and not has_inline:
                raise ValueError(f"--{name} is a switch and takes no value, got {following}")
            args[name] = _literal(following[0]) if following else True
        elif not following:
            if flag is not None:
                raise ValueError(f"--{name} expects a value")
            args[name] = True
        else:
            coerce = flag.coerce if flag else _literal
            values = [coerce(v) for v in following]
            many = flag.takes_list if flag else len(values) > 1
            if not many and len(values) > 1:
                raise ValueError(f"--{name} takes one value, got {following}")
            args[name] = values if many else values[0]
    return args


def to_argv(args: dict[str, Any], catalog: Catalog | None = None) -> list[str]:
    """The inverse of `parse_argv`: `{"tp-size": 2}` -> `["--tp-size", "2"]`."""
    argv: list[str] = []
    for key, value in args.items():
        flag = catalog.get(key) if catalog else None
        if value is None:
            continue
        if value is False:
            if flag is not None and flag.kind == "bool":
                argv.append(f"--no-{key}")
            continue
        if value is True:
            argv.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            if not value:
                continue
            argv.append(f"--{key}")
            argv.extend(str(item) for item in value)
        else:
            argv.extend([f"--{key}", str(value)])
    return argv
