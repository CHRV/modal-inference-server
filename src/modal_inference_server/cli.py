"""Command line interface: serve, deploy and query SGLang models on Modal.

    modal-sglang <action> [Modal options] [--from SOURCE]... -- [serve] [SGLang flags]

Modal options (GPU, image, secrets, autoscaling...) go left of `--`; right of
it is a plain `sglang serve` command line, passed through as-is:

    modal-sglang run --gpu L4:2 -- serve --model-path Qwen/Qwen3-8B --tp 2
    modal-sglang deploy --from cookbook:Qwen3.6-35B-A3B@H100 --min-containers 1
    modal-sglang show --yaml --from hf:Qwen/Qwen3-8B -- --context-length 32768
    modal-sglang chat --app sglang-qwen-qwen3-8b "hello"

Because the server class is serialized by value, none of this needs a `.py`
file on disk — the config assembled from these flags is enough.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import textwrap
import threading
import types
import typing
from typing import Any

import modal

from .client import chat, stream_chat, wait_for_server
from .config import ConfigError, SGLangConfig, UnsupportedGPUError
from .flags import Catalog, catalog_for_image, extract_catalog
from .server import build_app
from .sources import Composed, Layer, compose

__all__ = ["main", "build_parser", "compose_from_args"]

DEFAULT_SERVER_NAME = "SGLangServer"

# Help text for the options generated from SGLangConfig. Fields missing here
# still get a flag — see `test_cli.py`, which fails if any field is unreachable.
FIELD_HELP: dict[str, str] = {
    "gpu": "Modal GPU, TYPE or TYPE:COUNT, e.g. L4, H100:2 (a count alone is kept "
           "from --from when omitted)",
    "name": "Modal App name (defaults to a slug of the model)",
    "port": "port SGLang listens on inside the container",
    "region": "region where containers run, e.g. us-east",
    "routing_region": "region of the Modal proxies in front of the server",
    "cloud": "cloud provider: aws, gcp, oci, auto",
    "allow_unsupported_gpu": "skip the compute-capability check",
    "validate_flags": "fail on unknown/invalid SGLang flags (--no-validate-flags: warn only)",
    "gpu_snapshot": "restore replicas from a GPU memory snapshot (single GPU, deploy only)",
    "image": "SGLang container image to run",
    "pip": "extra package to install (repeatable)",
    "hf_cache_volume": "Modal Volume caching model weights",
    "hf_cache_path": "where that Volume is mounted",
    "volumes": "extra Volume as PATH=VOLUME_NAME (repeatable)",
    "secrets": "Modal Secret to inject (repeatable)",
    "env": "environment variable as KEY=VALUE (repeatable)",
    "min_containers": "replicas to keep warm at all times",
    "max_containers": "cap on concurrent replicas",
    "target_concurrency": "requests per replica the autoscaler aims for",
    "scaledown_window": "seconds an idle replica waits before shutting down",
    "startup_timeout": "seconds allowed for a replica to become healthy",
    "exit_grace_period": "seconds to drain in-flight requests on shutdown",
    "unauthenticated": "serve without Modal proxy auth",
    "warmup_requests": "requests sent on boot before a replica is ready",
}

# Fields whose CLI flag reads better under a different name.
FLAG_OVERRIDES: dict[str, str] = {
    "secrets": "--secret",
    "volumes": "--volume",
}

# Actions that build a config, and so take SGLang flags after `--`.
CONFIG_COMMANDS = ("run", "deploy", "show")

EPILOG = """\
SGLang flags go after `--`, exactly as for `sglang serve`; aliases and argparse
prefixes work (--tp, --model). `--no-FLAG` switches off an inherited switch.

sources for --from (repeatable, later wins, CLI wins over all):
  ./config.yaml | https://...yaml     a config file (JSON works too)
  cookbook:MODEL[@GPU][/VARIANT]      SGLang Cookbook preset (see `presets`)
  hf:ORG/NAME                         cookbook preset for that model, else the model
  hf:ORG/NAME/path/config.yaml        a config file in a Hugging Face repo
"""


def _key_value(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    key, _, value = text.partition("=")
    return key, value


def _flag_for(name: str) -> str:
    return FLAG_OVERRIDES.get(name, "--" + name.replace("_", "-"))


def _unwrap_optional(hint: Any) -> Any:
    """`str | None` -> `str`, leaving `dict[str, str]` and friends intact.

    Unwrapping must be limited to unions: `get_args(dict[str, str])` returns
    the key and value types, not union members, so unwrapping blindly turns
    every container into its first parameter.
    """
    if typing.get_origin(hint) in (typing.Union, types.UnionType):
        present = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        return present[0] if present else hint
    return hint


def _modal_fields() -> list[dataclasses.Field]:
    return [f for f in dataclasses.fields(SGLangConfig) if f.name != "sglang"]


def add_config_options(parser: argparse.ArgumentParser) -> None:
    """Generate one option per Modal field of SGLangConfig, plus --from/--unset.

    Driving this off the dataclass means a new field is reachable from the CLI
    the moment it is declared, with no second list to keep in sync. Every
    option defaults to "not given" so lower layers and defaults apply.
    """
    sources = parser.add_argument_group("config sources")
    sources.add_argument(
        "--from", dest="sources", action="append", default=[], metavar="SOURCE",
        help="start from a config file, URL, cookbook: or hf: preset (repeatable)",
    )
    sources.add_argument(
        "--unset", action="append", default=[], metavar="KEY",
        help="drop an inherited SGLang flag or Modal field, e.g. --unset speculative-algorithm",
    )

    hints = typing.get_type_hints(SGLangConfig)
    group = parser.add_argument_group("Modal options")
    for field in _modal_fields():
        flag = _flag_for(field.name)
        help_text = FIELD_HELP.get(field.name, f"set {field.name}")
        base = _unwrap_optional(hints[field.name])

        if base is bool:
            group.add_argument(flag, dest=field.name, default=None,
                               action=argparse.BooleanOptionalAction, help=help_text)
        elif typing.get_origin(base) is dict:
            group.add_argument(flag, dest=field.name, default=None, action="append",
                               type=_key_value, metavar="KEY=VALUE", help=help_text)
        elif typing.get_origin(base) is tuple:
            group.add_argument(flag, dest=field.name, default=None, action="append",
                               metavar="VALUE", help=help_text)
        else:
            group.add_argument(flag, dest=field.name, default=None, type=base, help=help_text)


def modal_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """The Modal fields given on the command line, ignoring the others."""
    hints = typing.get_type_hints(SGLangConfig)
    out: dict[str, Any] = {}
    for field in _modal_fields():
        value = getattr(args, field.name, None)
        if value is None:
            continue
        base = _unwrap_optional(hints[field.name])
        if typing.get_origin(base) is dict:
            value = dict(value)
        elif typing.get_origin(base) is tuple:
            value = tuple(value)
        out[field.name] = value
    return out


def compose_from_args(args: argparse.Namespace, sglang_argv: list[str]) -> Composed:
    """Stack --from sources, then the CLI's own Modal options and SGLang flags."""
    layers = [Layer.ref(source) for source in args.sources]
    layers.append(Layer("cli", modal=modal_overrides(args), argv=sglang_argv or None))
    return compose(layers, unset=args.unset)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal-sglang",
        description="Serve SGLang models on Modal.",
        usage="modal-sglang <action> [Modal options] [--from SOURCE]... -- [serve] [SGLang flags]",
    )
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="<action>")

    def config_command(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subcommands.add_parser(
            name, help=help_text, description=help_text, epilog=EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            usage=f"modal-sglang {name} [options] [--from SOURCE]... -- [serve] [SGLang flags]",
        )
        return sub

    show = config_command("show", "print the resolved config and launch command (offline)")
    output = show.add_mutually_exclusive_group()
    output.add_argument("--yaml", action="store_true",
                        help="print the merged config as YAML, reusable with --from")
    output.add_argument("--json", action="store_true", help="same, as JSON")
    show.add_argument("--full", action="store_true",
                      help="with --yaml/--json, include Modal fields left at their default")
    add_config_options(show)

    run = config_command("run", "run an ephemeral server in the foreground")
    run.add_argument("--detach", action="store_true",
                     help="keep the App running if this client disconnects")
    add_config_options(run)

    deploy = config_command("deploy", "deploy a persistent server")
    add_config_options(deploy)

    ask = subcommands.add_parser("chat", help="send a prompt to a running server")
    ask.add_argument("prompt", help="the user message to send")
    target = ask.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", help="server URL")
    target.add_argument("--app", help="deployed Modal App name to look the URL up from")
    ask.add_argument("--server-name", default=DEFAULT_SERVER_NAME,
                     help=f"server name within the App (default: {DEFAULT_SERVER_NAME})")
    ask.add_argument("--model", help="value to send as `model`")
    ask.add_argument("--max-tokens", type=int, default=256)
    ask.add_argument("--session-id", help="sticky-routing key for KV cache reuse")
    ask.add_argument("--no-stream", action="store_true", help="wait for the full reply")
    ask.add_argument("--timeout", type=float, default=15 * 60,
                     help="seconds to wait for readiness")

    flags = subcommands.add_parser("flags", help="list or search the SGLang flags (offline)")
    flags.add_argument("query", nargs="*", help="words to look for in names and help")
    flags.add_argument("--image", default=SGLangConfig.image, help="SGLang image to describe")
    flags.add_argument("--json", action="store_true", help="machine-readable output")

    presets = subcommands.add_parser("presets", help="list SGLang Cookbook presets (offline)")
    presets.add_argument("query", nargs="*", help="model name, or words to search for")
    presets.add_argument("--json", action="store_true", help="machine-readable output")

    sync = subcommands.add_parser(
        "sync-flags", help="extract the flag catalog of an SGLang image (runs on Modal, CPU)")
    sync.add_argument("--image", default=SGLangConfig.image)

    schema = subcommands.add_parser(
        "schema", help="print the JSON Schema of config files, SGLang flags included")
    schema.add_argument("--image", default=SGLangConfig.image)
    return parser


def _say(message: str = "") -> None:
    """Print a status line that survives being piped to a file."""
    print(message, flush=True)


def _complain(message: str) -> None:
    sys.stdout.flush()  # keep reports above the complaints about them
    print(message, file=sys.stderr, flush=True)


def _report(config: SGLangConfig) -> bool:
    """Print the config's warnings and errors on stderr; True if it has no errors."""
    errors, warnings = config.check()
    for warning in warnings:
        _complain(f"warning: {warning}")
    for error in errors:
        _complain(f"error: {error}")
    return not errors


def _checked(args: argparse.Namespace, sglang_argv: list[str]) -> SGLangConfig | None:
    """Compose the config, print warnings, and return None if it has errors."""
    config = compose_from_args(args, sglang_argv).config
    if not _report(config):
        return None
    config.validate()  # the GPU guard raises UnsupportedGPUError
    return config


def _show(args: argparse.Namespace, sglang_argv: list[str]) -> int:
    composed = compose_from_args(args, sglang_argv)
    config = composed.config
    if args.yaml or args.json:
        data = config.to_dict(full=args.full)
        _say(json.dumps(data, indent=2) if args.json else config.to_yaml(full=args.full).rstrip())
        return 0 if _report(config) else 1

    catalog = config.catalog
    origin = composed.provenance
    _say(f"app name:  {config.app_name}")
    _say(f"gpu:       {config.gpu} (compute capability {config.compute_capability})"
         f"{_from(origin, 'modal.gpu')}")
    _say(f"image:     {config.image}{_from(origin, 'modal.image')}")
    _say(f"catalog:   {'SGLang ' + catalog.version if catalog else 'none (flags unchecked)'}")
    _say("sglang:")
    resolved = config.resolved_args()
    width = max(map(len, resolved), default=0)
    value_width = min(max((len(_render(v)) for v in resolved.values()), default=0), 40)
    for key, value in resolved.items():
        source = origin.get(f"sglang.{key}") or ("managed" if key in ("host", "port") else "auto")
        _say(f"  {key:<{width}}  {_render(value):<{value_width}}  <- {source}")
    _say(f"command:   {' '.join(config.launch_command())}")

    _report(config)
    try:
        config.validate()
    except UnsupportedGPUError as error:
        _complain(f"\nthis config would be rejected:\n{error}")
        return 1
    except ConfigError:
        return 1
    return 0


def _from(provenance: dict[str, str], key: str) -> str:
    return f"   <- {provenance[key]}" if key in provenance else ""


def _render(value: Any) -> str:
    if value is False:
        return "(off)"
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value).lower() if isinstance(value, bool) else str(value)


def _run(args: argparse.Namespace, sglang_argv: list[str]) -> int:
    config = _checked(args, sglang_argv)
    if config is None:
        return 1
    if config.gpu_snapshot:
        _complain("warning: Modal never snapshots ephemeral Apps, so `run` boots normally; "
                  "use `deploy` to get GPU snapshot cold starts")
    app = build_app(config)
    # The interrupt has to be caught outside the context manager: catching it
    # inside lets it surface again while Modal tears the App down, which ends
    # a plain ctrl-c on a traceback.
    try:
        with modal.enable_output(), app.run(detach=args.detach):
            _say(f"\nserving at {app.sglang.get_url()}")
            _say("press ctrl-c to stop")
            threading.Event().wait()
    except KeyboardInterrupt:
        _say("\nstopping")
    return 0


def _deploy(args: argparse.Namespace, sglang_argv: list[str]) -> int:
    config = _checked(args, sglang_argv)
    if config is None:
        return 1
    app = build_app(config)
    with modal.enable_output():
        app.deploy(name=config.app_name)
    _say(f"\ndeployed at {app.sglang.get_url()}")
    return 0


def _chat(args: argparse.Namespace) -> int:
    url = args.url or modal.Server.from_name(args.app, args.server_name).get_url()

    wait_for_server(url, timeout=args.timeout)
    messages = [{"role": "user", "content": args.prompt}]
    if args.no_stream:
        print(chat(url, messages, model=args.model, max_tokens=args.max_tokens,
                   session_id=args.session_id))
    else:
        for piece in stream_chat(url, messages, model=args.model,
                                 max_tokens=args.max_tokens, session_id=args.session_id):
            print(piece, end="", flush=True)
        print()
    return 0


def _catalog_or_fail(image: str) -> Catalog:
    catalog = catalog_for_image(image)
    if catalog is None:
        raise ConfigError(f"no flag catalog for {image}; run `modal-sglang sync-flags "
                          f"--image {image}` first")
    return catalog


def _flags(args: argparse.Namespace) -> int:
    catalog = _catalog_or_fail(args.image)
    found = catalog.search(" ".join(args.query)) if args.query else list(catalog.flags.values())
    if args.json:
        _say(json.dumps({f.name: f.to_dict() for f in found}, indent=2))
        return 0
    for flag in found:
        spelled = ", ".join("--" + s for s in (flag.name, *flag.aliases))
        kind = "switch" if flag.kind == "flag" else (
            "true/false" if flag.kind == "bool" else
            (flag.type or "value") + (" ..." if flag.takes_list else ""))
        default = "" if flag.default in (None, False, []) else f"  (default: {flag.default})"
        _say(f"{spelled}  <{kind}>{default}{'  [deprecated]' if flag.deprecated else ''}")
        if flag.choices:
            _say(f"    one of: {', '.join(map(str, flag.choices))}")
        if flag.help:
            _say(textwrap.fill(flag.help, width=96, initial_indent="    ",
                               subsequent_indent="    "))
    _complain(f"{len(found)} flag(s), SGLang {catalog.version}")
    return 0


def _presets(args: argparse.Namespace) -> int:
    from .presets import find_preset, search

    query = " ".join(args.query)
    exact = find_preset(query) if query else None
    if exact is not None:
        if args.json:
            _say(json.dumps(dataclasses.asdict(exact), indent=2))
            return 0
        _say(f"{exact.model}  ({exact.model_path})")
        if exact.description:
            _say(textwrap.fill(exact.description, width=96, initial_indent="  ",
                               subsequent_indent="  "))
        _say(f"  source: {exact.source}")
        for key, spec in exact.configs.items():
            attributes = ", ".join(f"{k}={v}" for k, v in spec.get("attributes", {}).items())
            _say(f"  cookbook:{exact.model}@{key:<30} {spec['modal']['gpu']:<9} {attributes}")
        return 0

    found = search(query)
    if args.json:
        _say(json.dumps([{"model": p.model, "model_path": p.model_path, "gpus": p.gpus}
                         for p in found], indent=2))
        return 0
    width = max((len(p.model) for p in found), default=0)
    for preset in found:
        _say(f"{preset.model:<{width}}  {', '.join(preset.gpus):<24}  {preset.model_path}")
    _complain(f"{len(found)} preset(s); `modal-sglang presets MODEL` shows the variants")
    return 0


def _sync_flags(args: argparse.Namespace) -> int:
    _say(f"extracting flags from {args.image} (CPU sandbox on Modal)...")
    with modal.enable_output():
        catalog = extract_catalog(args.image)
    path = catalog.save()
    catalog_for_image.cache_clear()
    _say(f"{len(catalog.flags)} flags for SGLang {catalog.version} -> {path}")
    return 0


def _schema(args: argparse.Namespace) -> int:
    from .schema import json_schema

    _say(json.dumps(json_schema(catalog_for_image(args.image)), indent=2))
    return 0


COMMANDS = {"chat": _chat, "flags": _flags, "presets": _presets,
            "sync-flags": _sync_flags, "schema": _schema}
CONFIG_HANDLERS = {"show": _show, "run": _run, "deploy": _deploy}


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split at the first `--`: (modal-sglang argv, SGLang argv)."""
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1:]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    ours, sglang_argv = split_argv(argv)
    args = parser.parse_args(ours)
    if sglang_argv and args.command not in CONFIG_HANDLERS:
        parser.error(f"`{args.command}` takes no SGLang flags after --")
    try:
        if args.command in CONFIG_HANDLERS:
            return CONFIG_HANDLERS[args.command](args, sglang_argv)
        return COMMANDS[args.command](args)
    except UnsupportedGPUError as error:
        _complain(f"error: {error}")
        return 1
    except ConfigError as error:
        _complain(f"error: {error}")
        return 2
    except KeyboardInterrupt:
        return 130  # conventional exit code for SIGINT
    except BrokenPipeError:  # `modal-sglang flags | head`
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115 — silence the flush at exit
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
