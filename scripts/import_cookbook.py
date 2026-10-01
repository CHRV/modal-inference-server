"""Convert SGLang Cookbook recipes into modal-sglang presets.

    uv run python scripts/import_cookbook.py                 # fetch from GitHub
    uv run python scripts/import_cookbook.py --local ./sgl-cookbook

The cookbook (github.com/sgl-project/sgl-cookbook) keeps machine-readable
recipes in `data/models/generated/<sglang-version>/*.yaml`. Each model there
becomes one file in `src/modal_inference_server/presets/cookbook/`, holding a
complete `{modal, sglang}` config per (Modal GPU, variant). Recipes Modal
cannot run — AMD or Grace hardware, multi-node, prefill/decode
disaggregation, more than 8 GPUs — are skipped and listed at the end.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import yaml

from modal_inference_server.flags import catalog_for_image
from modal_inference_server.config import SGLangConfig
from modal_inference_server.presets import PRESET_DIR
from modal_inference_server.sglang_args import normalize, parse_argv

REPO = "sgl-project/sgl-cookbook"
RECIPES = "data/models/generated"

# Cookbook hardware name -> Modal GPU type. Anything else has no Modal equivalent.
MODAL_GPUS = {"H100": "H100", "H200": "H200", "B200": "B200", "A100": "A100-80GB"}
MAX_GPUS = 8  # per Modal container

# Values the cookbook still uses but SGLang has since renamed. The import
# reports any other value the flag catalog rejects, so this only grows when
# that report says so.
RENAMED_VALUES = {
    ("reasoning-parser", "nano_v3"): "nemotron_3",
}

# Cookbook model attributes that map onto SGLang flags.
LLM_ATTRIBUTE_FLAGS = {
    "reasoning_parser": "reasoning-parser",
    "tool_parser": "tool-call-parser",
    "chat_template": "chat-template",
}


def _version_key(name: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", name))


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "modal-sglang"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def recipe_files(local: Path | None) -> dict[str, tuple[str, str]]:
    """Latest version of each recipe file: {file name: (repo path, yaml text)}."""
    if local is not None:
        paths = [str(p.relative_to(local)) for p in (local / RECIPES).glob("*/*.yaml")]
        read = lambda path: (local / path).read_text()  # noqa: E731
    else:
        tree = json.loads(_get(f"https://api.github.com/repos/{REPO}/git/trees/main?recursive=1"))
        paths = [i["path"] for i in tree["tree"]
                 if i["path"].startswith(RECIPES + "/") and i["path"].endswith(".yaml")]
        read = lambda path: _get(  # noqa: E731
            f"https://raw.githubusercontent.com/{REPO}/main/{path}").decode()

    latest: dict[str, str] = {}
    for path in paths:
        name = path.rsplit("/", 1)[1]
        if name not in latest or _version_key(path) > _version_key(latest[name]):
            latest[name] = path
    return {name: (path, read(path)) for name, path in sorted(latest.items())}


def convert_config(model: dict, config: dict, gpu: str, catalog, skipped: Counter):
    """One cookbook configuration -> `{attributes, modal, sglang}`, or None."""
    if config.get("prefill") or config.get("decode"):
        skipped["prefill/decode disaggregation"] += 1
        return None
    if config.get("attributes", {}).get("nodes", "single") != "single":
        skipped["multi-node"] += 1
        return None

    engine = config["engine"]
    tp, dp, ep = engine.get("tp") or 1, engine.get("dp"), engine.get("ep")
    dp_attention = bool(engine.get("enable_dp_attention"))
    n_gpus = tp * (dp if dp and not dp_attention else 1)
    if n_gpus > MAX_GPUS:
        skipped[f"more than {MAX_GPUS} GPUs"] += 1
        return None

    sglang: dict = {"model-path": config.get("quantized_model_path") or model["model_path"]}
    sglang["tp-size"] = tp
    if dp:
        sglang["dp-size"] = dp
    if ep:
        sglang["ep-size"] = ep
    if dp_attention:
        sglang["enable-dp-attention"] = True
    for attribute, flag in LLM_ATTRIBUTE_FLAGS.items():
        value = model.get("attributes", {}).get("llm", {}).get(attribute)
        if value:
            sglang[flag] = value
    sglang.update(parse_argv([str(t) for t in engine.get("extra_args") or []], catalog))
    sglang = normalize(sglang, catalog)
    for key, value in sglang.items():
        sglang[key] = RENAMED_VALUES.get((key, value), value) if isinstance(value, str) else value

    modal: dict = {"gpu": f"{gpu}:{n_gpus}"}
    if engine.get("env_vars"):
        modal["env"] = {k: str(v) for k, v in engine["env_vars"].items()}

    attributes = {k: v for k, v in (config.get("attributes") or {}).items() if k != "nodes"}
    return {"attributes": attributes, "modal": modal, "sglang": sglang}


def convert_file(path: str, text: str, catalog, skipped: Counter) -> list[dict]:
    data = yaml.safe_load(text)
    presets = []
    for family in data.get("families", []):
        for model in family.get("models", []):
            if "llm" not in model.get("attributes", {}):
                skipped["not an LLM (diffusion)"] += 1
                continue
            configs = {}
            for hardware, spec in model.get("hardware", {}).items():
                gpu = MODAL_GPUS.get(hardware)
                for config in spec.get("configurations", []):
                    if gpu is None:
                        skipped[f"{hardware} is not a Modal GPU"] += 1
                        continue
                    converted = convert_config(model, config, gpu, catalog, skipped)
                    if converted:
                        configs[f"{gpu}/{config['name']}"] = converted
            if not configs:
                continue
            presets.append({
                "model": model["name"],
                "model_path": model["model_path"],
                "vendor": data.get("vendor"),
                "family": family.get("name"),
                "description": family.get("description"),
                "source": f"https://github.com/{REPO}/blob/main/{path}",
                "attributes": model.get("attributes", {}).get("llm", {}),
                "configs": configs,
            })
    return presets


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert SGLang Cookbook recipes into presets.")
    parser.add_argument("--local", type=Path, help="use a local sgl-cookbook checkout")
    parser.add_argument("--out", type=Path, default=PRESET_DIR)
    args = parser.parse_args()

    catalog = catalog_for_image(SGLangConfig.image)
    skipped: Counter = Counter()
    presets = []
    for path, text in recipe_files(args.local).values():
        presets.extend(convert_file(path, text, catalog, skipped))

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    for preset in presets:
        slug = re.sub(r"[^a-z0-9.]+", "-", preset["model"].lower()).strip("-")
        header = (f"# Generated by scripts/import_cookbook.py from {preset['source']}\n"
                  "# Do not edit: re-run the script to refresh.\n")
        (args.out / f"{slug}.yaml").write_text(
            header + yaml.safe_dump(preset, sort_keys=False, width=100))

    problems: Counter = Counter()
    for preset in presets:
        for spec in preset["configs"].values():
            config = SGLangConfig.from_dict({"modal": spec["modal"], "sglang": spec["sglang"]})
            errors, _ = config.check()
            problems.update(f"{preset['model']}: {error}" for error in errors)

    n_configs = sum(len(p["configs"]) for p in presets)
    print(f"wrote {len(presets)} models / {n_configs} configs to {args.out}")
    for reason, count in skipped.most_common():
        print(f"  skipped {count:4d}  {reason}", file=sys.stderr)
    for problem, count in problems.most_common():
        print(f"  INVALID {count:4d}  {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
