# modal-inference-server

A small library and CLI for serving [SGLang](https://github.com/sgl-project/sglang)
models on [Modal](https://modal.com), built on `@app.server` (Modal Servers).

A deployment has two halves, each with one owner:

| | what | on the command line | in a config file |
| --- | --- | --- | --- |
| **Modal** | GPU, image, secrets, volumes, autoscaling | left of `--` | `modal:` |
| **SGLang** | every `sglang serve` flag, SGLang's own spelling | right of `--` | `sglang:` |

The SGLang half is passed through as-is, so **every** SGLang flag works, including
ones released after this library. A flag catalog extracted from the SGLang image
checks them offline, so a typo costs a millisecond instead of a container boot.

## Quickstart

```bash
modal-sglang run --gpu L4 -- serve --model-path Qwen/Qwen2.5-0.5B-Instruct
```

Right of `--` is exactly what you would type after `sglang serve` (the word `serve` is
optional). Left of it, Modal:

```bash
modal-sglang run    --gpu H100:2 --secret huggingface -- --model Qwen/Qwen3-8B --tp 2
modal-sglang deploy --gpu H100:2 --min-containers 1   -- --model Qwen/Qwen3-8B --tp 2
modal-sglang chat   --app sglang-qwen-qwen3-8b "hello"
```

Use `run --detach` for anything that boots a model. A plain `run` ties the App's
lifetime to your local client, so a dropped connection kills a replica that is minutes
into loading weights — and the traceback you get is a local gRPC error that looks
nothing like the real cause (`Stopping app - local client disconnected`).

The server speaks the OpenAI API:

```bash
curl $URL/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct",
       "messages": [{"role": "user", "content": "hi"}]}'
```

Interactive docs live at `$URL/docs`. A `503` means no replica is live yet, and SGLang
itself answers `503` on `/health` while it loads weights — both are cold start, not
failure. `wait_for_server` polls through them, and a replica is only marked ready once
`/health` returns 200 and a warmup completion has gone through, so the first real
request never pays for model load.

## Config files, presets and `--from`

A config file is YAML (JSON works too). `sglang:` takes the same keys as SGLang's own
`--config` file, so you can paste in either direction:

```yaml
# examples/qwen3-8b.yaml
modal:
  gpu: L40S
  secrets: [huggingface]
  target_concurrency: 16
sglang:
  model-path: Qwen/Qwen3-8B
  reasoning-parser: qwen3
  context-length: 32768
  enable-metrics: true
```

Values follow one rule everywhere: `true` is a bare switch, `false` turns it off, `null`
deletes an inherited value, and a list gives several values (`lora-paths: [a, b]`).

`--from` loads a config, and anything else on the command line edits it on the fly:

```bash
modal-sglang deploy --from examples/qwen3-8b.yaml --gpu H100 -- --context-length 65536
```

| `--from` source | example |
| --- | --- |
| file | `./qwen.yaml`, `cfg.json`, or a plain SGLang `--config` file |
| URL | `https://…/cfg.yaml` (`github.com/…/blob/…` is fetched raw) |
| SGLang Cookbook | `cookbook:Qwen3.6-35B-A3B`, `cookbook:Qwen3.6-35B-A3B@H200/speculative-mtp` |
| Hugging Face model | `hf:Qwen/Qwen3-8B` (the cookbook recipe if there is one, else just the model) |
| file in a HF repo | `hf:org/repo/path/config.yaml` (uses `HF_TOKEN` if set) |

Layers stack, weakest first: **defaults < `extends` < each `--from` in order < the
command line**. A config file can itself say `extends: <source>`. Within that order:

- an SGLang flag replaces the earlier value, and `--no-FLAG` switches one off;
- `--unset KEY` drops any inherited SGLang flag or Modal field;
- `env` and `volumes` merge key by key;
- a GPU without a count keeps the earlier count: `--gpu H200` over `H100:2` is
  `H200:2`. Write `H200:1` to force one.

### SGLang Cookbook presets

The [SGLang Cookbook](https://docs.sglang.io/cookbook/intro)'s recipes ship converted
— 111 models, 432 (GPU, variant) configs that Modal can run:

```bash
modal-sglang presets qwen                  # search
modal-sglang presets Qwen3.6-35B-A3B       # the variants of one model
modal-sglang show --from cookbook:Qwen3.6-35B-A3B@H100/speculative-mtp
```

`@GPU` defaults to the GPU you ask for (`--gpu B200` picks the B200 recipe), then to
the cheapest one the recipe covers; `/variant` defaults to `default`.
`scripts/import_cookbook.py` refreshes them, skipping what Modal cannot run (AMD and
Grace hardware, multi-node, more than 8 GPUs), and it fails if a recipe no longer
passes the flag catalog.

## Checking before booting: `show`

`show` resolves everything offline. It prints where each value came from, the exact
`sglang.launch_server` argv, and every problem it can find:

```
$ modal-sglang show --from cookbook:Qwen3.6-35B-A3B --gpu H200 -- --context-length 32768
app name:  sglang-qwen-qwen3-6-35b-a3b
gpu:       H200:1 (compute capability 9.0)   <- cli
image:     lmsysorg/sglang:v0.5.19-cu129
catalog:   SGLang 0.5.19
sglang:
  model-path        Qwen/Qwen3.6-35B-A3B  <- cookbook:Qwen3.6-35B-A3B@H200/default
  tp-size           1                     <- cookbook:Qwen3.6-35B-A3B@H200/default
  reasoning-parser  qwen3                 <- cookbook:Qwen3.6-35B-A3B@H200/default
  tool-call-parser  qwen3_coder           <- cookbook:Qwen3.6-35B-A3B@H200/default
  context-length    32768                 <- cli
  host              0.0.0.0               <- managed
  port              8000                  <- managed
command:   python -m sglang.launch_server --model-path Qwen/Qwen3.6-35B-A3B ...

$ modal-sglang show -- --model m --tp-sise 2
error: unknown SGLang flag --tp-sise (did you mean --tp-size, --pp-size, --ep-size?)
```

`show --yaml` prints the merged config instead, ready to commit and reuse with `--from`:

```bash
modal-sglang show --yaml --from hf:Qwen/Qwen3-8B --gpu H100 -- --context-length 65536 > prod.yaml
```

## GPU memory snapshots

`--gpu-snapshot` (`gpu_snapshot: true`) makes later replicas restore a snapshot of a
booted, warmed-up server, GPU memory included, instead of booting from scratch:

```bash
modal-sglang deploy --gpu-snapshot --gpu L4 -- --model Qwen/Qwen2.5-0.5B-Instruct
```

The first replica boots SGLang, sends the warmup requests, then calls
`/release_memory_occupation` so the snapshot carries no empty KV cache pages. Modal
then takes the snapshot. Restored replicas only call `/resume_memory_occupation`. The
SGLang flags this needs (`enable-memory-saver`, `enable-weights-cpu-backup`, plus
`enable-draft-weights-cpu-backup` with a draft model) are added unless you set them,
and so is `TORCHINDUCTOR_COMPILE_THREADS=1`.

Limits, enforced or warned about by `show`:

- **single GPU only**: `gpu: H100:2`, or `tp`/`dp`/`pp` above 1, is an error, because
  Modal's GPU snapshots do not support multi-GPU servers yet;
- **deployed Apps only**: Modal never snapshots ephemeral Apps, so `run` boots normally;
- what it saves is imports, JIT compilation and CUDA graph capture, not reading
  weights from storage;
- snapshots are per GPU type. A spec like `H100` may be upgraded to an H200, which
  needs its own snapshot; `H100!` pins it;
- Modal marks the feature alpha. It takes 2–3 snapshots per GPU type before every
  worker type is covered, and redeploying invalidates them.

## The SGLang flag catalog

SGLang's flags are not listed by hand anywhere in this library.
`modal-sglang sync-flags --image IMAGE` builds SGLang's own argparse parser inside
the image (a CPU Sandbox on Modal) and saves every option, with its aliases, `nargs`,
type, choices, default and help, to `flags/sglang-<version>.yaml`. The catalog for the
default image ships with the package. It drives:

- aliases and argparse prefixes: `--tp`, `--model`, `--speculative-algo` all work, and
  are stored under SGLang's first spelling (`tp-size`, `model-path`, …);
- the argv shape: `nargs` lists, `--x/--no-x` booleans;
- validation: unknown flags (with suggestions), choices, types, deprecations, and a
  warning when `tp × pp × dp` does not match the GPU count.

```bash
modal-sglang flags speculative          # search names and help
modal-sglang flags --json kv cache      # machine-readable
```

An image without a catalog still works: flags pass through unchecked, with a warning.
`--no-validate-flags` downgrades flag errors to warnings for one run.

## For LLMs and editors

- `modal-sglang schema` prints a JSON Schema of config files, with every SGLang flag
  of that version documented. `config.schema.json` ships with the package; point
  yaml-language-server at it for completion in editors.
- `modal-sglang show --json` / `--yaml` is the resolved config, and `flags --json` /
  `presets --json` are the catalogs, all machine-readable.
- Errors name the fix: `did you mean --tp-size`, `available: H100, H200, B200`.

## Python API

```python
# examples/serve_qwen.py
from modal_inference_server import SGLangConfig, build_app

config = SGLangConfig.for_model(
    "Qwen/Qwen2.5-0.5B-Instruct",
    gpu="L4",
    sglang={"mem-fraction-static": 0.8},  # any `sglang serve` flag
)
app = build_app(config)
```

```bash
uv run modal serve  examples/serve_qwen.py          # hot-reloading dev server
uv run modal run --detach examples/serve_qwen.py    # boot it, stream one prompt
uv run modal deploy examples/serve_qwen.py          # keep it running
```

`SGLangConfig.load(*sources, **overrides)` composes sources exactly like `--from`.
`SGLangConfig` is frozen, and `with_()` derives a variant. `sglang=` is merged rather
than replaced, so one definition can fan out into several deployments:

```python
base = SGLangConfig.load("cookbook:Qwen3.6-35B-A3B")
prod = base.with_(gpu="H200", min_containers=1, sglang={"context-length": 65536})
```

`to_dict()` / `from_dict()` / `to_yaml()` convert to and from the config-file shape.

Autoscaling can also be retuned on a *deployed* server without redeploying:

```python
import modal
modal.Server.from_name("sglang-qwen-qwen2-5-0-5b-instruct", "SGLangServer").update_autoscaler(
    min_containers=1, target_concurrency=16
)
```

## GPU support

SGLang ships **prebuilt** CUDA kernels, and the 0.5 series dropped sm75 (Turing).
A T4 therefore dies at the first RMSNorm with `no kernel image is available for
execution on the device`, and no combination of flags rescues it — the binary simply
has no code for that architecture. `build_server` checks this locally and raises
`UnsupportedGPUError` up front rather than letting you pay for a six-minute boot:

```
T4 has compute capability 7.5, but the prebuilt kernels in
lmsysorg/sglang:v0.5.19-cu129 need at least 8.0 (sgl-kernel dropped sm75 in the
0.5 series) ... Use L4, A10G or newer, or pin an SGLang 0.4.x image and set
allow_unsupported_gpu=True.
```

`L4` (sm89) is the default: the cheapest Modal GPU current SGLang actually runs on.

If you do pin an SGLang 0.4.x image for a T4, set `allow_unsupported_gpu: true` and the
Turing-safe flags below are filled in automatically for any flag you have not set:

| GPU | `dtype` | `attention-backend` | `sampling-backend` | `disable-cuda-graph` |
| --- | --- | --- | --- | --- |
| `T4` (sm75) | `float16` | `triton` | `pytorch` | `true` |
| everything else | SGLang's own defaults | | | |

Setting a flag explicitly always wins over the table, including `false`.

## Layout

```
src/modal_inference_server/
  config.py         SGLangConfig, launch_command() argv, checks, GPU capability guard
  sglang_args.py    SGLang argv <-> {flag: value}
  flags.py          flag catalog: lookup, validation, extraction from an image
  sources.py        --from / extends resolution and layer merging
  presets.py        SGLang Cookbook presets (presets/cookbook/*.yaml)
  schema.py         JSON Schema for config files
  image.py          container image and Volume mounts
  server.py         build_app() / build_server()     — the @app.server class
  client.py         stdlib OpenAI client: wait_for_server, chat, stream_chat
  cli.py            modal-sglang run / deploy / show / chat / flags / presets / ...
scripts/
  import_cookbook.py   regenerate presets from github.com/sgl-project/sgl-cookbook
```

## Tests

```bash
uv run python -m unittest discover -s tests            # no network
uv run modal run --detach examples/serve_qwen.py       # end-to-end on a real L4
```

The offline suite covers argv parsing and the launch command, the flag catalog, source
resolution and merge order, every bundled cookbook preset against the catalog, CLI
option generation, and one test that unpickles the server class in a process where
this library is deliberately unimportable. The container never has the library
installed, so that has to hold.

## License

MIT — see [LICENSE](LICENSE).
