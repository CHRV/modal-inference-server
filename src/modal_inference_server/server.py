"""Turn an `SGLangConfig` into a deployable Modal App."""

from __future__ import annotations

import modal

from .config import SGLangConfig
from .image import build_image, build_volumes

__all__ = ["build_app", "build_server", "make_server_class"]


def build_app(
    config: SGLangConfig,
    *,
    app: modal.App | None = None,
    name: str | None = None,
) -> modal.App:
    """Build a Modal App serving `config` over an OpenAI-compatible HTTP API.

    The returned App is ready for `modal serve`, `modal deploy` or `modal run`.
    The server handle is also available as `app.sglang`.

    Pass `app` to attach the server to an App you already own.
    """
    app = app or modal.App(config.app_name)
    app.sglang = build_server(config, app=app, name=name)
    return app


def build_server(
    config: SGLangConfig,
    *,
    app: modal.App,
    name: str | None = None,
) -> modal.Server:
    """Register an SGLang `@app.server` on `app` and return its handle."""
    config.validate()
    return app.server(
        image=build_image(config),
        gpu=config.gpu_spec,
        volumes=build_volumes(config),
        secrets=[modal.Secret.from_name(s) for s in config.secrets],
        port=config.port,
        name=name,
        # The class is created at runtime, so it has to travel by value.
        serialized=True,
        compute_region=config.region,
        routing_region=config.routing_region,
        cloud=config.cloud,
        min_containers=config.min_containers,
        max_containers=config.max_containers,
        target_concurrency=config.target_concurrency,
        scaledown_window=config.scaledown_window,
        startup_timeout=config.startup_timeout,
        exit_grace_period=config.exit_grace_period,
        unauthenticated=config.unauthenticated,
        enable_memory_snapshot=config.gpu_snapshot,
        experimental_options={"enable_gpu_snapshot": True} if config.gpu_snapshot else None,
    )(make_server_class(config))


def make_server_class(config: SGLangConfig) -> type:
    """Build the class that supervises `sglang.launch_server` in the container.

    Everything the container needs is captured below as plain data (strings,
    ints, lists), so the class cloudpickles by value: the image never needs
    this library installed.
    """
    argv = config.launch_command()
    port = config.port
    model_name = config.model_name
    boot_timeout = config.startup_timeout
    warmup_requests = config.warmup_requests
    gpu_snapshot = config.gpu_snapshot

    class SGLangServer:
        # With a GPU snapshot, `start` runs once, before the snapshot: boot,
        # warm up, then have SGLang release its GPU memory so the snapshot
        # carries no KV cache pages. Restored replicas only run `wake`.
        # Modal rejects `snap=True` without snapshots, hence the variable.
        @modal.enter(snap=gpu_snapshot)
        def start(self):
            import subprocess
            import time
            import urllib.request

            print("starting SGLang:", " ".join(argv), flush=True)
            # Environment comes from the image, which Modal has already applied.
            self.process = subprocess.Popen(argv)

            # Modal marks a replica ready as soon as `enter` returns, so we
            # block here until SGLang actually answers, and fail loudly with
            # the exit code if it dies during model load.
            deadline = time.monotonic() + boot_timeout
            while time.monotonic() < deadline:
                if (code := self.process.poll()) is not None:
                    raise RuntimeError(f"SGLang exited with code {code} during startup")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=5
                    ):
                        break
                except Exception:  # noqa: BLE001 — server not up yet
                    time.sleep(2)
            else:
                self.process.terminate()
                raise TimeoutError(f"SGLang was not healthy within {boot_timeout}s")

            self._warmup()
            if gpu_snapshot:
                self._post("/release_memory_occupation")
                print("SGLang asleep, ready to snapshot", flush=True)
            else:
                print("SGLang ready", flush=True)

        @modal.enter()
        def wake(self):
            if gpu_snapshot:
                self._post("/resume_memory_occupation")
                print("SGLang restored from snapshot and ready", flush=True)

        def _post(self, path):
            import urllib.request

            request = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=300):
                pass

        def _warmup(self):
            """Pay for kernel autotuning and the first CUDA graphs here, not
            on a user's first request."""
            import json
            import urllib.request

            payload = json.dumps(
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 8,
                }
            ).encode()
            for _ in range(warmup_requests):
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=120):
                    pass

        @modal.exit()
        def stop(self):
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except Exception:  # noqa: BLE001
                self.process.kill()

    SGLangServer.__name__ = SGLangServer.__qualname__ = "SGLangServer"
    return SGLangServer
