"""Dev deployment: a 0.5B model on a single L4.

    modal serve examples/serve_qwen.py    # hot-reloading dev server
    modal run   examples/serve_qwen.py    # boot it and send one prompt
    modal deploy examples/serve_qwen.py   # keep it running
"""

from modal_inference_server import SGLangConfig, build_app, stream_chat, wait_for_server

config = SGLangConfig.for_model(
    "Qwen/Qwen2.5-0.5B-Instruct",
    gpu="L4",
    target_concurrency=4,
    sglang={"mem-fraction-static": 0.8},  # any `sglang serve` flag, SGLang spelling
)

app = build_app(config)


@app.local_entrypoint()
def main(prompt: str = "Explain what an LLM inference server does, in 3 sentences."):
    url = app.sglang.get_url()
    print(f"server: {url}")
    wait_for_server(url)
    for chunk in stream_chat(
        url, [{"role": "user", "content": prompt}], model=config.model_name
    ):
        print(chunk, end="", flush=True)
    print()
