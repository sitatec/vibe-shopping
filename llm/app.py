import modal
from configs import (
    vllm_image,
    hf_cache_vol,
    vllm_cache_vol,
    MODEL_NAME,
    MODEL_REVISION,
    MINUTE,
    N_GPU,
    API_KEY,
    VLLM_PORT,
)


app = modal.App("vibe-shopping")


@app.function(
    name="vibe-shopping-llm",
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    cpu=5, # 10vCPUs
    memory=16,  # 16 GB RAM
    scaledown_window=(
        1 * MINUTE
        # how long should we stay up with no requests? Keep it low to minimize credit usage for now.
    ),
    timeout=10 * MINUTE,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[API_KEY],
)
@modal.concurrent(
    max_inputs=50  # maximum number of concurrent requests per aut-scaling replica
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTE)
def serve():
    import subprocess
    import os

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--tokenizer-mode",
        "mistral",
        "--config-format",
        "mistral",
        "--load-format",
        "mistral",
        "--tool-call-parser",
        "mistral",
        "--enable-auto-tool-choice",
        "--limit-mm-per-prompt",
        "image=5",
        "--tensor-parallel-size",
        str(N_GPU),
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        os.environ["API_KEY"],
    ]

    subprocess.Popen(cmd)
