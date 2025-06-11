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
    flashinfer_cache_vol,
)


app = modal.App("vibe-shopping-llm")


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=(
        5 * MINUTE
        # how long should we stay up with no requests? Keep it low to minimize credit usage for now.
    ),
    timeout=10 * MINUTE,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/root/.cache/flashinfer": flashinfer_cache_vol,
    },
    secrets=[API_KEY],
)
@modal.concurrent(
    max_inputs=50  # maximum number of concurrent requests per aut-scaling replica
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTE)
def serve_llm():
    import subprocess
    import os
    import requests

    chat_template_path = "/root/chat_template.jinja"
    if not os.path.exists(chat_template_path):
        print("Downloading chat template...")
        url = "https://raw.githubusercontent.com/edwardzjl/chat-templates/refs/heads/main/qwen2_5/chat_template.jinja"
        response = requests.get(url)
        response.raise_for_status()
        with open(chat_template_path, "w") as f:
            f.write(response.text)

    min_pixels = 128 * 28 * 28  # min 128 tokens
    max_pixels = 340 * 28 * 28  # max 340 tokens (~512x512 image)

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--uvicorn-log-level=info",
        "--tool-call-parser",
        "hermes",
        "--enable-auto-tool-choice",
        "--limit-mm-per-prompt",
        "image=10",
        "--tensor-parallel-size",
        str(N_GPU),
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        os.environ["API_KEY"],
        "--enforce-eager",
        "--chat-template",
        chat_template_path,
        # Minimize token usage
        "--mm-processor-kwargs",
        f"{{'min_pixels': {min_pixels}, 'max_pixels': {max_pixels}}}",
        # Extend context length to 65536 tokens
        "--rope-scaling",
        '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}',
        "--max-model-len",
        "65536",
    ]

    subprocess.Popen(cmd)


###### ------ FOR TESTING PURPOSES ONLY ------ ######
@app.local_entrypoint()
def test(test_timeout=25 * MINUTE, twice: bool = True):
    import os
    import json
    import time
    import urllib
    import dotenv

    dotenv.load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    print(f"Running health check for server at {serve_llm.get_web_url()}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(
                serve_llm.get_web_url() + "/health"
            ) as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {serve_llm.get_web_url()}"

    print(f"Successful health check for server at {serve_llm.get_web_url()}")

    messages = [{"role": "user", "content": "Testing! Is this thing on?"}]
    print(f"Sending a sample message to {serve_llm.get_web_url()}", *messages, sep="\n")

    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"messages": messages, "model": MODEL_NAME})
    req = urllib.request.Request(
        serve_llm.get_web_url() + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        print(json.loads(response.read().decode()))

    if twice:
        print("Sending the same message again to test caching.")
        with urllib.request.urlopen(req) as response:
            print(json.loads(response.read().decode()))
