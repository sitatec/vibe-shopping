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
    },
    secrets=[API_KEY],
)
@modal.concurrent(
    max_inputs=50  # maximum number of concurrent requests per aut-scaling replica
)
@modal.web_server(port=VLLM_PORT, startup_timeout=25 * MINUTE)
def serve_llm():
    import subprocess
    import os
    import requests

    chat_template_path = "/root/chat-template.jinja"

    if not os.path.exists(chat_template_path):
        template_json = requests.get(
            "https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/resolve/main/chat_template.json"
        ).json()
        with open(chat_template_path, "w") as f:
            f.write(template_json["chat_template"])

    cmd = [
        "vllm",
        "serve",
        "--model",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--uvicorn-log-level=info",
        "--tool-call-parser",
        "mistral",
        "--enable-auto-tool-choice",
        "--chat-template",
        chat_template_path,
        "--limit-mm-per-prompt",
        "image=20",
        "--tensor-parallel-size",
        str(N_GPU),
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        os.environ["API_KEY"],
        "--tensor-parallel-size",
        str(N_GPU),
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
