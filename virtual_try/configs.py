import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.7.0",
        "torchvision",
        "diffusers==0.33.1",
        "transformers==4.52.4",
        "accelerate==1.7.0",
        "huggingface_hub[hf_transfer]==0.32.4",
        "git+https://github.com/luca-medeiros/lang-segment-anything.git@e9af744d999d85eb4d0bd59a83342ecdc2bd2461",
        "https://github.com/mit-han-lab/nunchaku/releases/download/v0.3.0/nunchaku-0.3.0+torch2.7-cp312-cp312-linux_x86_64.whl#sha256=ed28665515075050c8ef1bacd16845b85aa4335f6c760d6fa716d3b090909d8d7",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
API_KEY = modal.Secret.from_name("vibe-shopping-secrets", required_keys=["VT_API_KEY"])
MINUTE = 60
PORT = 8000
