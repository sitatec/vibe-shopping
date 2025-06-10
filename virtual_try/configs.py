import modal
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.7.0",
        "torchvision",
        "diffusers==0.33.1",
        "transformers==4.52.4",
        "accelerate==1.7.0",
        "opencv-python-headless",
        "huggingface_hub[hf_transfer]==0.32.4",
        "git+https://github.com/sitatec/lang-segment-anything.git",
        "https://github.com/mit-han-lab/nunchaku/releases/download/v0.3.1dev20250609/nunchaku-0.3.1.dev20250609+torch2.7-cp312-cp312-linux_x86_64.whl#sha256=1518f6c02358545fd0336a6a74547e2c875603b381d5ce75b1664f981105b141",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(str(Path(__file__).resolve()), "/root/configs.py")
    .add_local_file(str(Path(__file__).parent / "auto_masker.py"), "/root/auto_masker.py")
)

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, environment_name="vibe-shopping"
)
SECRETS = modal.Secret.from_name(
    "vibe-shopping-secrets",
    required_keys=["VT_API_KEY", "HF_TOKEN"],
    environment_name="vibe-shopping",
)
MINUTE = 60

modal_class_config = {
    "image": image,
    "gpu": "A100-40GB",
    "volumes": {
        "/root/.cache/huggingface": hf_cache_vol,
    },
    "secrets": [SECRETS],
    "scaledown_window": (
        10 * MINUTE
        # how long should we stay up with no requests? Keep it low to minimize credit usage for now.
    ),
    "timeout": 10 * MINUTE,  # how long should we wait for container start?
}
