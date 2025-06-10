import modal
from pathlib import Path

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "vllm==0.9.0.1",
        "huggingface_hub[hf_transfer]==0.32.4",
        "flashinfer-python==0.2.6.post1"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
        }
    )
    .add_local_file(str(Path(__file__).resolve()), "/root/configs.py")
)

MODEL_NAME = "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic"
MODEL_REVISION = "3f96d104cdf17d4697995d2848efe6d313494ce5"

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True, environment_name="vibe-shopping"
)
vllm_cache_vol = modal.Volume.from_name(
    "vllm-cache", create_if_missing=True, environment_name="vibe-shopping"
)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-cache", create_if_missing=True, environment_name="vibe-shopping"
)

N_GPU = 1
API_KEY = modal.Secret.from_name(
    "vibe-shopping-secrets", required_keys=["API_KEY"], environment_name="vibe-shopping"
)
MINUTE = 60
VLLM_PORT = 8000
