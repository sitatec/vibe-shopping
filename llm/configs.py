import modal

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.0.1",
        "huggingface_hub[hf_transfer]==0.32.4",
        "flashinfer-python==0.2.5",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1", 
        }
    )
)

MODELS_DIR = "/model_weights"
MODEL_NAME = "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic"
MODEL_REVISION = "3f96d104cdf17d4697995d2848efe6d313494ce5"

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)


N_GPU = 1
API_KEY = modal.secret.Secret("vllm_api_key")

MINUTE = 60 

VLLM_PORT = 8000