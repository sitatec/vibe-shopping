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
        "flashinfer-python==0.2.6.post1",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
        }
    )
    .add_local_file(str(Path(__file__).resolve()), "/root/configs.py")
)

# MODEL_NAME = "RedHatAI/Qwen2.5-VL-72B-Instruct-quantized.w4a16"
# MODEL_REVISION = "fa775a982552b2e44f2a12b6cdc793348cd4f1c4"
MODEL_NAME = "BCCard/Qwen2.5-VL-32B-Instruct-FP8-Dynamic"
MODEL_REVISION = "9ee1aef586e9a8ccceaee339ffaf44676eb2092c"

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

CHAT_TEMPLATE = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\nTools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments based on the provided signatures within <tool-call></tool-call> XML tags:\\n<tool-call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool-call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_call) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_call %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool-call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool-call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
