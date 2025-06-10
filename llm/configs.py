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

CHAT_TEMPLATE = '{%- set today = strftime_now("%Y-%m-%d") %}\n{%- set default_system_message = "You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYour knowledge base was last updated on 2023-10-01. The current date is " + today + ".\\n\\nWhen you\'re not sure about some information, you say that you don\'t have the information and don\'t make up anything.\\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \\"What are some good restaurants around me?\\" => \\"Where are you?\\" or \\"When is the next flight to Tokyo\\" => \\"Where do you travel from?\\")" %}\n\n{{- bos_token }}\n\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- if messages[0][\'content\'] is string %}\n        {%- set system_message = messages[0][\'content\'] %}\n        {%- set loop_messages = messages[1:] %}\n    {%- else %}\n        {%- set system_message = messages[0][\'content\'][0][\'text\'] %}\n        {%- set loop_messages = messages[1:] %}\n    {%- endif %}\n{%- else %}\n    {%- set system_message = default_system_message %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- elif tools is not none %}\n    {%- set parallel_tool_prompt = "You are a helpful assistant that can call tools. If you call one or more tools, format them in a single JSON array or objects, where each object is a tool call, not as separate objects outside of an array or multiple arrays. Use the format [{\\"name\\": tool call name, \\"arguments\\": tool call arguments}, additional tool calls] if you call more than one tool. If you call tools, do not attempt to interpret them or otherwise provide a response until you receive a tool call result that you can interpret for the user." %}\n    {%- if system_message is defined %}\n        {%- set system_message = parallel_tool_prompt + "\\n\\n" + system_message %}\n    {%- else %}\n        {%- set system_message = parallel_tool_prompt %}\n    {%- endif %}\n{%- endif %}\n{{- \'[SYSTEM_PROMPT]\' + system_message + \'[/SYSTEM_PROMPT]\' }}\n\n{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}\n\n{%- set filtered_messages = [] %}\n{%- for message in loop_messages %}\n    {%- if message["role"] not in ["tool", "tool_results"] and not message.get("tool_calls") %}\n        {%- set filtered_messages = filtered_messages + [message] %}\n    {%- endif %}\n{%- endfor %}\n\n{%- for message in filtered_messages %}\n    {%- if (message["role"] == "user") != (loop.index0 % 2 == 0) %}\n        {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}\n    {%- endif %}\n{%- endfor %}\n\n{%- for message in loop_messages %}\n    {%- if message["role"] == "user" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- "[AVAILABLE_TOOLS] [" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- \'{"type": "function", "function": {\' }}\n                {%- for key, val in tool.items() if key != "return" %}\n                    {%- if val is string %}\n                        {{- \'"\' + key + \'": "\' + val + \'"\' }}\n                    {%- else %}\n                        {{- \'"\' + key + \'": \' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- ", " }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- "}}" }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- else %}\n                    {{- "]" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- "[/AVAILABLE_TOOLS]" }}\n        {%- endif %}\n        {%- if message[\'content\'] is string %}\n        {{- \'[INST]\' + message[\'content\'] + \'[/INST]\' }}\n        {%- else %}\n                {{- \'[INST]\' }}\n                {%- for block in message[\'content\'] %}\n                        {%- if block[\'type\'] == \'text\' %}\n                                {{- block[\'text\'] }}\n                        {%- elif block[\'type\'] == \'image\' or block[\'type\'] == \'image_url\' %}\n                                {{- \'[IMG]\' }}\n                            {%- else %}\n                                {{- raise_exception(\'Only text and image blocks are supported in message content!\') }}\n                            {%- endif %}\n                    {%- endfor %}\n                {{- \'[/INST]\' }}\n            {%- endif %}\n    {%- elif message["role"] == "tool_calls" or message.tool_calls is defined %}\n        {%- if message.tool_calls is defined %}\n            {%- set tool_calls = message.tool_calls %}\n        {%- else %}\n            {%- set tool_calls = message.content %}\n        {%- endif %}\n        {{- "[TOOL_CALLS] [" }}\n        {%- for tool_call in tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length < 9 %}\n                {{- raise_exception("Tool call IDs should be alphanumeric strings with length >= 9! (1)" + tool_call.id) }}\n            {%- endif %}\n            {{- \', "id": "\' + tool_call.id[-9:] + \'"}\' }}\n            {%- if not loop.last %}\n                {{- ", " }}\n            {%- else %}\n                {{- "]" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message[\'role\'] == \'assistant\' %}\n        {%- if message[\'content\'] is string %}\n            {{- message[\'content\'] + eos_token }}\n        {%- else %}\n            {{- message[\'content\'][0][\'text\'] + eos_token }}\n        {%- endif %}\n    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- \'[TOOL_RESULTS] {"content": \' + content|string + ", " }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length < 9 %}\n            {{- raise_exception("Tool call IDs should be alphanumeric strings with length >= 9! (2)" + message.tool_call_id) }}\n        {%- endif %}\n        {{- \'"call_id": "\' + message.tool_call_id[-9:] + \'"}[/TOOL_RESULTS]\' }}\n    {%- else %}\n        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}\n    {%- endif %}\n{%- endfor %}\n'
