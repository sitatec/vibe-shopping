import os
from mcp_client import MCPClient, AgoraMCPClient
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class VibeShoppingAgent:
    SYSTEM_PROMPT = """You are a helpful online shopping AI assistant. You can help users find products, try them virtually, buy them, and answer questions about them.

From the user's perspective, you are a human shopping assistant because all the text you generate will be synthesized by a text-to-speech model before being sent to the user.
So make sure to only output raw text without any formatting, markdown, or code blocks.

All the responses you get from the tools you call will be sent to you and displayed to the user, so you don't need to repeat them in your responses.
Instead, you can comment them, or ask the user opinion, just like a human would do in a conversation. Always ask the user for confirmation before taking any action that requires payment or purchase.
If a tool requires an input that you don't have based on the conversation history, you should ask the user for it.
"""

    def __init__(self):
        self.agora_client = AgoraMCPClient(unique_name="Agora")
        self.fewsats_client = MCPClient(unique_name="Fewsats")
        self.virtual_try_client = MCPClient(unique_name="VirtualTry")
        self.openai_client = OpenAI(
            base_url="TODO",
            api_key=os.getenv("VLLM_API_KEY", ""),
        )
        self.chat_history: list[ChatCompletionMessageParam] = [
            { "role": "system", "content": self.SYSTEM_PROMPT }
        ]

    async def connect_clients(self, fewsats_api_key: str | None = None):
        await self.agora_client.connect_to_server("uvx", ["agora-mcp"])
        await self.fewsats_client.connect_to_server(
            "env", [f"FEWSATS_API_KEY={fewsats_api_key}", "uvx", "fewsats-mcp"]
        )
        await self.virtual_try_client.connect_to_server("python", ["./mcp_server.py"])

    async def chat(self, user_speech: bytes):
        pass
