import os

import numpy as np

from mcp_client import MCPClient, AgoraMCPClient
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from mcp_host.stt import speech_to_text


class VibeShoppingAgent:
    SYSTEM_PROMPT: str = """You are a helpful online shopping AI assistant. You can help users find products, try them virtually, buy them, and answer questions about them.

From the user's perspective, you are a human shopping assistant because all the text you generate will be synthesized by a text-to-speech model before being sent to the user.
So make sure to only output raw text without any formatting, markdown, or code blocks.

All the responses you get from the tools you call will be sent to you and displayed to the user(but the user can't see failed call results), so you don't need to repeat them in your responses.
Instead, you can comment them, or ask the user opinion, just like a human would do in a conversation. Always ask the user for confirmation before taking any action that requires payment or purchase.
If a tool requires an input that you don't have based on your knowledge and the conversation history, you should ask the user for it.
"""

    def __init__(self, model_name: str = "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic"):
        self.agora_client = AgoraMCPClient(unique_name="Agora")
        self.fewsats_client = MCPClient(unique_name="Fewsats")
        self.virtual_try_client = MCPClient(unique_name="VirtualTry")
        self.openai_client = OpenAI(
            base_url="TODO",
            api_key=os.getenv("VLLM_API_KEY", ""),
        )
        self.chat_history: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=self.SYSTEM_PROMPT),
        ]
        self.model_name = model_name

    async def connect_clients(self, fewsats_api_key: str | None = "FAKE_API_KEY"):
        await self.agora_client.connect_to_server("uvx", ["agora-mcp"])
        await self.fewsats_client.connect_to_server(
            "env", [f"FEWSATS_API_KEY={fewsats_api_key}", "uvx", "fewsats-mcp"]
        )
        await self.virtual_try_client.connect_to_server("python", ["./mcp_server.py"])

    async def chat(
        self,
        user_speech: tuple[int, np.ndarray],
        chat_history: list[ChatCompletionMessageParam],
    ):
        # Normally, we should handle the chat history internally with self.chat_history, but since we are not persisting it,
        # we will rely on gradio's session state to keep the chat history per user session.
        chat_history = (
            # If no history is provided, start with the system prompt
            chat_history or self.chat_history
        )

        user_text_message = speech_to_text(user_speech)
        chat_history.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=user_text_message,
            )
        )
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,
            stream=True,
        )