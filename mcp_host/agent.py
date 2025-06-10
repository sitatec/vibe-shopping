from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Generator

import numpy as np
from PIL import Image
from openai import OpenAI

from mcp_client import MCPClient, AgoraMCPClient
from mcp_host.stt import speech_to_text
from mcp_host.tts import stream_text_to_speech
from utils import ImageUploader

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionMessageToolCallParam,
        ChatCompletionToolParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartTextParam,
        ChatCompletionContentPartImageParam,
    )
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


class VibeShoppingAgent:
    SYSTEM_PROMPT: str = """You are a helpful online shopping AI assistant. You can help users find products, try them virtually, buy them, and answer questions about them.

From the user's perspective, you are a human shopping assistant because all the text you generate will be synthesized by a text-to-speech model before being sent to the user.
So make sure to only output raw text without any formatting, markdown, or code blocks.

When you get a response from a tool, if it contains something displayable and relevant to the user, you should display it instead of reading it out loud. \
Then, you can comment on it, or ask the user's opinion, just like a human would do in a conversation. 
Always ask the user for confirmation before taking any action that requires payment or purchase.
If a tool requires an input that you don't have based on your knowledge and the conversation history, you should ask the user for it. For example, if the user asks to try a product, but you don't have the target image, you should ask the user to provide it.
"""

    def __init__(
        self,
        model_name: str = "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",
        openai_api_key: str = os.getenv("OPENAI_API_KEY", ""),
        openai_api_base_url: str = os.getenv("OPENAI_API_BASE_URL", ""),
        image_uploader: ImageUploader = ImageUploader(),
    ):
        self.agora_client = AgoraMCPClient(unique_name="Agora")
        self.fewsats_client = MCPClient(unique_name="Fewsats")
        self.virtual_try_client = MCPClient(unique_name="VirtualTry")

        self.openai_client = OpenAI(
            base_url=openai_api_base_url,
            api_key=openai_api_key,
        )
        self.chat_history: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=self.SYSTEM_PROMPT),
        ]
        self.model_name = model_name

        self._mcp_clients: list[MCPClient] = [
            self.agora_client,
            self.fewsats_client,
            self.virtual_try_client,
        ]
        self.display_tool = _build_display_tool_definition()
        self.image_uploader = image_uploader

    async def connect_clients(
        self, fewsats_api_key: str = os.getenv("FEWSATS_API_KEY", "FAKE_API_KEY")
    ):
        await self.agora_client.connect_to_server("uvx", ["agora-mcp"])
        await self.fewsats_client.connect_to_server(
            "env", [f"FEWSATS_API_KEY={fewsats_api_key}", "uvx", "fewsats-mcp"]
        )
        await self.virtual_try_client.connect_to_server("python", ["./mcp_server.py"])

        self.tools = (
            await self.agora_client.tools
            + await self.fewsats_client.tools
            + await self.virtual_try_client.tools
            + [self.display_tool]
        )

    def _get_mcp_client_for_tool(self, tool_name: str) -> MCPClient | None:
        try:
            # Iterate through the clients to find the one that owns the tool and stop at the first match
            return next(
                client for client in self._mcp_clients if client.owns_tool(tool_name)
            )
        except StopIteration:
            return None

    async def chat(
        self,
        user_speech: tuple[int, np.ndarray],
        chat_history: list[ChatCompletionMessageParam],
        # Update the UI with a list of products or an image from url, or clear the UI
        update_ui: Callable[
            [list[dict[str, str]] | None, str | None, bool | None], None
        ],
        voice: str | None = None,
        input_image: Image.Image | None = None,
        input_mask: Image.Image | None = None,
    ) -> AsyncGenerator[np.ndarray, None]:
        # Normally, we should handle the chat history internally with self.chat_history, but since we are not persisting it,
        # we will rely on gradio's session state to keep the chat history per user session.
        chat_history = (
            # If no history is provided, start with the system prompt
            chat_history
            or [
                ChatCompletionSystemMessageParam(
                    role="system", content=self.SYSTEM_PROMPT
                )
            ]
        ).copy()  # Ensure we don't modify the original history

        user_message_contents: list[ChatCompletionContentPartParam] = []
        if input_image is not None:
            user_message_contents.extend(
                list(self._build_input_image_content(input_image, "input_image"))
            )
            if input_mask is not None:
                user_message_contents.extend(
                    list(self._build_input_image_content(input_mask, "input_mask"))
                )

        user_text_message = speech_to_text(user_speech)
        user_message_contents.append(
            {
                "type": "text",
                "text": user_text_message,
            }
        )
        chat_history.append(
            {
                "role": "user",
                "content": user_message_contents,
            }
        )

        while True:
            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            tool_responses: list[ChatCompletionToolMessageParam] = []
            text_chunks: list[str] = []

            async for ai_speech in self._send_to_llm(
                chat_history=chat_history,
                voice=voice,
                tool_calls=tool_calls,
                tool_responses=tool_responses,
                text_chunks=text_chunks,
                update_ui=update_ui,
            ):
                yield ai_speech

            chat_history.extend(
                [
                    {
                        "role": "assistant",
                        "content": "".join(text_chunks),
                        "tool_calls": tool_calls,
                    },
                    *tool_responses,
                ]
            )

            if not tool_responses:
                break

    async def _send_to_llm(
        self,
        chat_history: list[ChatCompletionMessageParam],
        voice: str | None,
        tool_calls: list[ChatCompletionMessageToolCallParam],
        tool_responses: list[ChatCompletionToolMessageParam],
        text_chunks: list[str],
        update_ui: Callable[
            [list[dict[str, str]] | None, str | None, bool | None], None
        ],
    ) -> AsyncGenerator[np.ndarray, None]:
        llm_stream = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,
            stream=True,
            tools=self.tools,
        )
        pending_tool_calls: dict[int, ChoiceDeltaToolCall] = {}

        def text_stream() -> Generator[str, None, None]:
            for chunk in llm_stream:
                delta = chunk.choices[0].delta

                if delta.content:
                    text_chunks.append(delta.content)
                    yield delta.content

                for tool_call in delta.tool_calls or []:
                    index = tool_call.index

                    if index not in pending_tool_calls:
                        pending_tool_calls[index] = tool_call

                    pending_tool_calls[
                        index
                    ].function.arguments += tool_call.function.arguments  # type: ignore

        for ai_speech in stream_text_to_speech(text_stream(), voice=voice):
            yield ai_speech

        for tool_call in pending_tool_calls.values():
            assert tool_call.function is not None, "Tool call function must not be None"

            call_id: str = tool_call.id  # type: ignore
            tool_name: str = tool_call.function.name  # type: ignore
            tool_args: str = tool_call.function.arguments  # type: ignore

            tool_calls.append(  # type: ignore
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args,
                    },
                }
            )

            mcp_client = self._get_mcp_client_for_tool(tool_name)
            if mcp_client is None:
                print(f"Tool {tool_name} not found in any MCP client.")
                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Unable to find tool '{tool_name}'.",
                    }
                )
            else:
                try:
                    if tool_name == "display":
                        args = json.loads(tool_args) if tool_args else {}
                        update_ui(
                            args.get("products"),
                            args.get("image_url"),
                            args.get("clear_ui"),
                        )
                        tool_response: ChatCompletionToolMessageParam = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": "Content displayed successfully.",
                        }
                    else:
                        tool_response = await mcp_client.call_tool(
                            call_id=call_id,
                            tool_name=tool_name,
                            tool_args=json.loads(tool_args) if tool_args else None,
                        )
                    tool_responses.append(tool_response)
                except Exception as e:
                    print(f"Error calling tool {tool_name}: {e}")
                    tool_responses.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": f"Error calling tool '{tool_name}', Error: {str(e)[:500]}",
                        }
                    )

    def _build_input_image_content(
        self, input_image: Image.Image, image_label: str
    ) -> tuple[ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam]:
        """
        Build the input image content for the chat message.
        """
        image_url = self.image_uploader.upload_image(
            input_image, f"{image_label}.{(input_image.format or 'webp').lower()}"
        )
        return (
            {
                "type": "text",
                "text": f"{image_label}_url: {image_url}\n{image_label}:",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        )


def _build_display_tool_definition() -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": "display",
            "description": """Show content to the user, or clear the UI if none of the inputs are provided.
You can use this tool whenever you want to show tool results or when the user requests to see something that you have access to, like a list of products, specific product(s) from the conversation history, an image, or cart items.

You can only pass one argument at a time, either products or image_url, or clear_ui.
""",
            "parameters": {
                "type": "object",
                "properties": {
                    "products": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "image_url": {"type": "string"},
                                "price": {"type": "string"},
                            },
                            "required": ["name", "image_url", "price"],
                        },
                        "description": "A list of products to display from search results, cart items, or conversation history.",
                    },
                    "image_url": {
                        "type": "string",
                        "description": "An optional URL of an image to display.",
                    },
                    "clear_ui": {
                        "type": "boolean",
                        "description": (
                            "If true, clear the UI instead of displaying anything."
                        ),
                    },
                },
            },
        },
    }
