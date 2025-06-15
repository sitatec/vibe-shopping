from __future__ import annotations

import json
import os
import re
import time
from typing import TYPE_CHECKING, Generator

from gradio_client import Client
import numpy as np
from PIL import Image
from openai import OpenAI
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from mcp_client import MCPClient, AgoraMCPClient
from mcp_host.tts.gradio_api_tts import (
    stream_text_to_speech as gradio_api_stream_text_to_speech,
)
from utils import ImageUploader

IS_HF_ZERO_GPU = os.getenv("SPACE_ID", "").startswith("sitatech/")
if IS_HF_ZERO_GPU:
    from mcp_host.tts.hf_zero_gpu_tts import stream_text_to_speech
    from mcp_host.stt.hf_zero_gpu_stt import speech_to_text
else:
    from mcp_host.tts.fastrtc_tts import stream_text_to_speech
    from mcp_host.stt.openai_stt import speech_to_text

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionToolMessageParam,
        ChatCompletionMessageToolCallParam,
        ChatCompletionToolParam,
        ChatCompletionContentPartParam,
        ChatCompletionContentPartTextParam,
        ChatCompletionContentPartImageParam,
    )

ChatOutputType = Generator[
    # Sample rate, audio data
    tuple[int, np.ndarray]
    # Update the UI with a list of products or an image from url, or clear the UI
    | tuple[list[dict[str, str]] | None, str | None, bool | None],
    None,
    None,
]


# TODO: Refactor to improve readability
class VibeShoppingAgent:
    SYSTEM_PROMPT: str = """You are a helpful online shopping AI assistant. 
<context>    
    Your task is to help users find products, try them virtually and buy them. 
    You have access to many tools (functions) you can call to to perform different tasks. You are also capable of displaying products and images in the user interface using the Display tools, so the user can see them.
</context>

<instructions-and-rules>
    When you get a response from a function, if it contains something displayable (products, images), you must display it, don't read it out loud.
    Then, when you get a response from the display tool confirming the content is displayed, you can say what you think about the displayed item(s), tell how they fit to the user request, or ask the user's opinion, just like a human would do in a conversation.
    Every image you are shown will be followed by its URL for reference, so you can use it when you need to display an image in the UI.

    Always ask the user for confirmation before taking any action that requires payment.
    If a function requires an input that you don't have based on your knowledge and the conversation history, you should ask the user for it. For example, if the user asks to try on a product, but you don't have the target image, you should ask the user to provide it.

    When calling a function, let the user know what you are doing while they are waiting. 
    Something like: One moment, I will search for products matching your request \n<tool-call>\n<call-function-to-search-products>\n</tool-call>\n
    Then when you get the response from the function, you can say Here are some products I found for you \n<tool-call>\n<call-function-to-display-products>\n</tool-call>\n
    
    When you receive a response from a tool you can see the content and images, but the user can't at that point, until you display it then the user will see it.
    It's normal if you don't see the content displayed by the Display tool, since you have already seen them. The display tool is used to show the content to the user, not to you.
</instructions-and-rules>

<constraints>
    The maximum number of products you can search at once is 10, don't exceed this limit.
    Text formatting is forbidden! So make sure to only output raw plain text. Do not output markdown or emoji.
    Always display all products search results so the user can see them, not read them.
    When calling a function, you must refer to its description to use it correctly.
</constraints>

<example-1>
    User: Can you find me a modern sofa?
    Assistant: Yes sure! Please wait while I search for a beautiful modern sofa for you.
    <tool-call>
    {"name": "Agora.search_products", "arguments": {"q": "modern sofa", "count": 10}}
    </tool-call>
    Tool:
    <tool-response>
    <products-images-grid-here>
    [{"_id": "id1", "name": "Sofa", "brand": "Modernism", "store":"The Modernism Store", "images": ["https://example.com/image.png"], "price": "29$"},
    {"_id": "id2", "name": "Stylish Green Sofa", "images": ["https://example.com/sofa.png"], "price": "$299.99"}
    ...
    {"_id": "id10", "name": "Luxury Sofa", "brand": "Luxury Furniture", "store":"The Luxury Furniture Store", "images": ["https://example.com/luxury-sofa.png"], "price": "$999.99"}]
    </tool-response>
    Assistant: I've found some great options you might like! Here they are:
    <tool-call>
    {"name": "Display.display_products", "arguments": {"products": [{ "name": "Sofa", "image_url": "https://example.com/image.png", "price": "29$"}, { "name": "Stylish Green Sofa", "image_url": "https://example.com/sofa.png", "price": "$299.99"}, ... { "name": "Luxury Sofa", "image_url": "https://example.com/luxury-sofa.png", "price": "$999.99"}]}}
    </tool-call>
    Tool:
    <tool-response>
    Content displayed successfully.
    </tool-response>
    Assistant:
    Personally, I think the Stylish Green Sofa looks really nice and fits the modern style you asked for. What do you think? Would you like to see more details or try it virtually?
</example-1>
<example-2>
    User: I would like to buy a dress for a professional dinner
    Assistant: Sure! I can help you find a nice dress for that occasion. One second please.
    <tool-call>
    {"name": "Agora.search_products", "arguments": {"q": "Event Dresses", "count": 10}}
    </tool-call>
    Tool:
    <tool-response>
    <products-images-grid-here>
    [{"_id": "id1", "name": "Elegant Black Dress", "brand": "Elegance", "store":"The Elegance Store", "images": ["https://example.com/elegant-black-dress.png"], "price": "$199.99"},
    ...
    {"_id": "id10", "name": "Stylish Red Dress",  "brand": "Dress Mania", "store":"Dress Mania", "images": ["https://example.com/stylish-red-dress.png"], "price": "$249.99"}]
    </tool-response>
    Assistant: Here are some beautiful dresses I found for you:
    <tool-call>
    {"name": "Display.display_products", "arguments": {"products": [{ "name": "Elegant Black Dress", "image_url": "https://example.com/elegant-black-dress.png", "price": "$199.99"}, ... { "name": "Stylish Red Dress", "image_url": "https://example.com/stylish-red-dress.png", "price": "$249.99"}]}}
    </tool-call>
    Tool:
    <tool-response>
    Content displayed successfully.
    </tool-response>
    Assistant: If you like standing out, The Stylish Red Dress is a great choice, it looks very elegant and professional. Would you like to try it on?
    User: Yes, I would like to try it on
    Assistant: Great! Please upload a photo of yourself so I can help you try it on.
    User: 
    <uploaded-photo-wearing-pants-and-t-shirt>
    input_image_url = "https://example.com/user-photo.png"
    Assistant: Amazing! I received your photo, but for this photo, I can't automatically apply the dress, you need to draw a mask around the area where you want the dress to be applied.
    User: 
    <uploaded-photo>
    <uploaded-mask>
    input_image_url = "https://example.com/user-photo.png"
    input_mask_url = "https://example.com/user-photo-mask.png"
    <tool-call>
    {"name": "VirtualTry.try_item_with_masking", "arguments": {"prompt": "<prompt-as-described-in-tool-description>", "item_image_url": "https://example.com/stylish-red-dress.png", "target_image_url": "https://example.com/user-photo.png", "mask_image_url": "https://example.com/user-photo-mask.png"}}
    </tool-call>
    Tool:
    <tool-response>
    <result-image>
    image_url: "https://example.com/virtual-try-on-result.png"
    </tool-response>
    Assistant: Hum, you look gorgeous in the Stylish Red Dress! Take a look at this:
    <tool-call>
    {"name": "Display.display_image", "arguments": {"image_url": "https://example.com/virtual-try-on-result.png"}}
    </tool-call>
</example-2>
<example-3>
    User: I would like to buy a t-shirt
    Assistant: Sure! I can help you find a nice t-shirt. One second please.
    <tool-call>
    {"name": "Agora.search_products", "arguments": {"q": "t-shirt", "count": 10}}
    </tool-call>
    Tool:
    <tool-response>
    <products-images-grid-here>
    [{"_id": "id1", "name": "Stylish T-Shirt", "brand": "Fashion", "store":"The Fashion Store", "images": ["https://example.com/stylish-t-shirt.png"], "price": "$29.99"},
    ...
    {"_id": "id10", "name": "Casual T-Shirt", "brand": "Casual Wear", "store":"Casual Wear Store", "images": ["https://example.com/casual-t-shirt.png"], "price": "$19.99"}]
    </tool-response>
    Assistant: Here are some beautiful t-shirts for you to choose from:
    <tool-call>
    {"name": "Display.display_products", "arguments": {"products": [{ "name": "Stylish T-Shirt", "image_url": "https://example.com/stylish-t-shirt.png", "price": "$29.99"}, ... { "name": "Casual T-Shirt", "image_url": "https://example.com/casual-t-shirt.png", "price": "$19.99"}]}}
    </tool-call>
    Tool:
    <tool-response>
    Content displayed successfully.
    </tool-response>
    Assistant: Which one do you like? Would you like to try it on?
    User: Yes, I would like to try on the Casual one
    Assistant: Great! Please upload a photo of yourself so I can help you try it on.
    User:
    <uploaded-photo-wearing-pants-and-t-shirt>
    input_image_url = "https://example.com/nice-user-photo.png"
    Assistant: I've seen your photo, let me apply the Casual T-Shirt on it (PS: I like the pants you are wearing, they will look great with the t-shirt).
    <tool-call>
    {"name": "VirtualTry.try_item_with_auto_masking", "arguments": {"prompt": "<prompt-as-described-in-tool-description>", "item_image_url": "https://example.com/casual-t-shirt.png", "target_image_url": "https://example.com/nice-user-photo.png", "masking_prompt": "t-shirt"}}
    </tool-call>
    Tool:
    <tool-response>
    <result-image>
    image_url: "https://example.com/virtual-try-on-casual-t-shirt-result.png"
    </tool-response>
    Assistant: Here is how you look in the Casual T-Shirt:
    <tool-call>
    {"name": "Display.display_image", "arguments": {"image_url": "https://example.com/virtual-try-on-casual-t-shirt-result.png"}}
    </tool-call>
</example-3>
<example-4>
    User: I would like to buy a new laptop for my son's birthday, he loves gaming, can you help me find one?
    Assistant: Oh wow, happy birthday to your son! I can definitely help you find a great laptop that he will like. Give me a moment to search for some gaming laptops.
    <tool-call>
    {"name": "Agora.search_products", "arguments": {"q": "gaming laptop", "count": 10}}
    </tool-call>
    Tool:
    <tool-response>
    <products-images-grid-here>
    [{"_id": "id1", "name": "Gaming Laptop", "brand": "GamerLand", "store":"The GamerLand Store", "images": ["https://example.com/gaming-laptop.png"], "price": "$999.99"},
    {"_id": "id2", "name": "New Gen Gaming Laptop", "brand": "NextGen", "store":"NextGen Store", "images": ["https://example.com/new-gen-gaming-laptop.png"], "price": "$1299.99"},
    ...
    {"_id": "id9", "name": "Gaming Laptop Pro", "brand": "ProGamer", "store":"ProGamer Store", "images": ["https://example.com/gaming-laptop-pro.png"], "price": "$1199.99"},
    {"_id": "id10", "name": "High-Performance Gaming Laptop", "brand": "High-Performance", "store":"High-Performance", "images": ["https://example.com/high-performance-laptop.png"], "price": "$1499.99"}]
    </tool-response>
    Assistant: I've found some awesome gaming laptops that I think your son will love:
    <tool-call>
    {"name": "Display.display_products", "arguments": {"products": [{ "name": "Gaming Laptop", "image_url": "https://example.com/gaming-laptop.png", "price": "$999.99"}, { "name": "New Gen Gaming Laptop", "image_url": "https://example.com/new-gen-gaming-laptop.png", "price": "$1299.99"}, ... { "name": "Gaming Laptop Pro", "image_url": "https://example.com/gaming-laptop-pro.png", "price": "$1199.99"}, { "name": "High-Performance Gaming Laptop", "image_url": "https://example.com/high-performance-laptop.png", "price": "$1499.99"}]}}
    </tool-call>
    Tool:
    <tool-response>
    Content displayed successfully.
    </tool-response>
    Assistant:
    The High-Performance Gaming Laptop is a great choice for gaming, it has a powerful GPU and a fast processor. Do you like any of these options?
</example-4>
"""

    def __init__(
        self,
        model_name: str = "RedHatAI/Qwen2.5-VL-72B-Instruct-quantized.w4a16",
        openai_api_key: str = os.getenv("OPENAI_API_KEY", ""),
        openai_api_base_url: str = os.getenv("OPENAI_API_BASE_URL", ""),
        image_uploader: ImageUploader = ImageUploader(),
    ):
        self.agora_client = AgoraMCPClient(unique_name="Agora")
        # self.fewsats_client = MCPClient(unique_name="Fewsats")
        self.virtual_try_client = MCPClient(unique_name="VirtualTry")

        self.openai_client = OpenAI(
            base_url=openai_api_base_url,
            api_key=openai_api_key,
        )
        self.chat_history: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            }
        ]
        self.model_name = model_name

        self._mcp_clients: list[MCPClient] = [
            self.agora_client,
            # self.fewsats_client,
            self.virtual_try_client,
        ]
        self.display_tool = _build_display_tool_definitions()
        self.image_uploader = image_uploader
        self.clients_connected = False

        # For custom tool call parsing, read the comment where we handle model output
        # in the _send_to_llm method for details on why we need this.
        self.tool_call_pattern = re.compile(r"<tool-call>(.*?)</tool-call>", re.DOTALL)

    def connect_clients(
        self, fewsats_api_key: str = os.getenv("FEWSATS_API_KEY", "FAKE_API_KEY")
    ):
        self.agora_client.connect_to_server("uvx", ["agora-mcp"])
        # Excluding Payments with FEWSATS for now
        # self.fewsats_client.connect_to_server(
        #     "env", [f"FEWSATS_API_KEY={fewsats_api_key}", "uvx", "fewsats-mcp"]
        # )
        self.virtual_try_client.connect_to_server("python", ["./mcp_server.py"])

        self.tools = (
            self.display_tool
            + self.agora_client.tools
            # + self.fewsats_client.tools
            + self.virtual_try_client.tools
        )
        self.clients_connected = True

    def _get_mcp_client_for_tool(self, tool_name: str) -> MCPClient | None:
        try:
            # Iterate through the clients to find the one that owns the tool and stop at the first match
            return next(
                client for client in self._mcp_clients if client.owns_tool(tool_name)
            )
        except StopIteration:
            return None

    def chat(
        self,
        user_speech: tuple[int, np.ndarray],
        chat_history: list[ChatCompletionMessageParam],
        voice: str | None = None,
        input_image: Image.Image | None = None,
        input_mask: Image.Image | None = None,
        gradio_client: Client | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
    ) -> ChatOutputType:
        if voice == "debug_echo_user_speech":
            time.sleep(1)  # Simulate some processing delay
            print(f"Debug echo user speech: {user_speech}")
            yield user_speech
            return

        # Normally, we should handle the chat history internally with self.chat_history, but since we are not persisting it,
        # we will rely on gradio's session state to keep the chat history per user session.
        if not chat_history:
            # If history is empty, start with the system prompt
            chat_history.append(
                {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT}
            )

        user_message_contents: list[ChatCompletionContentPartParam] = []
        if input_image is not None:
            user_message_contents.extend(
                list(self._build_input_image_content(input_image, "input_image"))
            )
            if input_mask is not None:
                user_message_contents.extend(
                    list(self._build_input_image_content(input_mask, "input_mask"))
                )

        t = time.time()
        user_text_message = speech_to_text(user_speech).strip()
        print(f"Speech to text took {time.time() - t:.2f} seconds")

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
        print(f"User message: {user_text_message}")
        print("Entering Agent loop")
        t1 = time.time()
        while True:
            tool_calls: list[ChatCompletionMessageToolCallParam] = []
            tool_responses: list[ChatCompletionToolMessageParam] = []
            text_chunks: list[str] = []

            for ai_speech_or_ui_update in self._send_to_llm(
                chat_history=chat_history,
                voice=voice,
                tool_calls=tool_calls,
                tool_responses=tool_responses,
                text_chunks=text_chunks,
                gradio_client=gradio_client,
                temperature=temperature,
                top_p=top_p,
            ):
                yield ai_speech_or_ui_update
                print(
                    f"AI speech received. Time taken since agent loop started: {time.time() - t1:.2f} seconds"
                )

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
                print("No tool responses, ending chat loop.")
                break
            print(f"Num tool responses: {len(tool_responses)}")
            print("Continuing Agent loop")
        print(f"Agent loop completed in {time.time() - t1:.2f} seconds")
        print(f"Time taken for the entire chat: {time.time() - t:.2f} seconds")

    def _send_to_llm(
        self,
        chat_history: list[ChatCompletionMessageParam],
        voice: str | None,
        tool_calls: list[ChatCompletionMessageToolCallParam],
        tool_responses: list[ChatCompletionToolMessageParam],
        text_chunks: list[str],
        gradio_client: Client | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> ChatOutputType:
        llm_stream = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,
            stream=True,
            tools=self.tools,
            temperature=temperature,
            top_p=top_p,
        )

        pending_tool_calls: dict[int, ChoiceDeltaToolCall] = {}
        pending_custom_tool_calls: list[ChoiceDeltaToolCall] = []
        response_log = ""
        custom_tool_parser_buffer = ""

        def text_stream() -> Generator[str, None, None]:
            nonlocal response_log, custom_tool_parser_buffer

            for chunk in llm_stream:
                delta = chunk.choices[0].delta

                response_log += delta.content or ""
                response_log += "".join(
                    tool_call.model_dump_json(indent=2)
                    for tool_call in delta.tool_calls or []
                )

                if delta.content:
                    # For some reason, the display tools are either not being called, or the output is malformed,
                    # but when I prompt the Qwen to output tool calls in a specific format different from the default,
                    # it works. Maybe related to hermses tool call parser used in vLLM, but I think it is qween that wasn't
                    # trained on tool use with complex arguments (arrays of objects, etc.) or maybe it was mainly trained on search-like tool calling, so it doesn't generalize well.

                    # Manually parse tool calls with the following format: <tool-call>...</tool-call> as a fallback, not replacement incase the model use the default format for some cases.
                    if custom_tool_parser_buffer:
                        custom_tool_parser_buffer += delta.content
                        if not custom_tool_parser_buffer.startswith("<tool"):
                            # We received a chunk that ends with "<" earlier, since it wasn't possible to know
                            # at that point if it was a tool call or not, we buffered it. Now we know it was not a tool call.
                            yield custom_tool_parser_buffer
                            custom_tool_parser_buffer = ""
                            continue

                        if "</tool-call>" in custom_tool_parser_buffer:
                            # Complete tool call found, parse it
                            matches = self.tool_call_pattern.findall(
                                custom_tool_parser_buffer
                            )
                            for match in matches:
                                try:
                                    tool_call_data = json.loads(match)
                                    pending_custom_tool_calls.append(
                                        ChoiceDeltaToolCall(
                                            id=f"custom_{len(pending_custom_tool_calls)}",
                                            index=len(pending_custom_tool_calls),
                                            function=ChoiceDeltaToolCallFunction(
                                                name=tool_call_data["name"],
                                                arguments=json.dumps(
                                                    tool_call_data.get("arguments", {})
                                                ),
                                            ),
                                        )
                                    )
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing tool call: {e}")
                            # yield all text content outside of tool calls
                            parts = self.tool_call_pattern.split(
                                custom_tool_parser_buffer
                            )
                            for i, text in enumerate(parts):
                                # The split parts follow the pattern of alternating text and tool calls:
                                # # text, tool-call, text, tool-call, ... but text can be empty.
                                if i % 2 == 0:
                                    # This is text content outside of tool calls
                                    text = text.strip()
                                    if text:
                                        text_chunks.append(text)
                                        yield text
                            # Clear the buffer after processing
                            custom_tool_parser_buffer = ""
                        continue
                    else:
                        # Check if the content contains a tool call
                        if "<tool" in delta.content or delta.content.endswith("<"):
                            custom_tool_parser_buffer = delta.content[
                                delta.content.index("<") :  # Start from the first "<"
                            ]
                            # yield any text content before the tool call
                            remaining_text = delta.content[: delta.content.index("<")]
                            if remaining_text:
                                text_chunks.append(remaining_text)
                                yield remaining_text
                            continue

                    text_chunks.append(delta.content)
                    yield delta.content

                for tool_call in delta.tool_calls or []:
                    index = tool_call.index

                    if index not in pending_tool_calls:
                        pending_tool_calls[index] = tool_call

                    if tool_call.function is not None:
                        pending_fun = pending_tool_calls[index].function
                        if pending_fun is not None:
                            if tool_call.function.arguments is not None:
                                pending_fun.arguments = (
                                    pending_fun.arguments or ""
                                ) + tool_call.function.arguments
                        else:
                            pending_tool_calls[index].function = tool_call.function

        if gradio_client is not None:
            print("Using online Gradio client for text-to-speech.")
            for audio_chunk in gradio_api_stream_text_to_speech(
                text_stream(), client=gradio_client, voice=voice
            ):
                yield audio_chunk
        else:
            for ai_speech in stream_text_to_speech(text_stream(), voice=voice):
                yield ai_speech

        print("LLM stream completed. \nResponse log:\n", response_log)
        for tool_call in list(pending_tool_calls.values()) + pending_custom_tool_calls:
            print(f"Processing tool call: {tool_call}")
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

            try:
                print(f"Calling tool {tool_name} with args: {tool_args}")
                if tool_name.startswith("Display."):
                    args = json.loads(tool_args) if tool_args else {}
                    yield (
                        args.get("products"),
                        args.get("image_url"),
                        tool_name == "Display.clear_display",
                    )
                    tool_response: ChatCompletionToolMessageParam = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": (
                            "Content displayed to the user successfully!"
                            if tool_name != "clear_display"
                            else "Display cleared."
                        ),
                    }
                else:
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
                        tool_response = mcp_client.call_tool(
                            call_id=call_id,
                            tool_name=tool_name,
                            tool_args=json.loads(tool_args) if tool_args else None,
                        )
                print("Tool responded")
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
                "text": f"{image_label}_url: {image_url}",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        )


def _build_display_tool_definitions() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "Display.display_products",
                "description": """
    Display a list of products. Use this to show search results, cart items, or products from conversation history.
    
    Args:
        products: A list of products to display. Each product should have a name, image URL, and formatted price.
        example:
            products: [
                {
                    "name": "Stylish Green Shirt",
                    "image_url": "https://example.com/images/green-shirt.jpg",
                    "price": "$29.99"
                },
                {
                    "name": "Comfortable Jeans",
                    "image_url": "https://example.com/images/jeans.jpg",
                    "price": "$49.99"
                }
            ]
""",
                "parameters": {
                    "properties": {
                        "products": {
                            "items": {
                                "properties": {
                                    "name": {
                                        "title": "Product Name",
                                        "type": "string",
                                    },
                                    "image_url": {
                                        "title": "Product Image URL",
                                        "type": "string",
                                    },
                                    "price": {
                                        "title": "Product Price",
                                        "type": "string",
                                    },
                                },
                                "required": ["name", "image_url", "price"],
                                "title": "Product",
                                "type": "object",
                            },
                            "title": "Product List",
                            "type": "array",
                        }
                    },
                    "required": ["products"],
                    "title": "display_productsArguments",
                    "type": "object",
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Display.display_image",
                "description": "Display a single standalone image. Use this for virtual try-on results, a specific product image requested by the user, or any other relevant single image.\n\nArgs:\n    image_url: The URL of the image to display.",
                "parameters": {
                    "properties": {
                        "image_url": {
                            "title": "Image URL",
                            "type": "string",
                        },
                    },
                    "required": ["image_url"],
                    "title": "display_imageArguments",
                    "type": "object",
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Display.clear_display",
                "description": "Clear any content currently displayed in the user interface. Removes everything from the visual display area.\n\nArgs: None",
                "parameters": {
                    "properties": {},
                    "title": "clear_displayArguments",
                    "type": "object",
                },
            },
        },
    ]
