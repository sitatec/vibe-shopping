from __future__ import annotations

import base64
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, cast
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent, ImageContent

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionToolMessageParam,
        ChatCompletionContentPartParam,
        ChatCompletionToolParam,
    )


from utils import ImageUploader


class MCPClient:
    def __init__(
        self, unique_name: str, image_uploader: ImageUploader = ImageUploader()
    ):
        """Initialize the MCP client.
        Args:
            unique_name: The name of the client, serves as Namespace to prevent conflicts with function names
            if multiple mcp servers expose tools with the same name.
            Also used to identify the client that need to handle the call.
            Keep it short and descriptive as it will be added to the tool name before passing
            it to the LLM. e.g: "FileSystem", "Agora", "Amazon", etc.
            The tool names will be something like "FileSystem.list_files",
            image_uploader: An optional image uploader instance to handle image uploads. If not provided,
            a default uploader will be used that assumes the client is running on Huggingface Spaces.
        """
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.name = unique_name
        self.image_uploader = image_uploader

    def ensure_initialized(self):
        if not self.session:
            raise RuntimeError(
                "Session is not initialized. Call connect_to_server first."
            )

    @property
    async def tools(self) -> list[ChatCompletionToolParam]:
        self.ensure_initialized()
        response = await self.session.list_tools()  # type: ignore

        return [
            {
                "type": "function",
                "function": {
                    "name": f"{self.name}.{tool.name}",
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]

    def owns_tool(self, tool_name: str) -> bool:
        """
        Whether the tool belongs to this client's server or not
        Args:
            tool_name: The name of the tool to check.

        Returns:
            bool: True if the tool belongs to this client's server, False otherwise.
        """
        self.ensure_initialized()
        return tool_name.startswith(f"{self.name}.")

    async def connect_to_server(
        self,
        server_command: str,
        args: list[str] = [],
        env: dict[str, str] | None = None,
    ):
        server_params = StdioServerParameters(
            command=server_command,
            args=args or [],
            env=env,
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        print(
            f"\nSuccessfully Connected to {self.name} MCPClient! \nAvailable tools:",
            [tool["function"]["name"] for tool in await self.tools],  # type: ignore
            "\n",
        )

    async def pre_tool_call(
        self, call_id: str, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        return tool_name, tool_args

    async def call_tool(
        self, call_id: str, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> ChatCompletionToolMessageParam:
        self.ensure_initialized()
        tool_name = tool_name.split(".", 1)[-1]  # Remove the namespace prefix if exists

        tool_name, tool_args = await self.pre_tool_call(call_id, tool_name, tool_args)
        print(
            f"Send tool call to mcp server: {tool_name} with args: {tool_args} (call_id: {call_id})"
        )
        response = await self.session.call_tool(tool_name, tool_args)  # type: ignore
        print(
            f"Response from mcp server: {tool_name} with args: {tool_args} (call_id: {call_id}) :"
        )
        print(response.model_dump_json(indent=2))
        content = await self.post_tool_call(call_id, tool_name, response, tool_args)
        print(f"Processed tool call response content: {content}")
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,  # type: ignore
        }

    async def post_tool_call(
        self,
        call_id: str,
        tool_name: str,
        response: CallToolResult,
        tool_args: dict[str, Any] | None = None,
    ) -> list[ChatCompletionContentPartParam]:
        contents: list[ChatCompletionContentPartParam] = []
        for content in response.content:
            if isinstance(content, TextContent):
                contents.append(
                    {
                        "type": "text",
                        "text": content.text,
                    }
                )
            elif isinstance(content, ImageContent):
                # We need to give a reference (in this case a URL) to the LLM for
                # any image content we show it, so that when needed it can show it to the user.
                image_url = self.image_uploader.upload_image(
                    image=base64.b64decode(content.data),
                    filename=f"{call_id}.{content.mimeType.split('/')[1]}",  # e.g. "call_id.png"
                )
                contents.append(
                    {
                        "type": "text",
                        "text": f"image_url: {image_url}\nimage:",
                    }
                )
                # Put the image content after the url
                contents.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": content.data,
                        },
                    }
                )
            else:
                raise ValueError(f"Unsupported content type: {content.type}")

        return contents

    async def close(self):
        if self.session:
            await self.exit_stack.aclose()


class AgoraMCPClient(MCPClient):
    FIELDS_TO_REMOVE = [
        # The date for priceHistory is always empty, so not very useful.
        "priceHistory",
        # The scores below are related to embeddings, we want our LLM to decide which item is more relevant
        # based user request and chat history.
        "_adjustedScore",
        "_combinedScoreData",
        "_rankingScore",
        "agoraScore",
    ]

    async def post_tool_call(
        self,
        call_id: str,
        tool_name: str,
        response: CallToolResult,
        tool_args: dict[str, Any] | None = None,
    ) -> list[ChatCompletionContentPartParam]:
        contents = cast(list[TextContent], response.content)
        status_code = contents[0].text
        if status_code != "200":
            return await super().post_tool_call(call_id, tool_name, response, tool_args)
        
        content = contents[1]  # The second content part should be the JSON response
        json_data = json.loads(content.text)
        if json_data.get("status") != "success" or "Products" not in json_data:
            # If not successful or not a product search/list response, return the
            # original response with status and possibly an error message.
            # Not robust though, since response schema can be changed by the server.
            # We could also rely on tool_name to check if it is list/search but the tool name can also change.
            # We need to rely on mcp server versioning.
            return await super().post_tool_call(call_id, tool_name, response, tool_args)

        new_content: list[ChatCompletionContentPartParam] = []
        products: list[dict[str, Any]] = json_data["Products"]
        for product in products:
            # Remove all the fields we don't need to reduce token usage and preserver focused context.
            for key in self.FIELDS_TO_REMOVE:
                product.pop(key, None)

            # We add the product data first then show its image if available.
            # The LLM will see:
            # product_details: {...}
            # product_image:
            # <image_content> or <Not available message>
            new_content.append(
                {
                    "type": "text",
                    "text": f"product_details: {json.dumps(product)}\nproduct_image:",
                }
            )
            if product.get("images"):
                new_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": product["images"][0],
                        },
                    }
                )
            else:
                new_content.append(
                    {
                        "type": "text",
                        "text": f'No image available for "{product["name"]}" with _id {product.get("_id")}.',
                    }
                )

        return new_content
