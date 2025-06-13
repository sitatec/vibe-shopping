from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import TYPE_CHECKING, Any, cast
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent, ImageContent
import aiohttp

if TYPE_CHECKING:
    from openai.types.chat import (
        ChatCompletionToolMessageParam,
        ChatCompletionContentPartParam,
        ChatCompletionToolParam,
    )


from utils import ImageUploader


__all__ = ["MCPClient", "AgoraMCPClient"]


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
        self._event_loop = get_or_create_event_loop()

    def ensure_initialized(self):
        if not self.session:
            raise RuntimeError(
                "Session is not initialized. Call connect_to_server first."
            )

    @property
    def tools(self) -> list[ChatCompletionToolParam]:
        self.ensure_initialized()
        response = self._event_loop.run_until_complete(
            self.session.list_tools(),  # type: ignore
        )

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

    def connect_to_server(
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

        stdio_transport = self._event_loop.run_until_complete(
            self.exit_stack.enter_async_context(stdio_client(server_params))
        )
        self.stdio, self.write = stdio_transport
        self.session = self._event_loop.run_until_complete(
            self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        )

        self._event_loop.run_until_complete(self.session.initialize())

        print(
            f"\nSuccessfully Connected to {self.name} MCPClient! \nAvailable tools:",
            [tool["function"]["name"] for tool in self.tools],
            "\n",
        )

    def pre_tool_call(
        self, call_id: str, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        return tool_name, tool_args

    def call_tool(
        self, call_id: str, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> ChatCompletionToolMessageParam:
        self.ensure_initialized()
        tool_name = tool_name.split(".", 1)[-1]  # Remove the namespace prefix if exists

        tool_name, tool_args = self.pre_tool_call(call_id, tool_name, tool_args)
        print(
            f"Send tool call to mcp server: {tool_name} with args: {tool_args} (call_id: {call_id})"
        )
        response = self._event_loop.run_until_complete(
            self.session.call_tool(  # type: ignore
                tool_name, tool_args, read_timeout_seconds=timedelta(seconds=20)
            )
        )
        content = self.post_tool_call(call_id, tool_name, response, tool_args)
        print(
            f"Received response from mcp server: {tool_name} with args: {tool_args} (call_id: {call_id}) "
        )
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,  # type: ignore
        }

    def post_tool_call(
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

    def close(self):
        if self.session:
            self._event_loop.run_until_complete(self.exit_stack.aclose())


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

    def __init__(
        self, unique_name: str, image_uploader: ImageUploader = ImageUploader()
    ):
        self._http_session: aiohttp.ClientSession | None = None
        super().__init__(unique_name, image_uploader)

    @property
    def tools(self) -> list[ChatCompletionToolParam]:
        return [
            tool
            for tool in super().tools
            # For new limit to search tools only during testing to reduce token usage
            # and complexity for the LLM. Later on use more powerful LLMs that can handle more tools.
            if "search" in tool["function"]["name"]
        ]

    def post_tool_call(
        self,
        call_id: str,
        tool_name: str,
        response: CallToolResult,
        tool_args: dict[str, Any] | None = None,
    ) -> list[ChatCompletionContentPartParam]:
        contents = cast(list[TextContent], response.content)
        status_code = contents[0].text
        if status_code != "200":
            return super().post_tool_call(call_id, tool_name, response, tool_args)

        content = contents[1]  # The second content part should be the JSON response
        json_data = json.loads(content.text)
        if json_data.get("status") != "success" or "Products" not in json_data:
            # If not successful or not a product search/list response, return the
            # original response with status and possibly an error message.
            # Not robust though, since response schema can be changed by the server.
            # We could also rely on tool_name to check if it is list/search but the tool name can also change.
            # We need to rely on mcp server versioning.
            return super().post_tool_call(call_id, tool_name, response, tool_args)

        new_content: list[ChatCompletionContentPartParam] = []
        products: list[dict[str, Any]] = json_data["Products"]

        if len(products) > 5:
            print(f"Received {len(products)} products, limiting to 5 for context.")
            products = products[:5]  # Limit to first 5 products

        # Check if images exist for each product and prepare the content
        image_exists_tasks = self._event_loop.run_until_complete(
            self._check_product_images(products)
        )

        for product, image_exists in zip(products, image_exists_tasks):
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
            if image_exists:
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

    async def _check_product_images(self, products: list[dict[str, Any]]) -> list[bool]:
        """
        Check if the products have images available.
        Args:
            products: A list of product dictionaries to check.

        Returns:
            list[bool]: A list of booleans indicating whether each product has an image.
        """
        tasks = [self._check_product_image_exists(product) for product in products]
        return await asyncio.gather(*tasks)

    async def _check_product_image_exists(self, product: dict[str, Any]) -> bool:
        """
        Check if the product has an image available.
        Args:
            product: The product dictionary to check.

        Returns:
            bool: True if the product has an image, False otherwise.
        """
        if "images" in product and product["images"]:
            if not self._http_session:
                self._http_session = aiohttp.ClientSession(
                    # We are only using this client to check if images exist with a HEAD request before sending them to the LLM.
                    # So we can use a low timeout to reduce latency.
                    timeout=aiohttp.ClientTimeout(total=3)
                )
            try:
                response = await self._http_session.head(
                    product["images"][0], allow_redirects=True
                )
                return response.status == 200
            except Exception as e:
                print(f"Error checking image URL: {e}")
                return False
        return False

    def close(self):
        if self._http_session:
            self._event_loop.run_until_complete(self._http_session.close())
        super().close()


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get or create an event loop for the current thread (will be shared by all MCP clients in the same thread).
    """
    try:
        return asyncio.get_event_loop()
    except RuntimeError as e:
        if "event loop" in str(e).lower():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise e
