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


from utils import ImageUploader, make_image_grid_with_index_labels


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
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    }
                )
                contents.append(
                    {
                        "type": "text",
                        "text": f"image_url: {image_url}",
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

        if len(products) > 10:
            print(f"Received {len(products)} products, limiting to 10 for context.")
            products = products[:10]  # Limit to first 10 products

        products_image_bytes = self._event_loop.run_until_complete(
            self._download_product_images(products)
        )
        # Instead of add each image to the content, we will add a single image
        # grid with all the product images to reduce token usage and decrease latency
        #  This will affect the LLM's performance but minor in the case every day objects like clothes, shoes, etc.
        image = make_image_grid_with_index_labels(products_image_bytes)
        new_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": self.image_uploader.upload_image(
                        image=image, filename=f"{call_id}.png"
                    ),
                },
            }
        )

        # Remove all the fields we don't need to reduce token usage and preserve focused context.
        for product in products:
            for key in self.FIELDS_TO_REMOVE:
                product.pop(key, None)

        new_content.append(
            {
                "type": "text",
                "text": f"Every image is labeled with its corresponding index in the products list:\n{json.dumps(products)}"
            }
        )

        return new_content

    async def _download_product_images(
        self, products: list[dict[str, Any]]
    ) -> list[bytes | None]:
        """
        Download the first image of each product if available.
        Args:
            products: A list of product dictionaries, each containing an "images" key with a list of image URLs.
        Returns:
            list[bytes | None]: A list of image bytes for each product, or None if the image could not be downloaded.
        """
        tasks = [self._download_product_image(product) for product in products]
        return await asyncio.gather(*tasks)

    async def _download_product_image(self, product: dict[str, Any]) -> bytes | None:
        """
        Download the first image of a product if available.
        Args:
            product: A product dictionary containing an "images" key with a list of image URLs.
        Returns:
            bytes | None: The image bytes if the image is successfully downloaded, None otherwise.
        """
        if "images" in product and product["images"]:
            if not self._http_session:
                self._http_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=5)
                )
            try:
                response = await self._http_session.get(
                    product["images"][0], allow_redirects=True
                )
                response.raise_for_status()
                return await response.read()
            except Exception as e:
                print(f"Error downloading image: {e}")
                return None
        return None

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
