from contextlib import AsyncExitStack
from typing import Any, TYPE_CHECKING, cast
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

if TYPE_CHECKING:
    from mcp.types import CallToolResult, TextContent, ImageContent
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionContentPartParam,
    )


class MCPClient:
    def __init__(
        self,
    ):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()

    def ensure_initialized(self):
        if not self.session:
            raise RuntimeError(
                "Session is not initialized. Call connect_to_server first."
            )

    @property
    async def tools(self):
        self.ensure_initialized()
        return await self.session.list_tools()  # type: ignore

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

        response = await self.session.list_tools()
        print(
            "\nConnected successfully! \nAvailable tools:",
            [tool.name for tool in response.tools],
            "\n",
        )

    async def pre_tool_call(
        self, call_id: str, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        return tool_name, tool_args

    async def call_tool(
        self, call_id: str, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> ChatCompletionMessageParam:
        self.ensure_initialized()

        tool_name, tool_args = await self.pre_tool_call(call_id, tool_name, tool_args)
        response = await self.session.call_tool(tool_name, tool_args)  # type: ignore
        content = await self.post_tool_call(call_id, tool_name, response, tool_args)

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        }  # type: ignore

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
                contents.append({"type": "text", "text": content.text})
            elif isinstance(content, ImageContent):
                contents.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": content.data},
                    }
                )
            else:
                raise ValueError(f"Unsupported content type: {content.type}")

        return contents

    async def close(self):
        if self.session:
            await self.exit_stack.aclose()


class AgoraMCPClient(MCPClient):
    async def post_tool_call(
        self,
        call_id: str,
        tool_name: str,
        response: CallToolResult,
        tool_args: dict[str, Any] | None = None,
    ) -> list[ChatCompletionContentPartParam]:
        content = cast(TextContent, response.content[0])
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
            for key in [
                # The date for priceHistory is always empty, so not very useful.
                "priceHistory",
                # The scores below are related to embeddings, we want our LLM to decide which item is more relevant
                # based user request and chat history.
                "_adjustedScore",
                "_combinedScoreData",
                "_rankingScore",
                "agoraScore",
            ]:
                del product[key]
            # We add the product data first then show its image if available.
            new_content.append({"type": "text", "text": json.dumps(product)})
            if product.get("images"):
                new_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": product["images"][0]},
                    }
                )
            else:
                new_content.append(
                    {
                        "type": "text",
                        "text": f'No image available for "{product["name"]}" with ID {product.get("_id")}.',
                    }
                )

        return new_content
