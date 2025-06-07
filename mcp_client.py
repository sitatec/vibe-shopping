from contextlib import AsyncExitStack
from typing import Any, TYPE_CHECKING, cast
import json

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai.types.chat import (
    ChatCompletionToolParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.shared_params import FunctionDefinition

if TYPE_CHECKING:
    from mcp.types import CallToolResult, TextContent, ImageContent
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionContentPartParam,
    )


class MCPClient:
    def __init__(self, unique_name: str):
        """Initialize the MCP client.
        Args:
            unique_name: The name of the client, serves as Namespace to prevent conflicts with function names
            if multiple mcp servers expose tools with the same name.
            Also used to identify the client that need to handle the call.
            Keep it short and descriptive as it will be added to the tool name before passing
            it to the LLM. e.g: "FileSystem", "Agora", "Amazon", etc.
            The tool names will be something like "FileSystem.list_files",
        """
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.name = unique_name

    def ensure_initialized(self):
        if not self.session:
            raise RuntimeError(
                "Session is not initialized. Call connect_to_server first."
            )

    @property
    async def tools(self):
        self.ensure_initialized()
        response = await self.session.list_tools()  # type: ignore

        return [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=f"{self.name}.{tool.name}",
                    description=tool.description or "",
                    parameters=tool.inputSchema,
                ),
            )
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
            "\nConnected successfully! \nAvailable tools:",
            [tool.function.name for tool in await self.tools],  # type: ignore
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
                contents.append(
                    ChatCompletionContentPartTextParam(type="text", text=content.text)
                )
            elif isinstance(content, ImageContent):
                contents.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url", image_url=ImageURL(url=content.data)
                    )
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
            for key in self.FIELDS_TO_REMOVE:
                del product[key]
            # We add the product data first then show its image if available.
            new_content.append({"type": "text", "text": json.dumps(product)})
            if product.get("images"):
                new_content.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=product["images"][0]),
                    )
                )
            else:
                new_content.append(
                    ChatCompletionContentPartTextParam(
                        type="text",
                        text=f'No image available for "{product["name"]}" with ID {product.get("_id")}.',
                    )
                )

        return new_content
