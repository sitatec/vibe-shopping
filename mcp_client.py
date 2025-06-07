from abc import ABC
from contextlib import AsyncExitStack
from typing import Any, TYPE_CHECKING

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

if TYPE_CHECKING:
    from mcp.types import CallToolResult


class BaseMCPClient(ABC):
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
        self, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any] | None]:
        return tool_name, tool_args

    async def call_tool(
        self, tool_name: str, tool_args: dict[str, Any] | None = None
    ) -> CallToolResult:
        self.ensure_initialized()

        tool_name, tool_args = await self.pre_tool_call(tool_name, tool_args)
        response = await self.session.call_tool(tool_name, tool_args)  # type: ignore
        return await self.post_tool_call(response)

    async def post_tool_call(self, response: CallToolResult) -> CallToolResult:
        return response

    async def close(self):
        if self.session:
            await self.exit_stack.aclose()
