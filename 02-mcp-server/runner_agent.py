import asyncio
import sys
from pathlib import Path

from agents import Agent, Runner
from agents.mcp import MCPServerStdio


def read_only(context, tool):
    """Give the agent only the tools that the server marked as read-only."""
    return bool(tool.annotations and tool.annotations.read_only_hint)


async def ask(question):
    async with MCPServerStdio(
        name="shop",
        params={"command": sys.executable, "args": [str(Path(__file__).with_name("shop_server.py"))]},
        tool_filter=read_only,
        cache_tools_list=True,
    ) as shop:
        agent = Agent(
            name="Support assistant",
            instructions="You are a customer support assistant for an online electronics shop.",
            model="gpt-6.1-sol",
            mcp_servers=[shop],
        )
        result = await Runner.run(agent, question, max_turns=10)
        return result.final_output


if __name__ == "__main__":
    print(asyncio.run(ask("Is the USB-C hub in stock, and where is order ORD-10042?")))
