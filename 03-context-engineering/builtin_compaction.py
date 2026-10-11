"""The same agent, but the OpenAI API compacts the context on its side (OpenAI only).

Run: uv run python builtin_compaction.py
"""
import asyncio

from mcp import Client
from openai import APIError

import agent

COMPACT_THRESHOLD = 200_000  # tokens, the value used in OpenAI's compaction guide


def drop_before_compaction(items):
    """The latest compaction item carries everything that came before it, so older items can go."""
    positions = [i for i, item in enumerate(items) if item.get("type") == "compaction"]
    return items[positions[-1]:] if positions else items


async def run_turn(shop, tools, context, question):
    context.items.append(agent.customer_message(question))
    llm_tools = [agent.to_llm_tool(tool) for tool in tools.values()]

    for _ in range(agent.MAX_STEPS):
        response = await agent.llm.responses.create(
            model=agent.MODEL,
            instructions=agent.INSTRUCTIONS,
            input=context.items,
            tools=llm_tools,
            context_management=[{"type": "compaction", "compact_threshold": COMPACT_THRESHOLD}],
            store=False,
        )
        if response.usage:
            agent.record_usage(context, response.usage)
        if response.status != "completed":
            return f"The agent stopped early (reason: {response.status})."

        context.items += [item.to_dict(mode="json") for item in response.output]
        context.items = drop_before_compaction(context.items)

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text
        for call in calls:
            output = agent.cap(await agent.handle_call(shop, tools, call))
            context.items.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

    return "The agent hit the step limit without finishing."


async def main():
    async with Client(agent.SHOP_SERVER, read_timeout_seconds=30) as shop:
        tools = {tool.name: tool for tool in (await shop.list_tools()).tools}
        context = agent.Context()
        while True:
            question = await asyncio.to_thread(input, "\nCustomer: ")
            if question.strip().lower() in ("", "quit", "exit"):
                break
            try:
                answer = await run_turn(shop, tools, context, question)
            except APIError as error:
                answer = f"Sorry, something went wrong on our side. Please try again. ({type(error).__name__})"
            print("Agent:", answer)


if __name__ == "__main__":
    asyncio.run(main())
