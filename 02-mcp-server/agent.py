import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, MCPError, StdioServerParameters
from openai import AsyncOpenAI

llm = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
MODEL = "gpt-6.1-sol"
MAX_STEPS = 10

# Start the shop server as a subprocess and talk to it over stdin and stdout
SERVER_FILE = Path(__file__).with_name("shop_server.py")
SHOP_SERVER = StdioServerParameters(command=sys.executable, args=[str(SERVER_FILE)])


def to_llm_tool(tool):
    """Turn an MCP tool definition into a function tool for the OpenAI Responses API."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": False,  # the schema comes from the server, the server validates the input
    }


def needs_approval(tool):
    """Ask a person before running any tool that the server did not mark as read-only."""
    return not (tool.annotations and tool.annotations.read_only_hint)


async def ask_human(question):
    answer = await asyncio.to_thread(input, f"{question} [y/N] ")
    return answer.strip().lower() == "y"


async def run_tool(shop, tool, arguments):
    """Call one tool on the MCP server and return the text to send back to the LLM."""
    if needs_approval(tool):
        if not await ask_human(f"The agent wants to call {tool.name} with {arguments}. Allow?"):
            return (
                "Error: a human operator declined this action. Do not try again. "
                "Tell the customer that a support agent will contact them."
            )
    try:
        result = await shop.call_tool(tool.name, arguments)
    except MCPError as error:
        return f"Error: the tool call failed ({error}). Tell the customer to try again later."
    text = "\n".join(block.text for block in result.content if block.type == "text")
    # The Responses API has no error flag for tool results, so we say it in the text
    return f"Error: {text}" if result.is_error else text


async def run_agent(shop, tools, instructions, question):
    llm_tools = [to_llm_tool(tool) for tool in tools.values()]
    conversation = [{"role": "user", "content": question}]

    for _ in range(MAX_STEPS):
        response = await llm.responses.create(
            model=MODEL,
            instructions=instructions,
            input=conversation,
            tools=llm_tools,
        )

        if response.status != "completed":
            reason = response.incomplete_details.reason if response.incomplete_details else response.status
            return f"The agent stopped early (reason: {reason})."

        # Keep every output item (reasoning, messages, tool calls) in the conversation
        conversation += response.output

        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text

        for call in calls:
            tool = tools.get(call.name)
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = None
            if tool is None:
                output = f"Error: unknown tool '{call.name}'."
            elif not isinstance(arguments, dict):
                output = "Error: the arguments were not a valid JSON object. Try again."
            else:
                output = await run_tool(shop, tool, arguments)
            print(f"[tool] {call.name}({call.arguments}) -> {output[:60]!r}")
            conversation.append({"type": "function_call_output", "call_id": call.call_id, "output": output})

    return "The agent hit the step limit without finishing."


async def main():
    async with Client(SHOP_SERVER, read_timeout_seconds=30) as shop:
        listed = await shop.list_tools()
        tools = {tool.name: tool for tool in listed.tools}

        # The app decides which resources go into the context. Here: the return policy.
        policy = await shop.read_resource("shop://policies/returns")
        instructions = (
            "You are a customer support assistant for an online electronics shop. "
            "Use the tools to look up real data. Never guess order details or stock levels.\n\n"
            "Reference data from the shop server, the return policy:\n" + policy.contents[0].text
        )

        print("Connected to the shop server. Tools:", ", ".join(tools))
        while True:
            question = await asyncio.to_thread(input, "\nCustomer: ")
            if question.strip().lower() in ("", "quit", "exit"):
                break
            print("Agent:", await run_agent(shop, tools, instructions, question))


if __name__ == "__main__":
    asyncio.run(main())
