# AI Agents: Build Your First MCP Server

Code for the article "AI Agents: Build Your First MCP Server" by [Cheick Zida](https://medium.com/@cheickzida).

An MCP server for a small online electronics shop, with three tools (`get_order_status`, `check_stock`, `cancel_order`), a resource (the return policy) and a prompt, plus an agent that uses the server through an MCP client.

| File | What it is |
|------|------------|
| `shop_server.py` | The MCP server (MCP Python SDK 2, `MCPServer`) |
| `agent.py` | An agent loop with the OpenAI Responses API that gets its tools from the server and asks before any tool that changes data |
| `runner_agent.py` | A shorter agent with the OpenAI Agents SDK (read-only tools only) |
| `http_client.py` | A client for the server over Streamable HTTP |
| `host-configs/` | Config for Claude Desktop and Cursor (`claude_desktop_config.json`) and VS Code (`vscode_mcp.json`) |
| `test_*.py` | 20 tests: the server in memory and over stdio, the host configs, HTTP, and the agents against a fake OpenAI API |

## Run it

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q                      # 20 tests, no API key needed
uv run mcp dev shop_server.py         # open the server in the MCP Inspector (needs Node.js)
```

To run the agent, put your OpenAI key in a `.env` file (it is ignored by git):

```bash
echo 'OPENAI_API_KEY=your-key' > .env
uv run --env-file .env python agent.py
```

To run the server over HTTP:

```bash
uv run mcp run shop_server.py --transport streamable-http   # http://127.0.0.1:8000/mcp
uv run python http_client.py
```

The data is fake and the server has no authorization. Do not expose it to a network as it is.
