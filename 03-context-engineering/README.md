# AI Agents: Context Engineering for Long-Running Agents

Code for the article AI Agents: Context Engineering for Long-Running Agents by [Cheick Zida](https://medium.com/@cheickzida).

The shop support agent from [02-mcp-server](../02-mcp-server), now with a conversation that lasts across messages and a
context manager that keeps it small: a token meter, short tool results, policies loaded on demand, and compaction of
old turns into a summary.

| File | What it is |
|------|------------|
| `shop_server.py` | The MCP server, with two new tools: `search_products` (returns only the fields the agent needs) and `get_policy` (loads a policy only when a question needs it) |
| `agent.py` | The agent (OpenAI Responses API) with the context manager: `record_usage`, `cap`, `compact`, `run_turn` |
| `builtin_compaction.py` | The same agent with OpenAI's server-side compaction (`context_management`) instead of our own summary |
| `simulate.py` | Replays a scripted 30 message chat with and without compaction, with a fake LLM, token estimates from tiktoken and a simple model of prompt caching costs |
| `test_*.py` | Tests for the server and the agent, against a fake OpenAI API |

## Run it

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q              # 22 tests, no API key needed
uv run python simulate.py     # the numbers behind the article's chart
```

To chat with the agent, put your OpenAI key in a `.env` file (it is ignored by git):

```bash
echo 'OPENAI_API_KEY=your-key' > .env
uv run --env-file .env python agent.py
```

The limit in `agent.py` (`COMPACT_AT = 3_500` tokens) is small on purpose, so you can see compaction in a short chat.
Set yours from the token counts of real conversations.
