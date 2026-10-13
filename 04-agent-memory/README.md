# AI Agents: Make Your Agent Remember Every Customer

Code for the article [AI Agents: Make Your Agent Remember Every Customer](https://medium.com/@cheickzida/ai-agents-make-your-agent-remember-every-customer-59a05f80ddbb) by [Cheick Zida](https://medium.com/@cheickzida).

The shop support agent from [03-context-engineering](../03-context-engineering), now with long-term memory: after each
chat a small LLM decides what to add, update or delete, SQLite keeps the memories per customer, the agent recalls them
at the start of the next chat, and customers can ask to be forgotten.

| File | What it is |
|------|------------|
| `memory.py` | The memory store (SQLite), the extraction with Structured Outputs, recall, forget and expire |
| `agent.py` | The agent from article 3 (OpenAI Responses API, MCP tools, compaction), with memory added |
| `shop_server.py` | The MCP server from article 3 (orders, stock, products, policies) |
| `fake_api.py`, `test_*.py` | Tests against a fake OpenAI API and the real MCP server, no API key needed |

## Run it

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q              # 24 tests, no API key needed
```

To chat with the agent, put your OpenAI key in a `.env` file (it is ignored by git) and pass a customer ID. In a real
app the ID comes from your login system.

```bash
echo 'OPENAI_API_KEY=your-key' > .env
uv run --env-file .env python agent.py cus_1042
```

Run it twice: the second chat starts with what the first one remembered. Type `/forget` to delete everything about
the customer. Memories are stored in `memory.db` next to the code (ignored by git).
