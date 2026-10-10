# AI Agents: Tool Calling Done Right

Code for the article [AI Agents: Tool Calling Done Right](https://medium.com/@cheickzida/ai-agents-tool-calling-done-right-python-guide-342e929b6e6e) by [Cheick Zida](https://medium.com/@cheickzida).

A customer support agent for a small online electronics shop, built with the Claude API. It has three tools (`get_order_status`, `check_stock`, `cancel_order`), strict input schemas, error messages written for the model, and a human approval step before any cancellation.

| File | What it is |
|------|------------|
| `support_agent.py` | The full agent: tool definitions, the tool functions, `run_tool` and the agent loop |
| `runner_agent.py` | The shorter version with the SDK tool runner (`@beta_tool`) |
| `test_support_agent.py` | Runs both against a fake Messages API: parallel calls, errors, approval, `max_tokens`, the step limit |

## Run it

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q          # no API key needed
```

To talk to the agent, put your Anthropic key in a `.env` file (it is ignored by git):

```bash
echo 'ANTHROPIC_API_KEY=your-key' > .env
uv run --env-file .env python support_agent.py
```

The data is fake. The agent asks you in the terminal before it cancels an order.
