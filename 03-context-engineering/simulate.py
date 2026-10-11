"""Replay a scripted 30 turn support chat through the agent, with and without context management.

No API key and no real model: a fake Responses API plays the LLM and estimates input tokens with
tiktoken's o200k_base encoding. The tools are the real shop server. Run: uv run python simulate.py
"""
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import math
import os
import sys

os.environ.setdefault("OPENAI_API_KEY", "not-used")

import httpx2
import openai
import tiktoken

from mcp import Client

import agent
import shop_server

ENC = tiktoken.get_encoding("o200k_base")
logging.disable(logging.INFO)  # hide the HTTP request log lines

# (customer message, tool calls the fake LLM makes, the fake LLM's answer)
CHAT = [
    ("Hi, I'm Ana. Where is my order ORD-10042?", [("get_order_status", {"order_id": "ORD-10042"})],
     "Hi Ana, thanks for reaching out. Your order ORD-10042 has shipped with DHL, and the tracking number is JD014600003SE. It contains one wireless mouse, 29.99 USD in total. You can follow the package on the DHL website with that number."),
    ("When will it arrive?", [("get_policy", {"topic": "shipping"})],
     "Standard shipping takes 3 to 5 business days after the order ships. Tracking can take up to 24 hours to show the first scan, so if the DHL page looks empty, please check again tomorrow. Your mouse should arrive within the week."),
    ("Can I still return it if I open the box?", [("get_policy", {"topic": "returns"})],
     "You can return unused items within 30 days of delivery, in the original box with all accessories. If you open it and the mouse turns out to be faulty, you can still return it within 30 days. Just send us a photo of the problem first."),
    ("Do you sell USB-C chargers?", [("search_products", {"query": "usb-c charger"})],
     "Yes, we sell several USB-C chargers. The USB-C charger 65W costs 39 USD but is sold out right now. The USB-C charger 30W costs 24 USD and is in stock. We also have a car charger with two USB-C ports for 19 USD."),
    ("Is the 30W one in stock?", [("search_products", {"query": "charger 30w"})],
     "Yes, the USB-C charger 30W is in stock, with 7 units available right now. It costs 24 USD. If you order it with something else and the total goes over 35 USD, standard shipping is free."),
    ("What about my other order, ORD-10043?", [("get_order_status", {"order_id": "ORD-10043"})],
     "Your order ORD-10043 is still processing, so it has not shipped yet. It contains a USB-C hub and an HDMI cable, 54.50 USD in total. Since it has not shipped, you can still cancel it or change the delivery address once if you need to."),
    ("Is the USB-C hub in stock? I want a second one.", [("check_stock", {"product_name": "usb-c hub"})],
     "I'm sorry, the USB-C hub is sold out right now, with 0 units in stock. Your existing order ORD-10043 already includes one hub and is still being prepared. I can't reserve a second one, but you can check back in a few days."),
    ("What warranty does the hub have?", [("get_policy", {"topic": "warranty"})],
     "The USB-C hub has a 1 year warranty from the delivery date. It covers defects in materials and workmanship, but not drops, water damage or normal wear. If something goes wrong, we first try troubleshooting, then send a replacement of the same model."),
    ("Do you have power banks?", [("search_products", {"query": "power bank"})],
     "Yes, we have two power banks. The 10000 mAh model costs 35 USD and the 20000 mAh model costs 49 USD. Both are in stock."),
    ("Which one is lighter?", [],
     "The 10000 mAh power bank is the lighter of the two, so it is the better choice for a bag or a pocket. The 20000 mAh model is heavier but holds twice the charge."),
    ("Do you ship to Canada?", [("get_policy", {"topic": "shipping"})],
     "Yes, we ship to the United States and Canada. Orders to Canada may have import fees, and those are paid by the customer on delivery. Standard shipping still takes 3 to 5 business days after the order ships."),
    ("How long does a refund take?", [("get_policy", {"topic": "returns"})],
     "Once your return reaches our warehouse, the refund goes back to your original payment method within 5 business days. We email you when we receive the item, so you will know when the 5 days start."),
    ("Do you sell HDMI cables?", [("search_products", {"query": "hdmi"})],
     "Yes, we sell a 2 m HDMI cable for 11 USD. It is the same cable that is in your order ORD-10043. Let me know if you would like to know how many are in stock."),
    ("How many are in stock?", [("check_stock", {"product_name": "hdmi cable"})],
     "We have 40 HDMI cables in stock, so there is no problem ordering more. They usually ship the same day if you order before 2 pm shop time on a business day."),
    ("Is my mouse order still on the way?", [("get_order_status", {"order_id": "ORD-10042"})],
     "Yes, ORD-10042 is shipped with DHL and on its way. The tracking number is JD014600003SE. Standard shipping takes 3 to 5 business days, so it should arrive soon."),
    ("What if the package says delivered but I don't have it?", [("get_policy", {"topic": "shipping"})],
     "If the tracking says delivered but you don't have the package, please check with your neighbors and wait one more business day. If it still has not appeared, contact us and we will open a claim with the carrier. We then send a replacement or a refund within 7 days."),
    ("Do you have laptop stands?", [("search_products", {"query": "laptop stand"})],
     "Yes, we have an aluminium laptop stand for 42 USD, and it is in stock. Because it costs more than 35 USD, standard shipping is free for it."),
    ("What warranty does the stand have?", [("get_policy", {"topic": "warranty"})],
     "Laptop stands have a 2 year warranty from the delivery date, which is longer than most of our products. It covers defects, but not drops, water damage or normal wear."),
    ("Any wireless chargers?", [("search_products", {"query": "wireless charger"})],
     "Yes, we have a wireless charger pad for 29 USD, and it is in stock. It works with any phone that supports wireless charging."),
    ("Is shipping free if I order it?", [("get_policy", {"topic": "shipping"})],
     "Standard shipping is free for orders over 35 USD. The wireless charger pad alone is 29 USD, so shipping would cost 4.99 USD. If you add a cable or another item and pass 35 USD, shipping becomes free."),
    ("What's in order ORD-10043 again?", [("get_order_status", {"order_id": "ORD-10043"})],
     "Your order ORD-10043 contains a USB-C hub and an HDMI cable, 54.50 USD in total. It is still processing, so it has not shipped yet."),
    ("Can I return a sale item?", [("get_policy", {"topic": "returns"})],
     "Yes, you can return items you bought on sale. The refund is the price you actually paid, not the full price. The other return rules are the same: within 30 days, unused, in the original box."),
    ("Do you sell magnetic cables?", [("search_products", {"query": "magnetic cable"})],
     "Yes, we sell a magnetic charging cable for 15 USD, and it is in stock. The magnetic tip stays in the phone, so you only connect the cable."),
    ("And travel chargers?", [("search_products", {"query": "travel charger"})],
     "Yes, we have a travel charger with 3 ports for 45 USD. It is in stock, and since it costs more than 35 USD, standard shipping is free."),
    ("Is the wireless mouse in stock?", [("check_stock", {"product_name": "wireless mouse"})],
     "Yes, the wireless mouse is in stock, with 12 units available. It is the same model as in your order ORD-10042."),
    ("If my mouse is faulty, what do I do?", [("get_policy", {"topic": "warranty"})],
     "If your mouse is faulty, send us the order ID ORD-10042, a short description of the problem, and a photo or a video. We first try a few troubleshooting steps with you. If that does not fix it, we send a replacement of the same model within 3 business days of approving the claim."),
    ("Does express shipping cost more?", [("get_policy", {"topic": "shipping"})],
     "Yes, express shipping takes 1 to 2 business days and costs 12.99 USD. Standard shipping takes 3 to 5 business days and is free over 35 USD."),
    ("Which chargers are under 30 USD?", [("search_products", {"query": "charger", "limit": 10})],
     "Here are our chargers under 30 USD: the USB-C charger 30W for 24 USD, the wireless charger pad for 29 USD, and the car charger with two USB-C ports for 19 USD. All three are in stock."),
    ("Thanks. Can you remind me of both my order numbers?", [],
     "Sure. Your orders are ORD-10042, the wireless mouse, which has shipped with DHL, and ORD-10043, the USB-C hub and HDMI cable, which is still processing."),
    ("Great, that's all.", [],
     "You're welcome, Ana. If anything else comes up with your orders, just write to us. Have a nice day."),
]

SUMMARY = (
    "- Customer: Ana. Orders: ORD-10042 (wireless mouse, shipped with DHL, tracking JD014600003SE) and "
    "ORD-10043 (USB-C hub and HDMI cable, 54.50 USD, processing).\n"
    "- Ana asked about shipping times, returns of opened items, the hub warranty, and chargers and power banks.\n"
    "- The USB-C hub is sold out. No action taken on any order. No open questions."
)


# Input prices relative to one uncached input token of gpt-6.1-sol (OpenAI models page and prompt caching guide)
CACHE_READ = 0.05  # gpt-6.1-sol reads cached tokens at 5% of the input price
CACHE_WRITE = 1.25  # writing tokens to the cache costs 1.25 times the input price
MIN_CACHED = 1_024  # the shortest prefix that can be cached
SUMMARY_PRICE = 0.05  # gpt-6-luna input costs 0.10 USD per million tokens, gpt-6.1-sol 2.00 USD
USD_PER_TOKEN = 2.00 / 1_000_000


def count(text):
    return len(ENC.encode(text))


def parts(body):
    """The request as (key, tokens) pairs: instructions and tools first, then one pair per input item.
    The token count is an estimate of what the model reads: the text of each part plus a few tokens per item."""
    head = (body.get("instructions") or "") + json.dumps(body.get("tools", []))
    result = [(head, count(head))]
    items = body["input"] if isinstance(body["input"], list) else [{"content": body["input"]}]
    for item in items:
        content = item.get("content", "")
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content)
        text = content + item.get("arguments", "") + item.get("output", "") + item.get("name", "")
        result.append((json.dumps(item, sort_keys=True), count(text) + 4))
    return result


class PromptCache:
    """A simple model of prompt caching: every request writes its whole prompt to the cache, and a later
    request reads the longest earlier prompt that is an exact prefix of its own. Everything else is written."""

    def __init__(self):
        self.written = set()

    def cost(self, request_parts):
        digest, size, prefixes = hashlib.sha256(), 0, []
        for key, tokens in request_parts:
            digest.update(key.encode())
            size += tokens
            prefixes.append((digest.hexdigest(), size))
        if size < MIN_CACHED:
            return size  # too short to cache, full price
        cached = next((n for d, n in reversed(prefixes) if d in self.written and n >= MIN_CACHED), 0)
        self.written.add(prefixes[-1][0])
        return cached * CACHE_READ + (size - cached) * CACHE_WRITE


class FakeLLM:
    def __init__(self):
        self.turn, self.step = 0, 0
        self.requests = []  # (turn, input tokens) of every agent request
        self.summary_tokens, self.cost, self.cache = 0, 0.0, PromptCache()

    def handler(self, request):
        body = json.loads(request.content)
        request_parts = parts(body)
        tokens = sum(t for _, t in request_parts)
        if body["instructions"] == agent.SUMMARY_PROMPT:
            self.summary_tokens += tokens
            self.cost += tokens * SUMMARY_PRICE * CACHE_WRITE
            output = [message(SUMMARY)]
        else:
            self.requests.append((self.turn, tokens))
            self.cost += self.cache.cost(request_parts)
            _, calls, answer = CHAT[self.turn]
            if self.step < len(calls):
                name, args = calls[self.step]
                output = [{"type": "function_call", "id": f"fc_{self.turn}", "call_id": f"call_{self.turn}",
                           "name": name, "arguments": json.dumps(args), "status": "completed"}]
                self.step += 1
            else:
                output = [message(answer)]
                self.turn, self.step = self.turn + 1, 0
        return httpx2.Response(200, json={
            "id": "resp", "object": "response", "created_at": 0, "model": agent.MODEL, "status": "completed",
            "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": tokens, "output_tokens": 50, "total_tokens": tokens + 50,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        })


def message(text):
    return {"type": "message", "id": "msg", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


async def run(managed):
    """Play the whole chat once. Returns the fake LLM (its requests, tokens and cost) and the context events."""
    fake = FakeLLM()
    saved = agent.llm, agent.COMPACT_AT
    agent.llm = openai.AsyncOpenAI(api_key="test", max_retries=0, http_client=openai.DefaultAsyncHttpxClient(
        transport=httpx2.MockTransport(fake.handler)))
    if not managed:
        agent.COMPACT_AT = math.inf
    try:
        async with Client(shop_server.mcp) as shop:
            tools = {tool.name: tool for tool in (await shop.list_tools()).tools}
            context, log = agent.Context(), io.StringIO()
            with contextlib.redirect_stdout(log):
                for question, _, _ in CHAT:
                    await agent.run_turn(shop, tools, context, question)
    finally:
        agent.llm, agent.COMPACT_AT = saved
    events = [line for line in log.getvalue().splitlines() if "compacted" in line]
    return fake, events


def last_request_per_turn(fake):
    return {turn + 1: tokens for turn, tokens in fake.requests}


async def main():
    managed, events = await run(True)
    full, _ = await run(False)
    m, f = last_request_per_turn(managed), last_request_per_turn(full)
    print("turn  keep_everything  managed")
    for turn in f:
        print(f"{turn:<4}  {f[turn]:>15,}  {m[turn]:>7,}")
    f_total = sum(t for _, t in full.requests)
    m_total = sum(t for _, t in managed.requests) + managed.summary_tokens
    print(f"input tokens for the whole chat: keep everything {f_total:,}, managed {m_total:,}")
    print(f"input cost with prompt caching: keep everything {full.cost * USD_PER_TOKEN:.4f} USD, "
          f"managed {managed.cost * USD_PER_TOKEN:.4f} USD")
    print(*events, sep="\n")
    if "--json" in sys.argv:
        print(json.dumps({"full": f, "managed": m, "requests": managed.requests}))


if __name__ == "__main__":
    asyncio.run(main())
