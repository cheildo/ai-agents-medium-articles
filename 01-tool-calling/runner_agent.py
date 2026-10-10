import json

import anthropic
from anthropic import beta_tool
from anthropic.lib.tools import ToolError

client = anthropic.Anthropic()
MODEL = "claude-opus-5-5"
STOCK = {"wireless mouse": 12, "usb-c hub": 0, "hdmi cable": 40}


@beta_tool(strict=True)
def check_stock(product_name: str) -> str:
    """Check how many units of a product are available right now.

    Use this when the customer asks if a product is in stock or can be ordered.
    Returns the number of units in stock. Zero means the product is sold out.

    Args:
        product_name: The product name as the customer wrote it, for example "usb-c hub".
    """
    name = product_name.strip().lower()
    if name not in STOCK:
        known = ", ".join(sorted(STOCK))
        raise ToolError(f"Unknown product '{product_name}'. Products we sell: {known}.")
    return json.dumps({"product": name, "units_in_stock": STOCK[name]})


def ask(question):
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        max_iterations=10,
        system="You are a customer support assistant for an online electronics shop.",
        tools=[check_stock],
        messages=[{"role": "user", "content": question}],
    )
    final_message = runner.until_done()
    return "".join(block.text for block in final_message.content if block.type == "text")


if __name__ == "__main__":
    print(ask("Is the HDMI cable in stock?"))
