import json

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5-5"
MAX_STEPS = 10

# Fake data so the example runs without a real database
ORDERS = {
    "ORD-10042": {
        "status": "shipped",
        "carrier": "DHL",
        "tracking_number": "JD014600003SE",
        "items": ["Wireless mouse"],
        "total_usd": 29.99,
    },
    "ORD-10043": {
        "status": "processing",
        "items": ["USB-C hub", "HDMI cable"],
        "total_usd": 54.50,
    },
}

STOCK = {"wireless mouse": 12, "usb-c hub": 0, "hdmi cable": 40}

SYSTEM_PROMPT = (
    "You are a customer support assistant for an online electronics shop. "
    "Use the tools to look up real data. Never guess order details or stock levels."
)

TOOLS = [
    {
        "name": "get_order_status",
        "description": (
            "Look up the current status of a customer's order. "
            "Use this whenever the customer asks where their order is, when it will arrive, "
            "or what they ordered. Returns the status (processing, shipped, delivered or cancelled), "
            "the items, the total price, and the tracking number if the order has shipped. "
            "It does not change the order in any way."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID from the confirmation email, for example 'ORD-10042'.",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_stock",
        "description": (
            "Check how many units of a product are available right now. "
            "Use this when the customer asks if a product is in stock or can be ordered. "
            "Returns the number of units in stock. Zero means the product is sold out."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "The product name as the customer wrote it, for example 'usb-c hub'.",
                }
            },
            "required": ["product_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "cancel_order",
        "description": (
            "Cancel an order that has not shipped yet. "
            "Only use this after the customer has clearly asked to cancel. "
            "Orders that already shipped cannot be cancelled. "
            "A human operator must approve every cancellation, so the call can be declined."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to cancel, for example 'ORD-10043'.",
                },
                "reason": {
                    "type": "string",
                    "enum": ["customer_request", "duplicate_order", "wrong_item"],
                    "description": "Why the order is being cancelled.",
                },
            },
            "required": ["order_id", "reason"],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """An error message written for the model, so it can correct itself."""


def find_order(order_id):
    order_id = order_id.strip().upper()
    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(
            f"No order found with ID '{order_id}'. Order IDs look like 'ORD-10042'. "
            "Ask the customer to check the ID in their confirmation email."
        )
    return order_id, order


def get_order_status(order_id):
    order_id, order = find_order(order_id)
    return {"order_id": order_id, **order}


def check_stock(product_name):
    name = product_name.strip().lower()
    if name not in STOCK:
        known = ", ".join(sorted(STOCK))
        raise ToolError(f"Unknown product '{product_name}'. Products we sell: {known}.")
    return {"product": name, "units_in_stock": STOCK[name]}


def confirm(question):
    answer = input(f"{question} [y/N] ")
    return answer.strip().lower() == "y"


def cancel_order(order_id, reason):
    order_id, order = find_order(order_id)
    if order["status"] == "cancelled":
        raise ToolError(f"Order {order_id} is already cancelled. There is nothing more to do.")
    if order["status"] != "processing":
        raise ToolError(
            f"Order {order_id} is already {order['status']} and can no longer be cancelled. "
            "Offer the customer a return instead."
        )
    if not confirm(f"The agent wants to cancel {order_id} (reason: {reason}). Allow?"):
        raise ToolError(
            "A human operator declined this cancellation. Do not try again. "
            "Tell the customer that a support agent will contact them."
        )
    order["status"] = "cancelled"
    return {"order_id": order_id, "status": "cancelled"}


TOOL_FUNCTIONS = {
    "get_order_status": get_order_status,
    "check_stock": check_stock,
    "cancel_order": cancel_order,
}


def run_tool(name, tool_input):
    """Run one tool and return (content, is_error)."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Unknown tool '{name}'.", True
    try:
        return json.dumps(func(**tool_input)), False
    except ToolError as error:
        return str(error), True
    except Exception as error:
        # A bug in our code. Log the details for us, give the model a short message.
        print(f"[tool crashed] {name}: {error!r}")
        return "The tool failed because of an internal problem. Tell the customer to try again later.", True


def run_agent(user_message):
    messages = [{"role": "user", "content": user_message}]

    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason in ("max_tokens", "refusal"):
            return f"The agent stopped early (reason: {response.stop_reason})."

        # Keep the full reply, including the tool_use blocks
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                content, is_error = run_tool(block.name, block.input)
                print(f"[tool] {block.name}({block.input}) error={is_error}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                        "is_error": is_error,
                    }
                )

        # All results go back together, in one user message
        messages.append({"role": "user", "content": tool_results})

    return "The agent stopped because it reached the step limit."


if __name__ == "__main__":
    question = "Hi, where is my order ORD-10042? And is the USB-C hub back in stock?"
    print(run_agent(question))
