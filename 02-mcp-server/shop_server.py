import logging
from typing import Annotated, Literal

from pydantic import Field

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

# On stdio, stdout carries the protocol messages, so never print() in a server.
# The logging module writes to stderr, which is safe.
logger = logging.getLogger("shop")

mcp = MCPServer(
    "shop",
    version="1.0.0",
    instructions="Order and stock tools for the customer support team of an online electronics shop.",
)

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

RETURN_POLICY = """\
# Return policy
- Customers can return any item within 30 days of delivery for a full refund.
- Items must be unused and in the original box.
- Orders that have not shipped yet can be cancelled instead of returned.
- Refunds go back to the original payment method within 5 business days.
"""


def find_order(order_id: str) -> tuple[str, dict]:
    order_id = order_id.strip().upper()
    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(
            f"No order found with ID '{order_id}'. Order IDs look like 'ORD-10042'. "
            "Ask the customer to check the ID in their confirmation email."
        )
    return order_id, order


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def get_order_status(
    order_id: Annotated[
        str, Field(max_length=20, description="The order ID from the confirmation email, for example 'ORD-10042'.")
    ],
) -> dict:
    """Look up the current status of a customer's order.

    Use this whenever the customer asks where their order is, when it will arrive, or what they ordered.
    Returns the status (processing, shipped, delivered or cancelled), the items, the total price,
    and the tracking number if the order has shipped. It does not change the order in any way.
    """
    order_id, order = find_order(order_id)
    return {"order_id": order_id, **order}


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def check_stock(
    product_name: Annotated[
        str, Field(max_length=100, description="The product name as the customer wrote it, for example 'usb-c hub'.")
    ],
) -> dict:
    """Check how many units of a product are available right now.

    Use this when the customer asks if a product is in stock or can be ordered.
    Returns the number of units in stock. Zero means the product is sold out.
    """
    name = product_name.strip().lower()
    if name not in STOCK:
        known = ", ".join(sorted(STOCK))
        raise ToolError(f"Unknown product '{product_name}'. Products we sell: {known}.")
    return {"product": name, "units_in_stock": STOCK[name]}


@mcp.tool(annotations=ToolAnnotations(destructive_hint=True, open_world_hint=False))
def cancel_order(
    order_id: Annotated[str, Field(max_length=20, description="The order ID to cancel, for example 'ORD-10043'.")],
    reason: Annotated[
        Literal["customer_request", "duplicate_order", "wrong_item"],
        Field(description="Why the order is being cancelled."),
    ],
) -> dict:
    """Cancel an order that has not shipped yet.

    Only use this after the customer has clearly asked to cancel.
    Orders that already shipped cannot be cancelled, offer a return instead.
    """
    order_id, order = find_order(order_id)
    if order["status"] == "cancelled":
        raise ToolError(f"Order {order_id} is already cancelled. There is nothing more to do.")
    if order["status"] != "processing":
        raise ToolError(
            f"Order {order_id} is already {order['status']} and can no longer be cancelled. "
            "Offer the customer a return instead."
        )
    order["status"] = "cancelled"
    logger.info("Cancelled %s (reason: %s)", order_id, reason)
    return {"order_id": order_id, "status": "cancelled"}


@mcp.resource("shop://policies/returns", mime_type="text/markdown")
def return_policy() -> str:
    """The shop's return and refund policy."""
    return RETURN_POLICY


@mcp.prompt()
def support_reply(customer_message: str) -> str:
    """Draft a reply to a customer message, using the shop tools and the return policy."""
    return (
        "You work in customer support for an online electronics shop. "
        "Read the customer message below. Look up every order and product it mentions with the tools, "
        "follow the return policy (resource shop://policies/returns), and write a short, friendly reply.\n\n"
        f"Customer message:\n{customer_message}"
    )


if __name__ == "__main__":
    mcp.run()
