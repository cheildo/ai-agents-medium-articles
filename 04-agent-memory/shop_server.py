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
    version="1.1.0",
    instructions="Order, stock and policy tools for the customer support team of an online electronics shop.",
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

# The product catalog, the way a shop database returns it: many fields the LLM never needs
CATALOG = [
    {
        "sku": f"SKU-{2000 + n}",
        "name": name,
        "category": category,
        "price_usd": price,
        "units_in_stock": (n * 7) % 23,
        "description": f"{name} for home and office. " + "Tested for daily use, ships in recyclable packaging. " * 3,
        "weight_grams": 80 + n * 15,
        "warehouse_bin": f"B{n:03d}",
        "supplier_id": f"SUP-{300 + n % 4}",
        "image_urls": [f"https://cdn.example.com/{2000 + n}/{i}.jpg" for i in range(4)],
        "created_at": "2025-03-01T09:00:00Z",
    }
    for n, (name, category, price) in enumerate([
        ("USB-C charger 65W", "chargers", 39.0), ("USB-C charger 30W", "chargers", 24.0),
        ("Wireless charger pad", "chargers", 29.0), ("Car charger dual USB-C", "chargers", 19.0),
        ("Power bank 10000 mAh", "chargers", 35.0), ("Power bank 20000 mAh", "chargers", 49.0),
        ("Travel charger with 3 ports", "chargers", 45.0), ("Magnetic charging cable", "cables", 15.0),
        ("USB-C to USB-C cable 2 m", "cables", 12.0), ("HDMI cable 2 m", "cables", 11.0),
        ("Laptop stand aluminium", "accessories", 42.0), ("Wireless mouse", "accessories", 29.99),
    ])
]

POLICIES = {
    "returns": """\
# Return policy
- Customers can return any item within 30 days of delivery for a full refund.
- Items must be unused, with all accessories, in the original box. Opened software and gift cards cannot be returned.
- Damaged or faulty items can be returned within 30 days even if opened. The customer sends a photo of the damage first.
- Orders that have not shipped yet can be cancelled instead of returned.
- The customer starts a return from the "My orders" page. We email a prepaid return label within one business day.
- Return shipping is free for faulty items and for orders over 50 USD. For other returns we deduct 5.99 USD.
- Refunds go back to the original payment method within 5 business days after the item reaches our warehouse.
- Items bought on sale are refunded at the price the customer paid.
- After 30 days we cannot accept a return, but faulty items may still be covered by the warranty.
""",
    "shipping": """\
# Shipping policy
- Orders placed before 2 pm (shop time) on a business day ship the same day. Later orders ship the next business day.
- Standard shipping takes 3 to 5 business days and is free for orders over 35 USD, otherwise 4.99 USD.
- Express shipping takes 1 to 2 business days and costs 12.99 USD.
- We ship with DHL and UPS. The tracking number is emailed when the order ships and shown on the "My orders" page.
- Tracking can take up to 24 hours to show the first scan.
- If a package shows as delivered but did not arrive, the customer should check with neighbors and wait one more
  business day, then contact us. We open a claim with the carrier and send a replacement or a refund within 7 days.
- We cannot change the delivery address after an order ships. Before it ships, support can change it once.
- We ship to the United States and Canada only. Canadian orders may have import fees, paid by the customer.
""",
    "warranty": """\
# Warranty policy
- Every product has a 1 year warranty from the delivery date. Laptop stands and chargers have 2 years.
- The warranty covers defects in materials and workmanship. It does not cover drops, water damage, or normal wear.
- To make a claim, the customer needs the order ID, a short description of the problem, and a photo or video.
- We first try to fix the problem with troubleshooting steps. If that fails, we send a replacement of the same model.
- If the model is no longer sold, we send the closest current model or refund the price paid.
- Replacements ship within 3 business days after we approve the claim. The customer does not need to send the
  faulty item back unless we ask for it.
- A replacement keeps the remaining warranty of the original item, or 90 days, whichever is longer.
""",
}


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
def search_products(
    query: Annotated[str, Field(max_length=100, description="Words to look for, for example 'usb-c charger'.")],
    limit: Annotated[int, Field(ge=1, le=10, description="How many products to return.")] = 3,
) -> dict:
    """Search the product catalog by name or category.

    Use this when the customer asks which products we sell, or for a product by type.
    Returns the name, price and stock of the best matches. Only the fields the agent needs.
    """
    words = query.lower().split()
    matches = [p for p in CATALOG if all(w in f"{p['name']} {p['category']}".lower() for w in words)]
    return {
        "total_matches": len(matches),
        "products": [
            {"name": p["name"], "price_usd": p["price_usd"], "units_in_stock": p["units_in_stock"]}
            for p in matches[:limit]
        ],
    }


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


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def get_policy(
    topic: Annotated[Literal["returns", "shipping", "warranty"], Field(description="Which policy to read.")],
) -> str:
    """Read one of the shop's policies.

    Call this before you answer any question about returns, refunds, shipping, delivery or warranty.
    Never answer those questions from memory.
    """
    return POLICIES[topic]


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


if __name__ == "__main__":
    mcp.run()
