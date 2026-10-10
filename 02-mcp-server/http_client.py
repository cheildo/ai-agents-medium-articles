import asyncio

from mcp import Client

URL = "http://localhost:8000/mcp"


async def main(url=URL):
    async with Client(url) as shop:
        result = await shop.call_tool("check_stock", {"product_name": "hdmi cable"})
        print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
