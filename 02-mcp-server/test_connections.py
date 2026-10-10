"""Checks the host config files and the HTTP transport the article shows."""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcp import Client, StdioServerParameters

import http_client

HERE = Path(__file__).parent


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("file, key", [("claude_desktop_config.json", "mcpServers"), ("vscode_mcp.json", "servers")])
async def test_host_config_launches_the_server(file, key):
    entry = json.loads((HERE / "host-configs" / file).read_text())[key]["shop"]
    args = [str(HERE) if a == "/absolute/path/to/shop-mcp" else a for a in entry["args"]]
    async with Client(StdioServerParameters(command=entry["command"], args=args, cwd="/")) as c:
        assert len((await c.list_tools()).tools) == 3


@pytest.mark.anyio
async def test_streamable_http(capsys):
    port = 8765
    server = subprocess.Popen(
        [sys.executable, "-c", f"import shop_server; shop_server.mcp.run(transport='streamable-http', port={port})"],
        cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        await http_client.main(f"http://localhost:{port}/mcp")
        assert '"units_in_stock": 40' in capsys.readouterr().out
    finally:
        server.terminate()
        server.wait()
