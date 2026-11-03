"""Spawn the MCP server as a subprocess and discover its tools over stdio, exactly as an
MCP client (Claude Desktop, an IDE) would. Needs no database or models: services are lazy."""

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]


async def _list_tools():
    params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server"], cwd=str(ROOT),
                                   env={**os.environ, "OPENAI_API_KEY": ""})
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.list_tools()
        return {t.name: t for t in result.tools}


def test_tool_discovery_over_stdio():
    tools = asyncio.run(_list_tools())
    assert set(tools) == {"search_documents", "list_documents", "list_tables", "describe_table",
                          "run_readonly_sql", "ask_research_agent"}
    assert "sql" in tools["run_readonly_sql"].inputSchema["properties"]
    assert tools["search_documents"].description
