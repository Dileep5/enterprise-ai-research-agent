import asyncio

import pytest

from app import mcp_server
from tests.conftest import FakeDatabase


@pytest.fixture
def mcp_services(ingested_services):
    services = ingested_services(database=FakeDatabase())
    mcp_server.set_services_factory(lambda: services)
    yield services
    mcp_server.set_services_factory(mcp_server.get_services)


def test_tools_are_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert names == {"search_documents", "list_documents", "list_tables", "describe_table",
                     "run_readonly_sql", "ask_research_agent"}


def test_search_documents_tool(mcp_services):
    out = mcp_server.search_documents("restocking fee for opened tents", top_k=3)
    assert out["sufficient_evidence"]
    assert out["passages"][0]["source"] == "return_policy.txt"
    assert len(out["passages"]) == 3


def test_search_documents_clamps_top_k(mcp_services):
    assert len(mcp_server.search_documents("tents", top_k=1000)["passages"]) <= 20


def test_run_readonly_sql_rejects_writes(mcp_services):
    out = mcp_server.run_readonly_sql("UPDATE orders SET status = 'x'")
    assert out["error"] == "unsafe_sql"


def test_run_readonly_sql_returns_rows(mcp_services):
    out = mcp_server.run_readonly_sql("SELECT count(*) FROM orders")
    assert out["rows"] == [[42]]


def test_list_documents_tool(mcp_services):
    sources = {d["source"] for d in mcp_server.list_documents()}
    assert "annual_report_2025.pdf" in sources
