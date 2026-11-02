"""HTTP API tests using FastAPI's TestClient with injected fake services."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import SAMPLE_DOCS, FakeDatabase, FakeLLM


def _answer_handler(kind, system, user):
    if kind == "route":
        return json.dumps({"route": "documents", "reason": "test"})
    if kind == "sql":
        return "SELECT count(*) AS value FROM orders"
    return "Gold members get 60 days [1]."


@pytest.fixture
def client_factory(make_services):
    def _make(**kwargs):
        services = make_services(**kwargs)
        return TestClient(create_app(lambda: services)), services

    return _make


def test_health_reports_component_status(client_factory):
    client, _ = client_factory(database=FakeDatabase(unavailable=True))
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["database_reachable"] is False
    assert body["llm_configured"] is False


def test_upload_and_list_documents(client_factory):
    client, _ = client_factory()
    with open(SAMPLE_DOCS / "return_policy.txt", "rb") as fh:
        res = client.post("/documents", files=[("files", ("return_policy.txt", fh, "text/plain"))])
    assert res.status_code == 200
    result = res.json()["results"][0]
    assert result["status"] == "ingested" and result["chunks"] > 0

    docs = client.get("/documents").json()
    assert docs[0]["source"] == "return_policy.txt"

    deleted = client.delete(f"/documents/{result['doc_id']}")
    assert deleted.status_code == 200
    assert client.delete(f"/documents/{result['doc_id']}").status_code == 404


def test_upload_multiple_files_partial_failure(client_factory):
    client, _ = client_factory()
    files = [
        ("files", ("good.txt", b"Valid policy text about refunds.", "text/plain")),
        ("files", ("bad.exe", b"MZ...", "application/octet-stream")),
    ]
    res = client.post("/documents", files=files)
    assert res.status_code == 200
    statuses = {r["filename"]: r["status"] for r in res.json()["results"]}
    assert statuses == {"good.txt": "ingested", "bad.exe": "error"}


@pytest.mark.parametrize(
    "filename, content, status, code",
    [
        ("virus.exe", b"MZ", 415, "unsupported_file_type"),
        ("empty.txt", b"", 422, "empty_document"),
        ("broken.pdf", b"%PDF-1.4 garbage", 422, "document_parse_error"),
    ],
)
def test_upload_rejects_malformed_files(client_factory, filename, content, status, code):
    client, _ = client_factory()
    res = client.post("/documents", files=[("files", (filename, content, "application/octet-stream"))])
    assert res.status_code == status
    assert res.json()["error"] == code


def test_upload_size_limit(client_factory, settings):
    settings.max_upload_mb = 1
    client, _ = client_factory()
    res = client.post("/documents", files=[("files", ("big.txt", b"a" * (1024 * 1024 + 10), "text/plain"))])
    assert res.status_code == 413


def test_query_documents_end_to_end(client_factory):
    client, services = client_factory(llm=FakeLLM(_answer_handler))
    for path in SAMPLE_DOCS.iterdir():
        services.ingestion.ingest_path(path)
    res = client.post("/query", json={"question": "What is the return window for Gold members?"})
    assert res.status_code == 200
    body = res.json()
    assert body["route"] == "documents"
    assert body["citations"] and body["citations"][0]["source"]
    assert body["sql"] is None
    assert body["latency_ms"] >= 0


def test_query_sql_includes_sql_details(client_factory):
    client, _ = client_factory(llm=FakeLLM(_answer_handler))
    body = client.post("/query", json={"question": "How many orders?", "mode": "sql"}).json()
    assert body["sql"]["query"].startswith("SELECT")
    assert body["sql"]["rows"] == [[42]]


@pytest.mark.parametrize(
    "payload",
    [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 2001},
     {"question": "valid question", "mode": "web"}, {"question": "ok?", "extra": 1}],
)
def test_query_validation_errors(client_factory, payload):
    client, _ = client_factory()
    assert client.post("/query", json=payload).status_code == 422


def test_query_malformed_json(client_factory):
    client, _ = client_factory()
    res = client.post("/query", content=b"{not json", headers={"content-type": "application/json"})
    assert res.status_code == 422


def test_sql_endpoint_blocks_writes(client_factory):
    client, _ = client_factory()
    res = client.post("/sql", json={"sql": "DROP TABLE orders"})
    assert res.status_code == 400
    assert res.json()["error"] == "unsafe_sql"


def test_sql_endpoint_database_down(client_factory):
    client, _ = client_factory(database=FakeDatabase(unavailable=True))
    res = client.post("/sql", json={"sql": "SELECT * FROM orders"})
    assert res.status_code == 503
    assert res.json()["error"] == "database_unavailable"


def test_unhandled_errors_return_generic_500(client_factory):
    client, services = client_factory()

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    services.agent.run = boom
    client = TestClient(client.app, raise_server_exceptions=False)
    res = client.post("/query", json={"question": "anything here"})
    assert res.status_code == 500
    assert "secret" not in res.text
