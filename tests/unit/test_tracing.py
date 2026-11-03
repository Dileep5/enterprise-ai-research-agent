"""Langfuse wiring. No live Langfuse server is contacted: the client is created with an
unreachable host and tracing is asserted at the callback level only."""

from app.observability.tracing import build_callbacks
from tests.conftest import make_settings


def test_tracing_disabled_without_keys(tmp_path):
    assert build_callbacks(make_settings(tmp_path)) == []


def test_tracing_enabled_with_keys(tmp_path):
    settings = make_settings(tmp_path, langfuse_public_key="pk-lf-test", langfuse_secret_key="sk-lf-test",
                             langfuse_host="http://127.0.0.1:9")
    callbacks = build_callbacks(settings)
    assert len(callbacks) == 1
    assert type(callbacks[0]).__name__ == "LangchainCallbackHandler"


def test_agent_passes_callbacks_to_graph(make_services, tmp_path):
    services = make_services()
    sentinel = object()
    services.agent.callbacks = [sentinel]
    captured = {}
    original = services.agent.graph.invoke

    def spy(state, config):
        captured.update(config)
        return original(state, config={**config, "callbacks": []})

    services.agent.graph.invoke = spy
    services.agent.run("What is the refund policy?", mode="documents")
    assert captured["callbacks"] == [sentinel] and captured["run_name"] == "research-agent"
