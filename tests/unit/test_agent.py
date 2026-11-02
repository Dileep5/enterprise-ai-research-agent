"""LangGraph workflow tests with a scripted LLM and an in-memory database."""

import json

import pytest

from app.agent.citations import cited_indices
from app.agent.graph import INSUFFICIENT_MESSAGE
from app.agent.router import heuristic_route, parse_route, route_question
from app.agent.sql_generation import extract_sql
from tests.conftest import FailingLLM, FakeDatabase, FakeLLM


def scripted(route="documents", sql="SELECT count(*) AS value FROM orders", answer="See [1]."):
    def handler(kind, system, user):
        if kind == "route":
            return json.dumps({"route": route, "reason": "test"})
        if kind == "sql":
            return f"```sql\n{sql}\n```"
        return answer(user) if callable(answer) else answer

    return handler


# ----------------------------------------------------------------- routing
@pytest.mark.parametrize(
    "question, expected",
    [
        ("What is the return window for Gold members?", "documents"),
        ("According to the handbook, when are cycle counts done?", "documents"),
        ("What was total revenue by store in 2025?", "sql"),
        ("How many orders were cancelled?", "sql"),
        ("Which products are below their reorder point in inventory?", "sql"),
        ("Compare our actual online revenue share with the target in the annual report", "both"),
    ],
)
def test_heuristic_router(question, expected):
    assert heuristic_route(question)[0] == expected


def test_parse_route_handles_wrapped_and_invalid_json():
    assert parse_route('Sure! {"route": "sql", "reason": "numbers"}') == ("sql", "numbers")
    assert parse_route('{"route": "web"}') is None
    assert parse_route("not json") is None


def test_router_falls_back_to_heuristic_when_llm_fails():
    route, reason = route_question("Total revenue by store?", FailingLLM())
    assert route == "sql" and reason.startswith("keyword heuristic")


def test_router_falls_back_on_garbage_output():
    llm = FakeLLM(lambda *_: "I think SQL maybe")
    route, reason = route_question("What is the refund policy?", llm)
    assert route == "documents" and "heuristic" in reason


def test_extract_sql_variants():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("Here:\n```\nSELECT 2\n```") == "SELECT 2"
    assert extract_sql("SELECT 3;") == "SELECT 3"


def test_cited_indices_dedupes_in_order():
    assert cited_indices("A [2]. B [1][2]. C [10].") == [2, 1, 10]


# -------------------------------------------------------- document route
def test_document_question_returns_grounded_answer_with_citations(ingested_services):
    llm = FakeLLM(scripted(route="documents",
                           answer="Gold members have a 60-day return window [1]."))
    services = ingested_services(llm=llm)
    state = services.agent.run("What is the return window for Gold loyalty members?")

    assert state["route"] == "documents"
    assert not state["insufficient_evidence"]
    assert state["citations"][0]["id"] == 1
    assert state["citations"][0]["source"] in {"return_policy.txt", "loyalty_program.docx"}
    assert state["steps"] == ["route", "retrieve", "synthesize"]
    # The answer prompt must contain numbered evidence.
    answer_prompt = [u for k, u in llm.calls if k == "answer"][0]
    assert "[1] (" in answer_prompt


def test_unrelated_question_short_circuits_without_llm_answer(ingested_services):
    llm = FakeLLM(scripted(route="documents"))
    services = ingested_services(llm=llm)
    state = services.agent.run("Who won the 1998 FIFA World Cup final?")
    assert state["insufficient_evidence"]
    assert state["answer"] == INSUFFICIENT_MESSAGE
    assert not [c for c in llm.calls if c[0] == "answer"]


def test_llm_declaring_insufficient_evidence_is_respected(ingested_services):
    llm = FakeLLM(scripted(route="documents", answer="INSUFFICIENT_EVIDENCE"))
    state = ingested_services(llm=llm).agent.run("What restocking fee applies to opened tents?")
    assert state["insufficient_evidence"] and state["citations"] == []


def test_invalid_citation_numbers_are_dropped_with_warning(ingested_services):
    llm = FakeLLM(scripted(route="documents", answer="Fee is 15% [1][99]."))
    state = ingested_services(llm=llm).agent.run("What restocking fee applies to opened tents?")
    assert [c["id"] for c in state["citations"]] == [1]
    assert any("do not exist" in w for w in state["warnings"])


def test_uncited_answer_is_flagged(ingested_services):
    llm = FakeLLM(scripted(route="documents", answer="The fee is 15%."))
    state = ingested_services(llm=llm).agent.run("What restocking fee applies to opened tents?")
    assert any("did not cite" in w for w in state["warnings"])


def test_llm_outage_returns_extractive_fallback(ingested_services):
    services = ingested_services(llm=FailingLLM())
    state = services.agent.run("What restocking fee applies to opened tents?", mode="documents")
    assert "language model is unavailable" in state["answer"]
    assert state["citations"]
    assert any("Answer generation unavailable" in w for w in state["warnings"])


# ------------------------------------------------------------- SQL route
def test_sql_question_executes_guarded_query(make_services):
    db = FakeDatabase()
    llm = FakeLLM(scripted(route="sql", answer="There are 42 orders [SQL]."))
    state = make_services(llm=llm, database=db).agent.run("How many orders are there?")
    assert state["route"] == "sql"
    assert state["sql_result"]["rows"] == [[42]]
    assert db.executed[0].endswith("LIMIT 200")
    assert state["answer"] == "There are 42 orders [SQL]."
    assert state["steps"] == ["route", "generate_sql", "execute_sql", "synthesize"]


def test_sql_error_triggers_one_retry_with_error_feedback(make_services):
    db = FakeDatabase(fail_first=True)
    llm = FakeLLM(scripted(route="sql", answer="42 [SQL]"))
    state = make_services(llm=llm, database=db).agent.run("How many orders?")
    assert state["sql_attempts"] == 2
    retry_prompt = [u for k, u in llm.calls if k == "sql"][1]
    assert 'column "revenue" does not exist' in retry_prompt
    assert state["sql_result"] is not None


def test_unsafe_generated_sql_is_blocked_and_never_executed(make_services):
    db = FakeDatabase()
    llm = FakeLLM(scripted(route="sql", sql="DELETE FROM orders"))
    state = make_services(llm=llm, database=db).agent.run("Delete all orders")
    assert db.executed == []
    assert state["sql_result"] is None
    assert state["insufficient_evidence"]
    assert "Only SELECT" in state["sql_error"]
    assert state["sql_attempts"] == 2  # retried once, still unsafe


def test_database_outage_is_reported_gracefully(make_services):
    llm = FakeLLM(scripted(route="sql"))
    state = make_services(llm=llm, database=FakeDatabase(unavailable=True)).agent.run("Total revenue?")
    assert "unavailable" in state["answer"]
    assert state["sql_attempts"] == 1  # non-retryable


def test_sql_route_without_llm_reports_configuration_problem(make_services):
    services = make_services()  # default fake LLM is unavailable
    state = services.agent.run("Total revenue by store", mode="sql")
    assert state["insufficient_evidence"]
    assert "not configured" in state["sql_error"].lower() or "OPENAI" in state["sql_error"]


# ------------------------------------------------------------ both route
def test_multi_source_question_uses_documents_and_sql(ingested_services):
    llm = FakeLLM(scripted(route="both", answer="Actual share is 30% [SQL] vs a 35% target [1]."))
    db = FakeDatabase()
    services = ingested_services(llm=llm, database=db)
    state = services.agent.run("Compare online revenue share with the target in the annual report")
    assert state["route"] == "both"
    assert state["steps"] == ["route", "retrieve", "generate_sql", "execute_sql", "synthesize"]
    assert state["sql_result"] and state["citations"]
    answer_prompt = [u for k, u in llm.calls if k == "answer"][0]
    assert "[SQL]" in answer_prompt and "[1] (" in answer_prompt


def test_multi_source_sql_generation_receives_document_context(ingested_services):
    llm = FakeLLM(scripted(route="both", answer="46 combinations [SQL]; rule from [1]."))
    services = ingested_services(llm=llm, database=FakeDatabase())
    services.agent.run("The handbook explains when products must be reordered; how many need reordering?")
    sql_prompt = [u for k, u in llm.calls if k == "sql"][0]
    assert "Document context" in sql_prompt
    assert "reorder point" in sql_prompt


def test_sql_route_with_empty_result_falls_back_to_documents(ingested_services):
    from app.sql.database import SQLResult

    db = FakeDatabase({"default": SQLResult("SELECT NULL", ["supplier"], [[None]], 1, False)})
    llm = FakeLLM(scripted(route="sql", answer="TrailTech Electronics has a 60 day lead time [1]."))
    state = ingested_services(llm=llm, database=db).agent.run("Which supplier has the longest lead time?")
    assert state["steps"] == ["route", "generate_sql", "execute_sql", "retrieve", "synthesize"]
    assert state["citations"][0]["source"] == "supplier_directory.csv"
    assert not state["insufficient_evidence"]


def test_sql_answer_refusal_falls_back_to_documents(ingested_services):
    calls = {"answer": 0}

    def handler(kind, system, user):
        if kind == "route":
            return json.dumps({"route": "sql", "reason": "looks numeric"})
        if kind == "sql":
            return "SELECT count(*) AS value FROM orders"
        calls["answer"] += 1
        # First synthesis sees only the SQL result and refuses; second sees documents.
        return "INSUFFICIENT_EVIDENCE" if calls["answer"] == 1 else "Net Promoter Score rose to 58 [1]."

    state = ingested_services(llm=FakeLLM(handler)).agent.run("What was the Net Promoter Score?")
    assert state["steps"] == ["route", "generate_sql", "execute_sql", "synthesize", "retrieve", "synthesize"]
    assert not state["insufficient_evidence"]
    assert state["citations"][0]["source"] == "annual_report_2025.pdf"


def test_forced_mode_skips_router(make_services):
    llm = FakeLLM(scripted(route="documents", answer="42 [SQL]"))
    state = make_services(llm=llm).agent.run("anything", mode="sql")
    assert state["route"] == "sql" and state["route_reason"] == "forced by request"
    assert not [c for c in llm.calls if c[0] == "route"]
