import pytest

from app.agent.grounding import extract_numbers, unsupported_numbers
from tests.conftest import FakeLLM
from tests.unit.test_agent import scripted


def test_extract_numbers_ignores_citations_and_formats():
    assert extract_numbers("Revenue was $180,895.90 [1] and 31% [SQL].") == [180895.90, 31.0]


@pytest.mark.parametrize("answer", ["Revenue was 180,895.90 [SQL].", "Revenue was $180895.9.", "31% online [1]."])
def test_numbers_present_in_evidence_are_supported(answer):
    evidence = "| store | revenue |\n| A | 180895.9 |\nE-commerce represented 31% of revenue."
    assert unsupported_numbers(answer, evidence) == []


def test_hallucinated_number_is_flagged():
    assert unsupported_numbers("Gold members get a 90-day window [1].", "Gold: 60-day return window.") == ["90"]


def test_citation_aware_check_uses_the_cited_source_only():
    sources = {"1": "Opened tents: 15% restocking fee.", "2": "Safety stock raised by 25%."}
    evidence = "\n".join(sources.values())
    assert unsupported_numbers("The fee is 25% [1].", evidence, sources=sources) == ["25"]
    assert unsupported_numbers("The fee is 15% [1]. Stock rises 25% [2].", evidence, sources=sources) == []


def test_question_numbers_and_small_ordinals_are_allowed():
    assert unsupported_numbers("1. In 2025 the top 3 stores ...", "no numbers", "top 3 stores in 2025") == []


def test_agent_flags_unsupported_numbers(ingested_services):
    llm = FakeLLM(scripted(route="documents", answer="Opened tents carry a 25% restocking fee [1]."))
    state = ingested_services(llm=llm).agent.run("What restocking fee applies to opened tents?")
    assert state["grounding"]["numbers_unsupported"] == ["25"]
    assert any("not found in the evidence" in w for w in state["warnings"])


def test_agent_grounded_answer_has_no_flags(ingested_services):
    llm = FakeLLM(scripted(route="documents", answer="Opened tents returned after 14 days carry a 15% fee [1]."))
    state = ingested_services(llm=llm).agent.run("What restocking fee applies to opened tents?")
    assert state["grounding"] == {"numbers_unsupported": [], "checked": True}
