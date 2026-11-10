"""The compliance graph offline: mock LLM, networkx graph, the real Postgres checkpointer in the test database."""
import pytest
from django.core.files.base import ContentFile
from django.core.management import call_command
from pydantic import ValidationError

from apps.agents import graph
from apps.agents.mocking import mock_verify
from apps.agents.models import AgentRun
from apps.agents.schemas import Draft, Plan, Verification
from apps.ingest.models import Chunk, Document
from apps.kg.services.build import build_graph
from apps.rag.services.bm25 import clear_cache
from apps.rules.models import RuleConflict
from apps.rules.services.pipeline import run_rules
from core import mock_llm
from core.embeddings import embed_texts
from core.models import EvalRun

pytestmark = pytest.mark.django_db

DOCUMENTS = {
    "airworthiness_directive_excerpt": [
        ("2. Required action", "Replace the side-stay before the accumulated flight hours on part 65-45210-3 reach 12000. "
         "The replacement part is PN-77-4418. A ferry flight with the part over the limit is not permitted."),
    ],
    "maintenance_planning_procedure": [
        ("2.1 Replacement criteria", "IF part 65-45210-3 is installed AND its flight hours are at least 12000, THEN "
         "replace it with PN-77-4418 before the next revenue flight. Exception: the part may stay fitted for 50 "
         "flight hours when PN-77-4418 is on order and the inspection in section 2.2 finds no crack."),
        ("2.2 Inspection", "Do a detailed visual inspection of the side-stay at every C-check."),
        ("3. Deferral", "A missing non-essential placard may be deferred under the MEL. The side-stay replacement "
         "itself cannot be deferred under the MEL."),
    ],
    "engineering_order_EO-2291": [
        ("1. Side-stay limit", "IF part 65-45210-3 is installed on the narrow-body fleet, THEN replace it at 11500 flight hours."),
    ],
}


@pytest.fixture
def corpus():
    for doc_index, (title, sections) in enumerate(DOCUMENTS.items()):
        document = Document(title=title, original_name=f"{title}.docx", sha256=str(doc_index) * 64, size_bytes=10,
                            file_type="docx", status=Document.Status.DONE)
        document.file.save(f"{title}.docx", ContentFile(b"x"), save=False)
        document.save()
        for ordinal, (section, text) in enumerate(sections):
            Chunk.objects.create(
                document=document, ordinal=ordinal, text=text, text_hash=f"{doc_index}-{ordinal}",
                page_start=ordinal + 1, section_title=section, section_path=section, token_count=30,
                metadata={}, embedding=embed_texts([text])[0].tolist(),
            )
    clear_cache()
    build_graph(extract=True)
    run_rules()
    assert RuleConflict.objects.filter(is_conflict=True).exists()


def _nodes(run: AgentRun) -> list[str]:
    return list(run.steps.values_list("node", flat=True))


def test_plan_must_include_search_and_an_answer_needs_claims():
    with pytest.raises(ValidationError, match="hybrid_search is required"):
        Plan.model_validate({"tools": ["kg_query"], "search_query": "AD 2023-14-09", "reason": "x"})
    with pytest.raises(ValidationError, match="at most once"):
        Plan.model_validate({"tools": ["hybrid_search", "hybrid_search"], "search_query": "abc", "reason": "x"})
    with pytest.raises(ValidationError, match="at least one claim"):
        Draft.model_validate({"status": "answered", "answer": "Yes.", "claims": []})


def test_out_of_scope_goes_straight_to_a_refusal(corpus):
    run = graph.execute(graph.create_run("What is a good recipe for banana bread?"))
    assert run.status == AgentRun.Status.REFUSED
    assert _nodes(run) == ["classify", "refuse"]
    assert "only answer questions about the maintenance documents" in run.final["answer"]


def test_lookup_pauses_at_approval_and_resumes_from_the_postgres_checkpoint(corpus):
    run = graph.execute(graph.create_run("Which replacement part number goes with AD 2023-14-09 on the side-stay?"))
    assert run.status == AgentRun.Status.AWAITING_APPROVAL
    assert run.intent == "lookup"
    assert _nodes(run) == ["classify", "plan", "hybrid_retrieve", "graph_lookup", "rule_check", "draft",
                           "verify_citations", "human_approval"]
    assert run.steps.get(node="human_approval").status == "waiting"
    assert graph.checkpoint_count(run) > 0
    retrieve = run.steps.get(node="hybrid_retrieve")
    assert retrieve.tool_calls[0]["tool"] == "hybrid_search" and retrieve.tool_calls[0]["result"]["mode"] == "hybrid_rerank"
    assert run.steps.get(node="classify").llm_calls[0]["schema"] == "Classification"

    run = graph.resume(AgentRun.objects.get(id=run.id), approved=True, note="looks right")
    assert run.status == AgentRun.Status.DONE
    assert _nodes(run)[-2:] == ["human_approval", "final"]
    assert run.steps.get(node="human_approval").status == "done"
    assert run.verified is True and run.loops == 0
    citation = run.final["citations"][0]
    assert citation["verified"] and citation["document"] and citation["page"] and citation["section"]
    assert "77-4418" in run.final["answer"]
    assert run.final["approval_note"] == "looks right"


def test_rejection_keeps_the_note(corpus):
    run = graph.execute(graph.create_run("What inspection is done on the side-stay at every C-check?"))
    run = graph.resume(run, approved=False, note="cite the AD instead")
    assert run.status == AgentRun.Status.REJECTED
    assert run.final["status"] == "rejected" and run.final["approval_note"] == "cite the AD instead"


def test_conflicting_applicable_rule_is_surfaced_not_resolved(corpus):
    run = graph.execute(graph.create_run(
        "Part 65-45210-3 is at 11,800 flight hours on a narrow-body aircraft. Is it due for replacement now?"))
    assert run.intent == "compliance_check"
    conflicts = run.review["conflicts"]
    assert conflicts, "the 11500 vs 12000 conflict should be found by rule_check"
    assert "conflict" in run.draft["answer"].lower()
    quotes = " ".join(claim["quote"] for claim in run.draft["claims"])
    assert "11500" in quotes and "12000" in quotes
    run = graph.resume(run, approved=True)
    assert run.final["conflicts"] and run.verified is True


def test_failed_verification_loops_back_and_stops_at_the_cap(corpus, monkeypatch, settings):
    assert settings.AGENT_MAX_LOOPS == 2
    judged = []

    def never_supported(prompt):
        judged.append(prompt)
        return {"verdicts": [{"claim": 1, "supported": False, "reason": "forced failure"}],
                "refined_query": "side-stay replacement part PN-77-4418"}

    monkeypatch.setitem(mock_llm._HANDLERS, Verification, never_supported)
    run = graph.execute(graph.create_run("Which replacement part number goes with AD 2023-14-09 on the side-stay?"))
    nodes = _nodes(run)
    assert run.status == AgentRun.Status.AWAITING_APPROVAL
    assert run.loops == 2 and run.verified is False
    assert nodes.count("verify_citations") == 3 and nodes.count("hybrid_retrieve") == 3 and nodes.count("draft") == 3
    assert len(judged) == 3
    searches = [step.tool_calls[0]["args"]["query"] for step in run.steps.filter(node="hybrid_retrieve")]
    assert searches[1] == searches[2] == "side-stay replacement part PN-77-4418"
    assert [step.loop for step in run.steps.filter(node="verify_citations")] == [0, 1, 2]
    assert "after 2 re-retrieval(s)" in run.warning and "forced failure" in run.warning
    run = graph.resume(run, approved=True, note="approved despite the warning")
    assert run.status == AgentRun.Status.DONE
    assert run.final["verified"] is False and run.final["warning"]
    assert run.final["citations"][0]["verified"] is False


def test_one_failed_check_then_a_pass_uses_one_loop(corpus, monkeypatch):
    calls = {"n": 0}

    def fail_once(prompt):
        calls["n"] += 1
        if calls["n"] > 1:
            return mock_verify(prompt)
        return {"verdicts": [{"claim": 1, "supported": False, "reason": "forced"}],
                "refined_query": "PN-77-4418 replacement part"}

    monkeypatch.setitem(mock_llm._HANDLERS, Verification, fail_once)
    run = graph.execute(graph.create_run("Which replacement part number goes with AD 2023-14-09 on the side-stay?"))
    assert run.loops == 1 and run.verified is True and run.warning == ""


def test_a_quote_that_is_not_in_its_source_fails_without_the_judge(corpus, monkeypatch):
    def invented(prompt):
        return {"status": "answered", "answer": "Use PN-77-4418.", "conflict_note": "",
                "claims": [{"text": "Use PN-77-4418.", "source": 1, "quote": "The part must be PN-77-4418 always."}]}

    monkeypatch.setitem(mock_llm._HANDLERS, Draft, invented)
    run = graph.execute(graph.create_run("Which replacement part number goes with AD 2023-14-09 on the side-stay?"))
    verify = run.steps.filter(node="verify_citations").first()
    assert verify.output["verification"]["checks"][0]["quote_found"] is False
    assert verify.output["verification"]["judged"] == 0
    assert run.loops == 2 and run.verified is False


def test_page_shows_provider_trace_approval_and_final_answer(client, corpus):
    page = client.get("/agents/").content.decode()
    assert "LLM <strong>mock</strong>" in page and "LLM_PROVIDER=mock" in page
    response = client.post("/agents/", {"question": "Can the side-stay replacement on part 65-45210-3 be deferred under the MEL?"})
    run = AgentRun.objects.get()
    assert response.status_code == 302 and response["Location"] == f"/agents/run/{run.id}/"
    body = client.get(response["Location"]).content.decode()
    assert "Pending approval" in body and "verify_citations" in body and "rule_lookup" in body
    assert client.post(f"/agents/run/{run.id}/decide/", {"decision": "reject"}).status_code == 302
    assert AgentRun.objects.get().status == AgentRun.Status.AWAITING_APPROVAL
    client.post(f"/agents/run/{run.id}/decide/", {"decision": "approve", "note": "ok"})
    body = client.get(f"/agents/run/{run.id}/").content.decode()
    assert AgentRun.objects.get().status == AgentRun.Status.DONE
    assert "Final answer" in body and "cannot be deferred" in body
    trace = client.get(f"/agents/run/{run.id}/trace/?poll=1")
    assert trace["HX-Refresh"] == "true"


def test_eval_agent_scores_every_case(corpus):
    call_command("eval_agent")
    run = EvalRun.objects.get(kind="agent")
    rows = {row["id"]: row for row in run.metrics["rows"]}
    assert len(rows) == 6
    assert rows["out-of-scope"]["status"] == "refused" and rows["out-of-scope"]["answer_ok"]
    assert rows["conflict-limit"]["conflict_ok"] is True
    assert "LLM: `mock`" in run.report_md and "| case | intent correct |" in run.report_md
