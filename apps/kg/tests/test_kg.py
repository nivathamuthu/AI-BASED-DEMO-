"""Extraction cache, relation filtering, entity resolution, stores, GraphRAG and the page."""
import pytest
from django.core.files.base import ContentFile
from django.core.management import call_command

from apps.ingest.models import Chunk, Document
from apps.kg.models import GraphBuild, KGExtraction
from apps.kg.schemas import KGOutput
from apps.kg.services.build import build_graph
from apps.kg.services.eval import run_eval
from apps.kg.services.extract import extract_chunk, filter_relations
from apps.kg.services.graphrag import graph_answer
from apps.kg.services.normalize import entity_key, same_identifier
from apps.kg.services.resolve import resolve
from apps.rag.services.bm25 import clear_cache
from core.embeddings import embed_texts
from core.graph_store import GraphStoreError, Neo4jStore, get_store
from core.llm import Usage, structured_call
from core.models import EvalRun, LLMCall

pytestmark = pytest.mark.django_db

TEXTS = [
    "AD 2023-14-09 applies to the narrow-body fleet when part 65-45210-3 is installed in the side-stay.",
    "IF part 65-45210-3 is installed AND its flight hours are at least 12000, "
    "THEN replace it with PN-77-4418 before the next revenue flight.",
    "Replacement bushing P/N 77-4418, issued as a matched set with the pin.",
    "Do a detailed visual inspection of the side-stay at every C-check. "
    "Record the part number 65-45210-3 against AD 2023-14-09.",
    "Crew meals are loaded in the forward galley.",
]


@pytest.fixture
def corpus():
    document = Document(
        title="Side-stay pack", original_name="pack.txt", sha256="b" * 64, size_bytes=10,
        file_type="txt", status=Document.Status.DONE,
    )
    document.file.save("pack.txt", ContentFile(b"pack"), save=False)
    document.save()
    chunks = [
        Chunk.objects.create(
            document=document, ordinal=index, text=text, text_hash=f"h{index}", page_start=index + 1,
            section_title="2. Required action", section_path="2. Required action", token_count=20,
            metadata={}, embedding=embed_texts([text])[0].tolist(),
        )
        for index, text in enumerate(TEXTS)
    ]
    clear_cache()
    return chunks


def test_part_number_spellings_share_a_key():
    assert entity_key("PN-77-4418") == entity_key("P/N 77-4418") == entity_key("part number 77-4418")
    assert entity_key("12000 FH") == entity_key("12000 flight hours")
    assert entity_key("The Side-Stay") == entity_key("side-stay")
    assert entity_key("PN-77-4418") != entity_key("PN-77-4410")
    assert same_identifier("PN-77-4418", "P/N 77-4418") and same_identifier("77-4418", "PN-77-4418")
    assert not same_identifier("65-45210-3", "S1000 through S1899")
    assert not same_identifier("PN-77-4418", "PN-77-4410")
    assert same_identifier("side-stay", "main landing gear side-stay")


def test_relations_with_unknown_endpoints_or_invented_evidence_are_dropped():
    output = KGOutput.model_validate({
        "entities": [{"name": "AD 2023-14-09", "type": "Regulation"}, {"name": "PN-77-4418", "type": "Part"}],
        "relations": [
            {"source": "AD 2023-14-09", "relation": "APPLIES_TO", "target": "P/N 77-4418",
             "evidence_quote": "AD 2023-14-09 covers PN-77-4418."},
            {"source": "AD 2023-14-09", "relation": "APPLIES_TO", "target": "the aircraft",
             "evidence_quote": "AD 2023-14-09 covers PN-77-4418."},
            {"source": "AD 2023-14-09", "relation": "GOVERNED_BY", "target": "PN-77-4418",
             "evidence_quote": "A sentence that is not in the chunk."},
        ],
    })
    _entities, kept, dropped = filter_relations(output, "AD 2023-14-09 covers PN-77-4418.")
    assert len(kept) == 1
    assert {row["reason"].split(":")[0] for row in dropped} == {
        "endpoint not in entity list", "evidence_quote is not in the chunk",
    }


def test_extraction_is_cached_per_prompt_version(corpus):
    chunk = corpus[1]
    first, ran = extract_chunk(chunk)
    assert ran and first.prompt_version == "v2"
    assert {entity["name"] for entity in first.entities} >= {"65-45210-3", "PN-77-4418", "Replacement"}
    assert any(row["relation"] == "REPLACES" and row["source"] == "PN-77-4418" for row in first.relations)
    calls = LLMCall.objects.filter(task="kg.extract").count()
    _again, ran_again = extract_chunk(chunk)
    assert ran_again is False
    assert LLMCall.objects.filter(task="kg.extract").count() == calls
    _forced, ran_forced = extract_chunk(chunk, force=True)
    assert ran_forced is True


def test_naive_prompt_loses_relations_to_the_endpoint_check(corpus):
    v1, _ = extract_chunk(corpus[0], version="v1")
    v2, _ = extract_chunk(corpus[0], version="v2", force=True)
    assert len(v1.dropped) >= 2
    assert v2.dropped == []


def test_resolution_merges_duplicates_and_keeps_provenance(corpus):
    for chunk in corpus:
        extract_chunk(chunk)
    graph = resolve(KGExtraction.objects.select_related("chunk__document"))
    bushing_part = next(node for node in graph.nodes if node["name"] == "PN-77-4418")
    assert "P/N 77-4418" in bushing_part["aliases"]
    assert {row["chunk_id"] for row in bushing_part["provenance"]} == {corpus[1].id, corpus[2].id}
    assert {row["page"] for row in bushing_part["provenance"]} == {2, 3}
    assert any(entry.get("name") == "PN-77-4418" for entry in graph.merge_log)
    assert graph.merged_count >= 1
    for edge in graph.edges:
        assert edge["provenance"] and edge["provenance"][0]["document"] == "Side-stay pack"


def test_serial_range_and_other_part_numbers_never_merge_into_a_part(corpus):
    chunk = corpus[1]
    KGExtraction.objects.create(
        chunk=chunk, prompt_version="v2", llm_provider="google", llm_model="gemini-x",
        entities=[
            {"name": "65-45210-3", "type": "Part", "aliases": ["S1000 through S1899", "P/N 65-45210-3"]},
            {"name": "77-4418", "type": "Part", "aliases": ["P/N 77-4418", "PN-77-4410"]},
            {"name": "bushing", "type": "Component", "aliases": ["bushing P/N 77-4418"]},
            {"name": "PN-77-4418", "type": "Part", "aliases": []},
        ],
        relations=[],
    )
    graph = resolve(KGExtraction.objects.select_related("chunk__document"))
    by_name = {node["name"]: node for node in graph.nodes}
    side_stay_part = by_name["65-45210-3"]
    assert side_stay_part["aliases"] == ["P/N 65-45210-3"]
    assert "S1000 through S1899" not in str(graph.nodes)
    bushing_part = next(node for node in graph.nodes if "P/N 77-4418" in node["aliases"])
    assert {bushing_part["name"], *bushing_part["aliases"]} == {"77-4418", "P/N 77-4418", "PN-77-4418"}
    assert "PN-77-4410" not in str(graph.nodes)
    assert by_name["bushing"]["aliases"] == []
    rejected = {entry["rejected_alias"]: entry["reason"] for entry in graph.merge_log if "rejected_alias" in entry}
    assert rejected["S1000 through S1899"] == "identifiers differ: 65 45210 3 vs s1000 s1899"
    assert set(rejected) == {"S1000 through S1899", "PN-77-4410", "bushing P/N 77-4418"}


def test_build_graph_writes_the_networkx_store_without_new_llm_calls(corpus):
    build = build_graph(extract=True)
    assert build.backend == "networkx"
    assert build.node_count == len(get_store().nodes()) > 5
    assert build.edge_count == len(get_store().edges()) > 5
    calls = LLMCall.objects.count()
    call_command("build_graph")
    assert LLMCall.objects.count() == calls
    assert GraphBuild.objects.count() == 2


def test_graphrag_answers_a_multi_hop_question_with_a_path(corpus):
    build_graph(extract=True)
    result = graph_answer("Which tasks are governed by AD 2023-14-09 for part 65-45210-3?")
    assert result["known"] is True
    assert "Replacement" in result["answer"]
    assert "inspection" in result["answer"]
    assert {seed["name"] for seed in result["seeds"]} >= {"AD 2023-14-09", "65-45210-3"}
    assert result["path"]
    assert result["citations"] and "Side-stay pack" in result["citations"][0]


def test_graphrag_abstains_without_entities_or_context(corpus):
    build_graph(extract=True)
    result = graph_answer("What colour is the cabin carpet?")
    assert result["known"] is False
    assert result["citations"] == []


def test_kg_eval_records_both_methods_and_the_models(corpus):
    build_graph(extract=True)
    run = run_eval()
    assert run.kind == "kg"
    assert {"hybrid", "graphrag", "rows"} <= set(run.metrics)
    assert len(run.metrics["rows"]) == 8
    assert run.llm_provider == "mock" and run.embeddings_provider == "mock"
    assert run.config["graph_backend"] == "networkx"
    assert "LLM:" in run.report_md and "Graph:" in run.report_md
    assert EvalRun.objects.filter(kind="kg").count() == 1


def test_kg_page_graph_frame_and_node_panel(client, corpus):
    build_graph(extract=True)
    page = client.get("/kg/").content.decode()
    assert "merged duplicates" in page and "dropped relations" in page
    frame = client.get("/kg/graph/?type=Part")
    assert frame.status_code == 200
    html = frame.content.decode()
    assert "network.on" in html and "PN-77-4418" in html and "Replacement" not in html
    node = client.get("/kg/node/", {"id": entity_key("PN-77-4418").replace(" ", "-")}).content.decode()
    assert "Side-stay pack, p.2" in node
    assert "P/N 77-4418" in node
    answer = client.post("/kg/", {"question": "Which tasks are governed by AD 2023-14-09 for part 65-45210-3?"})
    body = answer.content.decode()
    assert "Traversed path" in body and "Hybrid only" in body


def test_invalid_extraction_is_retried():
    class Flaky:
        provider = "mock"
        model = "flaky"
        calls = 0

        def generate(self, schema, messages):
            self.calls += 1
            relation = "CAUSES" if self.calls == 1 else "APPLIES_TO"
            return {
                "entities": [{"name": "AD 2023-14-09", "type": "Regulation"}, {"name": "PN-77-4418", "type": "Part"}],
                "relations": [{"source": "AD 2023-14-09", "relation": relation, "target": "PN-77-4418",
                               "evidence_quote": "x"}],
            }, Usage()

    backend = Flaky()
    result = structured_call(KGOutput, "CHUNK:\nx", task="test.kg_retry", backend=backend)
    assert result.relations[0].relation == "APPLIES_TO"
    assert LLMCall.objects.get(task="test.kg_retry").attempts == 2


def test_store_selection_and_unreachable_neo4j(settings):
    settings.GRAPH_BACKEND = "sqlite"
    with pytest.raises(GraphStoreError):
        get_store()
    with pytest.raises(GraphStoreError):
        Neo4jStore("bolt://127.0.0.1:1", "neo4j", "x").nodes()


def test_ingest_hands_the_document_to_the_graph(settings):
    from apps.ingest.services.intake import ingest_bytes

    settings.KG_AUTO_BUILD = True
    ingest_bytes("note.txt", b"AD 2023-14-09 applies when part 65-45210-3 is installed in the side-stay.\n")
    assert KGExtraction.objects.exists()
    assert GraphBuild.objects.exists()
