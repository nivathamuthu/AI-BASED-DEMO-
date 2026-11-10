"""GraphRAG: question -> entities -> 1-2 hop traversal -> linked chunks + hybrid chunks -> answer."""
from dataclasses import asdict, dataclass

import numpy as np
from django.conf import settings

from apps.ingest.models import Chunk
from apps.kg.schemas import GraphAnswer
from apps.kg.services.normalize import entity_key, has_digit, normalize_text
from apps.rag.services.answer import UNKNOWN
from apps.rag.services.bm25 import content_tokens, tokenize
from apps.rag.services.hits import Hit
from apps.rag.services.retrieve import is_weak, retrieve
from core.embeddings import embed_query, embed_texts
from core.graph_store import get_store
from core.llm import structured_call
from core.prompts import render_prompt

MAX_SEEDS = 6
MAX_FACTS = 15
MAX_GRAPH_CHUNKS = 4
MAX_CONTEXT = 8

_ENTITY_CACHE: dict = {"key": None, "vectors": None}


@dataclass
class Seed:
    id: str
    name: str
    type: str
    reason: str
    score: float = 1.0


def clear_entity_cache() -> None:
    _ENTITY_CACHE["key"] = None
    _ENTITY_CACHE["vectors"] = None


def _name_vectors(nodes: list[dict]) -> np.ndarray:
    key = (settings.EMBEDDINGS_PROVIDER, tuple(node["id"] for node in nodes))
    if _ENTITY_CACHE["key"] != key:
        _ENTITY_CACHE["vectors"] = embed_texts([f"{node['name']} ({node['type']})" for node in nodes])
        _ENTITY_CACHE["key"] = key
    return _ENTITY_CACHE["vectors"]


def link_entities(question: str, nodes: list[dict]) -> list[Seed]:
    """Exact name or alias match first, then the nearest entity names by embedding."""
    padded = f" {normalize_text(question)} "
    seeds: dict[str, Seed] = {}
    for node in nodes:
        for form in [node["name"], *node.get("aliases", [])]:
            key = entity_key(form)
            if len(key) < 3 or (not has_digit(key) and len(key) < 4):
                continue
            if f" {key} " in padded:
                reason = "name" if form == node["name"] else f"alias {form}"
                seeds.setdefault(node["id"], Seed(node["id"], node["name"], node["type"], reason))
                break
    if nodes:
        vectors = _name_vectors(nodes)
        similarity = vectors @ embed_query(question)
        for index in np.argsort(-similarity)[:3]:
            score = float(similarity[index])
            node = nodes[int(index)]
            if score >= settings.KG_ENTITY_MIN_SIMILARITY and node["id"] not in seeds:
                seeds[node["id"]] = Seed(node["id"], node["name"], node["type"], f"embedding {score:.2f}", score)
    exact = [seed for seed in seeds.values() if not seed.reason.startswith("embedding")]
    fuzzy = [seed for seed in seeds.values() if seed.reason.startswith("embedding")]
    longest_first = sorted(exact, key=lambda seed: -len(seed.name))
    return (longest_first + fuzzy)[:MAX_SEEDS]


_TYPE_WORDS = {
    "aircraft": "Aircraft", "fleet": "Aircraft", "component": "Component", "assembly": "Component",
    "task": "Task", "action": "Task", "regulation": "Regulation", "directive": "Regulation",
    "procedure": "Procedure", "part": "Part", "interval": "Interval", "limit": "Interval",
}


def answer_type(question: str) -> str | None:
    """The first entity-type word in the question ("Which intervals apply to the tasks ..." -> Interval)."""
    for word in tokenize(question):
        singular = word.removesuffix("s")
        if singular in _TYPE_WORDS:
            return _TYPE_WORDS[singular]
    return None


def rank_facts(question: str, seeds: list[Seed], nodes: list[dict], edges: list[dict]) -> list[dict]:
    """Prefer edges next to a seed, edges into a node that bridges several seeds, and question words."""
    seed_ids = {seed.id for seed in seeds}
    by_id = {node["id"]: node for node in nodes}
    seed_neighbours: dict[str, set[str]] = {}
    for edge in edges:
        if edge["source"] in seed_ids:
            seed_neighbours.setdefault(edge["target"], set()).add(edge["source"])
        if edge["target"] in seed_ids:
            seed_neighbours.setdefault(edge["source"], set()).add(edge["target"])
    wanted = content_tokens(question)
    asked = answer_type(question)
    scored = []
    for edge in edges:
        source, target = by_id.get(edge["source"]), by_id.get(edge["target"])
        if not source or not target:
            continue
        bridge = max(len(seed_neighbours.get(edge["source"], ())), len(seed_neighbours.get(edge["target"], ())))
        words = content_tokens(f"{source['name']} {target['name']}")
        typed = 4 if asked in (source["type"], target["type"]) else 0
        score = (3 if edge.get("hop", 1) == 1 else 0) + 2 * max(0, bridge - 1) + typed + len(wanted & words)
        scored.append((score, edge, source, target))
    scored.sort(key=lambda item: (-item[0], item[1].get("hop", 1), item[1]["source"], item[1]["relation"]))
    facts = []
    for score, edge, source, target in scored[:MAX_FACTS]:
        origin = edge["provenance"][0] if edge.get("provenance") else {}
        facts.append({
            "source": source["name"],
            "source_type": source["type"],
            "relation": edge["relation"],
            "target": target["name"],
            "target_type": target["type"],
            "hop": edge.get("hop", 1),
            "score": score,
            "evidence": edge.get("evidence", ""),
            "chunk_ids": [row["chunk_id"] for row in edge.get("provenance", [])],
            "citation": _citation(origin),
        })
    return facts


def _citation(origin: dict) -> str:
    parts = [origin.get("document", "")]
    if origin.get("page"):
        parts.append(f"p.{origin['page']}")
    if origin.get("section"):
        parts.append(f"§ {origin['section']}")
    return ", ".join(part for part in parts if part)


def _chunk_hits(chunk_ids: list[int]) -> list[Hit]:
    rows = {chunk.id: chunk for chunk in Chunk.objects.select_related("document").filter(id__in=chunk_ids)}
    hits = []
    for chunk_id in chunk_ids:
        chunk = rows.get(chunk_id)
        if chunk is None:
            continue
        hits.append(Hit(
            chunk_id=chunk.id, text=chunk.text, document_id=chunk.document_id,
            document_title=chunk.document.title, page_start=chunk.page_start,
            section=chunk.section_path, score=1.0, lexical=1,
        ))
    return hits


def graph_answer(question: str, k: int = 5) -> dict:
    """Answer from graph facts plus chunks. The payload carries the seeds and the traversed path."""
    store = get_store()
    nodes = store.nodes()
    seeds = link_entities(question, nodes)
    facts: list[dict] = []
    if seeds:
        sub_nodes, sub_edges = store.neighborhood([seed.id for seed in seeds], hops=2)
        facts = rank_facts(question, seeds, sub_nodes, sub_edges)
    graph_chunk_ids: list[int] = []
    for fact in facts:
        for chunk_id in fact["chunk_ids"]:
            if chunk_id not in graph_chunk_ids:
                graph_chunk_ids.append(chunk_id)
    graph_hits = _chunk_hits(graph_chunk_ids[:MAX_GRAPH_CHUNKS])
    hybrid_hits = retrieve(question, "hybrid", k=k)
    context: list[Hit] = []
    for hit in graph_hits + hybrid_hits:
        if hit.chunk_id not in {item.chunk_id for item in context}:
            context.append(hit)
    context = context[:MAX_CONTEXT]

    base = {"question": question, "seeds": [asdict(seed) for seed in seeds], "facts": facts}
    if not facts and is_weak(hybrid_hits, "hybrid"):
        return {**base, **_payload(UNKNOWN, False, [], context, [])}

    facts_block = "\n".join(
        f"[F{index}] {fact['source']} ({fact['source_type']}) {fact['relation']} "
        f"{fact['target']} ({fact['target_type']}) - {fact['citation']}"
        for index, fact in enumerate(facts, start=1)
    ) or "(no facts)"
    chunk_block = "\n\n".join(
        f"[{index}] {hit.text}\n(source: {hit.citation()})" for index, hit in enumerate(context, start=1)
    )
    prompt = render_prompt(
        "answer_with_graph", "v1",
        {"facts": facts_block, "chunk": chunk_block, "question": question, "domain": "aviation maintenance"},
        GraphAnswer,
    )
    output = structured_call(
        GraphAnswer, prompt, task="kg.graph_answer", prompt_name="answer_with_graph", prompt_version="v1"
    )
    if not output.known or output.answer.strip().lower() == "unknown":
        return {**base, **_payload(UNKNOWN, False, [], context, [])}
    used = [facts[i - 1] for i in output.fact_indexes if 1 <= i <= len(facts)]
    cited = [context[i - 1] for i in output.citation_indexes if 1 <= i <= len(context)]
    fact_chunks = {chunk_id for fact in used for chunk_id in fact["chunk_ids"]}
    cited += [hit for hit in context if hit.chunk_id in fact_chunks and hit not in cited]
    return {**base, **_payload(output.answer, True, cited or context[:1], context, used)}


def _payload(answer: str, known: bool, cited: list[Hit], context: list[Hit], used: list[dict]) -> dict:
    return {
        "answer": answer,
        "known": known,
        "citations": [hit.citation() for hit in cited] if known else [],
        "path": used,
        "hits": [
            {"chunk_id": hit.chunk_id, "document": hit.document_title, "page": hit.page_start,
             "section": hit.section, "text": hit.text}
            for hit in context
        ],
    }
