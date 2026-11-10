"""Merge per-chunk extractions into one graph with provenance on every node and edge."""
from collections import Counter
from dataclasses import dataclass, field

from apps.kg.services.normalize import entity_key, identifier_tokens, same_identifier


@dataclass
class ResolvedGraph:
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    merge_log: list[dict] = field(default_factory=list)
    merged_count: int = 0
    dropped_count: int = 0
    chunk_count: int = 0


class _UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def _provenance(extraction) -> dict:
    chunk = extraction.chunk
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "document": chunk.document.title,
        "page": chunk.page_start,
        "section": chunk.section_path,
    }


def _accepted_aliases(entity: dict, chunk_id: int, rejected: list[dict] | None) -> list[str]:
    """Aliases that carry the same identifier as the name. A serial range or another part number never merges."""
    accepted = []
    for alias in entity.get("aliases", []):
        if not entity_key(alias):
            continue
        if same_identifier(entity["name"], alias):
            accepted.append(alias)
        elif rejected is not None:
            rejected.append({
                "name": entity["name"],
                "rejected_alias": alias,
                "chunk_id": chunk_id,
                "reason": f"identifiers differ: {' '.join(identifier_tokens(entity['name'])) or 'none'}"
                          f" vs {' '.join(identifier_tokens(alias)) or 'none'}",
            })
    return accepted


def resolve(extractions) -> ResolvedGraph:
    """`extractions` is an iterable of KGExtraction with chunk and document loaded."""
    extractions = list(extractions)
    union = _UnionFind()
    surfaces: dict[str, Counter] = {}
    types: dict[str, Counter] = {}
    provenance: dict[str, list[dict]] = {}

    rejected: list[dict] = []

    for extraction in extractions:
        for entity in extraction.entities:
            key = entity_key(entity["name"])
            if not key:
                continue
            union.find(key)
            for alias in _accepted_aliases(entity, extraction.chunk_id, rejected):
                union.union(key, entity_key(alias))

    for extraction in extractions:
        source = _provenance(extraction)
        for entity in extraction.entities:
            key = entity_key(entity["name"])
            if not key:
                continue
            root = union.find(key)
            surfaces.setdefault(root, Counter())[entity["name"]] += 1
            for alias in _accepted_aliases(entity, extraction.chunk_id, None):
                surfaces[root][alias] += 0
            types.setdefault(root, Counter())[entity["type"]] += 1
            rows = provenance.setdefault(root, [])
            if source["chunk_id"] not in {row["chunk_id"] for row in rows}:
                rows.append(source)

    graph = ResolvedGraph(chunk_count=len(extractions))
    graph.merge_log.extend(rejected)
    node_id = {root: root.replace(" ", "-") for root in surfaces}
    for root, counter in surfaces.items():
        name = min(counter, key=lambda form: (-counter[form], len(form)))
        aliases = sorted(form for form in counter if form != name)
        kind = types[root].most_common(1)[0][0]
        graph.nodes.append({
            "id": node_id[root],
            "name": name,
            "type": kind,
            "aliases": aliases,
            "provenance": provenance[root],
        })
        if aliases:
            graph.merge_log.append({
                "node": node_id[root],
                "name": name,
                "merged": [
                    {"form": form, "reason": "same normalized name" if entity_key(form) == entity_key(name) else "alias"}
                    for form in aliases
                ],
            })
            graph.merged_count += len(aliases)
        if len(types[root]) > 1:
            graph.merge_log.append({
                "node": node_id[root],
                "name": name,
                "type_votes": dict(types[root]),
                "chosen_type": kind,
            })

    edges: dict[tuple[str, str, str], dict] = {}
    for extraction in extractions:
        source = _provenance(extraction)
        graph.dropped_count += len(extraction.dropped or [])
        for relation in extraction.relations:
            start = union.find(entity_key(relation["source"]))
            end = union.find(entity_key(relation["target"]))
            if start not in node_id or end not in node_id or start == end:
                continue
            key = (node_id[start], relation["relation"], node_id[end])
            edge = edges.setdefault(key, {
                "source": key[0],
                "relation": key[1],
                "target": key[2],
                "evidence": relation["evidence_quote"],
                "provenance": [],
            })
            if source["chunk_id"] not in {row["chunk_id"] for row in edge["provenance"]}:
                edge["provenance"].append({**source, "evidence": relation["evidence_quote"]})
    graph.edges = list(edges.values())
    return graph
