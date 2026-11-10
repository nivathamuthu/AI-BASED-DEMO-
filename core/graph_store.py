"""Graph store with two backends, selected by GRAPH_BACKEND.

    neo4j    - the compose service (default). Nodes are :Entity plus a type label.
    networkx - a node-link JSON file under MEDIA_ROOT/kg/, for zero-setup runs and tests.

Both take the same node and edge dicts:
    node {id, name, type, aliases, provenance: [{chunk_id, document_id, document, page, section}]}
    edge {source, relation, target, evidence, provenance: [... plus evidence]}
"""
from __future__ import annotations

import json
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from django.conf import settings

from apps.kg.schemas import ENTITY_TYPES, RELATION_TYPES


class GraphStoreError(Exception):
    """The configured backend cannot be reached or used."""


class GraphStore(Protocol):
    name: str

    def replace(self, nodes: list[dict], edges: list[dict]) -> None: ...
    def nodes(self) -> list[dict]: ...
    def edges(self) -> list[dict]: ...
    def neighborhood(self, ids: list[str], hops: int) -> tuple[list[dict], list[dict]]: ...


def get_store() -> GraphStore:
    backend = settings.GRAPH_BACKEND
    if backend == "networkx":
        return NetworkXStore(Path(settings.MEDIA_ROOT) / "kg" / "graph.json")
    if backend == "neo4j":
        return Neo4jStore(settings.NEO4J_URI, settings.NEO4J_USER, settings.NEO4J_PASSWORD)
    raise GraphStoreError(f"Unknown GRAPH_BACKEND {backend!r}; use neo4j or networkx")


def hop_distances(seed_ids: list[str], edges: list[dict], hops: int) -> dict[str, int]:
    """Undirected breadth-first distance from the seeds, up to `hops`."""
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge["source"], set()).add(edge["target"])
        adjacency.setdefault(edge["target"], set()).add(edge["source"])
    distance = {seed: 0 for seed in seed_ids}
    queue = deque(seed_ids)
    while queue:
        current = queue.popleft()
        if distance[current] >= hops:
            continue
        for neighbour in adjacency.get(current, ()):
            if neighbour not in distance:
                distance[neighbour] = distance[current] + 1
                queue.append(neighbour)
    return distance


def _within(seed_ids: list[str], nodes: list[dict], edges: list[dict], hops: int):
    distance = hop_distances(seed_ids, edges, hops)
    kept_edges = []
    for edge in edges:
        near = min(distance.get(edge["source"], 99), distance.get(edge["target"], 99))
        if near < hops and edge["source"] in distance and edge["target"] in distance:
            kept_edges.append({**edge, "hop": near + 1})
    kept_nodes = [{**node, "hop": distance[node["id"]]} for node in nodes if node["id"] in distance]
    return kept_nodes, kept_edges


class NetworkXStore:
    name = "networkx"

    def __init__(self, path: Path):
        self.path = path

    def replace(self, nodes, edges):
        import networkx as nx

        graph = nx.MultiDiGraph()
        for node in nodes:
            graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in edges:
            graph.add_edge(edge["source"], edge["target"], key=edge["relation"],
                           **{k: v for k, v in edge.items() if k not in {"source", "target"}})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(graph, edges="links")
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        _load_file.cache_clear()

    def _data(self) -> tuple[list[dict], list[dict]]:
        if not self.path.is_file():
            return [], []
        return _load_file(str(self.path), self.path.stat().st_mtime_ns)

    def nodes(self):
        return self._data()[0]

    def edges(self):
        return self._data()[1]

    def neighborhood(self, ids, hops):
        nodes, edges = self._data()
        return _within(ids, nodes, edges, hops)


@lru_cache(maxsize=2)
def _load_file(path: str, _mtime: int) -> tuple[list[dict], list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [{"id": row["id"], **{k: v for k, v in row.items() if k != "id"}} for row in data["nodes"]]
    edges = [{k: v for k, v in row.items() if k != "key"} for row in data["links"]]
    return nodes, edges


class Neo4jStore:
    name = "neo4j"

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.auth = (user, password)

    def _session(self, work):
        from neo4j.exceptions import AuthError, ServiceUnavailable

        try:
            with _driver(self.uri, self.auth).session() as session:
                return work(session)
        except (ServiceUnavailable, AuthError, OSError) as exc:
            raise GraphStoreError(
                f"Neo4j at {self.uri} is not usable ({exc.__class__.__name__}: {exc}). "
                "Start it with `docker compose up -d neo4j`, or set GRAPH_BACKEND=networkx."
            ) from exc

    def _run(self, query: str, **params) -> list[dict]:
        return self._session(lambda session: [record.data() for record in session.run(query, **params)])

    def replace(self, nodes, edges):
        """Swap the whole graph in one write transaction. The driver retries it on deadlocks."""
        self._run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        node_rows = [{**node, "provenance": json.dumps(node["provenance"])} for node in nodes]
        edge_batches = {
            relation: [{**edge, "provenance": json.dumps(edge["provenance"])}
                       for edge in edges if edge["relation"] == relation]
            for relation in RELATION_TYPES
        }

        def write(tx):
            tx.run("MATCH (e:Entity) DETACH DELETE e")
            tx.run(
                "UNWIND $rows AS row CREATE (e:Entity {id: row.id, name: row.name, type: row.type, "
                "aliases: row.aliases, provenance: row.provenance})",
                rows=node_rows,
            )
            for kind in ENTITY_TYPES:
                tx.run(f"MATCH (e:Entity {{type: $kind}}) SET e:`{kind}`", kind=kind)
            for relation, batch in edge_batches.items():
                if batch:
                    tx.run(
                        "UNWIND $rows AS row MATCH (a:Entity {id: row.source}), (b:Entity {id: row.target}) "
                        f"CREATE (a)-[:`{relation}` {{evidence: row.evidence, provenance: row.provenance}}]->(b)",
                        rows=batch,
                    )

        self._session(lambda session: session.execute_write(write))

    def nodes(self):
        rows = self._run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.type AS type, "
                         "e.aliases AS aliases, e.provenance AS provenance ORDER BY e.id")
        return [{**row, "provenance": json.loads(row["provenance"] or "[]")} for row in rows]

    def edges(self):
        rows = self._run("MATCH (a:Entity)-[r]->(b:Entity) RETURN a.id AS source, type(r) AS relation, "
                         "b.id AS target, r.evidence AS evidence, r.provenance AS provenance")
        return [{**row, "provenance": json.loads(row["provenance"] or "[]")} for row in rows]

    def neighborhood(self, ids, hops):
        hops = 2 if hops >= 2 else 1
        rows = self._run(
            "MATCH (s:Entity) WHERE s.id IN $ids "
            f"MATCH p = (s)-[*1..{hops}]-(:Entity) "
            "UNWIND relationships(p) AS r "
            "WITH DISTINCT r LIMIT 600 "
            "RETURN startNode(r).id AS source, type(r) AS relation, endNode(r).id AS target, "
            "r.evidence AS evidence, r.provenance AS provenance",
            ids=ids,
        )
        edges = [{**row, "provenance": json.loads(row["provenance"] or "[]")} for row in rows]
        wanted = sorted(set(ids) | {e["source"] for e in edges} | {e["target"] for e in edges})
        node_rows = self._run(
            "MATCH (e:Entity) WHERE e.id IN $ids RETURN e.id AS id, e.name AS name, e.type AS type, "
            "e.aliases AS aliases, e.provenance AS provenance",
            ids=wanted,
        )
        nodes = [{**row, "provenance": json.loads(row["provenance"] or "[]")} for row in node_rows]
        return _within(ids, nodes, edges, hops)


@lru_cache(maxsize=2)
def _driver(uri: str, auth: tuple[str, str]):
    from neo4j import GraphDatabase

    return GraphDatabase.driver(uri, auth=auth, connection_timeout=5)
