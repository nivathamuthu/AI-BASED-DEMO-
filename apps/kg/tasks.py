"""Extract a finished document's chunks and rebuild the graph, off the request path."""
import logging

from celery import shared_task

from apps.ingest.models import Chunk
from apps.kg.services.build import build_graph
from apps.kg.services.extract import extract_chunks
from core.graph_store import GraphStoreError

logger = logging.getLogger(__name__)


@shared_task
def task_kg_document(document_id: int) -> dict:
    stats = extract_chunks(Chunk.objects.filter(document_id=document_id).order_by("ordinal"))
    try:
        build = build_graph()
    except GraphStoreError as exc:
        logger.warning("kg build skipped for document %s: %s", document_id, exc)
        return {**stats, "build": "skipped"}
    return {**stats, "nodes": build.node_count, "edges": build.edge_count}
