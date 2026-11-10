"""Rebuild the graph store from cached KGExtraction rows."""
import logging
from contextlib import contextmanager

from django.conf import settings

from apps.ingest.models import Chunk
from apps.kg.models import GraphBuild, KGExtraction
from apps.kg.services.extract import extract_chunks
from apps.kg.services.resolve import resolve
from core.graph_store import get_store

logger = logging.getLogger(__name__)


@contextmanager
def _build_lock():
    """One rebuild at a time across web, worker and commands, so the last writer saw every extraction."""
    if settings.CELERY_TASK_ALWAYS_EAGER:
        yield
        return
    import redis

    try:
        lock = redis.Redis.from_url(settings.CELERY_BROKER_URL).lock("docintel:kg-build", timeout=600, blocking_timeout=300)
        acquired = lock.acquire()
    except redis.RedisError as exc:
        logger.warning("kg build lock unavailable, building without it: %s", exc)
        yield
        return
    try:
        yield
    finally:
        if acquired:
            lock.release()


def build_graph(extract: bool = False, force: bool = False, version: str | None = None) -> GraphBuild:
    """Optionally extract missing chunks first, then resolve and write the whole graph."""
    version = version or settings.KG_PROMPT_VERSION
    if extract:
        stats = extract_chunks(Chunk.objects.order_by("id"), version=version, force=force)
        logger.info("kg extraction %s", stats)
    with _build_lock():
        extractions = KGExtraction.objects.select_related("chunk__document").order_by("chunk_id")
        graph = resolve(extractions)
        store = get_store()
        store.replace(graph.nodes, graph.edges)
        build = GraphBuild.objects.create(
            backend=store.name,
            prompt_version=version,
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            merged_count=graph.merged_count,
            dropped_count=graph.dropped_count,
            chunk_count=graph.chunk_count,
            merge_log=graph.merge_log,
        )
    _clear_entity_cache()
    logger.info("kg build %s: %s nodes, %s edges, %s merged, %s dropped",
                store.name, build.node_count, build.edge_count, build.merged_count, build.dropped_count)
    return build


def _clear_entity_cache() -> None:
    from apps.kg.services.graphrag import clear_entity_cache

    clear_entity_cache()
