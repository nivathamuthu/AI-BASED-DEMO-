"""Reciprocal rank fusion. k=60 is the value from Cormack, Clarke and Buettcher."""


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked id lists. Earlier ranks contribute more. Ties keep the first-seen order."""
    scores: dict[int, float] = {}
    order: list[int] = []
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            if item_id not in scores:
                order.append(item_id)
                scores[item_id] = 0.0
            scores[item_id] += 1.0 / (k + rank)
    order.sort(key=lambda item_id: scores[item_id], reverse=True)
    return [(item_id, scores[item_id]) for item_id in order]
