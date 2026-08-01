"""Submission must not issue work proportional to batch size in queries.

A 500-report batch is allowed by the API contract. If the per-report cost is a query
(or worse, a network round trip), the endpoint stops being a fast 202.
"""

import pytest
from sqlalchemy import event

from app.core.db import engine

pytestmark = pytest.mark.integration


class _Counter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *_args, **_kwargs) -> None:
        self.count += 1


@pytest.fixture
def count_queries():
    counter = _Counter()
    event.listen(engine, "before_cursor_execute", counter)
    yield counter
    event.remove(engine, "before_cursor_execute", counter)


def test_query_count_does_not_scale_with_batch_size(api, make_document, count_queries):
    small = [make_document() for _ in range(2)]
    baseline_start = count_queries.count
    api.post(
        "/v1/document-processing-runs",
        json={"documents": [{"document_id": d} for d in small]},
    )
    small_cost = count_queries.count - baseline_start

    large = [make_document() for _ in range(12)]
    large_start = count_queries.count
    api.post(
        "/v1/document-processing-runs",
        json={"documents": [{"document_id": d} for d in large]},
    )
    large_cost = count_queries.count - large_start

    # Batched lookups and a single multi-row INSERT make submission cost the same
    # number of statements whatever the batch size. A per-report SELECT, or a
    # per-item flush/savepoint, would show up here immediately.
    assert large_cost <= small_cost, (
        f"statement count scaled with batch size: {small_cost} for 2 reports, "
        f"{large_cost} for 12 — a per-report round trip has crept back in"
    )
