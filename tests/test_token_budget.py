"""Tests for the token/character budgeting component of the retrieval pipeline.

These tests validate the behaviour of :class:`ContextBudgetManager` and ensure that the
`ResultRankingPipeline` correctly incorporates budgeting without breaking the existing
`RankedContext` API.
"""

import pytest

from retrieval.context_budget import ContextBudgetManager, ContextBudgetResult
from retrieval.retrieval_schemas import SearchResult
from retrieval.ranking_pipeline import ResultRankingPipeline, RankingOptions, RankedContext

# Helper to create a simple SearchResult with given content length
def make_result(content: str, chunk_id: str = "") -> SearchResult:
    # Minimal Document stub – we can use a simple dict mimicking langchain Document
    class DummyDoc:
        def __init__(self, text: str):
            self.page_content = text
            self.metadata = {"source_name": f"source_{chunk_id}", "category": "general", "access_level": "public"}
    return SearchResult(document=DummyDoc(content), score=0.0, chunk_id=chunk_id)


def test_budget_under_limit_includes_all_chunks():
    chunks = [make_result("a" * 100, "c1"), make_result("b" * 200, "c2"), make_result("c" * 150, "c3")]
    manager = ContextBudgetManager(default_max_chars=1000)
    res: ContextBudgetResult = manager.apply_budget(chunks)
    assert res.selected_chunks == chunks
    assert res.total_chars_used == 450
    assert res.remaining_budget == 550
    assert res.skipped_chunks_count == 0


def test_budget_exact_limit_selects_all_and_exhausts_budget():
    chunks = [make_result("x" * 300, "c1"), make_result("y" * 700, "c2")]
    manager = ContextBudgetManager(default_max_chars=1000)
    res = manager.apply_budget(chunks)
    assert res.selected_chunks == chunks
    assert res.total_chars_used == 1000
    assert res.remaining_budget == 0
    assert res.skipped_chunks_count == 0


def test_budget_skips_chunk_exceeding_remaining():
    chunks = [make_result("a" * 400, "c1"), make_result("b" * 700, "c2"), make_result("c" * 100, "c3")]
    manager = ContextBudgetManager(default_max_chars=800)
    res = manager.apply_budget(chunks)
    # First chunk fits (400), remaining 400, second chunk (700) exceeds -> skipped, third fits
    assert [c.chunk_id for c in res.selected_chunks] == ["c1", "c3"]
    assert res.total_chars_used == 500
    assert res.remaining_budget == 300
    assert res.skipped_chunks_count == 1


def test_budget_empty_input():
    manager = ContextBudgetManager(default_max_chars=600)
    res = manager.apply_budget([])
    assert res.selected_chunks == []
    assert res.total_chars_used == 0
    assert res.remaining_budget == 600
    assert res.skipped_chunks_count == 0


def test_budget_configurable_limit():
    chunks = [make_result("x" * 250), make_result("y" * 250)]
    manager = ContextBudgetManager(default_max_chars=500)
    # Override with a smaller limit
    res = manager.apply_budget(chunks, max_context_chars=400)
    assert len(res.selected_chunks) == 1
    assert res.total_chars_used == 250
    assert res.remaining_budget == 150


def test_budget_order_preserved():
    chunks = [make_result("a" * 100, "c1"), make_result("b" * 100, "c2"), make_result("c" * 100, "c3")]
    manager = ContextBudgetManager(default_max_chars=250)
    res = manager.apply_budget(chunks)
    # Only first two should be selected, preserving order
    assert [c.chunk_id for c in res.selected_chunks] == ["c1", "c2"]


def test_ranking_pipeline_uses_budget_and_preserves_api():
    # Build a pipeline with a small max_context_chars option
    options = RankingOptions(top_k=10, max_context_chars=300)
    pipeline = ResultRankingPipeline()
    # Create three chunks of 150 chars each – only first two fit into 300
    chunks = [make_result("x" * 150, "c1"), make_result("y" * 150, "c2"), make_result("z" * 150, "c3")]
    ranked: RankedContext = pipeline.process(query="test", results=chunks, options=options)
    # Verify budgeting fields are populated
    assert ranked.total_chars_used == 300
    assert ranked.remaining_budget == 0
    assert ranked.skipped_chunks_count == 1
    assert ranked.max_context_chars == 300
    # Ensure the original fields still exist and are correct
    assert ranked.total_candidates == 3
    assert ranked.retained_count == 2
    assert len(ranked.results) == 2
    assert [c.chunk_id for c in ranked.results] == ["c1", "c2"]
