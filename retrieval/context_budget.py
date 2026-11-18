"""Context and token budgeting for the retrieval pipeline.

Enforces character and token limits on retrieved document chunks before passing them
downstream to LLM prompt generation, ensuring prompt window safety while preserving
complete document chunks and relative ranking order.
"""

from dataclasses import dataclass, field
import logging
from typing import Any

from retrieval.retrieval_schemas import SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ContextBudgetResult:
    """Outcome of context budgeting applied to a ranked set of document chunks.

    Attributes:
        selected_chunks: Complete SearchResult chunks that fit within the context budget.
        total_chars_used: Total characters used across all selected chunks.
        remaining_budget: Remaining character budget after selecting chunks.
        skipped_chunks_count: Count of candidate chunks skipped because they exceeded remaining budget.
        max_context_chars: The maximum character budget applied.
    """

    selected_chunks: list[SearchResult] = field(default_factory=list)
    total_chars_used: int = 0
    remaining_budget: int = 0
    skipped_chunks_count: int = 0
    max_context_chars: int = 6000

    @property
    def selected_count(self) -> int:
        """Return the number of selected chunks."""
        return len(self.selected_chunks)


class ContextBudgetManager:
    """Manages context and token budgeting for retrieval results.

    Applies character constraints to ranked candidate chunks:
    - Never splits chunks (preserves complete semantic units).
    - Preserves relative ranking order among selected chunks.
    - Accurately tracks characters used, remaining budget, and skipped chunks.
    """

    def __init__(self, default_max_chars: int = 6000) -> None:
        """Initialize ContextBudgetManager with default maximum character budget.

        Args:
            default_max_chars: Default maximum context characters allowed (default: 6000).

        Raises:
            ValueError: If default_max_chars is negative.
        """
        if not isinstance(default_max_chars, int) or isinstance(default_max_chars, bool) or default_max_chars < 0:
            raise ValueError(f"default_max_chars must be a non-negative integer, got {default_max_chars!r}")
        self.default_max_chars = default_max_chars

    def apply_budget(
        self,
        chunks: list[SearchResult],
        max_context_chars: int | None = None,
    ) -> ContextBudgetResult:
        """Select complete document chunks that fit within the character budget.

        Chunks are evaluated in their existing ranked order. A chunk is included
        if its character count (len(chunk.page_content)) is less than or equal
        to the remaining budget. If a chunk exceeds the remaining budget, it is skipped
        (not split), preserving whole chunks and relative ranking order.

        Args:
            chunks: Ranked candidate SearchResult chunks.
            max_context_chars: Optional override for maximum allowed characters.
                If None, uses self.default_max_chars.

        Returns:
            ContextBudgetResult containing selected chunks, total chars used,
            remaining budget, and skipped chunk count.

        Raises:
            ValueError: If max_context_chars is negative or invalid.
        """
        budget_limit = self.default_max_chars if max_context_chars is None else max_context_chars

        if not isinstance(budget_limit, int) or isinstance(budget_limit, bool) or budget_limit < 0:
            raise ValueError(f"max_context_chars must be a non-negative integer, got {budget_limit!r}")

        if not chunks:
            logger.debug("No candidate chunks provided to ContextBudgetManager. Returning empty result.")
            return ContextBudgetResult(
                selected_chunks=[],
                total_chars_used=0,
                remaining_budget=budget_limit,
                skipped_chunks_count=0,
                max_context_chars=budget_limit,
            )

        selected: list[SearchResult] = []
        total_used = 0
        remaining = budget_limit
        skipped_count = 0

        for chunk in chunks:
            chunk_len = len(chunk.page_content) if chunk.page_content else 0
            if chunk_len <= remaining:
                selected.append(chunk)
                total_used += chunk_len
                remaining -= chunk_len
                logger.debug(
                    "Selected chunk '%s' (len: %d). Total used: %d, Remaining: %d",
                    chunk.chunk_id or chunk.source_name,
                    chunk_len,
                    total_used,
                    remaining,
                )
            else:
                skipped_count += 1
                logger.debug(
                    "Skipping chunk '%s' (len: %d) because it exceeds remaining budget (%d)",
                    chunk.chunk_id or chunk.source_name,
                    chunk_len,
                    remaining,
                )

        logger.info(
            "Context budgeting applied: %d/%d chunks selected (%d chars used, %d remaining, %d skipped, limit: %d)",
            len(selected),
            len(chunks),
            total_used,
            remaining,
            skipped_count,
            budget_limit,
        )

        return ContextBudgetResult(
            selected_chunks=selected,
            total_chars_used=total_used,
            remaining_budget=remaining,
            skipped_chunks_count=skipped_count,
            max_context_chars=budget_limit,
        )
