"""Result filtering, exact deduplication, and near-duplicate overlap suppression.

Removes redundant or overlapping chunks, applies metadata criteria (category,
access level, source diversity), and enforces score cutoffs.
"""

from collections import Counter
import hashlib
import logging
import re
from typing import Any

from retrieval.retrieval_schemas import SearchResult

logger = logging.getLogger(__name__)


def _compute_token_set(text: str) -> set[str]:
    """Tokenize text into a set of lowercased alphanumeric words."""
    words = re.findall(r"\w+", text.lower())
    return set(words)


def _compute_jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Calculate Jaccard similarity between two token sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class ResultFilter:
    """Filters, deduplicates, and prunes retrieved search results."""

    def deduplicate_exact(self, results: list[SearchResult]) -> list[SearchResult]:
        """Eliminate duplicate chunks based on content hash, chunk ID, or exact text.

        Preserves the occurrence with the highest similarity score.

        Args:
            results: List of SearchResult candidates.

        Returns:
            Deduplicated list of SearchResults.
        """
        if not results:
            return []

        # Sort by score descending to prioritize keeping the highest scoring duplicate
        sorted_results = sorted(results, key=lambda r: float(r.score), reverse=True)

        unique_results: list[SearchResult] = []
        seen_hashes: set[str] = set()
        seen_chunk_ids: set[str] = set()
        seen_texts: set[str] = set()

        for r in sorted_results:
            meta = r.metadata or {}
            content_hash = meta.get("content_hash")
            chunk_id = r.chunk_id or meta.get("chunk_id")
            clean_text = r.page_content.strip()

            if not content_hash and clean_text:
                content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

            # Check exact duplicates
            if content_hash and content_hash in seen_hashes:
                logger.debug("Suppressing duplicate chunk with content_hash: %s", content_hash[:8])
                continue

            if chunk_id and chunk_id in seen_chunk_ids:
                logger.debug("Suppressing duplicate chunk with chunk_id: %s", chunk_id)
                continue

            if clean_text in seen_texts:
                logger.debug("Suppressing duplicate chunk with identical text")
                continue

            if content_hash:
                seen_hashes.add(content_hash)
            if chunk_id:
                seen_chunk_ids.add(chunk_id)
            if clean_text:
                seen_texts.add(clean_text)

            unique_results.append(r)

        return unique_results

    def deduplicate_overlapping(
        self,
        results: list[SearchResult],
        overlap_threshold: float = 0.85,
    ) -> list[SearchResult]:
        """Suppress near-duplicate chunks sharing high token overlap from the same source.

        Args:
            results: List of SearchResult candidates (assumed sorted by score descending).
            overlap_threshold: Jaccard token overlap threshold (0.0 to 1.0) above which
                lower-scoring chunks from the same source are dropped.

        Returns:
            Pruned list of SearchResults with near-duplicates removed.
        """
        if not results or overlap_threshold >= 1.0:
            return results

        pruned_results: list[SearchResult] = []
        # Group retained token sets by source: source_name -> list of (token_set, SearchResult)
        retained_by_source: dict[str, list[tuple[set[str], SearchResult]]] = {}

        for candidate in results:
            source = candidate.source_name
            cand_tokens = _compute_token_set(candidate.page_content)

            is_near_duplicate = False
            if source in retained_by_source:
                for existing_tokens, _ in retained_by_source[source]:
                    jaccard = _compute_jaccard_similarity(cand_tokens, existing_tokens)
                    if jaccard >= overlap_threshold:
                        logger.debug(
                            "Suppressing near-duplicate chunk %s (overlap: %.2f with retained chunk)",
                            candidate.chunk_id,
                            jaccard,
                        )
                        is_near_duplicate = True
                        break

            if not is_near_duplicate:
                pruned_results.append(candidate)
                if source not in retained_by_source:
                    retained_by_source[source] = []
                retained_by_source[source].append((cand_tokens, candidate))

        return pruned_results

    def filter_by_metadata(
        self,
        results: list[SearchResult],
        min_score: float = 0.0,
        category: str | None = None,
        allowed_access_levels: list[str] | tuple[str, ...] | None = None,
        allowed_sources: list[str] | None = None,
        excluded_sources: list[str] | None = None,
        max_per_source: int | None = None,
    ) -> list[SearchResult]:
        """Apply post-search metadata filtering and source diversity constraints.

        Args:
            results: List of SearchResult candidates.
            min_score: Minimum allowable similarity score threshold.
            category: Optional category filter requirement.
            allowed_access_levels: Optional list of permitted access levels.
            allowed_sources: Optional whitelist of source names.
            excluded_sources: Optional blacklist of source names.
            max_per_source: Maximum chunks allowed from any single source document.

        Returns:
            Filtered list of SearchResult objects.
        """
        if not results:
            return []

        filtered: list[SearchResult] = []
        source_counts: Counter[str] = Counter()

        allowed_levels_set = (
            {str(lvl).strip().lower() for lvl in allowed_access_levels}
            if allowed_access_levels
            else None
        )
        allowed_src_set = {str(s).strip() for s in allowed_sources} if allowed_sources else None
        excluded_src_set = {str(s).strip() for s in excluded_sources} if excluded_sources else None
        target_category = str(category).strip().lower() if category else None

        for r in results:
            # 1. Score threshold check
            if float(r.score) < min_score:
                logger.debug("Filtering out %s: score %.4f < min_score %.4f", r.chunk_id, r.score, min_score)
                continue

            # 2. Category check
            if target_category and target_category not in ("all", "none"):
                if r.category.lower() != target_category:
                    continue

            # 3. Access level check
            if allowed_levels_set:
                if r.access_level.lower() not in allowed_levels_set:
                    continue

            # 4. Allowed sources check
            if allowed_src_set is not None:
                if r.source_name not in allowed_src_set:
                    continue

            # 5. Excluded sources check
            if excluded_src_set is not None:
                if r.source_name in excluded_src_set:
                    continue

            # 6. Source diversity limit check
            if max_per_source is not None and max_per_source > 0:
                if source_counts[r.source_name] >= max_per_source:
                    logger.debug("Source diversity cap reached for %s", r.source_name)
                    continue

            source_counts[r.source_name] += 1
            filtered.append(r)

        return filtered
