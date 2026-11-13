"""Rule-based document category classification.

Classifies chunks and sources into predefined categories using keyword pattern matching
with word boundaries and strict priority ordering.
"""

import re
from urllib.parse import urlparse

CATEGORY_RULES: dict[str, list[str]] = {
    "policy": [
        "policy",
        "rules",
        "compliance",
        "terms",
        "code of conduct",
        "guidelines",
    ],
    "faq": [
        "faq",
        "frequently asked",
        "q&a",
        "question",
    ],
    "guide": [
        "guide",
        "how to",
        "tutorial",
        "manual",
        "handbook",
        "steps",
    ],
    "internal": [
        "internal",
        "confidential",
        "team",
        "meeting",
        "memo",
    ],
    "web_page": [],
    "general": [],
}


def _count_keyword_matches(text: str) -> dict[str, int]:
    """Count keyword occurrences for each category in CATEGORY_RULES using word boundaries.

    Args:
        text: Target text string to inspect.

    Returns:
        Dictionary mapping category names to total hit counts.
    """
    if not text:
        return {cat: 0 for cat in CATEGORY_RULES}

    # Normalize underscores and slashes to spaces so filenames like policy_document.pdf
    # or hr/guidelines are properly tokenized by word boundaries.
    normalized = re.sub(r"[_\-/\\]", " ", text.lower())
    counts: dict[str, int] = {}

    for cat, keywords in CATEGORY_RULES.items():
        hit_count = 0
        for kw in keywords:
            pattern = rf"\b{re.escape(kw.lower())}\b"
            hit_count += len(re.findall(pattern, normalized))
        counts[cat] = hit_count

    return counts


def detect_category(text: str, source: str = "", source_type: str = "") -> str:
    """Detect document category following strict precedence rules.

    Precedence:
    1. Keyword match on source name/filename first (highest count wins, ties follow CATEGORY_RULES order).
    2. Keyword match on first 2000 characters of content (highest count wins, ties follow CATEGORY_RULES order).
    3. 'web_page' if source_type == 'web'.
    4. 'general' otherwise.

    Args:
        text: Document content text.
        source: Source identifier, filename, or URL.
        source_type: Ingestion source type (e.g., 'web', 'pdf', 'docx').

    Returns:
        Detected category string name.
    """
    # 1. Check source name / filename
    if source:
        src_str = str(source)
        parsed = urlparse(src_str)
        probe_source = parsed.path if parsed.scheme in ("http", "https") else src_str

        source_counts = _count_keyword_matches(probe_source)
        best_source_cat: str | None = None
        best_source_hits = 0
        for cat in CATEGORY_RULES:
            hits = source_counts.get(cat, 0)
            if hits > best_source_hits:
                best_source_hits = hits
                best_source_cat = cat

        if best_source_hits > 0 and best_source_cat:
            return best_source_cat

    # 2. Check first 2000 characters of content
    content_prefix = text[:2000] if text else ""
    if content_prefix:
        content_counts = _count_keyword_matches(content_prefix)
        best_content_cat: str | None = None
        best_content_hits = 0
        for cat in CATEGORY_RULES:
            hits = content_counts.get(cat, 0)
            if hits > best_content_hits:
                best_content_hits = hits
                best_content_cat = cat

        if best_content_hits > 0 and best_content_cat:
            return best_content_cat

    # 3. Web source type fallback
    if source_type and source_type.lower() == "web":
        return "web_page"

    # 4. Default fallback
    return "general"
