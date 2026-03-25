"""Utility functions for URL parsing, text similarity, and date handling."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def extract_domain(url: str) -> str:
    """Extract the domain from a URL string."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        return domain.lower().strip()
    except Exception:
        return url.lower().strip()


def get_tld(domain: str) -> str:
    """Return the top-level domain suffix including the dot (e.g. '.edu')."""
    parts = domain.split(".")
    if len(parts) >= 3 and len(parts[-2]) <= 3:
        return "." + ".".join(parts[-2:])
    if len(parts) >= 2:
        return "." + parts[-1]
    return ""


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenization."""
    return _WORD_RE.findall(text.lower())


def term_frequency(tokens: list[str]) -> dict[str, float]:
    """Compute normalized term-frequency vector."""
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {t: c / total for t, c in counts.items()}


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as dicts."""
    if not vec_a or not vec_b:
        return 0.0
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tfidf_similarity(query: str, document: str) -> float:
    """Simple TF-based cosine similarity between a query and a document."""
    q_tf = term_frequency(tokenize(query))
    d_tf = term_frequency(tokenize(document))
    return cosine_similarity(q_tf, d_tf)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    set_a = set(tokenize(text_a))
    set_b = set(tokenize(text_b))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def parse_date(value: str | datetime | None) -> datetime | None:
    """Best-effort date parser. Accepts ISO-8601 strings or datetime objects."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    value = value.strip()
    if _ISO_RE.match(value):
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return None


def days_since(dt: datetime | None) -> int | None:
    """Return integer days between *dt* and now (UTC)."""
    if dt is None:
        return None
    now = datetime.now(timezone.utc)
    return max(0, (now - dt).days)
