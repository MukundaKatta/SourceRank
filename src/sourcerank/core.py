"""Core ranking engine for SourceRank."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from sourcerank.config import RankerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_CITATION_RE = re.compile(r"\([A-Z][a-z]+ (?:et al\.,? )?\d{4}\)|\[\d+\]")
_HEDGING = {"maybe", "perhaps", "possibly", "might", "could", "guess", "seems", "sort of", "kind of", "probably"}
_FACTUAL = re.compile(r"\d+%|\d{4}|study|data|found|significant|evidence|research|published")

# Authority TLD scores
_DEFAULT_AUTHORITY: Dict[str, float] = {
    ".gov": 0.95, ".edu": 0.90, ".org": 0.75, ".ac.uk": 0.88,
    "nature.com": 0.92, "science.org": 0.90, "pubmed": 0.88,
    "arxiv.org": 0.85, "ieee.org": 0.85, "springer.com": 0.80,
}
_LOW_AUTHORITY = {"blogspot", "wordpress", "medium.com", ".xyz", ".tk", ".buzz"}


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    value = str(value).strip()
    if _ISO_RE.match(value):
        try:
            dt = datetime.fromisoformat(value)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            pass
    return None


def _days_since(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    return max(0, (datetime.now(timezone.utc) - dt).days)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Source:
    """A single source document to be ranked."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: int = 0

# Keep backward compat
SourceDocument = Source


@dataclass
class QualitySignals:
    """Individual quality signal scores."""
    recency: float = 0.0
    authority: float = 0.0
    relevance: float = 0.0
    completeness: float = 0.0
    citation_density: float = 0.0
    factual_language: float = 0.0


@dataclass
class RankingResult:
    """A ranked source with scores."""
    source: Source
    composite_score: float
    signals: QualitySignals
    rank: int = 0


@dataclass
class ComparisonResult:
    """Result of comparing two sources."""
    winner: str  # "a" or "b"
    advantage: float
    a_score: float
    b_score: float


# ---------------------------------------------------------------------------
# SourceRanker
# ---------------------------------------------------------------------------

class SourceRanker:
    """Rank and score source documents by quality."""

    def __init__(self, config: Optional[RankerConfig] = None) -> None:
        self.config = config or RankerConfig()
        self._sources: List[Source] = []
        self._next_id = 1

    def add_source(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Source:
        """Add a source document and return it."""
        src = Source(text=text, metadata=metadata or {}, id=self._next_id)
        self._next_id += 1
        self._sources.append(src)
        return src

    # -- Individual signal scorers ------------------------------------------

    def score_recency(self, source: Source) -> float:
        """Score recency based on date in metadata."""
        date_str = source.metadata.get("date")
        dt = _parse_date(date_str)
        age = _days_since(dt)
        if age is None:
            return 0.3
        max_age = 3650  # 10 years
        if age >= max_age:
            return 0.0
        return round(1.0 - (age / max_age), 4)

    def score_authority(self, source: Source) -> float:
        """Score authority based on domain in metadata."""
        domain = source.metadata.get("domain", "").lower()
        if not domain:
            return 0.5
        for key, score in _DEFAULT_AUTHORITY.items():
            if key in domain:
                return score
        for low in _LOW_AUTHORITY:
            if low in domain:
                return 0.2
        return 0.5

    def score_citation_density(self, source: Source) -> float:
        """Score based on number of citations/references in text."""
        citations = _CITATION_RE.findall(source.text)
        count = len(citations)
        if count == 0:
            return 0.0
        return min(1.0, count * 0.25)

    def score_factual_language(self, source: Source) -> float:
        """Score based on factual vs hedging language."""
        text_lower = source.text.lower()
        words = _WORD_RE.findall(text_lower)
        total = max(len(words), 1)

        # Count factual indicators
        factual_hits = len(_FACTUAL.findall(text_lower))
        # Count hedging indicators
        hedge_hits = sum(1 for w in words if w in _HEDGING)

        factual_ratio = factual_hits / total
        hedge_ratio = hedge_hits / total

        score = 0.5 + factual_ratio * 5.0 - hedge_ratio * 5.0
        return round(max(0.0, min(1.0, score)), 4)

    def score_completeness(self, source: Source) -> float:
        """Score based on text length."""
        length = len(source.text)
        if length < 50:
            return 0.1
        if length >= 500:
            return 1.0
        return round(length / 500, 4)

    def _compute_composite(self, source: Source) -> tuple[float, QualitySignals]:
        """Compute all signals and composite score."""
        signals = QualitySignals(
            recency=self.score_recency(source),
            authority=self.score_authority(source),
            citation_density=self.score_citation_density(source),
            factual_language=self.score_factual_language(source),
            completeness=self.score_completeness(source),
        )
        w = self.config.weights
        composite = (
            w.recency * signals.recency
            + w.authority * signals.authority
            + w.completeness * signals.completeness
            + w.citation_density * signals.citation_density
            + w.factual_language * signals.factual_language
        )
        return round(composite, 6), signals

    # -- Ranking API --------------------------------------------------------

    def rank_sources(self) -> List[RankingResult]:
        """Rank all added sources. Returns list sorted by composite score."""
        if not self._sources:
            return []
        results = []
        for src in self._sources:
            score, signals = self._compute_composite(src)
            results.append(RankingResult(source=src, composite_score=score, signals=signals))
        results.sort(key=lambda r: r.composite_score, reverse=True)
        for i, r in enumerate(results):
            r.rank = i + 1
        return results

    def get_top_sources(self, k: int) -> List[RankingResult]:
        """Return top-k ranked sources."""
        return self.rank_sources()[:k]

    def compare_sources(self, a: Source, b: Source) -> ComparisonResult:
        """Compare two sources directly."""
        a_score, _ = self._compute_composite(a)
        b_score, _ = self._compute_composite(b)
        winner = "a" if a_score >= b_score else "b"
        return ComparisonResult(
            winner=winner,
            advantage=round(abs(a_score - b_score), 6),
            a_score=a_score,
            b_score=b_score,
        )

    def generate_report(self, rankings: List[RankingResult]) -> str:
        """Generate a text report of rankings."""
        lines = ["Source Ranking Report", "=" * 40, f"Total sources evaluated: {len(rankings)}", ""]
        for r in rankings:
            preview = r.source.text[:60] + "..." if len(r.source.text) > 60 else r.source.text
            lines.append(f"#{r.rank} (score: {r.composite_score:.4f})")
            lines.append(f"  Text: {preview}")
            lines.append(f"  Recency: {r.signals.recency:.2f} | Authority: {r.signals.authority:.2f}")
            lines.append(f"  Citations: {r.signals.citation_density:.2f} | Factual: {r.signals.factual_language:.2f}")
            lines.append("")
        return "\n".join(lines)
