"""Core ranking engine for SourceRank.

Provides the ``SourceRank`` class with methods to add, score, rank,
compare, and report on text sources across four quality dimensions:
recency, authority, citation density, and factual language.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Text analysis helpers (inlined for reliability)
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
    "%B %d, %Y", "%b %d, %Y",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y",
]

_AUTHORITY_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bDr\.\s", r"\bProf\.\s", r"\bProfessor\s", r"\bPhD\b",
        r"\bMD\b", r"\bM\.D\.\b",
        r"\buniversity\b", r"\binstitute\b", r"\blaboratory\b",
        r"\bhospital\b", r"\bjournal\b", r"\bpublished\s+in\b",
        r"\bpeer[\s-]reviewed\b", r"\bgovernment\b", r"\bministry\b",
        r"\bnational\b", r"\bfederal\b", r"\bworld\s+health\b",
    ]
]

_AUTHORITY_DOMAINS = [
    "nature.com", "science.org", "thelancet.com", "nejm.org",
    "ieee.org", "acm.org", "arxiv.org", "gov", "edu",
    "who.int", "nih.gov", "cdc.gov",
]

_CITATION_PATTERNS = [
    re.compile(r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&\s+[A-Z][a-z]+))?,?\s*\d{4}\)"),
    re.compile(r"\[\d+\]"),
    re.compile(r"\b(?:doi|DOI)\s*:\s*10\.\d{4,}"),
    re.compile(r"https?://(?:dx\.)?doi\.org/10\.\d{4,}"),
    re.compile(r"\(\d{4}\)"),
    re.compile(r"\b(?:ibid|op\.\s*cit|loc\.\s*cit)\b", re.IGNORECASE),
]

_HEDGING_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b", r"\bmight\b",
        r"\bcould\b", r"\bseems?\s+(?:to|like)\b", r"\bappears?\s+to\b",
        r"\bsort\s+of\b", r"\bkind\s+of\b", r"\bwho\s+knows\b",
        r"\bi\s+think\b", r"\bi\s+believe\b", r"\bi\s+guess\b",
        r"\bprobably\b", r"\ballegedly\b", r"\bsupposedly\b",
        r"\brumo(?:u)?rs?\b",
    ]
]

_FACTUAL_INDICATORS = [
    re.compile(r"\d+(?:\.\d+)?%"),
    re.compile(r"\b\d{4}\b"),
    re.compile(r"\b(?:according\s+to|based\s+on)\b", re.I),
    re.compile(r"\b(?:study|research|data|evidence|analysis)\b", re.I),
    re.compile(r"\b(?:found\s+that|showed\s+that|demonstrated)\b", re.I),
    re.compile(r"\b(?:statistic(?:al|s)?|significant(?:ly)?)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b", re.I),
]

CITATION_SATURATION = 10
RECENCY_HALF_LIFE_DAYS = 365


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    value = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _days_since(d: date) -> int:
    return max((date.today() - d).days, 0)


def _recency_decay(days: int, half_life: int) -> float:
    if days <= 0:
        return 1.0
    return math.exp(-0.693147 * days / half_life)


def _count_authority_signals(text: str) -> int:
    return sum(1 for pat in _AUTHORITY_PATTERNS if pat.search(text))


def _domain_is_authoritative(domain: str | None) -> bool:
    if not domain:
        return False
    d = domain.lower().strip()
    return any(d.endswith(t) for t in _AUTHORITY_DOMAINS)


def _count_citations(text: str) -> int:
    return sum(len(pat.findall(text)) for pat in _CITATION_PATTERNS)


def _factual_language_score(text: str) -> float:
    words = len(text.split())
    if words == 0:
        return 0.0
    factual = sum(len(pat.findall(text)) for pat in _FACTUAL_INDICATORS)
    hedging = sum(len(pat.findall(text)) for pat in _HEDGING_COMPILED)
    net = (factual - hedging) / (words / 100)
    clamped = max(-5.0, min(net, 5.0))
    return round((clamped + 5.0) / 10.0, 4)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """Internal representation of a source document."""

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    composite_score: float = 0.0


@dataclass
class RankedSource:
    """A source with its composite score and per-signal breakdown."""

    rank: int
    source: Source
    composite_score: float
    scores: dict[str, float]


@dataclass
class ComparisonResult:
    """Side-by-side comparison of two sources."""

    source_a: Source
    source_b: Source
    scores_a: dict[str, float]
    scores_b: dict[str, float]
    composite_a: float
    composite_b: float
    winner: str  # "a", "b", or "tie"
    advantage: float  # absolute difference


# ---------------------------------------------------------------------------
# Configuration (simplified for the four-signal API)
# ---------------------------------------------------------------------------


class SourceRankConfig(BaseModel):
    """Scoring weights and thresholds for the SourceRank engine.

    The four signal weights must sum to 1.0.
    """

    recency_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    authority_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    citation_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    factual_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    recency_half_life_days: int = Field(default=365, gt=0)
    citation_saturation: int = Field(default=10, gt=0)


# ---------------------------------------------------------------------------
# SourceRank
# ---------------------------------------------------------------------------


class SourceRank:
    """Evaluate and rank text sources by quality and reliability.

    Scores each source from 0 to 1 on four dimensions -- recency, authority,
    citation density, and factual language -- then produces a weighted
    composite score used for ranking.

    Parameters
    ----------
    config : SourceRankConfig | None
        Custom weights and thresholds.  Uses balanced defaults when *None*.

    Examples
    --------
    >>> ranker = SourceRank()
    >>> ranker.add_source("Some text", {"date": "2025-06-01"})
    >>> rankings = ranker.rank_sources()
    """

    def __init__(self, config: SourceRankConfig | None = None) -> None:
        self.config = config or SourceRankConfig()
        self._sources: list[Source] = []

    # -- adding sources -----------------------------------------------------

    def add_source(self, text: str, metadata: dict[str, Any] | None = None) -> Source:
        """Add a source for evaluation.

        Parameters
        ----------
        text : str
            The full text content of the source.
        metadata : dict, optional
            Arbitrary metadata.  Recognised keys include ``date``,
            ``author``, and ``domain``.

        Returns
        -------
        Source
            The newly created source object.
        """
        source = Source(text=text, metadata=metadata or {})
        self._sources.append(source)
        return source

    @property
    def sources(self) -> list[Source]:
        """Return a copy of the current source list."""
        return list(self._sources)

    # -- individual signal scorers ------------------------------------------

    def score_recency(self, source: Source) -> float:
        """Score a source 0-1 based on its publication date freshness.

        Uses exponential decay with the configured half-life.  Sources
        without a ``date`` in their metadata receive a default score of 0.3.
        """
        raw_date = source.metadata.get("date")
        parsed = _parse_date(raw_date)
        if parsed is None:
            return 0.3
        age_days = _days_since(parsed)
        return round(_recency_decay(age_days, self.config.recency_half_life_days), 4)

    def score_authority(self, source: Source) -> float:
        """Score a source 0-1 based on authority signals.

        Combines two sub-signals:
        - **Domain trust** (50%): whether the metadata ``domain`` matches a
          known authoritative suffix.
        - **Textual authority markers** (50%): titles (Dr., Prof.), institution
          names, and phrases like "peer-reviewed" found in the text.
        """
        domain = source.metadata.get("domain")
        domain_score = 1.0 if _domain_is_authoritative(domain) else 0.0

        signals = _count_authority_signals(source.text)
        author = source.metadata.get("author", "")
        signals += _count_authority_signals(author)
        text_score = min(signals / 3.0, 1.0)

        return round(0.5 * domain_score + 0.5 * text_score, 4)

    def score_citation_density(self, source: Source) -> float:
        """Score a source 0-1 based on in-text citation density.

        Counts citation patterns (APA-style, numeric brackets, DOI links)
        and maps the count to [0, 1] with configurable saturation.
        """
        count = _count_citations(source.text)
        return round(min(count / self.config.citation_saturation, 1.0), 4)

    def score_factual_language(self, source: Source) -> float:
        """Score a source 0-1 based on factual vs. hedging language.

        Higher scores indicate more data-driven, precise language and fewer
        hedging phrases like "maybe", "sort of", "who knows".
        """
        return _factual_language_score(source.text)

    # -- composite scoring --------------------------------------------------

    def _compute_scores(self, source: Source) -> dict[str, float]:
        """Compute all four signal scores and the weighted composite."""
        scores = {
            "recency": self.score_recency(source),
            "authority": self.score_authority(source),
            "citation_density": self.score_citation_density(source),
            "factual_language": self.score_factual_language(source),
        }
        composite = (
            self.config.recency_weight * scores["recency"]
            + self.config.authority_weight * scores["authority"]
            + self.config.citation_weight * scores["citation_density"]
            + self.config.factual_weight * scores["factual_language"]
        )
        scores["composite"] = round(composite, 4)
        source.scores = scores
        source.composite_score = scores["composite"]
        return scores

    # -- ranking ------------------------------------------------------------

    def rank_sources(self, sources: list[Source] | None = None) -> list[RankedSource]:
        """Rank all added sources (or a custom list) by composite score.

        Parameters
        ----------
        sources : list[Source] | None
            Sources to rank.  Defaults to all sources added via
            ``add_source``.

        Returns
        -------
        list[RankedSource]
            Sources ordered best-first with rank numbers starting at 1.
        """
        target = sources if sources is not None else self._sources
        for src in target:
            self._compute_scores(src)

        sorted_sources = sorted(target, key=lambda s: s.composite_score, reverse=True)
        return [
            RankedSource(
                rank=i + 1,
                source=src,
                composite_score=src.composite_score,
                scores=dict(src.scores),
            )
            for i, src in enumerate(sorted_sources)
        ]

    def get_top_sources(
        self, n: int, sources: list[Source] | None = None
    ) -> list[RankedSource]:
        """Return the top *n* ranked sources.

        Parameters
        ----------
        n : int
            Number of top sources to return.
        sources : list[Source] | None
            Sources to rank.  Defaults to all added sources.
        """
        rankings = self.rank_sources(sources)
        return rankings[:n]

    # -- comparison ---------------------------------------------------------

    def compare_sources(self, s1: Source, s2: Source) -> ComparisonResult:
        """Side-by-side comparison of two sources.

        Returns a ``ComparisonResult`` indicating which source is stronger
        and by how much.
        """
        scores_a = self._compute_scores(s1)
        scores_b = self._compute_scores(s2)
        comp_a = scores_a["composite"]
        comp_b = scores_b["composite"]

        if abs(comp_a - comp_b) < 1e-6:
            winner = "tie"
        elif comp_a > comp_b:
            winner = "a"
        else:
            winner = "b"

        return ComparisonResult(
            source_a=s1,
            source_b=s2,
            scores_a=scores_a,
            scores_b=scores_b,
            composite_a=comp_a,
            composite_b=comp_b,
            winner=winner,
            advantage=round(abs(comp_a - comp_b), 4),
        )

    # -- reporting ----------------------------------------------------------

    def generate_report(self, rankings: list[RankedSource]) -> str:
        """Generate a human-readable ranking report.

        Parameters
        ----------
        rankings : list[RankedSource]
            Output from ``rank_sources()`` or ``get_top_sources()``.

        Returns
        -------
        str
            A formatted multi-line report string.
        """
        lines: list[str] = []
        lines.append("Source Ranking Report")
        lines.append("=" * 60)
        lines.append("")

        for r in rankings:
            preview = r.source.text[:60].replace("\n", " ")
            if len(r.source.text) > 60:
                preview += "..."
            lines.append(
                f'#{r.rank:<3} Score: {r.composite_score:.2f}  | "{preview}"'
            )

            detail_parts = [
                f"  recency={r.scores.get('recency', 0):.2f}",
                f"  authority={r.scores.get('authority', 0):.2f}",
                f"  citations={r.scores.get('citation_density', 0):.2f}",
                f"  factual={r.scores.get('factual_language', 0):.2f}",
            ]
            lines.append("     " + " |".join(detail_parts))
            lines.append("")

        lines.append("=" * 60)
        lines.append(f"Total sources evaluated: {len(rankings)}")
        if rankings:
            best = rankings[0]
            lines.append(
                f"Best source: #{best.rank} (score {best.composite_score:.2f})"
            )

        return "\n".join(lines)
