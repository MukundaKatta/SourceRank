"""Tests for SourceRank core ranking engine."""

from __future__ import annotations

import pytest

from sourcerank import SourceRank, SourceRankConfig
from sourcerank.core import Source


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ranker() -> SourceRank:
    return SourceRank()


@pytest.fixture
def high_quality_text() -> str:
    return (
        "According to a 2025 study published in Nature, the global temperature "
        "rose by 1.2 degrees Celsius (Smith et al., 2025). Dr. Jane Thompson at "
        "the National Institute confirmed that 67% of models showed significant "
        "warming trends [1][2][3]."
    )


@pytest.fixture
def low_quality_text() -> str:
    return (
        "Some people think the weather is changing. It might be getting warmer, "
        "but who knows really. I guess maybe it could be a problem."
    )


# ---------------------------------------------------------------------------
# Test: scoring pipeline
# ---------------------------------------------------------------------------


class TestScoring:
    """Test individual signal scorers and composite ranking."""

    def test_high_quality_source_scores_higher(
        self, ranker: SourceRank, high_quality_text: str, low_quality_text: str
    ) -> None:
        s1 = ranker.add_source(high_quality_text, {"date": "2025-11-01", "domain": "nature.com"})
        s2 = ranker.add_source(low_quality_text, {"date": "2018-03-15", "domain": "blogspot.com"})

        rankings = ranker.rank_sources()
        assert rankings[0].source.text == high_quality_text
        assert rankings[0].composite_score > rankings[1].composite_score

    def test_recency_score_recent_is_higher(self, ranker: SourceRank) -> None:
        recent = Source(text="content", metadata={"date": "2026-01-01"})
        old = Source(text="content", metadata={"date": "2015-01-01"})
        assert ranker.score_recency(recent) > ranker.score_recency(old)

    def test_recency_score_no_date_returns_default(self, ranker: SourceRank) -> None:
        no_date = Source(text="content", metadata={})
        assert ranker.score_recency(no_date) == pytest.approx(0.3)

    def test_authority_score_with_domain(self, ranker: SourceRank) -> None:
        gov_source = Source(text="Federal report", metadata={"domain": "cdc.gov"})
        blog_source = Source(text="blog post", metadata={"domain": "myblog.xyz"})
        assert ranker.score_authority(gov_source) > ranker.score_authority(blog_source)

    def test_citation_density_with_citations(self, ranker: SourceRank) -> None:
        cited = Source(text="Study (Smith et al., 2023) and (Jones, 2024) found [1] evidence.")
        uncited = Source(text="No references at all in this plain text.")
        assert ranker.score_citation_density(cited) > ranker.score_citation_density(uncited)

    def test_factual_language_hedging_penalised(self, ranker: SourceRank) -> None:
        factual = Source(
            text="Data showed that 45% of participants demonstrated significant improvement in 2024."
        )
        hedging = Source(
            text="Maybe it could possibly seem like something might be sort of happening perhaps."
        )
        assert ranker.score_factual_language(factual) > ranker.score_factual_language(hedging)


# ---------------------------------------------------------------------------
# Test: ranking and top-k
# ---------------------------------------------------------------------------


class TestRanking:
    """Test rank_sources and get_top_sources."""

    def test_rank_sources_returns_all(self, ranker: SourceRank) -> None:
        ranker.add_source("Source A", {"date": "2025-01-01"})
        ranker.add_source("Source B", {"date": "2024-01-01"})
        ranker.add_source("Source C", {"date": "2023-01-01"})
        rankings = ranker.rank_sources()
        assert len(rankings) == 3
        assert rankings[0].rank == 1

    def test_get_top_sources(self, ranker: SourceRank) -> None:
        for i in range(5):
            ranker.add_source(f"Source {i}")
        top = ranker.get_top_sources(2)
        assert len(top) == 2

    def test_empty_ranker(self, ranker: SourceRank) -> None:
        rankings = ranker.rank_sources()
        assert rankings == []


# ---------------------------------------------------------------------------
# Test: compare and report
# ---------------------------------------------------------------------------


class TestCompareAndReport:
    """Test compare_sources and generate_report."""

    def test_compare_sources(
        self, ranker: SourceRank, high_quality_text: str, low_quality_text: str
    ) -> None:
        s1 = ranker.add_source(high_quality_text, {"date": "2025-11-01", "domain": "nature.com"})
        s2 = ranker.add_source(low_quality_text, {"date": "2018-03-15"})
        result = ranker.compare_sources(s1, s2)
        assert result.winner == "a"
        assert result.advantage > 0

    def test_generate_report_contains_scores(self, ranker: SourceRank) -> None:
        ranker.add_source(
            "A recent peer-reviewed study found that 30% of cases improved.",
            {"date": "2026-01-01"},
        )
        ranker.add_source("Maybe something happened.", {"date": "2020-01-01"})
        rankings = ranker.rank_sources()
        report = ranker.generate_report(rankings)
        assert "Source Ranking Report" in report
        assert "#1" in report
        assert "#2" in report
        assert "Total sources evaluated: 2" in report
