"""SourceRank — Citation and source quality ranker for RAG systems."""

__version__ = "0.1.0"

from sourcerank.core import (
    QualitySignals,
    RankingResult,
    SourceDocument,
    SourceRanker,
)
from sourcerank.config import RankerConfig

__all__ = [
    "QualitySignals",
    "RankingResult",
    "RankerConfig",
    "SourceDocument",
    "SourceRanker",
]
