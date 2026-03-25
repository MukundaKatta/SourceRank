"""SourceRank — Citation and source quality ranker for RAG systems."""

__version__ = "0.1.0"

from sourcerank.core import (
    QualitySignals,
    RankingResult,
    SourceDocument,
    SourceRanker,
)
from sourcerank.config import RankerConfig

# Aliases for backward compatibility
SourceRank = SourceRanker
SourceRankConfig = RankerConfig

__all__ = [
    "QualitySignals",
    "RankingResult",
    "RankerConfig",
    "SourceDocument",
    "SourceRanker",
    "SourceRank",
    "SourceRankConfig",
]
