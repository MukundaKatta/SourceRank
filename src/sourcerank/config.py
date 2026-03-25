"""Configuration for SourceRank signal weights and thresholds."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SignalWeights(BaseModel):
    """Weights for each quality signal (must sum to 1.0 for normalized scoring)."""

    recency: float = Field(default=0.20, ge=0.0, le=1.0)
    authority: float = Field(default=0.20, ge=0.0, le=1.0)
    completeness: float = Field(default=0.15, ge=0.0, le=1.0)
    citation_density: float = Field(default=0.20, ge=0.0, le=1.0)
    factual_language: float = Field(default=0.25, ge=0.0, le=1.0)

    def normalized(self) -> "SignalWeights":
        """Return a copy with weights normalized to sum to 1.0."""
        total = (
            self.recency + self.authority + self.completeness
            + self.citation_density + self.factual_language
        )
        if total == 0:
            return SignalWeights()
        return SignalWeights(
            recency=self.recency / total,
            authority=self.authority / total,
            completeness=self.completeness / total,
            citation_density=self.citation_density / total,
            factual_language=self.factual_language / total,
        )


class AuthorityConfig(BaseModel):
    """Domain-based authority scoring rules."""

    domain_scores: dict[str, float] = Field(
        default_factory=lambda: {
            ".gov": 0.95,
            ".edu": 0.90,
            ".org": 0.75,
            ".com": 0.50,
            ".net": 0.45,
            ".io": 0.45,
        }
    )
    default_score: float = Field(default=0.40, ge=0.0, le=1.0)


class DeduplicationConfig(BaseModel):
    """Settings for near-duplicate detection."""

    similarity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="Jaccard similarity threshold above which documents are considered duplicates.",
    )


class DiversityConfig(BaseModel):
    """Settings for MMR-based diverse selection."""

    lambda_param: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Trade-off between relevance (1.0) and diversity (0.0).",
    )


class RankerConfig(BaseModel):
    """Top-level configuration for SourceRanker."""

    weights: SignalWeights = Field(default_factory=SignalWeights)
    authority: AuthorityConfig = Field(default_factory=AuthorityConfig)
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    diversity: DiversityConfig = Field(default_factory=DiversityConfig)
    max_document_age_days: int = Field(
        default=365 * 3,
        description="Documents older than this get a recency score of 0.",
    )
    min_document_length: int = Field(
        default=50,
        description="Minimum character length for a document to be considered complete.",
    )
    ideal_document_length: int = Field(
        default=2000,
        description="Character length at which completeness score is 1.0.",
    )
