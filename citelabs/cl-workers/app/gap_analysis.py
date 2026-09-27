"""
Content gap analysis (interface only, not yet implemented).

Goal: for a query, identify the claims and topics that the sources an answer
engine cites (competitors, and AI visibility competitors such as government,
reference, media and community sites) cover, but the sandbox page does not.
Those are the gaps most likely costing the sandbox its citation.

Inputs already exist from Stage 1 (see app/observation/):
- sandbox_evidence: the sandbox page's evidence-preserving representation for
  the query (observation_queries.target_evidence, or the sandbox's own
  observation_sources row when it was cited)
- competitor_evidence_list: the evidence of every other source in the query's
  union pool (observation_sources.evidence), with its category and stability

TODO: implement find_content_gaps. Intended approach:
1. Split every evidence text into atomic claims (facts, statistics, definitions,
   product/pricing details, comparisons); the evidence is already bullet-pointed.
2. Cluster claims across sources by topic (embeddings, e.g. the Gemini
   embeddings the pipeline already uses), so the same fact stated by several
   sources counts once, with all its sources attached.
3. For each topic cluster, check whether the sandbox evidence covers it
   (semantic match against sandbox claims, with an LLM check for borderline cases).
4. Report uncovered clusters as Gaps, ranked by how much citation weight
   supports them: the stability of the sources that state them (how often
   those sources were cited across repeated runs) and how many sources do.
5. Keep it evidence-based: every Gap cites the source URLs and quotes the claims
   it is derived from, in line with the deterministic diagnosis layer in
   sandbox._diagnose_rag_result, rather than an LLM's unsupported judgment.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class EvidenceSource:
    """One source's evidence for a query (an observation_sources row)."""

    url: str
    domain: str
    category: str  # e.g. direct_competitor, government, media (see geo_metrics.DEFAULT_SOURCE_CATEGORIES)
    competitor_group: str  # target | direct_business | ai_visibility
    evidence: str
    stability: float = 0.0  # share of Stage 1 runs that cited this source (0-1)


@dataclass
class Gap:
    """A topic or claim the cited sources cover and the sandbox page does not."""

    query: str
    topic: str  # short label, e.g. "RBI e-mandate compliance"
    claims: List[str] = field(default_factory=list)  # quoted claims from the sources
    supporting_sources: List[str] = field(default_factory=list)  # URLs
    source_categories: List[str] = field(default_factory=list)
    weight: float = 0.0  # ranking score, e.g. summed stability of supporting sources
    sandbox_partial_match: Optional[str] = None  # closest sandbox claim, if any


def find_content_gaps(
    sandbox_evidence: Optional[str],
    competitor_evidence_list: Sequence[EvidenceSource],
    query: str = "",
) -> List[Gap]:
    """
    Topics/claims covered by the cited sources but missing from the sandbox page,
    most important first. Not implemented yet: see the module docstring.
    """
    raise NotImplementedError(
        "Content gap analysis is not implemented yet; see app/gap_analysis.py for the planned approach."
    )
