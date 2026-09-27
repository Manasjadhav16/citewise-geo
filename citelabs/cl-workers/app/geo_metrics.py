"""
Pure metric and helper functions for the two-stage GEO evaluation.

Everything here is deterministic and depends only on the standard library, so it
can be unit-tested without importing the heavy pipeline modules (sandbox.py pulls
in torch, Playwright, Pinecone, ...).

Stage 1 (real-world observation) and Stage 2 (controlled RAG simulation) both
feed these functions:
- Stage 1: repeated Gemini search-grounded answers per query. NOTE: that is an
  approximate proxy for Google AI Overview citation behaviour, not real AI
  Overview data (there is no public AI Overview API).
- Stage 2: the existing RAG simulation over the crawled source pool.

Rates are returned as percentages (0-100) to match the existing scores
(citation_rate, coverage, competitor_dominance in sandbox.calculate_final_scores).
"""
import itertools
import os
import re
import statistics
from dataclasses import dataclass
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


# ==========================================
# Source category taxonomy
# ==========================================

DEFAULT_SOURCE_CATEGORIES: Tuple[str, ...] = (
    "target_company",
    "direct_competitor",
    "indirect_competitor",
    "government",
    "reference",
    "media",
    "community",
    "industry_organization",
    "academic",
    "other",
)

TARGET_CATEGORY = "target_company"
DIRECT_COMPETITOR_CATEGORY = "direct_competitor"
FALLBACK_CATEGORY = "other"

# Competitor groups: a direct business competitor sells a competing product or
# service; every other non-target source (government, reference, media, community,
# industry organisations, academic, indirect competitors, other) is an
# "AI visibility competitor": it occupies citation space without selling against us.
GROUP_TARGET = "target"
GROUP_DIRECT_BUSINESS = "direct_business"
GROUP_AI_VISIBILITY = "ai_visibility"


def load_source_categories(env_value: Optional[str] = None) -> Tuple[str, ...]:
    """
    Category taxonomy from the SOURCE_CATEGORIES env var (comma-separated), or the
    default taxonomy when unset. target_company, direct_competitor and other are
    always included because the pipeline relies on them.
    """
    raw = env_value if env_value is not None else os.getenv("SOURCE_CATEGORIES", "")
    categories = [c.strip().lower() for c in raw.split(",") if c.strip()]
    if not categories:
        return DEFAULT_SOURCE_CATEGORIES
    for required in (TARGET_CATEGORY, DIRECT_COMPETITOR_CATEGORY, FALLBACK_CATEGORY):
        if required not in categories:
            categories.append(required)
    return tuple(dict.fromkeys(categories))


def competitor_group(category: str) -> str:
    """Map a source category to target / direct_business / ai_visibility."""
    if category == TARGET_CATEGORY:
        return GROUP_TARGET
    if category == DIRECT_COMPETITOR_CATEGORY:
        return GROUP_DIRECT_BUSINESS
    return GROUP_AI_VISIBILITY


# ==========================================
# Domains and heuristic categorisation
# ==========================================


def normalize_domain(url_or_domain: str) -> str:
    """
    Lowercased host without scheme, "www.", port or path.
    Accepts full URLs ("https://www.Example.com/a") or bare domains ("example.com/a").
    Returns "" when nothing usable is found.
    """
    if not url_or_domain:
        return ""
    value = url_or_domain.strip()
    if "://" not in value:
        value = "//" + value
    try:
        host = urlparse(value).hostname or ""
    except ValueError:
        return ""
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


_TRACKING_PARAM = re.compile(r"^(utm_[a-z]+|gclid|fbclid|srsltid|ref)$", re.IGNORECASE)


def canonical_url(url: str) -> str:
    """
    Identity for a source page: https, normalized host, no fragment (drops
    "#:~:text=" highlights), no tracking parameters, no trailing slash.
    Two citations of the same page map to the same string.
    """
    domain = normalize_domain(url)
    if not domain:
        return ""
    parsed = urlparse(url if "://" in url else "//" + url)
    query = "&".join(
        part for part in parsed.query.split("&") if part and not _TRACKING_PARAM.match(part.split("=", 1)[0])
    )
    path = parsed.path.rstrip("/")
    return f"https://{domain}{path}" + (f"?{query}" if query else "")


def _domain_matches(domain: str, candidates: Iterable[str]) -> bool:
    """True if domain equals a candidate or is a subdomain of one."""
    for candidate in candidates:
        candidate = normalize_domain(candidate)
        if candidate and (domain == candidate or domain.endswith("." + candidate)):
            return True
    return False


# Known-domain lists are deliberately small: the heuristics are a fallback and a
# cross-check for the LLM classifier, not the primary classifier.
REFERENCE_DOMAINS = (
    "wikipedia.org",
    "wikimedia.org",
    "britannica.com",
    "merriam-webster.com",
    "dictionary.com",
    "investopedia.com",
)
COMMUNITY_DOMAINS = (
    "reddit.com",
    "quora.com",
    "stackexchange.com",
    "stackoverflow.com",
    "news.ycombinator.com",
)
MEDIA_DOMAINS = (
    "nytimes.com",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "forbes.com",
    "bloomberg.com",
    "techcrunch.com",
    "theverge.com",
    "economictimes.indiatimes.com",
    "timesofindia.indiatimes.com",
    "livemint.com",
    "thehindu.com",
    "ndtv.com",
    "moneycontrol.com",
    "business-standard.com",
    "hindustantimes.com",
)

_GOVERNMENT_SUFFIX = re.compile(r"(\.gov|\.mil|\.gov\.[a-z]{2}|\.gouv\.[a-z]{2}|\.nic\.in)$")
_ACADEMIC_SUFFIX = re.compile(r"(\.edu|\.edu\.[a-z]{2}|\.ac\.[a-z]{2})$")


def heuristic_source_category(
    url: str,
    target_domain: Optional[str] = None,
    direct_competitor_domains: Collection[str] = (),
) -> Optional[str]:
    """
    Deterministic category guess from the domain alone. Returns None when the
    domain gives no reliable signal, leaving the decision to the LLM classifier.
    """
    domain = normalize_domain(url)
    if not domain:
        return None
    if target_domain and _domain_matches(domain, [target_domain]):
        return TARGET_CATEGORY
    if _domain_matches(domain, direct_competitor_domains):
        return DIRECT_COMPETITOR_CATEGORY
    if _GOVERNMENT_SUFFIX.search(domain):
        return "government"
    if _ACADEMIC_SUFFIX.search(domain):
        return "academic"
    if _domain_matches(domain, REFERENCE_DOMAINS):
        return "reference"
    if _domain_matches(domain, COMMUNITY_DOMAINS):
        return "community"
    if _domain_matches(domain, MEDIA_DOMAINS):
        return "media"
    return None


# ==========================================
# Token budget
# ==========================================


def compute_per_source_token_budget(
    num_sources: int,
    total_budget: int = 20000,
    per_source_cap: int = 1000,
    min_floor: int = 200,
) -> int:
    """
    Evidence token budget for each source in a query's pool:
    min(per_source_cap, total_budget // num_sources), never below min_floor.

    The floor wins over the total: with many sources (e.g. 150 sources at a
    200-token floor = 30,000 tokens) the query's total can exceed total_budget,
    because a representation below the floor is not meaningful. Callers that must
    stay within total_budget should cap the number of sources instead.

    Returns 0 when there are no sources.
    """
    if num_sources < 0:
        raise ValueError("num_sources must be >= 0")
    if total_budget <= 0 or per_source_cap <= 0 or min_floor <= 0:
        raise ValueError("total_budget, per_source_cap and min_floor must be > 0")
    if min_floor > per_source_cap:
        raise ValueError("min_floor cannot exceed per_source_cap")
    if num_sources == 0:
        return 0
    return max(min_floor, min(per_source_cap, total_budget // num_sources))


# ==========================================
# Mention vs citation, and citation position
# ==========================================


@dataclass(frozen=True)
class ResponseVisibility:
    """How the target appeared in one generated response."""

    mentioned: bool  # brand named in the answer text
    cited: bool  # target URL/domain cited as a source
    citation_position: Optional[int] = None  # 1-based rank among cited domains, None if not cited


def _percentage(count: int, total: int) -> float:
    return (count / total) * 100.0 if total else 0.0


def mention_rate(responses: Sequence[ResponseVisibility]) -> float:
    """AI Visibility / Mention Rate: % of responses that name the target brand."""
    return _percentage(sum(1 for r in responses if r.mentioned), len(responses))


def citation_rate(responses: Sequence[ResponseVisibility]) -> float:
    """
    Strict citation rate: % of responses that cite the target as a source.

    Not the same as sandbox.calculate_final_scores' "citation_rate", which counts
    did_sandbox_appear (mentioned OR cited) and feeds the GEO/AEO formulas; that
    one is kept unchanged so scores stay comparable with earlier runs.
    """
    return _percentage(sum(1 for r in responses if r.cited), len(responses))


def citation_position(cited_urls: Sequence[str], target_domain: str) -> Optional[int]:
    """
    1-based position of the target among the distinct cited domains, in order of
    first appearance. Repeated citations of one domain count once, so
    [a.com/x, a.com/y, target.com] puts the target at position 2.
    Returns None if the target is not cited.
    """
    target = normalize_domain(target_domain)
    if not target:
        return None
    seen: List[str] = []
    for url in cited_urls:
        domain = normalize_domain(url)
        if not domain or domain in seen:
            continue
        seen.append(domain)
        if domain == target or domain.endswith("." + target):
            return len(seen)
    return None


def _valid_positions(positions: Iterable[Optional[int]]) -> List[int]:
    valid = [p for p in positions if p is not None]
    if any(p < 1 for p in valid):
        raise ValueError("citation positions are 1-based")
    return valid


def mean_citation_position(positions: Iterable[Optional[int]]) -> Optional[float]:
    """Mean position over responses where the target was cited (None entries skipped)."""
    valid = _valid_positions(positions)
    return statistics.fmean(valid) if valid else None


def median_citation_position(positions: Iterable[Optional[int]]) -> Optional[float]:
    """Median position over responses where the target was cited (None entries skipped)."""
    valid = _valid_positions(positions)
    return float(statistics.median(valid)) if valid else None


# ==========================================
# Stage 1 source-set metrics (per query, across N repeated runs)
# ==========================================
# `runs` holds one collection of source identifiers (URLs or domains) per
# successful run. Callers should leave failed runs out: an empty set means
# "the run succeeded and cited nothing", which counts against stability.


def union_sources(runs: Sequence[Collection[str]]) -> List[str]:
    """
    Union source pool S1 ∪ ... ∪ SN, ordered by how many runs cited each source
    (most frequent first), ties broken by first appearance. No source is dropped.
    """
    counts: Dict[str, int] = {}
    for run in runs:
        for source in dict.fromkeys(run):
            counts[source] = counts.get(source, 0) + 1
    first_seen = {source: i for i, source in enumerate(counts)}
    return sorted(counts, key=lambda s: (-counts[s], first_seen[s]))


def source_stability(runs: Sequence[Collection[str]]) -> Dict[str, float]:
    """Per source: appearances / N, where N is the number of runs (0.0-1.0)."""
    if not runs:
        return {}
    counts: Dict[str, int] = {}
    for run in runs:
        for source in set(run):
            counts[source] = counts.get(source, 0) + 1
    return {source: counts[source] / len(runs) for source in union_sources(runs)}


def source_set_diversity(runs: Sequence[Collection[str]]) -> int:
    """|S1 ∪ ... ∪ SN|: number of distinct sources cited across all runs."""
    return len(set().union(*runs)) if runs else 0


def jaccard_similarity(a: Collection[str], b: Collection[str]) -> float:
    """|A ∩ B| / |A ∪ B|. Two empty sets are identical, so they score 1.0."""
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def mean_pairwise_jaccard(runs: Sequence[Collection[str]]) -> Optional[float]:
    """
    Source-Set Stability: average Jaccard similarity over all pairs of runs.
    None with fewer than two runs, since there is no pair to compare.
    """
    if len(runs) < 2:
        return None
    pairs = list(itertools.combinations(runs, 2))
    return sum(jaccard_similarity(a, b) for a, b in pairs) / len(pairs)


# ==========================================
# AI Visibility Competition breakdown
# ==========================================


def category_breakdown(
    citation_categories: Iterable[str],
    taxonomy: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    % share of citations by source category (one entry per citation).
    With a taxonomy, every category in it is present (0.0 if unused); categories
    outside the taxonomy are still counted, never dropped. Returns {} (or all
    zeros with a taxonomy) when there are no citations.
    """
    counts: Dict[str, int] = {category: 0 for category in taxonomy or ()}
    total = 0
    for category in citation_categories:
        counts[category] = counts.get(category, 0) + 1
        total += 1
    return {category: _percentage(count, total) for category, count in counts.items()}


# ==========================================
# Before/after comparison
# ==========================================


def compare_run_metrics(
    before: Mapping[str, Optional[float]],
    after: Mapping[str, Optional[float]],
) -> Dict[str, object]:
    """
    Pure core of compare_runs(before_run_id, after_run_id): the ID-based wrapper
    loads each run's stored metrics and passes them here.

    Returns geo_improvement and aeo_improvement (Score(after) - Score(before)) and
    a per-metric {before, after, delta} map over every metric in either run.
    A delta is None when either side is missing (e.g. no citation position
    because the target was never cited).
    """
    deltas: Dict[str, Dict[str, Optional[float]]] = {}
    for key in dict.fromkeys([*before.keys(), *after.keys()]):
        b, a = before.get(key), after.get(key)
        numeric = isinstance(b, (int, float)) and isinstance(a, (int, float))
        deltas[key] = {"before": b, "after": a, "delta": (a - b) if numeric else None}
    return {
        "geo_improvement": deltas.get("geo_score", {}).get("delta"),
        "aeo_improvement": deltas.get("aeo_score", {}).get("delta"),
        "deltas": deltas,
    }
