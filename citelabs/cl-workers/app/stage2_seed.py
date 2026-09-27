"""
Stage 2 seeding and tagging: pure functions that connect Stage 1's real-world
observations to the controlled RAG pipeline.

- build_stage1_pool / merge_source_pools: turn Stage 1's union source pools into
  the competitor list Stage 2 crawls, deduplicated by domain, with Stage 1's
  category labels preserved, then merged with Stage 2's own SERP/LLM discovery.
- annotate_discovered: category labels for runs without Stage 1 seeding (no
  pages are dropped, so unseeded runs crawl exactly what they did before).
- tag_rag_result: per-answer sandbox citation position and the category of every
  cited and retrieved source.
- summarize_stage2_metrics: run-level mention/strict citation rates, citation
  positions and the AI visibility competition breakdown.

Standard library only (plus geo_metrics), so it is testable without the pipeline.
"""
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .geo_metrics import (
    DIRECT_COMPETITOR_CATEGORY,
    FALLBACK_CATEGORY,
    TARGET_CATEGORY,
    ResponseVisibility,
    category_breakdown,
    citation_position,
    citation_rate,
    competitor_group,
    heuristic_source_category,
    load_source_categories,
    mean_citation_position,
    median_citation_position,
    mention_rate,
    normalize_domain,
)

# Stage 2's pipeline distinguishes only these two competitor types
TYPE_BUSINESS_COMPETITOR = "business_competitor"
TYPE_AUTHORITY_SOURCE = "authority_source"

DEFAULT_MAX_SOURCES = 40

# Same pattern as sandbox._extract_urls_from_text, kept here to stay import-light
_URL_PATTERN = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)


def max_sources_from_env() -> int:
    value = os.getenv("STAGE2_MAX_SOURCES", "").strip()
    return int(value) if value else DEFAULT_MAX_SOURCES


def pipeline_type(category: str) -> str:
    """Stage 1 category -> Stage 2 competitor type (drives retrieval weights and citation lists)."""
    return TYPE_BUSINESS_COMPETITOR if category == DIRECT_COMPETITOR_CATEGORY else TYPE_AUTHORITY_SOURCE


def _discovered_category(entry: Mapping[str, Any]) -> Tuple[str, str]:
    """Category for a SERP/LLM-discovered source, which has only a competitor type."""
    if entry.get("type") in (TYPE_BUSINESS_COMPETITOR, "direct"):
        return DIRECT_COMPETITOR_CATEGORY, "discovery"
    heuristic = heuristic_source_category(entry.get("url", ""))
    if heuristic and heuristic != TARGET_CATEGORY:
        return heuristic, "heuristic"
    return FALLBACK_CATEGORY, "fallback"


def annotate_discovered(discovered: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Add category fields to discovered sources without dropping or reordering any."""
    annotated = []
    for entry in discovered:
        category, source = _discovered_category(entry)
        annotated.append(
            {
                **entry,
                "source_category": category,
                "competitor_group": competitor_group(category),
                "category_source": source,
                "origin": entry.get("origin", "discovery"),
            }
        )
    return annotated


def build_stage1_pool(stage1_sources: Sequence[Mapping[str, Any]], target_domain: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Stage 1 union pool -> Stage 2 competitor entries, one per domain.

    `stage1_sources` are observation_sources rows from every query (url, domain,
    category, competitor_group, category_source, stability). Per domain, the page
    with the highest stability is kept (earlier rows win ties). The target's own
    domain is excluded because the sandbox page is always crawled separately, and
    sources whose URL could not be resolved cannot be crawled. Ordered by
    stability, most stable first.

    Returns (pool, skipped counts).
    """
    target = normalize_domain(target_domain)
    best: Dict[str, Dict[str, Any]] = {}
    skipped = {"unresolved_url": 0, "target_domain": 0, "same_domain_duplicate": 0}
    for row in stage1_sources:
        url = row.get("url")
        domain = normalize_domain(row.get("domain") or url or "")
        if not url or not domain:
            skipped["unresolved_url"] += 1
            continue
        if row.get("category") == TARGET_CATEGORY or domain == target or domain.endswith("." + target):
            skipped["target_domain"] += 1
            continue
        stability = float(row.get("stability") or 0.0)
        if domain in best:
            skipped["same_domain_duplicate"] += 1
            if stability <= best[domain]["stage1_stability"]:
                continue
        category = row.get("category") or FALLBACK_CATEGORY
        best[domain] = {
            "url": url,
            "type": pipeline_type(category),
            "source": "stage1",
            "origin": "stage1",
            "source_category": category,
            "competitor_group": row.get("competitor_group") or competitor_group(category),
            "category_source": row.get("category_source") or "stage1",
            "stage1_stability": stability,
        }
    pool = sorted(best.values(), key=lambda e: -e["stage1_stability"])  # stable sort keeps first-seen order on ties
    return pool, skipped


def merge_source_pools(
    stage1_pool: Sequence[Mapping[str, Any]],
    discovered: Sequence[Mapping[str, Any]],
    target_domain: str,
    max_sources: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Stage 1 pool first (its categories win), then discovered sources whose domain
    is not already present, deduplicated by domain. Capped at max_sources, which
    trims discovered sources before Stage 1 ones.

    Returns (merged pool, counts).
    """
    if max_sources < 1:
        raise ValueError("max_sources must be >= 1")
    target = normalize_domain(target_domain)
    merged: List[Dict[str, Any]] = [dict(e) for e in stage1_pool]
    seen = {normalize_domain(e["url"]) for e in merged}
    added = duplicates = 0
    for entry in annotate_discovered(discovered):
        domain = normalize_domain(entry.get("url", ""))
        if not domain or domain == target or domain.endswith("." + target):
            continue
        if domain in seen:
            duplicates += 1
            continue
        seen.add(domain)
        merged.append(entry)
        added += 1
    counts = {
        "stage1": len(stage1_pool),
        "discovered_added": added,
        "discovered_duplicate_domains": duplicates,
        "dropped_over_cap": max(0, len(merged) - max_sources),
    }
    return merged[:max_sources], counts


def category_lookup(pool: Iterable[Mapping[str, Any]]) -> Dict[str, Tuple[str, str]]:
    """domain -> (category, competitor group) for tagging RAG citations and chunks."""
    lookup: Dict[str, Tuple[str, str]] = {}
    for entry in pool:
        domain = normalize_domain(entry.get("url", ""))
        category = entry.get("source_category")
        if domain and category and domain not in lookup:
            lookup[domain] = (category, entry.get("competitor_group") or competitor_group(category))
    return lookup


def _categorize_domain(domain: str, sandbox_domain: str, lookup: Mapping[str, Tuple[str, str]]) -> Tuple[str, str]:
    if domain == sandbox_domain or domain.endswith("." + sandbox_domain):
        return TARGET_CATEGORY, competitor_group(TARGET_CATEGORY)
    if domain in lookup:
        return lookup[domain]
    heuristic = heuristic_source_category(domain) or FALLBACK_CATEGORY
    return heuristic, competitor_group(heuristic)


def extract_cited_urls(answer: str) -> List[str]:
    """URLs in the answer, in order of appearance (same pattern as the pipeline's extractor)."""
    urls = []
    for match in _URL_PATTERN.findall(answer or ""):
        urls.append(match if match.lower().startswith("http") else "https://" + match)
    return urls


def tag_rag_result(
    answer: str,
    sandbox_url: str,
    retrieved_chunk_urls: Sequence[str],
    lookup: Mapping[str, Tuple[str, str]],
) -> Dict[str, Any]:
    """
    Per-answer tags:
    - sandbox_citation_position: 1-based rank of the sandbox among distinct cited domains
    - cited_source_categories: one entry per distinct cited URL
    - retrieved_source_categories: category -> number of retrieved chunks
    """
    sandbox_domain = normalize_domain(sandbox_url)
    cited = list(dict.fromkeys(extract_cited_urls(answer)))
    cited_categories = []
    for url in cited:
        domain = normalize_domain(url)
        if not domain:
            continue
        category, group = _categorize_domain(domain, sandbox_domain, lookup)
        cited_categories.append({"url": url, "domain": domain, "category": category, "competitor_group": group})
    retrieved: Dict[str, int] = {}
    for url in retrieved_chunk_urls:
        domain = normalize_domain(url)
        if domain:
            category, _ = _categorize_domain(domain, sandbox_domain, lookup)
            retrieved[category] = retrieved.get(category, 0) + 1
    return {
        "sandbox_citation_position": citation_position(cited, sandbox_domain),
        "cited_source_categories": cited_categories,
        "retrieved_source_categories": retrieved,
    }


def is_error_result(result: Mapping[str, Any]) -> bool:
    """RAG rows produced by a failed question, not by a generated answer."""
    return (result.get("answer") or "").startswith("Error") and not result.get("chunks_used")


def summarize_stage2_metrics(
    rag_results: Sequence[Mapping[str, Any]],
    taxonomy: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run-level metrics added to Stage 2's final scores. The existing GEO/AEO
    formulas and their citation_rate (did_sandbox_appear: mentioned OR cited) are
    untouched; these are separate metrics. Failed questions are excluded.
    """
    taxonomy = tuple(taxonomy) if taxonomy else load_source_categories()
    evaluated = [r for r in rag_results if not is_error_result(r)]
    # cited follows the pipeline's own sandbox_citations (exact domain match); a
    # position only counts for answers that meet that definition
    visibilities = [
        ResponseVisibility(
            mentioned=bool((r.get("metrics") or {}).get("is_brand_mentioned")),
            cited=bool(r.get("sandbox_citations")),
            citation_position=r.get("sandbox_citation_position") if r.get("sandbox_citations") else None,
        )
        for r in evaluated
    ]
    categories = [c["category"] for r in evaluated for c in (r.get("cited_source_categories") or [])]
    groups = [c["competitor_group"] for r in evaluated for c in (r.get("cited_source_categories") or [])]
    return {
        "responses_evaluated": len(evaluated),
        "mention_rate": mention_rate(visibilities),
        "strict_citation_rate": citation_rate(visibilities),
        "mean_citation_position": mean_citation_position(v.citation_position for v in visibilities),
        "median_citation_position": median_citation_position(v.citation_position for v in visibilities),
        "category_breakdown": category_breakdown(categories, taxonomy=taxonomy),
        "competitor_group_breakdown": category_breakdown(groups, taxonomy=("target", "direct_business", "ai_visibility")),
    }
