"""
Source categorisation for Stage 1's union source pool.

Every source is categorised; none are filtered out. Categories are assigned per
domain (one LLM decision covers all of a domain's pages):

1. The target's own domain is always target_company (deterministic).
2. Otherwise the LLM classifier decides, using the same single-prompt chat
   pattern as sandbox.classify_business_category, many domains per call.
3. If the LLM call fails or returns an unknown category, the domain heuristics
   in geo_metrics decide, then "other".

The heuristic guess is stored next to the final label as a cross-check, so
disagreements stay visible in the raw data.
"""
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..geo_metrics import (
    FALLBACK_CATEGORY,
    TARGET_CATEGORY,
    competitor_group,
    heuristic_source_category,
    normalize_domain,
)

logger = logging.getLogger(__name__)

CATEGORY_DEFINITIONS: Dict[str, str] = {
    "target_company": "the business being analysed",
    "direct_competitor": "sells a product or service that competes directly with the target's",
    "indirect_competitor": "sells an adjacent or substitute product/service, or competes for the same customers in a different way",
    "government": "government bodies, regulators, public-sector portals",
    "reference": "encyclopedias, dictionaries, general reference and explainer sites (e.g. Wikipedia)",
    "media": "news outlets, magazines, editorial publishers, review and comparison blogs",
    "community": "forums, Q&A sites, social platforms, user communities (e.g. Reddit, Quora)",
    "industry_organization": "trade bodies, industry associations, standards organisations",
    "academic": "universities, research institutions, journals",
    "other": "none of the above",
}

# Category sources recorded with each label
SOURCE_TARGET_MATCH = "target_domain_match"
SOURCE_LLM = "llm"
SOURCE_HEURISTIC = "heuristic"
SOURCE_FALLBACK = "fallback"


@dataclass
class SourceCategory:
    domain: str
    category: str
    category_source: str  # target_domain_match | llm | heuristic | fallback
    llm_category: Optional[str]
    heuristic_category: Optional[str]
    competitor_group: str  # target | direct_business | ai_visibility

    @property
    def heuristic_agrees(self) -> Optional[bool]:
        """None when there is nothing to compare."""
        if self.llm_category is None or self.heuristic_category is None:
            return None
        return self.llm_category == self.heuristic_category


def build_classification_prompt(
    domains: Sequence[str],
    taxonomy: Sequence[str],
    target_domain: str,
    target_brand: Optional[str],
    business_category: Optional[str],
    page_intent: Optional[str],
) -> str:
    category_lines = "\n".join(
        f"- {c}: {CATEGORY_DEFINITIONS.get(c, 'custom category')}" for c in taxonomy if c != TARGET_CATEGORY
    )
    target_lines = [f"Target domain: {target_domain}"]
    if target_brand:
        target_lines.append(f"Target brand: {target_brand}")
    if business_category:
        target_lines.append(f"Target business category: {business_category}")
    if page_intent:
        target_lines.append(f"What the target page offers: {page_intent[:500]}")
    domain_lines = "\n".join(f"- {d}" for d in domains)
    return (
        "Classify each website domain below by its relationship to the target business.\n\n"
        + "\n".join(target_lines)
        + "\n\nCategories:\n"
        + category_lines
        + "\n\nA direct_competitor must sell a competing product or service. Publishers that review or "
        "compare products are media, not competitors. Use your knowledge of each domain; if unsure, use other.\n\n"
        "Domains:\n"
        + domain_lines
        + '\n\nRespond with ONLY a JSON object mapping each domain to one category, e.g. {"example.com": "media"}'
    )


def parse_classification_response(response: str, taxonomy: Sequence[str]) -> Dict[str, str]:
    """
    Extract {domain: category} from the model's reply. Tolerates code fences and
    surrounding text; drops entries whose category is not in the taxonomy.
    """
    match = re.search(r"\{.*\}", response or "", re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    allowed = set(taxonomy)
    parsed: Dict[str, str] = {}
    for domain, category in data.items():
        if not isinstance(category, str):
            continue
        category = category.strip().lower()
        if category in allowed:
            parsed[normalize_domain(str(domain))] = category
    return parsed


def resolve_category(
    domain: str,
    llm_category: Optional[str],
    heuristic_category: Optional[str],
    target_domain: str,
) -> SourceCategory:
    """Apply the precedence rules: target match > LLM > heuristic > other."""
    if heuristic_category == TARGET_CATEGORY or (target_domain and domain == normalize_domain(target_domain)):
        category, source = TARGET_CATEGORY, SOURCE_TARGET_MATCH
    elif llm_category and llm_category != TARGET_CATEGORY:
        # The LLM may not claim another domain is the target
        category, source = llm_category, SOURCE_LLM
    elif heuristic_category:
        category, source = heuristic_category, SOURCE_HEURISTIC
    else:
        category, source = FALLBACK_CATEGORY, SOURCE_FALLBACK
    return SourceCategory(
        domain=domain,
        category=category,
        category_source=source,
        llm_category=llm_category,
        heuristic_category=heuristic_category,
        competitor_group=competitor_group(category),
    )


async def categorize_domains(
    domains: Sequence[str],
    target_domain: str,
    taxonomy: Sequence[str],
    target_brand: Optional[str] = None,
    business_category: Optional[str] = None,
    page_intent: Optional[str] = None,
    batch_size: int = 25,
    llm=None,
) -> Dict[str, SourceCategory]:
    """Categorise every domain. A failed LLM batch falls back to heuristics, never raises."""
    unique = [d for d in dict.fromkeys(normalize_domain(d) for d in domains) if d]
    heuristics = {d: heuristic_source_category(d, target_domain=target_domain) for d in unique}
    to_classify = [d for d in unique if heuristics[d] != TARGET_CATEGORY]

    if llm is None and to_classify:
        from ..constants import get_model_for_role
        from ..llm_providers.factory import get_llm_provider

        llm = get_llm_provider(get_model_for_role("SOURCE_CLASSIFICATION"))

    async def classify_batch(batch: List[str]) -> Dict[str, str]:
        prompt = build_classification_prompt(batch, taxonomy, target_domain, target_brand, business_category, page_intent)
        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max(800, 40 * len(batch)),
            temperature=0.0,
        )
        return parse_classification_response(response, taxonomy)

    batches = [to_classify[i:i + batch_size] for i in range(0, len(to_classify), batch_size)]
    results = await asyncio.gather(*(classify_batch(b) for b in batches), return_exceptions=True)

    llm_labels: Dict[str, str] = {}
    for batch, result in zip(batches, results):
        if isinstance(result, Exception):
            logger.warning(f"[stage1] source classification batch of {len(batch)} failed, using heuristics: {result}")
            continue
        llm_labels.update(result)

    return {d: resolve_category(d, llm_labels.get(d), heuristics[d], target_domain) for d in unique}
