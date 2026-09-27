"""
Evidence-preserving compression for Stage 1 sources.

For each (query, source) pair, extract a compact representation of the evidence
on the page that could influence an answer engine's answer and citation decision
for that query: facts, statistics, definitions, entities, product/service details,
claims, specs, comparisons, expert statements and relevant excerpts. This is
deliberately not a generic page summary, and it is query-specific, so a source
cited for several queries is compressed once per query.

Token counts are estimated at ~4 characters per token; no tokenizer call is made.
The model is asked to stay within the per-source budget, and the output is
truncated to it as a hard limit.
"""
import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..geo_metrics import compute_per_source_token_budget

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return -(-len(text or "") // CHARS_PER_TOKEN)  # ceiling division


def enforce_token_budget(text: str, token_budget: int) -> str:
    """Hard cap: truncate to the budget, at a line or word boundary where possible."""
    text = (text or "").strip()
    max_chars = token_budget * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 2]  # room for the " …" marker
    boundary = max(cut.rfind("\n"), cut.rfind(" "))
    if boundary > max_chars * 0.8:
        cut = cut[:boundary]
    return cut.rstrip() + " …"


NO_EVIDENCE_SENTINEL = "NO_RELEVANT_EVIDENCE"


def is_no_evidence_reply(text: str) -> bool:
    """
    True when the model used the no-evidence escape hatch. Models vary the
    spelling ("NO_Relevant_EVIDENCE", "No relevant evidence."), so letters are
    compared case-insensitively with separators and punctuation ignored.
    """
    letters = re.sub(r"[^a-z]", "", (text or "").lower())
    return letters.startswith("norelevantevidence") and len(letters) <= len("norelevantevidence") + 40


def build_evidence_prompt(
    query: str,
    url: str,
    title: Optional[str],
    page_text: str,
    token_budget: int,
    cited_for_query: bool = True,
) -> str:
    word_budget = int(token_budget * 0.75)
    cited_note = (
        "An AI search engine cited this page when answering this query, so it very likely contains "
        "information that informed the answer: look for it carefully.\n\n"
        if cited_for_query
        else ""
    )
    return (
        "You are preparing evidence for an answer engine that must answer a user's search query "
        "and decide which sources to cite.\n\n"
        f"Query: {query}\n"
        f"Source: {url}" + (f" ({title})" if title else "") + "\n\n"
        + cited_note
        + "Create a compact representation containing the evidence on this page that could influence "
        "the answer or the citation decision for this query. Do not summarise the page in general.\n\n"
        "Extract, where present:\n"
        "- key facts, statistics and numbers (with units, dates, currencies)\n"
        "- definitions\n"
        "- important keywords, named entities, brands and products\n"
        "- product/service information: features, pricing, plans, eligibility, availability\n"
        "- claims the page makes, and technical specifications\n"
        "- comparisons with alternatives\n"
        "- expert statements or quotes, attributed\n"
        "- short verbatim excerpts most relevant to the query, in quotes\n\n"
        "Preserve exact figures and wording rather than paraphrasing. Omit navigation, boilerplate "
        "and anything irrelevant to the query. Use terse bullet points.\n\n"
        "Partial relevance counts: if the page covers the query's topic, products, entities or related "
        "statistics without answering it exactly, extract that evidence. Only if the page has nothing at "
        f"all to do with the query's subject, reply exactly: {NO_EVIDENCE_SENTINEL}\n"
        f"Stay under {word_budget} words.\n\n"
        f"PAGE CONTENT:\n{page_text}"
    )


@dataclass
class EvidenceResult:
    url: str
    evidence: Optional[str]
    token_budget: int
    token_estimate: int
    has_relevant_evidence: bool
    error: Optional[str] = None
    latency_seconds: Optional[float] = None


async def extract_evidence(
    query: str,
    url: str,
    title: Optional[str],
    page_text: str,
    token_budget: int,
    max_page_chars: int,
    llm,
    cited_for_query: bool = True,
) -> EvidenceResult:
    prompt = build_evidence_prompt(query, url, title, page_text[:max_page_chars], token_budget, cited_for_query)
    started = time.monotonic()
    # Output headroom above the budget: thinking models spend part of max_tokens
    # before answering. The budget itself is enforced on the returned text.
    response = await llm.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max(1024, token_budget * 2),
        temperature=0.0,
    )
    latency = round(time.monotonic() - started, 2)
    text = (response or "").strip()
    if not text:
        raise ValueError("empty evidence response")
    relevant = not is_no_evidence_reply(text)
    evidence = enforce_token_budget(text, token_budget) if relevant else None
    return EvidenceResult(
        url=url,
        evidence=evidence,
        token_budget=token_budget,
        token_estimate=estimate_tokens(evidence or ""),
        has_relevant_evidence=relevant,
        latency_seconds=latency,
    )


async def compress_query_sources(
    query: str,
    sources: Sequence[Dict[str, Optional[str]]],
    total_budget: int,
    per_source_cap: int,
    min_floor: int,
    batch_size: int,
    max_page_chars: int,
    llm=None,
    semaphore: Optional[asyncio.Semaphore] = None,
    cited_for_query: bool = True,
) -> List[EvidenceResult]:
    """
    Compress every source for one query. `sources` items need url, title and
    page_text (None when the crawl failed); results come back in the same order.

    The per-source budget is computed from the number of sources. At most
    `batch_size` calls are in flight at once, as a rolling window rather than
    fixed groups, so one slow call doesn't hold up the others. Pass a shared
    `semaphore` to apply one window across several queries. cited_for_query tells
    the model the observed engine cited these pages for the query. Calls run under
    asyncio.gather(return_exceptions=True): one failure never stops the rest.
    """
    if llm is None:
        from ..constants import get_model_for_role
        from ..llm_providers.factory import get_llm_provider

        llm = get_llm_provider(get_model_for_role("EVIDENCE_EXTRACTION"))

    budget = compute_per_source_token_budget(len(sources), total_budget, per_source_cap, min_floor)
    window = semaphore or asyncio.Semaphore(batch_size)

    async def compress(source: Dict[str, Optional[str]]) -> EvidenceResult:
        if not source.get("page_text"):
            raise ValueError(source.get("crawl_error") or "page content unavailable")
        async with window:
            return await extract_evidence(
                query, source["url"], source.get("title"), source["page_text"], budget, max_page_chars, llm,
                cited_for_query,
            )

    outcomes = await asyncio.gather(*(compress(s) for s in sources), return_exceptions=True)
    results: List[EvidenceResult] = []
    for source, outcome in zip(sources, outcomes):
        if isinstance(outcome, Exception):
            logger.warning(f"[stage1] evidence extraction failed for {source['url']}: {outcome}")
            outcome = EvidenceResult(source["url"], None, budget, 0, False, error=str(outcome)[:500])
        results.append(outcome)
    return results
