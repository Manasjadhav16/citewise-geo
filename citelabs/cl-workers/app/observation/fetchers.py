"""
Observation fetchers: where Stage 1's real-world citation data comes from.

ObservationFetcher is the pluggable interface, so the data source can be swapped
if a better one becomes available (for example a licensed AI Overview dataset).

GeminiGroundingFetcher is the current implementation. IMPORTANT: it is NOT real
Google AI Overview data. It records Gemini's own search-and-cite behaviour with
Google Search grounding, which uses the same underlying Google Search index that
AI Overviews draw from. That makes it an approximate proxy for AI Overview
citation behaviour, not an equivalent: there is no official public AI Overview API.
"""
import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from ..geo_metrics import normalize_domain

logger = logging.getLogger(__name__)

# Gemini returns source links as redirects through this host
GROUNDING_REDIRECT_HOST = "vertexaisearch.cloud.google.com"
_DOMAIN_LIKE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$", re.IGNORECASE)


@dataclass
class ObservedCitation:
    """One source attached to one observed answer."""

    raw_uri: str  # as returned by the fetcher (may be a redirect link)
    url: Optional[str]  # real source URL, None if it could not be resolved
    domain: str
    title: Optional[str]
    order: int  # 1-based: sources the answer cites, by first appearance, then the rest
    cited_in_answer: bool  # the answer text references this source
    is_web_source: bool  # False for search-engine utility results (e.g. "current time")


@dataclass
class Observation:
    """One answer to one query from the observed engine."""

    answer_text: str
    model: str
    citations: List[ObservedCitation] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    # "answer", or OUTCOME_NO_AI_ANSWER when the engine showed no AI answer for the
    # query (a valid observation, not a failure)
    outcome: str = "answer"
    credits_used: Optional[int] = None  # paid API credits this observation consumed
    fetch_meta: Dict[str, Any] = field(default_factory=dict)  # fetcher-specific provenance
    # Sanitized raw API responses, in call order, so parsed fields can be rebuilt offline
    raw_responses: List[Dict[str, Any]] = field(default_factory=list)


OUTCOME_ANSWER = "answer"
OUTCOME_NO_AI_ANSWER = "no_aio"
# The engine answered but the response could not be parsed; credits were spent
OUTCOME_PARSE_ERROR = "parse_error"


class ObservationFetcher(ABC):
    """Source of real-world (or proxy) AI answers with their cited sources."""

    name: str = "base"
    # Shown next to any Stage 1 result so its provenance is never misread
    data_disclaimer: str = ""
    # True when "no AI answer" is a possible outcome, so an activation rate is meaningful
    reports_activation: bool = False

    @property
    def settings(self) -> Dict[str, Any]:
        """Settings that define what was observed (stored with every run)."""
        return {}

    async def preflight(self, planned_observations: int) -> Dict[str, Any]:
        """Check the run can be afforded before it starts; raise to refuse it."""
        return {}

    @abstractmethod
    async def observe(self, query: str) -> Observation:
        """Ask one query once and return the answer with its cited sources."""


def is_utility_result(uri: str) -> bool:
    """Search-engine utility results (time, weather, calculators) are not web sources."""
    parsed = urlparse(uri or "")
    host = normalize_domain(uri or "")
    return host in ("google.com",) and parsed.path.startswith("/search")


def domain_from_title(title: Optional[str]) -> str:
    """Gemini usually titles grounding chunks with the source's domain."""
    candidate = (title or "").strip().lower()
    return normalize_domain(candidate) if _DOMAIN_LIKE.match(candidate) else ""


def build_citations(
    chunk_uris: List[str],
    chunk_titles: List[Optional[str]],
    cited_chunk_order: List[int],
    resolved: Dict[str, Optional[str]],
) -> List[ObservedCitation]:
    """
    Turn raw grounding chunks into ordered citations. Chunks the answer cites come
    first, in order of first appearance; chunks it never references follow in
    retrieval order.
    """
    order = list(cited_chunk_order) + [i for i in range(len(chunk_uris)) if i not in cited_chunk_order]
    cited = set(cited_chunk_order)
    citations: List[ObservedCitation] = []
    for position, index in enumerate(order, start=1):
        raw_uri, title = chunk_uris[index], chunk_titles[index]
        url = resolved.get(raw_uri) if GROUNDING_REDIRECT_HOST in raw_uri else (raw_uri or None)
        citations.append(
            ObservedCitation(
                raw_uri=raw_uri,
                url=url,
                domain=normalize_domain(url) if url else domain_from_title(title),
                title=title,
                order=position,
                cited_in_answer=index in cited,
                is_web_source=bool(raw_uri) and not is_utility_result(url or raw_uri),
            )
        )
    return citations


async def resolve_redirects(uris: List[str], timeout: float = 10.0, concurrency: int = 10) -> Dict[str, Optional[str]]:
    """
    Resolve grounding redirect links to real URLs by reading the Location header
    of a single HEAD request (the target page itself is not fetched).
    Unresolvable links map to None.
    """
    redirect_uris = list(dict.fromkeys(u for u in uris if GROUNDING_REDIRECT_HOST in u))
    semaphore = asyncio.Semaphore(concurrency)
    resolved: Dict[str, Optional[str]] = {}

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:

        async def resolve(uri: str) -> None:
            async with semaphore:
                try:
                    response = await client.head(uri)
                    location = response.headers.get("location")
                    resolved[uri] = location if response.is_redirect and location else None
                except Exception as exc:
                    logger.warning(f"[stage1] could not resolve grounding redirect: {exc}")
                    resolved[uri] = None

        await asyncio.gather(*(resolve(uri) for uri in redirect_uris))
    return resolved


class GeminiGroundingFetcher(ObservationFetcher):
    """
    Gemini + Google Search grounding: an approximate proxy for AI Overview
    citations, not real AI Overview data (see module docstring).
    """

    name = "gemini_grounding"
    data_disclaimer = (
        "Observed via Gemini with Google Search grounding: an approximate proxy for "
        "Google AI Overview citations, not real AI Overview data."
    )

    def __init__(self, model: str, provider=None):
        self.model = model
        self._provider = provider

    def _get_provider(self):
        if self._provider is None:
            from ..constants import get_model_for_role
            from ..llm_providers.factory import get_llm_provider

            self._provider = get_llm_provider(get_model_for_role("GROUNDED_OBSERVATION"))
        return self._provider

    async def observe(self, query: str) -> Observation:
        started = time.monotonic()
        response = await self._get_provider().grounded_generate(query, model=self.model)
        uris = [chunk.uri for chunk in response.chunks]
        resolved = await resolve_redirects(uris)
        return Observation(
            answer_text=response.text,
            model=response.model,
            citations=build_citations(
                uris, [chunk.title for chunk in response.chunks], response.cited_chunk_order, resolved
            ),
            search_queries=response.search_queries,
            latency_seconds=time.monotonic() - started,
        )


def get_observation_fetcher(name: str, model: str, serpapi_settings: Optional[Dict[str, str]] = None) -> ObservationFetcher:
    """Fetcher registry, selected by STAGE1_FETCHER."""
    if name == GeminiGroundingFetcher.name:
        return GeminiGroundingFetcher(model=model)
    from .serpapi_aio import SerpApiAIOFetcher

    if name == SerpApiAIOFetcher.name:
        return SerpApiAIOFetcher(**(serpapi_settings or {}))
    raise ValueError(f"Unknown Stage 1 observation fetcher: {name}")
