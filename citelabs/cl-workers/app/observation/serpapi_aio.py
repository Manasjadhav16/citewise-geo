"""
SerpApiAIOFetcher: Google AI Overview results via SerpApi.

Unlike the Gemini grounding fetcher (an approximate proxy), this fetcher records
Google AI Overview results as returned by SerpApi for a fixed country (gl),
language (hl) and device.

Flow per observation:
1. engine=google with gl/hl/device and no_cache=true.
2. If ai_overview holds only a page_token, call engine=google_ai_overview with
   that token immediately (it expires after about a minute), also no_cache=true.
3. Both responses are sanitized and kept with the observation (raw_responses).
4. parse_raw_responses() turns them into the answer and its citations:
   - citations are ai_overview.references, in order (references only);
   - the answer text is every text block's title/snippet, list items, nested
     blocks, and table cell text (from the table's `detailed` rows);
   - inline links inside the text (snippet_links) are recorded separately and
     are NOT citations;
   - a response without ai_overview is a valid "no AI Overview" outcome.

The same parse function is used live and to rebuild parsed fields from stored
raw responses (reparse_stored_response), so a stored observation can be
re-parsed without spending credits.

Strictness: a block type the parser does not explicitly handle, or a missing
required field, raises SerpApiFormatError. Live, that observation is recorded as
a parse_error (its credits still count) and the run continues; Stage 1 stops a
run after 3 consecutive parse errors. Nothing is guessed, because mis-parsed
citations would corrupt the measurements.

Known SerpApi limitation: table rows sometimes have fewer cells than the header
(cells that were links appear to be dropped). Such tables are flagged
(ragged_table). A brand that appears only in a dropped cell is not in the answer
text, so the mention rate can undercount.

Credits: each search request SerpApi completes counts as one credit (no_cache=true,
so free cached responses are never used). preflight() checks the account's
remaining searches before a run starts, and a shared budget stops further
requests once that number is reached.
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from ..geo_metrics import normalize_domain
from .fetchers import (
    OUTCOME_ANSWER,
    OUTCOME_NO_AI_ANSWER,
    OUTCOME_PARSE_ERROR,
    Observation,
    ObservationFetcher,
    ObservedCitation,
)

logger = logging.getLogger(__name__)

SEARCH_URL = "https://serpapi.com/search.json"
ACCOUNT_URL = "https://serpapi.com/account.json"
VALID_DEVICES = ("desktop", "mobile", "tablet")
# Worst case per observation: engine=google, then engine=google_ai_overview
MAX_CREDITS_PER_OBSERVATION = 2
# Text block types the parser understands; anything else is a parse error
KNOWN_BLOCK_TYPES = ("paragraph", "heading", "list", "table", "expandable")


class SerpApiError(RuntimeError):
    """SerpApi returned an error, or the request failed."""

    credits_used: int = 0  # credits already spent by the observation that failed
    raw_responses: Optional[List[Dict[str, Any]]] = None  # sanitized responses received before the failure


class SerpApiFormatError(SerpApiError):
    """The response does not have the structure this parser was written for."""


class SerpApiCreditsError(SerpApiError):
    """Not enough SerpApi credits for the planned run."""


# ==========================================
# Sanitizing for storage
# ==========================================

# Credentials, account identifiers, and links into the account's search archive
_DROP_KEYS = {
    "api_key", "apikey", "account_id", "account_email", "api_key_id",
    "json_endpoint", "raw_html_file", "prettify_html_file", "markdown_endpoint", "pixel_position_endpoint",
}


def sanitize_for_storage(data: Any, secrets: Tuple[str, ...] = ()) -> Any:
    """
    Copy of a SerpApi response that is safe to save: drops credentials, account
    identifiers and search-archive links, strips api_key query parameters from
    URLs, and replaces any secret string.
    """
    if isinstance(data, dict):
        return {k: sanitize_for_storage(v, secrets) for k, v in data.items() if k.lower() not in _DROP_KEYS}
    if isinstance(data, list):
        return [sanitize_for_storage(v, secrets) for v in data]
    if isinstance(data, str):
        value = data
        if value.startswith("http") and "api_key=" in value:
            parsed = urlparse(value)
            query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() != "api_key"]
            value = urlunparse(parsed._replace(query=urlencode(query)))
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    return data


# ==========================================
# Parsing
# ==========================================


@dataclass
class _TextAccumulator:
    texts: List[str] = field(default_factory=list)
    inline_links: List[Dict[str, str]] = field(default_factory=list)
    reference_indexes: set = field(default_factory=set)
    block_types: set = field(default_factory=set)
    tables_seen: int = 0
    ragged_rows: int = 0


def _fail(path: str, message: str):
    raise SerpApiFormatError(f"{path}: {message}")


def _common_fields(node: Dict[str, Any], path: str, acc: _TextAccumulator) -> None:
    """title/snippet text, inline links and reference indexes of a block or list item."""
    for key in ("title", "snippet"):
        value = node.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            _fail(path, f"'{key}' is a {type(value).__name__}, expected a string")
        if value.strip():
            acc.texts.append(value.strip())
    links = node.get("snippet_links")
    if links is not None:
        if not isinstance(links, list):
            _fail(path, "'snippet_links' is not a list")
        for j, link in enumerate(links):
            if not isinstance(link, dict) or not isinstance(link.get("link"), str):
                _fail(f"{path}.snippet_links[{j}]", "inline link without a 'link' string")
            acc.inline_links.append({"text": link.get("text") or "", "link": link["link"]})
    indexes = node.get("reference_indexes")
    if indexes is not None:
        if not isinstance(indexes, list):
            _fail(path, "'reference_indexes' is not a list")
        acc.reference_indexes.update(i for i in indexes if isinstance(i, int))


def _parse_table(block: Dict[str, Any], path: str, acc: _TextAccumulator) -> None:
    detailed = block.get("detailed")
    if not isinstance(detailed, list) or not detailed:
        _fail(path, "table block has no 'detailed' rows")
    rows: List[List[str]] = []
    for r, row in enumerate(detailed):
        if not isinstance(row, list):
            _fail(f"{path}.detailed[{r}]", "table row is not a list")
        cells = []
        for c, cell in enumerate(row):
            if not isinstance(cell, dict) or not isinstance(cell.get("snippet"), str):
                _fail(f"{path}.detailed[{r}][{c}]", "table cell without a 'snippet' string")
            _common_fields({"snippet_links": cell.get("snippet_links")}, f"{path}.detailed[{r}][{c}]", acc)
            cells.append(cell["snippet"].strip())
        rows.append(cells)
    header_width = len(rows[0])
    acc.tables_seen += 1
    acc.ragged_rows += sum(1 for cells in rows[1:] if len(cells) < header_width)
    for cells in rows:
        line = " | ".join(cell for cell in cells if cell)
        if line:
            acc.texts.append(line)


def _parse_items(items: Any, path: str, acc: _TextAccumulator) -> None:
    """Items of a list block: plain items (no type), or typed blocks."""
    if not isinstance(items, list):
        _fail(path, "'list' is not a list")
    for i, item in enumerate(items):
        item_path = f"{path}[{i}]"
        if not isinstance(item, dict):
            _fail(item_path, f"list item is a {type(item).__name__}, expected an object")
        if "type" in item:
            _parse_blocks([item], item_path, acc)
            continue
        _common_fields(item, item_path, acc)
        if "list" in item:
            _parse_items(item["list"], f"{item_path}.list", acc)
        if "text_blocks" in item:
            _parse_blocks(item["text_blocks"], f"{item_path}.text_blocks", acc)


def _parse_blocks(blocks: Any, path: str, acc: _TextAccumulator) -> None:
    if not isinstance(blocks, list):
        _fail(path, "text_blocks is not a list")
    for i, block in enumerate(blocks):
        block_path = f"{path}[{i}]"
        if not isinstance(block, dict):
            _fail(block_path, f"text block is a {type(block).__name__}, expected an object")
        block_type = block.get("type")
        if block_type not in KNOWN_BLOCK_TYPES:
            _fail(block_path, f"unknown text block type {block_type!r} (handled: {', '.join(KNOWN_BLOCK_TYPES)})")
        acc.block_types.add(block_type)
        _common_fields(block, block_path, acc)
        if block_type == "table":
            _parse_table(block, block_path, acc)
        if "list" in block:
            _parse_items(block["list"], f"{block_path}.list", acc)
        if "text_blocks" in block:
            _parse_blocks(block["text_blocks"], f"{block_path}.text_blocks", acc)


def parse_ai_overview(ai_overview: Any) -> Tuple[str, List[ObservedCitation], Dict[str, Any]]:
    """
    (answer text, citations in reference order, parse details) from a complete
    ai_overview object. Raises SerpApiFormatError on any unexpected structure.
    """
    if not isinstance(ai_overview, dict):
        raise SerpApiFormatError(f"ai_overview is a {type(ai_overview).__name__}, expected an object")
    if "error" in ai_overview:
        raise SerpApiError(f"ai_overview error: {ai_overview['error']}")
    blocks = ai_overview.get("text_blocks")
    references = ai_overview.get("references")
    if not isinstance(blocks, list):
        raise SerpApiFormatError(f"ai_overview has no text_blocks list (keys: {sorted(ai_overview)})")
    if not isinstance(references, list):
        raise SerpApiFormatError(f"ai_overview has no references list (keys: {sorted(ai_overview)})")

    acc = _TextAccumulator()
    _parse_blocks(blocks, "text_blocks", acc)

    citations = []
    for order, ref in enumerate(references, start=1):
        if not isinstance(ref, dict) or not isinstance(ref.get("link"), str) or not ref["link"].startswith("http"):
            raise SerpApiFormatError(f"reference #{order} has no http link: {json.dumps(ref)[:200]}")
        citations.append(
            ObservedCitation(
                raw_uri=ref["link"],
                url=ref["link"],
                domain=normalize_domain(ref["link"]),
                title=ref.get("title"),
                order=order,
                # Every reference is a citation of the overview; which ones the text
                # points at (reference_indexes) is recorded separately in the details
                cited_in_answer=True,
                is_web_source=True,
            )
        )
    details = {
        "text_block_count": len(blocks),
        "text_block_types": sorted(acc.block_types),
        "reference_count": len(references),
        "reference_indexes": [r.get("index") for r in references],
        "reference_indexes_used_in_text": sorted(acc.reference_indexes),
        "tables_seen": acc.tables_seen,
        "ragged_table": acc.ragged_rows > 0,  # some row has fewer cells than its header
        "ragged_rows": acc.ragged_rows,
        "inline_links": acc.inline_links,  # snippet_links: NOT citations
    }
    return "\n".join(acc.texts), citations, details


@dataclass
class ParsedResponses:
    outcome: str  # answer | no_aio
    answer_text: str
    citations: List[ObservedCitation]
    details: Dict[str, Any]


def _is_page_token_only(ai_overview: Any) -> bool:
    return isinstance(ai_overview, dict) and "page_token" in ai_overview and "text_blocks" not in ai_overview


def parse_raw_responses(raw_responses: Sequence[Dict[str, Any]]) -> ParsedResponses:
    """
    Parse one observation from its raw responses, in call order: the engine=google
    response, then the engine=google_ai_overview response when a page token was
    followed. Used live and for re-parsing stored responses.
    """
    if not raw_responses:
        raise SerpApiFormatError("no raw responses")
    ai_overview = raw_responses[0].get("ai_overview")
    if ai_overview is None:
        return ParsedResponses(OUTCOME_NO_AI_ANSWER, "", [], {"tables_seen": 0, "ragged_table": False, "ragged_rows": 0, "inline_links": []})
    if _is_page_token_only(ai_overview):
        if len(raw_responses) < 2:
            raise SerpApiFormatError("ai_overview has only a page_token and there is no google_ai_overview response")
        if "ai_overview" not in raw_responses[1]:
            raise SerpApiFormatError(f"google_ai_overview response has no ai_overview (keys: {sorted(raw_responses[1])})")
        ai_overview = raw_responses[1]["ai_overview"]
    text, citations, details = parse_ai_overview(ai_overview)
    return ParsedResponses(OUTCOME_ANSWER, text, citations, details)


def reparse_stored_response(raw_responses: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Rebuild an observation's parsed fields from its stored raw responses (no credits)."""
    try:
        parsed = parse_raw_responses(raw_responses)
    except SerpApiFormatError as exc:
        return {"outcome": OUTCOME_PARSE_ERROR, "parse_error": str(exc), "answer_text": "", "citations": [], "details": {}}
    return {
        "outcome": parsed.outcome,
        "parse_error": None,
        "answer_text": parsed.answer_text,
        "citations": [asdict(c) for c in parsed.citations],
        "details": parsed.details,
    }


# ==========================================
# Fetching
# ==========================================


class _CreditBudget:
    """Shared across a run's concurrent observations: never spend past the account's remaining searches."""

    def __init__(self, remaining: Optional[int]):
        self.remaining = remaining
        self.used = 0
        self._lock = asyncio.Lock()

    async def take(self) -> None:
        async with self._lock:
            if self.remaining is not None and self.used >= self.remaining:
                raise SerpApiCreditsError(
                    f"SerpApi credit budget reached ({self.used} used of {self.remaining} remaining at start); request not sent"
                )
            self.used += 1

    async def refund(self) -> None:
        async with self._lock:
            self.used -= 1


class SerpApiAIOFetcher(ObservationFetcher):
    """Google AI Overview results as returned by SerpApi, for fixed gl/hl/device."""

    name = "serpapi_aio"
    reports_activation = True

    def __init__(
        self,
        gl: str = "in",
        hl: str = "en",
        device: str = "desktop",
        api_key: Optional[str] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout: float = 60.0,
        raw_dump_dir: Optional[str] = None,
    ):
        if device not in VALID_DEVICES:
            raise ValueError(f"device must be one of {VALID_DEVICES}")
        self.gl, self.hl, self.device = gl, hl, device
        self.api_key = api_key or os.getenv("SERPAPI_KEY", "")
        if not self.api_key:
            raise ValueError("SERPAPI_KEY is not set")
        self._transport = transport
        self._timeout = timeout
        self._budget = _CreditBudget(None)
        # Optional: also write each sanitized raw response to files here
        self.raw_dump_dir = raw_dump_dir or os.getenv("STAGE1_SERPAPI_RAW_DIR") or None

    @property
    def settings(self) -> Dict[str, Any]:
        return {"fetcher": self.name, "gl": self.gl, "hl": self.hl, "device": self.device, "no_cache": True}

    @property
    def data_disclaimer(self) -> str:
        return (
            f"Google AI Overview results as returned by SerpApi for country (gl) = {self.gl}, "
            f"language (hl) = {self.hl}, device = {self.device}."
        )

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self._timeout, transport=self._transport)

    async def account(self) -> Dict[str, Any]:
        """SerpApi account status (does not consume search credits)."""
        async with self._client() as client:
            response = await client.get(ACCOUNT_URL, params={"api_key": self.api_key})
        try:
            data = response.json()
        except ValueError:
            raise SerpApiError(f"account endpoint returned HTTP {response.status_code} with a non-JSON body")
        if response.status_code != 200 or "error" in data:
            raise SerpApiError(f"account endpoint: HTTP {response.status_code}: {data.get('error')}")
        return data

    async def preflight(self, planned_observations: int) -> Dict[str, Any]:
        """
        Refuse the run if its worst case (2 credits per observation) exceeds the
        account's remaining searches; otherwise cap the run's spending at that number.
        """
        account = await self.account()
        remaining = account.get("total_searches_left")
        if not isinstance(remaining, int):
            raise SerpApiFormatError(f"account response has no integer total_searches_left (keys: {sorted(account)})")
        worst_case = planned_observations * MAX_CREDITS_PER_OBSERVATION
        if worst_case > remaining:
            raise SerpApiCreditsError(
                f"Run needs up to {worst_case} SerpApi credits ({planned_observations} observations x "
                f"{MAX_CREDITS_PER_OBSERVATION}) but only {remaining} remain"
            )
        self._budget = _CreditBudget(remaining)
        return {"serpapi_searches_left_before": remaining, "serpapi_worst_case_credits": worst_case}

    async def _search(self, client: httpx.AsyncClient, params: Dict[str, str]) -> Dict[str, Any]:
        """One billable request. A credit is counted only when SerpApi reports success."""
        await self._budget.take()
        try:
            response = await client.get(SEARCH_URL, params={**params, "no_cache": "true", "api_key": self.api_key})
        except httpx.HTTPError as exc:
            await self._budget.refund()
            raise SerpApiError(f"request to SerpApi failed: {type(exc).__name__}: {exc}")
        try:
            data = response.json()
        except ValueError:
            await self._budget.refund()
            raise SerpApiError(f"HTTP {response.status_code} with a non-JSON body")
        status = (data.get("search_metadata") or {}).get("status")
        if response.status_code != 200 or data.get("error") or status != "Success":
            await self._budget.refund()
            raise SerpApiError(f"HTTP {response.status_code}, status {status!r}: {data.get('error') or 'no error message'}")
        return data

    def _dump(self, engine: str, data: Dict[str, Any]) -> None:
        if not self.raw_dump_dir:
            return
        try:
            os.makedirs(self.raw_dump_dir, exist_ok=True)
            search_id = (data.get("search_metadata") or {}).get("id", str(int(time.time() * 1000)))
            with open(os.path.join(self.raw_dump_dir, f"serpapi_{engine}_{search_id}.json"), "w") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning(f"[stage1] could not write raw SerpApi response: {exc}")

    async def observe(self, query: str) -> Observation:
        meta: Dict[str, Any] = {**self.settings, "engines_called": [], "search_ids": [], "page_token_used": False}
        raws: List[Dict[str, Any]] = []
        try:
            return await self._observe(query, meta, raws)
        except SerpApiError as exc:
            exc.credits_used = len(meta["engines_called"])
            exc.raw_responses = raws
            raise

    async def _call(self, client: httpx.AsyncClient, params: Dict[str, str], meta: Dict[str, Any], raws: List[Dict[str, Any]]) -> Dict[str, Any]:
        data = await self._search(client, params)
        meta["engines_called"].append(params["engine"])
        meta["search_ids"].append((data.get("search_metadata") or {}).get("id"))
        clean = sanitize_for_storage(data, (self.api_key,))
        raws.append(clean)
        self._dump(params["engine"], clean)
        return data

    async def _observe(self, query: str, meta: Dict[str, Any], raws: List[Dict[str, Any]]) -> Observation:
        started = time.monotonic()
        async with self._client() as client:
            data = await self._call(
                client, {"engine": "google", "q": query, "gl": self.gl, "hl": self.hl, "device": self.device}, meta, raws
            )
            ai_overview = data.get("ai_overview")
            if _is_page_token_only(ai_overview):
                # Only a token: fetch the overview now, before the token expires
                meta["page_token_used"] = True
                await self._call(client, {"engine": "google_ai_overview", "page_token": ai_overview["page_token"]}, meta, raws)

        model = f"google_ai_overview via serpapi ({self.gl}/{self.hl}/{self.device})"
        credits = len(meta["engines_called"])
        try:
            parsed = parse_raw_responses(raws)
        except SerpApiFormatError as exc:
            # Credits were spent: keep the observation (and its raw responses) as a parse error
            logger.warning(f"[stage1] SerpApi parse error for {query!r}: {exc}")
            return Observation(
                answer_text="",
                model=model,
                outcome=OUTCOME_PARSE_ERROR,
                credits_used=credits,
                fetch_meta={**meta, "parse_error": str(exc)},
                raw_responses=raws,
                latency_seconds=time.monotonic() - started,
            )
        return Observation(
            answer_text=parsed.answer_text,
            model=model,
            citations=parsed.citations,
            outcome=parsed.outcome,
            credits_used=credits,
            fetch_meta={**meta, **parsed.details},
            raw_responses=raws,
            latency_seconds=time.monotonic() - started,
        )
