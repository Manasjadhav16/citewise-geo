"""
Competitive gap optimization (v1): what the sources Google's AI Overview cites
say about a query that the target's own pages do not.

For each query of a completed Stage 1 observation:
1. Sources: the TOP_CITED most-cited non-target pages of the query's union pool
   (by appearances, then best citation position), plus the target's own pages
   seen in the observation and the submitted page.
2. Text: already-crawled page text where the backend has it, else a text-only
   crawl. Each cited source's evidence is capped by the Stage 1 per-source token
   budget, b = max(b_min, min(cap, B / n)), with n = cited + target pages.
3. Evidence: one text-generation call per cited source returns structured items
   (fact, figure, definition, product_detail, comparison, claim), each with a
   verbatim quote. Items whose quote is not found in the page are dropped.
4. Claims: one call groups the items into claims; the code (not the model)
   counts the distinct sources behind each claim and ranks them.
5. Coverage: one call judges the most common claims against excerpts of the
   target pages (covered / partly / missing, with a target quote) and drafts a
   content addition for each gap. A "covered" verdict needs a quote that is found
   on a target page; suggested figures not on the target page are flagged.

Text generation only (no embedding calls). Calls go straight to the configured
Gemini model with capped exponential backoff on 503/429, and every attempt is
counted in the report.
"""
import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .geo_metrics import compute_per_source_token_budget, normalize_domain
from .observation.evidence import CHARS_PER_TOKEN, estimate_tokens
from .stage2_seed import _page_key as page_key

logger = logging.getLogger(__name__)

GAP_VERSION = "gap_v1"
TOP_CITED = 5
MAX_JUDGED_CLAIMS = 6
TOTAL_TOKEN_BUDGET = 20000  # B, per query (Stage 1 default)
PER_SOURCE_TOKEN_CAP = 1000
MIN_SOURCE_TOKENS = 200  # b_min
MAX_PAGE_CHARS = 40000  # page text passed to one extraction call
ITEM_TYPES = ("fact", "figure", "definition", "product_detail", "comparison", "claim")
STATUSES = ("covered", "partly", "missing")
TARGET_GROUP = "target"


# ---------- source selection ----------

def select_sources(
    sources: Sequence[Mapping[str, Any]], sandbox_url: str, top_n: int = TOP_CITED
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    (cited, targets) for one query. cited: the top_n non-target pages with a URL,
    by appearances, stability, best citation position, then URL. targets: the
    submitted page first, then the target's pages from the pool by appearances,
    one per page.
    """
    target_domain = normalize_domain(sandbox_url)

    def is_target(s: Mapping[str, Any]) -> bool:
        domain = normalize_domain(s.get("domain") or s.get("url") or "")
        return s.get("competitor_group") == TARGET_GROUP or domain == target_domain or domain.endswith("." + target_domain)

    candidates = [dict(s) for s in sources if s.get("url") and not is_target(s)]
    candidates.sort(key=lambda s: (
        -(s.get("appearances") or 0), -(s.get("stability") or 0.0),
        s.get("best_position") if s.get("best_position") is not None else 10**6, s["url"],
    ))
    cited = candidates[:top_n]

    targets: List[Dict[str, Any]] = [{"url": sandbox_url, "submitted": True, "appearances": None}]
    seen = {page_key(sandbox_url)}
    for s in sorted((s for s in sources if s.get("url") and is_target(s)), key=lambda s: -(s.get("appearances") or 0)):
        key = page_key(s["url"])
        if key in seen:
            targets[0]["appearances"] = targets[0]["appearances"] or s.get("appearances")
            continue
        seen.add(key)
        targets.append({**dict(s), "submitted": False})
    return cited, targets


def source_token_budget(n_sources: int) -> int:
    return compute_per_source_token_budget(n_sources, TOTAL_TOKEN_BUDGET, PER_SOURCE_TOKEN_CAP, MIN_SOURCE_TOKENS)


# ---------- quote verification ----------

_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "₹": "rs "})


def normalize_for_match(text: str) -> str:
    """Lowercase, unify quotes/dashes, and collapse everything but letters and digits to single spaces."""
    text = (text or "").translate(_QUOTE_CHARS).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def quote_in_text(quote: str, text: str, min_chars: int = 8) -> bool:
    """True when the quote (ignoring case, punctuation and spacing) appears in the text."""
    q = normalize_for_match(quote)
    return len(q) >= min_chars and q in normalize_for_match(text)


def numbers_in(text: str) -> List[str]:
    """Numbers as written (digits with , or . inside), for checking suggested figures."""
    return re.findall(r"\d+(?:[.,]\d+)*", text or "")


# ---------- model replies ----------

def parse_json_reply(text: str) -> Any:
    """JSON from a model reply, tolerating ```json fences and text around one object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def reply_entries(reply: Any, key: str) -> Tuple[List[Any], str]:
    """
    The list a reply holds under `key`, also accepting a bare list or an object
    whose only list value has another name. Returns (entries, shape) where shape
    describes what came back, for the report.
    """
    if isinstance(reply, list):
        return reply, "list"
    if isinstance(reply, Mapping):
        if isinstance(reply.get(key), list):
            return reply[key], f"object:{key}"
        lists = [k for k, v in reply.items() if isinstance(v, list)]
        if len(lists) == 1:
            return reply[lists[0]], f"object:{lists[0]}"
        return [], "object:" + (",".join(sorted(map(str, reply))) or "empty")
    return [], type(reply).__name__


@dataclass
class EvidenceItem:
    item_id: str  # "S<source>.<n>"
    source_label: str  # "S<source>"
    type: str
    text: str
    quote: str


def parse_evidence_items(
    reply: Any, source_label: str, page_text: str, token_budget: int
) -> Tuple[List[EvidenceItem], Dict[str, int]]:
    """
    Validated evidence items from an extraction reply: known type, non-empty text,
    and a quote found in the page. Items are kept in order until text + quote
    reach the source's token budget. Returns (items, counts of what was dropped).
    """
    raw, _ = reply_entries(reply, "items")
    dropped = {"unverified_quote": 0, "invalid": 0, "over_budget": 0}
    items: List[EvidenceItem] = []
    used = 0
    for entry in raw:
        if not isinstance(entry, Mapping):
            dropped["invalid"] += 1
            continue
        kind = str(entry.get("type") or "").strip().lower()
        text = str(entry.get("text") or "").strip()
        quote = str(entry.get("quote") or "").strip()
        if kind not in ITEM_TYPES or not text or not quote:
            dropped["invalid"] += 1
            continue
        if not quote_in_text(quote, page_text):
            dropped["unverified_quote"] += 1
            continue
        cost = estimate_tokens(text) + estimate_tokens(quote)
        if used + cost > token_budget:
            dropped["over_budget"] += 1
            continue
        used += cost
        items.append(EvidenceItem(f"{source_label}.{len(items) + 1}", source_label, kind, text, quote))
    return items, dropped


# ---------- claim counting ----------

@dataclass
class Claim:
    claim_id: str  # "C<n>"
    claim: str
    item_ids: List[str]
    sources: List[str]  # source labels, in first-seen order
    source_count: int
    weight: int  # summed appearances of the supporting sources (tie-break)
    quotes: List[Dict[str, str]] = field(default_factory=list)


def count_claims(
    raw_claims: Any,
    items: Sequence[EvidenceItem],
    appearances: Mapping[str, int],
) -> List[Claim]:
    """
    Claims from the grouping reply, counted by the code. Unknown item ids are
    ignored, an item belongs to the first claim that lists it, items no claim
    lists become single-item claims (their own text), and a claim's count is the
    number of distinct sources behind its items. Ranked by source count, then the
    sources' summed appearances, then first appearance.
    """
    by_id = {i.item_id: i for i in items}
    taken: set = set()
    groups: List[Tuple[str, List[str]]] = []
    entries, _ = reply_entries(raw_claims, "claims")
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        text = str(entry.get("claim") or "").strip()
        ids = [str(i).strip() for i in entry.get("item_ids") or [] if str(i).strip() in by_id]
        ids = [i for i in dict.fromkeys(ids) if i not in taken]
        if not text or not ids:
            continue
        taken.update(ids)
        groups.append((text, ids))
    for item in items:  # nothing the model left out is lost
        if item.item_id not in taken:
            groups.append((item.text, [item.item_id]))

    claims = []
    for order, (text, ids) in enumerate(groups):
        sources = list(dict.fromkeys(by_id[i].source_label for i in ids))
        claims.append((order, Claim(
            claim_id="",
            claim=text,
            item_ids=ids,
            sources=sources,
            source_count=len(sources),
            weight=sum(appearances.get(s, 0) for s in sources),
            quotes=[{"source": by_id[i].source_label, "quote": by_id[i].quote} for i in ids],
        )))
    claims.sort(key=lambda pair: (-pair[1].source_count, -pair[1].weight, pair[0]))
    ranked = [c for _, c in claims]
    for n, claim in enumerate(ranked, start=1):
        claim.claim_id = f"C{n}"
    return ranked


# ---------- target excerpts ----------

_STOPWORDS = set(
    "a an the and or of for to in on at by with from is are be as it its this that what which how do does "
    "can your you our we their they there per vs than into about over under more most best top".split()
)


def _terms(text: str) -> set:
    return {t for t in normalize_for_match(text).split() if t not in _STOPWORDS and (len(t) > 2 or t.isdigit())}


def split_passages(text: str, max_chars: int = 600) -> List[str]:
    """Paragraphs (by line), with long ones split at sentence ends into pieces of at most ~max_chars."""
    passages: List[str] = []
    for line in re.split(r"\n+", text or ""):
        line = line.strip()
        if not line:
            continue
        while len(line) > max_chars:
            cut = max(line.rfind(". ", 0, max_chars), line.rfind(" ", 0, max_chars))
            cut = cut + 1 if cut > max_chars // 2 else max_chars
            passages.append(line[:cut].strip())
            line = line[cut:].strip()
        if line:
            passages.append(line)
    return passages


def select_excerpt(page_text: str, query: str, claims: Sequence[Claim], token_budget: int) -> str:
    """
    The page's passages most relevant to the query and claims (term overlap, no
    embeddings), up to token_budget, kept in page order.
    """
    wanted = _terms(query)
    for claim in claims:
        wanted |= _terms(claim.claim)
    scored = []
    for index, passage in enumerate(split_passages(page_text)):
        overlap = len(_terms(passage) & wanted)
        if overlap:
            scored.append((overlap, index, passage))
    scored.sort(key=lambda s: (-s[0], s[1]))
    chosen, used = [], 0
    for _, index, passage in scored:
        cost = estimate_tokens(passage)
        if used + cost > token_budget:
            continue
        chosen.append((index, passage))
        used += cost
    return "\n".join(p for _, p in sorted(chosen))


# ---------- target pages ----------

@dataclass
class PageText:
    """A target page's text; main/chrome are set when the page was crawled with its sections."""

    text: str
    main: Optional[str] = None  # without header, nav, footer and aside
    chrome: Optional[str] = None  # header, nav, footer and aside only


def as_page(value: Union[str, PageText]) -> PageText:
    return value if isinstance(value, PageText) else PageText(value)


# ---------- suggestion honesty ----------

FIGURE_PLACEHOLDER = "[add your real figure]"
# Longest first, so "industry-leading" is removed before "leading"
PROMOTIONAL_TERMS = (
    "industry-leading", "industry leading", "market-leading", "market leading", "best-in-class", "best in class",
    "world-class", "world class", "state-of-the-art", "cutting-edge", "number one", "no. 1", "#1", "most trusted",
    "most reliable", "unmatched", "unbeatable", "unparalleled", "unrivalled", "unrivaled", "revolutionary",
    "seamlessly", "seamless", "fastest", "leading", "best", "top",
)


# Ordinary phrases that contain a listed word without promoting anything
_NOT_PROMOTIONAL_AFTER = {"best": r"(?!\s+practices?\b)", "leading": r"(?!\s+(?:to|up)\b)", "top": r"(?!\s+of\b)"}
_NOT_PROMOTIONAL_BEFORE = {"top": r"(?<!\bon\s)"}


def _term_pattern(term: str) -> "re.Pattern":
    return re.compile(
        _NOT_PROMOTIONAL_BEFORE.get(term, "") + r"(?<![\w#-])" + re.escape(term) + r"(?![\w-])"
        + _NOT_PROMOTIONAL_AFTER.get(term, ""),
        re.IGNORECASE,
    )


_TERM_PATTERNS = [(t, _term_pattern(t)) for t in PROMOTIONAL_TERMS]


_REMOVED = "\x00"  # marks where promotional wording was taken out


def _needs_an(word: str) -> bool:
    """By sound: "an" before a vowel, and before acronyms read letter by letter from A, E, F, H, I, L, M, N, O, R, S, X."""
    if len(word) > 1 and word.isupper():
        return word[0] in "AEFHILMNORSX"
    return word[:1].lower() in "aeiou"


def _fix_articles(text: str) -> str:
    """Re-choose "a"/"an" only where the following word changed because wording was removed."""
    def choose(match: "re.Match") -> str:
        article, word = match.group(1), match.group(2)
        wanted = "an" if _needs_an(word) else "a"
        return (wanted.capitalize() if article[:1].isupper() else wanted) + " " + word
    text = re.sub(r"\b([Aa]n?)\s*" + _REMOVED + r"[\s" + _REMOVED + r"]*(\w+)", choose, text)
    return text.replace(_REMOVED, "")


def check_suggestion(
    suggestion: str, target_text: str, competitor_names: Sequence[str] = ()
) -> Tuple[str, List[str]]:
    """
    Post-check for a suggested content addition. Promotional wording (superlatives
    such as "industry-leading", "best", "fastest") is removed unless a target page
    uses it; figures that are not on a target page become "[add your real figure]";
    naming a cited competitor is flagged. Returns (checked suggestion, flags).
    """
    flags: List[str] = []
    text = suggestion
    target_norm = " " + normalize_for_match(target_text) + " "
    removed = []
    for term, pattern in _TERM_PATTERNS:
        if " " + normalize_for_match(term) + " " in target_norm or not pattern.search(text):
            continue
        text = pattern.sub(_REMOVED, text)
        removed.append(term)
    if removed:
        flags.append("promotional_wording_removed:" + ",".join(removed))
    target_numbers = set(numbers_in(target_text))
    unknown = [n for n in numbers_in(text) if n not in target_numbers]
    if unknown:
        flags.append("suggestion_figures_not_on_target_page:" + ",".join(dict.fromkeys(unknown)))
        text = re.sub(r"(?:Rs\.?\s*|INR\s*|\u20b9\s*)?\d+(?:[.,]\d+)*\s*%?",
                      lambda m: m.group(0) if (numbers_in(m.group(0)) or [""])[0] in target_numbers else FIGURE_PLACEHOLDER,
                      text)
    named = [c for c in competitor_names if re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", text, re.IGNORECASE)]
    if named:
        flags.append("mentions_cited_competitor:" + ",".join(dict.fromkeys(named)))
    if removed or unknown:
        # commas that only separated removed adjectives ("the best, seamless checkout")
        text = re.sub(_REMOVED + r"\s*,", _REMOVED, re.sub(r",\s*" + _REMOVED, _REMOVED, text))
        text = _fix_articles(text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"[ \t]{2,}", " ", text))
        text = re.sub(r",\s*,", ",", re.sub(r"\(\s*\)", "", text)).strip()
    return text, flags


# ---------- quote quality ----------

MIN_QUOTE_WORDS = 8
MIN_QUOTE_CHARS = 40
_MENU_SEPARATORS = "|\u2022\u00b7\u00bb\u203a>/\u2013\u2014"


def quote_quality(quote: str, page: PageText, other_pages: Sequence[str] = ()) -> Optional[str]:
    """
    Why a target quote is too weak to prove coverage, or None if it is usable:
    too_short, menu_like (separators, or a run of capitalised words with no
    sentence punctuation), page_chrome (only in the header/nav/footer/aside),
    not_in_main_content, or repeated_on_other_target_pages (site-wide text).
    """
    words = normalize_for_match(quote).split()
    if len(words) < MIN_QUOTE_WORDS or len(" ".join(words)) < MIN_QUOTE_CHARS:
        return "too_short"
    raw_words = [w for w in quote.split() if any(ch.isalpha() for ch in w)]
    separators = sum(quote.count(ch) for ch in _MENU_SEPARATORS) + quote.count(" - ")
    capitalised = sum(1 for w in raw_words if w.lstrip("\"'(")[:1].isupper())
    if separators >= 2 or (
        len(raw_words) >= 6 and capitalised / len(raw_words) >= 0.6 and not re.search(r"[.!?;:,]", quote)
    ):
        return "menu_like"
    if page.main is not None:
        if not quote_in_text(quote, page.main):
            return "page_chrome" if page.chrome and quote_in_text(quote, page.chrome) else "not_in_main_content"
    if any(quote_in_text(quote, other) for other in other_pages):
        return "repeated_on_other_target_pages"
    return None


# ---------- coverage judgement ----------

def apply_judgements(
    claims: Sequence[Claim],
    reply: Any,
    target_texts: Mapping[str, Union[str, PageText]],
    source_urls: Mapping[str, str],
    competitor_names: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """
    Gap rows for the judged claims. A status the model did not give becomes
    missing. "covered" needs a quote found on a target page that passes
    quote_quality (main content, long enough, not menu or site-wide text); a
    weak or unverified quote downgrades it to "partly", flagged. Suggestions are
    kept only for missing/partly claims and go through check_suggestion.
    target_texts: target URL -> text (or PageText with sections).
    """
    verdicts = {}
    entries, _ = reply_entries(reply, "judgements")
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("claim_id"):
            verdicts.setdefault(str(entry["claim_id"]).strip(), entry)
    pages = {url: as_page(value) for url, value in target_texts.items()}
    all_target_text = "\n".join(p.text for p in pages.values())

    rows = []
    for claim in claims:
        verdict = verdicts.get(claim.claim_id, {})
        flags: List[str] = []
        status = str(verdict.get("status") or "").strip().lower()
        if status not in STATUSES:
            flags.append("no_verdict" if not verdict else "invalid_status")
            status = "missing"
        quote = str(verdict.get("target_quote") or "").strip() or None
        quote_url = quality = None
        if quote:
            quote_url = next((u for u, p in pages.items() if quote_in_text(quote, p.text)), None)
            if quote_url is None:
                flags.append("target_quote_not_found")
                quote = None
            else:
                others = [p.text for u, p in pages.items() if u != quote_url]
                quality = quote_quality(quote, pages[quote_url], others)
        if status == "covered" and quote is None:
            status = "partly"
            flags.append("covered_without_verified_quote")
        elif status == "covered" and quality:
            status = "partly"
            flags.append(f"weak_covered_quote:{quality}")
        elif quality:
            flags.append(f"weak_quote:{quality}")
        if status == "missing":
            quote, quote_url, quality = None, None, None

        suggestion = original = None
        if status in ("missing", "partly"):
            original = str(verdict.get("suggestion") or "").strip() or (
                f"Add a section that addresses: {claim.claim} Use your own verified details; add your real figures."
            )
            suggestion, suggestion_flags = check_suggestion(original, all_target_text, competitor_names)
            flags.extend(suggestion_flags)
        rows.append({
            "claim_id": claim.claim_id,
            "claim": claim.claim,
            "source_count": claim.source_count,
            "sources": [source_urls.get(s, s) for s in claim.sources],
            "source_labels": claim.sources,
            "source_quotes": claim.quotes,
            "target_status": status,
            "target_quote": quote,
            "target_url": quote_url,
            "target_quote_quality": "ok" if quote and not quality else quality,
            "suggestion": suggestion,
            "suggestion_original": original if original != suggestion else None,
            "flags": flags,
        })
    return rows


def rank_gaps(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Most-supported first; at equal support, missing before partly before covered."""
    order = {"missing": 0, "partly": 1, "covered": 2}
    ranked = sorted(rows, key=lambda r: (-r["source_count"], order.get(r["target_status"], 3)))
    return [{"rank": n, **r} for n, r in enumerate(ranked, start=1)]


# ---------- prompts ----------

def build_extraction_prompt(query: str, url: str, title: Optional[str], page_text: str, token_budget: int) -> str:
    return (
        "An AI search engine (Google AI Overview) cited the page below when answering this query.\n"
        f"Query: {query}\nSource: {url}" + (f" ({title})" if title else "") + "\n\n"
        "Extract the evidence on this page that is relevant to the query, as JSON:\n"
        '{"items": [{"type": "...", "text": "...", "quote": "..."}]}\n'
        f"type is one of: {', '.join(ITEM_TYPES)}.\n"
        "text: the evidence in one short sentence, keeping exact figures, units, dates and names.\n"
        "quote: a short verbatim excerpt from the page (5-30 words) that supports the item. Copy it exactly; "
        "do not paraphrase inside the quote.\n"
        "Only include what the page actually says. Skip navigation, boilerplate and anything unrelated to the query. "
        f'Keep the whole reply under about {int(token_budget * 0.75)} words. If nothing is relevant, return {{"items": []}}.\n\n'
        f"PAGE CONTENT:\n{page_text[:MAX_PAGE_CHARS]}"
    )


def build_cluster_prompt(query: str, items: Sequence[EvidenceItem]) -> str:
    lines = "\n".join(f"{i.item_id} [{i.type}] {i.text}" for i in items)
    return (
        f"Query: {query}\n\n"
        "Below are evidence items extracted from several cited sources. Item ids start with the source label "
        "(S1, S2, ...). Group items that state the same claim (the same fact, figure, definition, product detail "
        "or comparison), even when worded differently. Items that disagree on a figure are different claims.\n"
        "Return JSON: {\"claims\": [{\"claim\": \"one neutral sentence stating the shared claim\", "
        "\"item_ids\": [\"S1.2\", \"S3.1\"]}]}\n"
        "Use every item exactly once. Do not count sources yourself and do not add anything the items do not say.\n\n"
        f"ITEMS:\n{lines}"
    )


def build_judge_prompt(query: str, brand: str, claims: Sequence[Claim], excerpts: Sequence[Dict[str, str]]) -> str:
    claim_lines = "\n".join(f"{c.claim_id} (stated by {c.source_count} cited sources): {c.claim}" for c in claims)
    page_blocks = "\n\n".join(f"[{e['label']}] {e['url']}\n{e['excerpt'] or '(no relevant passage found)'}" for e in excerpts)
    return (
        f"Query: {query}\nTarget brand: {brand}\n\n"
        "The claims below are made by sources Google's AI Overview cited for this query. For each claim, judge "
        f"whether {brand}'s own pages (excerpts below) cover it:\n"
        "- covered: the target pages clearly state it (or the brand's own equivalent of it)\n"
        "- partly: the target pages touch on it but leave out the specific fact, figure or detail\n"
        "- missing: the target pages do not address it\n"
        "For covered or partly, give target_quote: a short verbatim excerpt (5-30 words) copied exactly from the "
        "target excerpts. For missing, target_quote is null.\n"
        "A quote from navigation, menus, headers or footers does not show coverage: quote a full sentence from the "
        "page's main content, or choose partly or missing.\n"
        f"For partly or missing, give suggestion: one or two sentences on what {brand} could add to its page to "
        "address the claim. The suggestion must state only verifiable facts about the brand's own offering: "
        "no superlatives or promotional wording (such as best, leading, industry-leading, top, fastest, number one, "
        "world-class, unmatched, most trusted, seamless), no claims about competitors, and no invented statistics, "
        "prices, dates, reviews or awards. Never present a competitor's figure as the brand's. Write "
        "\"[add your real figure]\" in place of any number. For covered, suggestion is null.\n"
        'Return JSON: {"judgements": [{"claim_id": "C1", "status": "covered|partly|missing", '
        '"target_quote": "..." or null, "suggestion": "..." or null}]}\n\n'
        f"CLAIMS:\n{claim_lines}\n\nTARGET PAGE EXCERPTS:\n{page_blocks}"
    )


# ---------- model client ----------

class ModelCallBudgetExceeded(RuntimeError):
    pass


class EmptyModelReply(ValueError):
    pass


class ModelQuotaExhausted(RuntimeError):
    """A daily quota ran out: further calls today would only fail, so none are made."""


class ModelCallTimeout(TimeoutError):
    pass


class RunDeadlineExceeded(RuntimeError):
    """The whole-run time limit passed: no further calls are made."""


def _is_timeout(error: Exception) -> bool:
    return isinstance(error, (asyncio.TimeoutError, TimeoutError)) or "timed out" in str(error).lower() \
        or "Timeout" in type(error).__name__


def _is_retryable(error: Exception) -> bool:
    text = str(error)
    return any(s in text for s in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand", "overloaded"))


def _is_unavailable(error: Exception) -> bool:
    """A 503-style overload: worth trying another model of the pool."""
    text = str(error)
    return any(s in text for s in ("503", "UNAVAILABLE", "high demand", "overloaded"))


def _is_daily_quota(error: Exception) -> bool:
    text = str(error)
    return ("429" in text or "RESOURCE_EXHAUSTED" in text) and ("PerDay" in text or "per day" in text.lower())


# Output-token ceilings: thinking models spend part of max_output_tokens before
# answering, and a cut-off reply is invalid JSON
EXTRACT_MAX_TOKENS = 8192
CLUSTER_MAX_TOKENS = 16384
JUDGE_MAX_TOKENS = 16384


class GeminiJSONModel:
    """
    JSON text generation on the configured EVIDENCE_EXTRACTION model, falling
    back through the existing Gemini model pool (gemini_2_5_flash.FALLBACK_MODELS):
    a 503 or a timeout moves the call, and later calls, to the next model in the
    pool, within the capped retries (max_retries, exponential backoff). A
    per-minute 429 is retried on the same model. A reply that is not valid JSON is
    retried once. A 429 for a daily quota is not retried, and no further calls are
    made. Each call is capped at call_timeout seconds, and no call starts after
    the run deadline. Every attempt is counted, with the model it used. Never
    calls embeddings.
    """

    def __init__(self, model_name: Optional[str] = None, max_retries: int = 3, base_delay: float = 4.0,
                 max_attempts: int = 40, json_retries: int = 1, call_timeout: float = 60.0,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        from .constants import get_model_for_role
        from .llm_providers.factory import get_llm_provider

        self.model_name = model_name or get_model_for_role("EVIDENCE_EXTRACTION")
        self.primary_model = self.model_name
        self._provider = get_llm_provider(self.model_name)
        self.model_pool = model_pool(self.model_name, getattr(self._provider, "fallback_models", None))
        self.max_retries, self.base_delay, self.max_attempts, self._sleep = max_retries, base_delay, max_attempts, sleep
        self.json_retries = json_retries
        self.call_timeout = call_timeout
        self.deadline: Optional[float] = None  # time.monotonic() after which no call starts
        self.quota_exhausted: Optional[str] = None
        self.calls: List[Dict[str, Any]] = []

    def _fall_back(self) -> str:
        """Move to the next model of the pool (wrapping around); returns the new model."""
        index = self.model_pool.index(self.model_name) if self.model_name in self.model_pool else -1
        self.model_name = self.model_pool[(index + 1) % len(self.model_pool)]
        return self.model_name

    def set_time_limit(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds

    def _call_timeout(self) -> float:
        if self.deadline is None:
            return self.call_timeout
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RunDeadlineExceeded("run time limit reached; call not made")
        return min(self.call_timeout, remaining)

    async def _request(self, prompt: str, max_tokens: int, json_mode: bool, timeout: float):
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.0, max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)),
        )
        await self._provider.rate_limiter.wait_for_slot(self.model_name)
        try:
            return await asyncio.wait_for(asyncio.to_thread(
                self._provider.client.models.generate_content,
                model=self.model_name, contents=prompt, config=config,
            ), timeout=timeout + 5)
        except asyncio.TimeoutError:
            raise ModelCallTimeout(f"model call timed out after {timeout:.0f}s")

    async def preflight(self) -> None:
        """
        One tiny text-generation call, before a report starts. Raises on failure
        (a daily-quota 429 also sets quota_exhausted); any reply counts as a pass.
        """
        if self.quota_exhausted:
            raise ModelQuotaExhausted(self.quota_exhausted)
        for attempt in range(1, self.max_retries + 2):
            started = time.monotonic()
            record: Dict[str, Any] = {"stage": "preflight", "attempt": attempt, "model": self.model_name}
            self.calls.append(record)
            try:
                await self._request('Reply with the JSON {"ok": true}.', 1024, False, min(self.call_timeout, 30.0))
                record.update(ok=True, seconds=round(time.monotonic() - started, 2))
                return
            except Exception as exc:
                record.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:600]}",
                              seconds=round(time.monotonic() - started, 2))
                if _is_daily_quota(exc):
                    self.quota_exhausted = f"daily quota exhausted on {self.model_name}"
                    raise
                if (_is_unavailable(exc) or _is_timeout(exc)) and attempt <= self.max_retries:
                    self._fall_back()
                    await self._sleep(self.base_delay * (2 ** (attempt - 1)))
                    continue
                raise

    async def generate_json(self, stage: str, prompt: str, max_tokens: int = 4096) -> Any:
        transient_retries = json_failures = 0
        attempt = 0
        while True:
            if self.quota_exhausted:
                raise ModelQuotaExhausted(f"not called: {self.quota_exhausted}")
            if len(self.calls) >= self.max_attempts:
                raise ModelCallBudgetExceeded(f"model call cap of {self.max_attempts} reached")
            timeout = self._call_timeout()
            attempt += 1
            started = time.monotonic()
            record: Dict[str, Any] = {"stage": stage, "attempt": attempt, "model": self.model_name}
            self.calls.append(record)
            try:
                response = await self._request(prompt, max_tokens, True, timeout)
                record["seconds"] = round(time.monotonic() - started, 2)
                try:
                    record["finish_reason"] = str(response.candidates[0].finish_reason)
                except Exception:
                    pass
                text = response.text or ""
                if not text.strip():
                    raise EmptyModelReply("empty model response")
                parsed = parse_json_reply(text)
                record["ok"] = True
                return parsed
            except Exception as exc:
                record.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:600]}")
                record.setdefault("seconds", round(time.monotonic() - started, 2))
                if _is_daily_quota(exc):
                    self.quota_exhausted = f"daily quota exhausted on {self.model_name}"
                    raise
                if isinstance(exc, (json.JSONDecodeError, EmptyModelReply)) and json_failures < self.json_retries:
                    json_failures += 1
                    continue
                if (_is_retryable(exc) or _is_timeout(exc)) and transient_retries < self.max_retries:
                    if _is_unavailable(exc) or _is_timeout(exc):
                        self._fall_back()  # a per-minute 429 stays on the same model
                    await self._sleep(self.base_delay * (2 ** transient_retries))
                    transient_retries += 1
                    continue
                raise


def model_pool(primary: str, fallback_models: Optional[Sequence[str]] = None) -> List[str]:
    """The primary model first, then the existing Gemini pool in its order, without repeats."""
    if fallback_models is None:
        from .llm_providers.gemini.gemini_2_5_flash import FALLBACK_MODELS as fallback_models
    return list(dict.fromkeys([primary, *fallback_models]))


def models_used(calls: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    used: Dict[str, int] = {}
    for c in calls:
        if c.get("model"):
            used[c["model"]] = used.get(c["model"], 0) + 1
    return used


def summarize_calls(calls: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_stage: Dict[str, int] = {}
    for c in calls:
        by_stage[c["stage"]] = by_stage.get(c["stage"], 0) + 1
    return {
        "total": len(calls),
        "succeeded": sum(1 for c in calls if c.get("ok")),
        "retries": sum(1 for c in calls if c.get("attempt", 1) > 1),
        "failed": sum(1 for c in calls if not c.get("ok")),
        "by_stage": by_stage,
    }


# ---------- orchestration ----------

Crawler = Callable[[str], Awaitable[Dict[str, Any]]]


async def _default_crawler(url: str) -> Dict[str, Any]:
    from . import crawl

    return await crawl.crawl_sections(url)  # text only, plus main content vs header/nav/footer


class TextStore:
    """
    Page text by URL: backend-provided crawled text first, then one crawl per page
    for the whole report. Target pages are crawled once even when cached, to
    separate main content from header/nav/footer (the cached text is flattened);
    the cached text is used if that crawl fails.
    """

    def __init__(self, cached: Mapping[str, str], crawler: Crawler):
        self._texts = {page_key(u): t for u, t in cached.items() if t}
        self._origin = {k: "cache" for k in self._texts}
        self._sections: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._errors: Dict[str, str] = {}
        self._crawled: set = set()
        self._crawler = crawler

    async def _crawl(self, url: str) -> None:
        key = page_key(url)
        self._crawled.add(key)
        try:
            result = await self._crawler(url)
        except Exception as exc:  # a crawl failure only removes this page
            result = {"error": f"{type(exc).__name__}: {exc}"}
        result = result or {}
        text = result.get("full_content") or ""
        if text.strip():
            self._texts[key], self._origin[key] = text, "crawled"
            if result.get("main_content") is not None:
                self._sections[key] = (result.get("main_content"), result.get("chrome_content") or "")
        elif key not in self._texts:
            self._errors[key] = result.get("error") or "no page text"

    async def get(self, url: str) -> Tuple[Optional[str], str, Optional[str]]:
        key = page_key(url)
        if key not in self._texts and key not in self._crawled:
            await self._crawl(url)
        if key in self._texts:
            return self._texts[key], self._origin[key], None
        return None, "unavailable", self._errors.get(key)

    async def get_page(self, url: str) -> Tuple[Optional[PageText], str, Optional[str]]:
        key = page_key(url)
        if key not in self._sections and key not in self._crawled:
            await self._crawl(url)
        text, origin, error = await self.get(url)
        if text is None:
            return None, origin, error
        main, chrome = self._sections.get(key, (None, None))
        return PageText(text, main, chrome), origin + ("+sections" if main is not None else ""), None


async def analyze_query(
    query: Mapping[str, Any], sandbox_url: str, brand: str, store: TextStore, model, extraction_concurrency: int = 3
) -> Dict[str, Any]:
    cited, targets = select_sources(query.get("sources") or [], sandbox_url)
    budget = source_token_budget(len(cited) + len(targets))
    result: Dict[str, Any] = {
        "query_index": query.get("query_index"),
        "query": query["query"],
        "per_source_token_budget": budget,
        "sources": [],
        "target_pages": [],
        "gaps": [],
        "other_claims": [],
        "errors": [],
    }

    window = asyncio.Semaphore(extraction_concurrency)

    async def extract(label: str, source: Dict[str, Any]) -> Tuple[List[EvidenceItem], Dict[str, Any]]:
        text, origin, crawl_error = await store.get(source["url"])
        row = {
            "label": label, "url": source["url"], "domain": source.get("domain"), "category": source.get("category"),
            "appearances": source.get("appearances"), "best_position": source.get("best_position"),
            "text_source": origin, "error": crawl_error, "items": 0, "dropped": None, "reply_shape": None,
        }
        if text is None:
            return [], row
        try:
            async with window:
                reply = await model.generate_json(
                    "extract", build_extraction_prompt(query["query"], source["url"], source.get("title"), text, budget),
                    max_tokens=EXTRACT_MAX_TOKENS,
                )
        except Exception as exc:
            row["error"] = f"extraction failed: {str(exc)[:200]}"
            return [], row
        items, dropped = parse_evidence_items(reply, label, text, budget)
        row.update(items=len(items), dropped=dropped, reply_shape=reply_entries(reply, "items")[1])
        return items, row

    extracted = await asyncio.gather(*(extract(f"S{n}", s) for n, s in enumerate(cited, start=1)))
    items = [i for its, _ in extracted for i in its]
    result["sources"] = [row for _, row in extracted]
    source_urls = {row["label"]: row["url"] for row in result["sources"]}
    appearances = {row["label"]: row["appearances"] or 0 for row in result["sources"]}

    target_texts: Dict[str, PageText] = {}
    for n, target in enumerate(targets, start=1):
        page, origin, crawl_error = await store.get_page(target["url"])
        result["target_pages"].append({
            "label": f"T{n}", "url": target["url"], "submitted": target.get("submitted", False),
            "appearances": target.get("appearances"), "text_source": origin, "error": crawl_error,
        })
        if page:
            target_texts[target["url"]] = page
    competitor_names = cited_brand_names(cited, brand)

    if not items:
        result["errors"].append("no verified evidence from the cited sources")
        return result
    try:
        grouped = await model.generate_json("cluster", build_cluster_prompt(query["query"], items), max_tokens=CLUSTER_MAX_TOKENS)
    except Exception as exc:
        result["errors"].append(f"claim grouping failed, items counted one by one: {str(exc)[:200]}")
        grouped = {}
    claims = count_claims(grouped, items, appearances)
    judged, rest = claims[:MAX_JUDGED_CLAIMS], claims[MAX_JUDGED_CLAIMS:]
    result["other_claims"] = [
        {"claim": c.claim, "source_count": c.source_count, "sources": [source_urls[s] for s in c.sources]} for c in rest
    ]

    if not target_texts:
        result["errors"].append("no target page text; coverage not judged")
        rows = apply_judgements(judged, {}, {}, source_urls, competitor_names)
        for row in rows:
            row["flags"].append("target_text_unavailable")
        result["gaps"] = rank_gaps(rows)
        return result

    excerpts = []
    for page in result["target_pages"]:
        target_page = target_texts.get(page["url"])
        if target_page:  # main content when known, so menus and footers are not offered as evidence
            excerpts.append({"label": page["label"], "url": page["url"],
                             "excerpt": select_excerpt(target_page.main or target_page.text, query["query"], judged, budget)})
    try:
        verdicts = await model.generate_json(
            "judge", build_judge_prompt(query["query"], brand, judged, excerpts), max_tokens=JUDGE_MAX_TOKENS
        )
    except Exception as exc:
        result["errors"].append(f"coverage judgement failed: {str(exc)[:200]}")
        verdicts = {}
    result["gaps"] = rank_gaps(apply_judgements(judged, verdicts, target_texts, source_urls, competitor_names))
    return result


_GENERIC_LABELS = {"blog", "www", "docs", "help", "support", "news", "business", "india", "online", "store", "shop"}


def cited_brand_names(cited: Sequence[Mapping[str, Any]], brand: str) -> List[str]:
    """Brand-like names of the cited sources (domain labels), for flagging suggestions that name them."""
    names = []
    for source in cited:
        label = normalize_domain(source.get("domain") or source.get("url") or "").split(".")[0]
        if len(label) >= 4 and label not in _GENERIC_LABELS and label != brand.lower():
            names.append(label)
    return list(dict.fromkeys(names))


RUN_TIME_LIMIT_SECONDS = 600.0


async def run_gap_analysis(
    payload: Mapping[str, Any], model=None, crawler: Optional[Crawler] = None,
    time_limit: float = RUN_TIME_LIMIT_SECONDS,
) -> Dict[str, Any]:
    """
    payload: {observation_id, sandbox_url, brand_name, queries: [{query_index,
    query, sources: [{url, domain, title, category, competitor_group,
    appearances, stability, best_position}]}], cached_texts: {url: text}}.
    Queries run one after another; extraction runs a few calls at a time. After
    time_limit seconds no model call starts and remaining queries are skipped,
    so the report comes back with what was done.
    """
    model = model or GeminiJSONModel()
    if hasattr(model, "set_time_limit"):
        model.set_time_limit(time_limit)
    store = TextStore(payload.get("cached_texts") or {}, crawler or _default_crawler)
    sandbox_url = payload["sandbox_url"]
    brand = payload.get("brand_name") or normalize_domain(sandbox_url).split(".")[0].capitalize()
    started = time.monotonic()
    queries = []
    deadline_reached = False
    for query in payload.get("queries") or []:
        if time.monotonic() - started >= time_limit:
            deadline_reached = True
            queries.append({"query_index": query.get("query_index"), "query": query["query"], "sources": [],
                            "target_pages": [], "gaps": [], "other_claims": [],
                            "errors": [f"not analyzed: the {time_limit:.0f}-second run limit was reached"]})
            continue
        queries.append(await analyze_query(query, sandbox_url, brand, store, model))
    deadline_reached = deadline_reached or any(
        "RunDeadlineExceeded" in (c.get("error") or "") for c in getattr(model, "calls", [])
    ) or any("run time limit" in e for q in queries for e in q["errors"])
    calls = getattr(model, "calls", [])
    return {
        "version": GAP_VERSION,
        "observation_id": payload.get("observation_id"),
        "sandbox_url": sandbox_url,
        "brand_name": brand,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": getattr(model, "model_name", None),
        "settings": {
            "top_cited_sources": TOP_CITED,
            "max_judged_claims": MAX_JUDGED_CLAIMS,
            "total_token_budget": TOTAL_TOKEN_BUDGET,
            "per_source_token_cap": PER_SOURCE_TOKEN_CAP,
            "min_source_tokens": MIN_SOURCE_TOKENS,
            "chars_per_token": CHARS_PER_TOKEN,
            "call_timeout_seconds": getattr(model, "call_timeout", None),
            "run_time_limit_seconds": time_limit,
            "model_pool": getattr(model, "model_pool", None),
        },
        "deadline_reached": deadline_reached,
        "model_calls": {**summarize_calls(calls), "models_used": models_used(calls),
                        "quota_exhausted": getattr(model, "quota_exhausted", None), "calls": list(calls)},
        "seconds": round(time.monotonic() - started, 1),
        "queries": queries,
    }


class GapPreflightError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


async def preflight_and_start(
    report_id: str, payload: Mapping[str, Any], model=None,
    launch: Callable[[Awaitable[None]], Any] = asyncio.create_task,
):
    """
    One tiny text-generation call; only if it succeeds is the report job launched.
    A daily-quota 429 raises GapPreflightError(429), anything else (503s after
    the call's own handling, timeouts) GapPreflightError(503). Nothing is started
    or saved on failure.
    """
    model = model or GeminiJSONModel()
    try:
        await model.preflight()
    except Exception as exc:
        if _is_daily_quota(exc) or getattr(model, "quota_exhausted", None):
            raise GapPreflightError(
                429, f"Gemini daily quota is exhausted for {model.model_name}; gap analysis not started, nothing saved."
            ) from exc
        raise GapPreflightError(
            503, f"Gemini preflight call failed ({type(exc).__name__}: {str(exc)[:200]}); gap analysis not started."
        ) from exc
    return launch(run_gap_analysis_job(report_id, payload, model=model))


async def run_gap_analysis_job(report_id: str, payload: Mapping[str, Any], model=None) -> None:
    """Background job: build the report and post it (or the failure) to the backend."""
    import httpx

    backend = os.getenv("BACKEND_BASE_URL", "http://localhost:4000")
    try:
        model = model or GeminiJSONModel()
        # Safety net over the run's own limit: crawls or a stuck thread cannot hold the job much longer
        report = await asyncio.wait_for(run_gap_analysis(payload, model=model), timeout=RUN_TIME_LIMIT_SECONDS + 60)
        body = {"status": "completed", "report": report}
    except Exception as exc:
        logger.exception(f"[gap_analysis] report {report_id} failed")
        calls = getattr(model, "calls", [])
        body = {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                "report": {"version": GAP_VERSION, "model_calls": {**summarize_calls(calls), "calls": list(calls)}}}
    body["observation_id"] = payload.get("observation_id")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{backend}/api/gap-analysis/{report_id}/result", json=body)
        if response.status_code >= 400:
            logger.error(f"[gap_analysis] storing report {report_id} failed: {response.status_code} {response.text[:300]}")
