"""
Stage 1 job: Real-World Observation.

1. Context: crawl the target page and reuse the Stage 2 analysis steps (domain
   summary, intent, business category) to condition query generation.
2. Query selection: caller-supplied queries, or a balanced subset of
   sandbox.generate_user_questions.
3. Repeated observation: each query is asked N times through the configured
   ObservationFetcher. Every run's raw citations are kept, failures included.
4. Union source pool per query: Sq = S1 ∪ ... ∪ SN over successful runs,
   with per-source stability. Nothing is averaged away or dropped.
5. Categorisation of every source domain.
6. Evidence-preserving compression of every pooled source, per query.
7. Per-query and run-level metrics.

A single failed observation, crawl or compression is recorded and skipped; it
never fails the job. The default fetcher is an approximate proxy for Google AI
Overview citations, not real AI Overview data (see observation/fetchers.py).
"""
import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import httpx

from ..geo_metrics import (
    ResponseVisibility,
    canonical_url,
    category_breakdown,
    citation_position,
    citation_rate,
    mean_citation_position,
    mean_pairwise_jaccard,
    median_citation_position,
    mention_rate,
    normalize_domain,
    source_set_diversity,
    source_stability,
    union_sources,
)
from .categorize import SourceCategory, categorize_domains
from .config import Stage1Config
from .evidence import compress_query_sources
from .fetchers import Observation, ObservationFetcher, get_observation_fetcher

logger = logging.getLogger(__name__)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:4000")


# ==========================================
# Pure helpers
# ==========================================


def select_balanced_queries(questions: Sequence[Dict[str, str]], count: int) -> List[Dict[str, str]]:
    """
    Pick `count` questions, taking one of each type in turn (intent, experience,
    transaction, ...) so the selection stays balanced. Blank questions and exact
    duplicates are skipped. Returns fewer than `count` if there aren't enough.
    """
    by_type: Dict[str, List[Dict[str, str]]] = {}
    seen = set()
    for question in questions:
        text = (question.get("q") or "").strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        by_type.setdefault(question.get("type") or "intent", []).append({"q": text, "type": question.get("type") or "intent"})
    selected: List[Dict[str, str]] = []
    while len(selected) < count and any(by_type.values()):
        for bucket in by_type.values():
            if bucket and len(selected) < count:
                selected.append(bucket.pop(0))
    return selected


def brand_terms(brand_name: Optional[str], target_domain: str) -> List[str]:
    """Names that count as a mention of the target: its brand and its domain."""
    terms = []
    if brand_name and brand_name.strip().lower() not in ("", "unknown", "none"):
        terms.append(brand_name.strip())
    if target_domain:
        terms.append(target_domain)
    return terms


def detect_visibility(answer_text: str, ordered_domains: Sequence[str], target_domain: str, terms: Sequence[str]) -> ResponseVisibility:
    """
    mentioned: a brand term appears in the answer as a whole word (case-insensitive).
    cited: the target's domain is among the answer's web sources.
    """
    text = answer_text or ""
    mentioned = any(
        re.search(rf"(?<![\w.]){re.escape(term)}(?![\w])", text, re.IGNORECASE) for term in terms if term
    )
    position = citation_position(list(ordered_domains), target_domain) if target_domain else None
    return ResponseVisibility(mentioned=mentioned, cited=position is not None, citation_position=position)


def query_source_metrics(run_url_sets: Sequence[Sequence[str]], run_domain_sets: Sequence[Sequence[str]], visibilities: Sequence[ResponseVisibility]) -> Dict[str, Any]:
    """Metrics for one query over its successful runs."""
    positions = [v.citation_position for v in visibilities]
    return {
        "successful_runs": len(run_url_sets),
        "source_set_diversity": source_set_diversity(run_url_sets),
        "domain_diversity": source_set_diversity(run_domain_sets),
        "source_set_stability": mean_pairwise_jaccard(run_url_sets),
        "domain_set_stability": mean_pairwise_jaccard(run_domain_sets),
        "mention_rate": mention_rate(visibilities),
        "citation_rate": citation_rate(visibilities),
        "mean_citation_position": mean_citation_position(positions),
        "median_citation_position": median_citation_position(positions),
    }


# ==========================================
# Backend callbacks (endpoints added in Phase 2; failures are logged only)
# ==========================================


async def _post(path: str, payload: Dict[str, Any], timeout: float) -> None:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{BACKEND_BASE_URL}{path}", json=payload)
            if response.status_code != 200:
                logger.warning(f"[stage1] POST {path} -> {response.status_code}: {response.text[:200]}")
    except Exception as exc:
        logger.warning(f"[stage1] POST {path} failed: {exc}")


async def _send_progress(observation_id: str, step: str, status: str, payload: Optional[Dict[str, Any]] = None) -> None:
    logger.info(f"[stage1] {observation_id} {step} {status} {payload or ''}")
    await _post(
        f"/api/observation/{observation_id}/progress",
        {
            "event_id": str(uuid.uuid4()),
            "observation_id": observation_id,
            "step": step,
            "status": status,
            "payload": payload or {},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
        timeout=10.0,
    )


# ==========================================
# Job steps
# ==========================================


async def _build_context(url: str) -> Dict[str, Any]:
    """Reuse Stage 2's analysis steps. Only a failed crawl is fatal; the rest degrade."""
    from .. import crawl, sandbox

    page_data = await crawl.crawl_url(url)
    if "error" in page_data:
        raise RuntimeError(f"Sandbox crawl failed: {page_data['error']}")

    context: Dict[str, Any] = {
        "page_title": page_data.get("title"),
        "page_text": page_data.get("full_content", ""),
        "domain_summary": None,
        "page_intent": None,
        "brand_name": None,
        "business_category": None,
    }
    try:
        context["domain_summary"] = await sandbox.generate_domain_summary(page_data=page_data)
    except Exception as exc:
        logger.warning(f"[stage1] domain summary failed, continuing: {exc}")
    try:
        analysis = await sandbox.analyze_sandbox_page(url=url, domain_summary=context["domain_summary"])
        if "error" not in analysis:
            context["page_intent"] = analysis.get("intent")
            context["brand_name"] = analysis.get("page_brand_name")
    except Exception as exc:
        logger.warning(f"[stage1] intent extraction failed, continuing: {exc}")
    try:
        context["business_category"] = await sandbox.classify_business_category(
            page_data={"title": page_data.get("title"), "h1": page_data.get("h1"), "full_content": context["page_text"]},
            page_intent=context["page_intent"] or "",
            domain_summary=context["domain_summary"],
        )
    except Exception as exc:
        logger.warning(f"[stage1] category classification failed, continuing: {exc}")
    return context


async def _select_queries(queries: Optional[List[str]], context: Dict[str, Any], config: Stage1Config) -> List[Dict[str, str]]:
    if queries:
        return [{"q": q.strip(), "type": "provided"} for q in queries if q and q.strip()]
    from .. import sandbox

    generated = await sandbox.generate_user_questions(
        page_intent=context["page_intent"] or context["page_title"] or "",
        business_category=context["business_category"],
        domain_summary=context["domain_summary"],
    )
    return select_balanced_queries(generated, config.query_count)


async def _observe_all(queries: List[Dict[str, str]], fetcher: ObservationFetcher, config: Stage1Config, observation_id: str) -> List[List[Dict[str, Any]]]:
    """Every (query, run) pair; failures become run records with ok=False."""
    semaphore = asyncio.Semaphore(config.observation_concurrency)
    completed = 0
    total = len(queries) * config.runs_per_query

    async def observe(query: str) -> Observation:
        nonlocal completed
        async with semaphore:
            try:
                return await fetcher.observe(query)
            finally:
                completed += 1
                if completed % max(1, config.runs_per_query) == 0 or completed == total:
                    await _send_progress(observation_id, "observation", "progress", {"completed": completed, "total": total})

    runs_by_query: List[List[Dict[str, Any]]] = []
    tasks = [[observe(q["q"]) for _ in range(config.runs_per_query)] for q in queries]
    outcomes = await asyncio.gather(*(asyncio.gather(*query_tasks, return_exceptions=True) for query_tasks in tasks))
    for query_outcomes in outcomes:
        runs = []
        for run_index, outcome in enumerate(query_outcomes):
            if isinstance(outcome, Exception):
                runs.append({"run_index": run_index, "ok": False, "error": str(outcome)[:500]})
            else:
                runs.append({"run_index": run_index, "ok": True, "observation": outcome})
        runs_by_query.append(runs)
    return runs_by_query


async def _crawl_sources(urls: Sequence[str], concurrency: int) -> Dict[str, Dict[str, Optional[str]]]:
    from .. import crawl

    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(url: str) -> Dict[str, Optional[str]]:
        async with semaphore:
            try:
                data = await crawl.crawl_text(url)
            except Exception as exc:
                return {"page_text": None, "title": None, "crawl_error": str(exc)[:300]}
            if "error" in data:
                return {"page_text": None, "title": None, "crawl_error": data["error"][:300]}
            return {"page_text": data.get("full_content") or None, "title": data.get("title"), "crawl_error": None}

    results = await asyncio.gather(*(fetch(u) for u in urls))
    return dict(zip(urls, results))


def _serialize_run(run: Dict[str, Any], visibility: Optional[ResponseVisibility]) -> Dict[str, Any]:
    if not run["ok"]:
        return {"run_index": run["run_index"], "ok": False, "error": run["error"], "citations": []}
    obs: Observation = run["observation"]
    return {
        "run_index": run["run_index"],
        "ok": True,
        "error": None,
        "model": obs.model,
        "latency_seconds": round(obs.latency_seconds, 2),
        "answer_text": obs.answer_text,
        "search_queries": obs.search_queries,
        "target": asdict(visibility) if visibility else None,
        "citations": [asdict(c) for c in obs.citations],
    }


async def run_stage1_job(
    observation_id: str,
    url: str,
    queries: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    fetcher: Optional[ObservationFetcher] = None,
) -> Dict[str, Any]:
    """Run Stage 1 end to end, report to the backend, and return the full raw result."""
    result: Dict[str, Any] = {"observation_id": observation_id, "sandbox_url": url, "status": "running"}
    try:
        config = Stage1Config.from_env(config_overrides)
        fetcher = fetcher or get_observation_fetcher(config.fetcher, config.grounding_model)
        target_domain = normalize_domain(url)
        result.update(
            {
                "fetcher": fetcher.name,
                "data_disclaimer": fetcher.data_disclaimer,
                "config": {k: v for k, v in asdict(config).items()},
            }
        )

        await _send_progress(observation_id, "context", "started")
        context = await _build_context(url)
        terms = brand_terms(context["brand_name"], target_domain)
        result["context"] = {k: v for k, v in context.items() if k != "page_text"}
        await _send_progress(observation_id, "context", "success", {"brand_name": context["brand_name"], "category": context["business_category"]})

        selected = await _select_queries(queries, context, config)
        if not selected:
            raise RuntimeError("No queries available for Stage 1")
        await _send_progress(observation_id, "query_selection", "success", {"queries": [q["q"] for q in selected]})

        await _send_progress(observation_id, "observation", "started", {"total": len(selected) * config.runs_per_query})
        runs_by_query = await _observe_all(selected, fetcher, config, observation_id)

        # Per-query source pools
        query_records: List[Dict[str, Any]] = []
        pool_meta: Dict[str, Dict[str, Any]] = {}  # source key -> url/domain/title
        for q_index, (query, runs) in enumerate(zip(selected, runs_by_query)):
            url_sets, domain_sets, visibilities, serialized = [], [], [], []
            for run in runs:
                if not run["ok"]:
                    serialized.append(_serialize_run(run, None))
                    continue
                web = sorted((c for c in run["observation"].citations if c.is_web_source and c.domain), key=lambda c: c.order)
                keys = []
                for c in web:
                    key = canonical_url(c.url) if c.url else c.domain
                    keys.append(key)
                    pool_meta.setdefault(key, {"url": c.url and canonical_url(c.url), "domain": c.domain, "title": c.title})
                visibility = detect_visibility(run["observation"].answer_text, [c.domain for c in web], target_domain, terms)
                url_sets.append(list(dict.fromkeys(keys)))
                domain_sets.append(list(dict.fromkeys(c.domain for c in web)))
                visibilities.append(visibility)
                serialized.append(_serialize_run(run, visibility))

            stability = source_stability(url_sets)
            query_records.append(
                {
                    "query_index": q_index,
                    "query": query["q"],
                    "query_type": query["type"],
                    "runs": serialized,
                    "pool": [{"key": key, "stability": stability[key], "appearances": round(stability[key] * len(url_sets))} for key in union_sources(url_sets)],
                    "metrics": query_source_metrics(url_sets, domain_sets, visibilities),
                    "_visibilities": visibilities,
                }
            )
        attempted = len(selected) * config.runs_per_query
        succeeded = sum(r["metrics"]["successful_runs"] for r in query_records)
        await _send_progress(observation_id, "observation", "success", {"attempted": attempted, "succeeded": succeeded})

        # Categorise every source domain
        await _send_progress(observation_id, "categorization", "started")
        categories: Dict[str, SourceCategory] = await categorize_domains(
            [meta["domain"] for meta in pool_meta.values()],
            target_domain=target_domain,
            taxonomy=config.source_categories,
            target_brand=context["brand_name"],
            business_category=context["business_category"],
            page_intent=context["page_intent"],
            batch_size=config.classification_batch_size,
        )
        await _send_progress(observation_id, "categorization", "success", {"domains": len(categories)})

        # Crawl each pooled page once, shared across queries. The sandbox page itself
        # was already crawled for context; other pages on the target's site are crawled.
        sandbox_key = canonical_url(url)
        crawlable: List[str] = []
        for record in query_records:
            for entry in record["pool"][: config.max_sources_per_query]:
                page_url = pool_meta[entry["key"]]["url"]
                if page_url and page_url != sandbox_key and page_url not in crawlable:
                    crawlable.append(page_url)
        await _send_progress(observation_id, "crawling_sources", "started", {"pages": len(crawlable)})
        pages = await _crawl_sources(crawlable, config.crawl_concurrency)
        target_page = {"page_text": context["page_text"] or None, "title": context["page_title"], "crawl_error": None}
        pages[sandbox_key] = target_page
        unresolved = {"page_text": None, "title": None, "crawl_error": "source URL could not be resolved"}
        await _send_progress(observation_id, "crawling_sources", "success", {"crawled": sum(1 for p in pages.values() if p["page_text"])})

        # Evidence per (query, source), plus the sandbox page for each query. One
        # rolling window of compression calls is shared across all queries.
        await _send_progress(observation_id, "evidence_compression", "started")
        window = asyncio.Semaphore(config.compression_batch_size)
        compression = dict(
            per_source_cap=config.per_source_token_cap,
            min_floor=config.min_source_tokens,
            batch_size=config.compression_batch_size,
            max_page_chars=config.max_page_chars,
            semaphore=window,
        )

        async def compress_record(record: Dict[str, Any]) -> None:
            in_budget = record["pool"][: config.max_sources_per_query]
            inputs = []
            for entry in in_budget:
                meta = pool_meta[entry["key"]]
                page = pages.get(meta["url"] or "", unresolved)
                inputs.append({"url": meta["url"] or meta["domain"], "title": page.get("title") or meta["title"], "page_text": page["page_text"], "crawl_error": page["crawl_error"]})
            evidence = await compress_query_sources(record["query"], inputs, total_budget=config.evidence_token_budget, **compression)
            evidence_by_key = {entry["key"]: ev for entry, ev in zip(in_budget, evidence)}
            crawl_errors = {entry["key"]: item["crawl_error"] for entry, item in zip(in_budget, inputs)}

            sources = []
            for entry in record["pool"]:
                meta = pool_meta[entry["key"]]
                cat = categories.get(meta["domain"])
                ev = evidence_by_key.get(entry["key"])
                sources.append(
                    {
                        **entry,
                        "url": meta["url"],
                        "domain": meta["domain"],
                        "title": meta["title"],
                        "category": cat.category if cat else "other",
                        "category_source": cat.category_source if cat else "fallback",
                        "llm_category": cat.llm_category if cat else None,
                        "heuristic_category": cat.heuristic_category if cat else None,
                        "competitor_group": cat.competitor_group if cat else "ai_visibility",
                        "over_source_cap": ev is None,
                        "crawl_error": crawl_errors.get(entry["key"]),
                        "evidence": ev.evidence if ev else None,
                        "evidence_token_budget": ev.token_budget if ev else None,
                        "evidence_token_estimate": ev.token_estimate if ev else None,
                        "has_relevant_evidence": ev.has_relevant_evidence if ev else None,
                        "evidence_error": ev.error if ev else None,
                        "evidence_latency_seconds": ev.latency_seconds if ev else None,
                    }
                )
            record["sources"] = sources
            del record["pool"]

            # The sandbox page's own evidence for this query (input for gap analysis),
            # compressed even when it was never cited
            record["target_evidence"] = None
            if not any(s["url"] == sandbox_key for s in sources) and target_page["page_text"]:
                [own] = await compress_query_sources(
                    record["query"],
                    [{"url": url, "title": context["page_title"], "page_text": target_page["page_text"]}],
                    total_budget=config.per_source_token_cap,
                    cited_for_query=False,
                    **compression,
                )
                record["target_evidence"] = asdict(own)

        await asyncio.gather(*(compress_record(record) for record in query_records))
        await _send_progress(observation_id, "evidence_compression", "success")

        # Run-level aggregates
        all_visibilities = [v for r in query_records for v in r.pop("_visibilities")]
        citation_categories = [
            (categories[c["domain"]].category if c["domain"] in categories else "other", categories[c["domain"]].competitor_group if c["domain"] in categories else "ai_visibility")
            for r in query_records
            for run in r["runs"]
            for c in run["citations"]
            if c["is_web_source"] and c["domain"]
        ]

        def mean_of(key: str) -> Optional[float]:
            values = [r["metrics"][key] for r in query_records if r["metrics"][key] is not None]
            return sum(values) / len(values) if values else None

        result["metrics"] = {
            "queries": len(query_records),
            "runs_attempted": attempted,
            "runs_succeeded": succeeded,
            "unique_sources": len(pool_meta),
            "unique_domains": len(categories),
            "mention_rate": mention_rate(all_visibilities),
            "citation_rate": citation_rate(all_visibilities),
            "mean_citation_position": mean_citation_position(v.citation_position for v in all_visibilities),
            "median_citation_position": median_citation_position(v.citation_position for v in all_visibilities),
            "mean_source_set_diversity": mean_of("source_set_diversity"),
            "mean_source_set_stability": mean_of("source_set_stability"),
            "mean_domain_set_stability": mean_of("domain_set_stability"),
            "category_breakdown": category_breakdown((c for c, _ in citation_categories), taxonomy=config.source_categories),
            "competitor_group_breakdown": category_breakdown((g for _, g in citation_categories), taxonomy=("target", "direct_business", "ai_visibility")),
        }
        result["source_categories"] = {d: {**asdict(c), "heuristic_agrees": c.heuristic_agrees} for d, c in categories.items()}
        result["queries"] = query_records
        result["status"] = "completed"
        await _send_progress(observation_id, "completed", "success", {k: v for k, v in result["metrics"].items() if not isinstance(v, dict)})
    except Exception as exc:
        logger.exception(f"[stage1] observation {observation_id} failed")
        result["status"] = "failed"
        result["error"] = str(exc)[:1000]
        await _send_progress(observation_id, "failed", "failed", {"error": result["error"]})

    _dump_result(result)
    await _post(f"/api/observation/{observation_id}/result", result, timeout=60.0)
    return result


def _dump_result(result: Dict[str, Any]) -> None:
    """Optional local copy of the raw result (STAGE1_RESULT_DIR), for debugging."""
    directory = os.getenv("STAGE1_RESULT_DIR")
    if not directory:
        return
    try:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{result['observation_id']}.json"), "w") as f:
            json.dump(result, f, indent=2, default=str)
    except Exception as exc:
        logger.warning(f"[stage1] could not write result file: {exc}")
