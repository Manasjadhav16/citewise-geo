"""
Async Job Runner for Sandbox Analysis
Handles the full 7-step pipeline asynchronously with progress updates
"""
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import httpx

from . import crawl, sandbox, stage2_seed

logger = logging.getLogger(__name__)

# Backend URL for progress updates
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:4000")


def _log_structured(level: str, job_id: str, step: str, message: str, **kwargs):
    """Structured JSON logging."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "job_id": job_id,
        "step": step,
        "message": message,
        "level": level,
        **kwargs,
    }
    print(json.dumps(log_entry))


async def _send_progress(
    run_id: str,
    step: str,
    status: str,
    payload: Dict[str, Any] = None,
    event_id: str = None,
):
    """Send progress update to backend."""
    if event_id is None:
        event_id = str(uuid.uuid4())

    progress_data = {
        "event_id": event_id,
        "job_id": run_id,
        "step": step,
        "status": status,
        "payload": payload or {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{BACKEND_BASE_URL}/api/sandbox/{run_id}/progress",
                json=progress_data,
            )
            if response.status_code != 200:
                logger.warning(
                    f"Failed to send progress update: {response.status_code} - {response.text}"
                )
    except Exception as exc:
        logger.error(f"Error sending progress update: {exc}")


async def _send_final_result(run_id: str, result: Dict[str, Any]):
    """Send final result to backend."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BACKEND_BASE_URL}/api/sandbox/{run_id}/result",
                json=result,
            )
            if response.status_code != 200:
                logger.error(
                    f"Failed to send final result: {response.status_code} - {response.text}"
                )
            else:
                _log_structured("INFO", run_id, "final_result", "Final result sent successfully")
    except Exception as exc:
        logger.error(f"Error sending final result: {exc}")


async def run_full_job(
    run_id: str,
    url: str,
    model_name: Optional[str] = None,
    seed: Optional[Dict[str, Any]] = None,
):
    """
    Run the complete 7-step sandbox analysis pipeline.

    Args:
        run_id: Unique run identifier
        url: Sandbox URL to analyze
        model_name: Optional model name to use for LLM calls
        seed: Optional two-stage GEO seed, built by the backend:
            {"mode": "stage1", "questions": [...], "stage1_sources": [...]}:
                questions are Stage 1's queries; the source pool is Stage 1's
                union pool merged with this run's own SERP/LLM discovery.
            {"mode": "baseline", "questions": [...], "source_pool": [...]}:
                an "after" run reusing its baseline's exact questions and source
                pool (no discovery), so only the pages themselves are re-crawled.
    """
    seed = seed or {}
    seed_mode = seed.get("mode")
    pool_counts: Dict[str, Any] = {}
    # DO NOT hardcode model_name - let each step use get_model_for_role() / get_model_for_step()
    # model_name parameter is kept for backward compatibility but should be None
    if model_name:
        logger.warning(f"[model_verification] WARNING: model_name={model_name} passed to run_sandbox_analysis. This may override MODEL_ROLES. Consider removing.")
    _log_structured("INFO", run_id, "job_started", f"Starting sandbox analysis for {url}")

    try:
        # ==========================================
        # STEP 1: Sandbox Crawl
        # ==========================================
        _log_structured("INFO", run_id, "crawling_sandbox_site", "Starting sandbox page crawl")
        await _send_progress(run_id, "crawling_sandbox_site", "started")

        try:
            page_data = await crawl.crawl_url(url)
            if "error" in page_data:
                raise Exception(f"Crawl error: {page_data['error']}")

            _log_structured(
                "INFO",
                run_id,
                "crawling_sandbox_site",
                "Crawl completed",
                title=page_data.get("title"),
                content_length=len(page_data.get("full_content", "")),
            )
            await _send_progress(
                run_id,
                "crawling_sandbox_site",
                "success",
                {"title": page_data.get("title"), "h1": page_data.get("h1")},
            )
        except Exception as exc:
            _log_structured("ERROR", run_id, "crawling_sandbox_site", f"Crawl failed: {exc}")
            await _send_progress(run_id, "crawling_sandbox_site", "failed", {"error": str(exc)})
            raise

        full_text = page_data.get("full_content", "")

        # ==========================================
        # STEP 1.5: Domain Summary Generation (NEW)
        # ==========================================
        _log_structured("INFO", run_id, "domain_summary_generation", "Starting domain summary generation")
        await _send_progress(run_id, "domain_summary_generation", "started")

        domain_summary = None
        try:
            domain_summary = await sandbox.generate_domain_summary(
                page_data=page_data,
                model_name=model_name,
            )
            
            _log_structured(
                "INFO",
                run_id,
                "domain_summary_generation",
                "Domain summary generated",
                summary_length=len(domain_summary),
            )
            await _send_progress(
                run_id,
                "domain_summary_generation",
                "success",
                {"summary_length": len(domain_summary)},
            )
        except Exception as exc:
            _log_structured("ERROR", run_id, "domain_summary_generation", f"Domain summary generation failed: {exc}")
            await _send_progress(run_id, "domain_summary_generation", "failed", {"error": str(exc)})
            # Don't fail the entire job - continue without summary (will use raw content)
            logger.warning(f"Domain summary generation failed, continuing without summary: {exc}")
            domain_summary = None

        # ==========================================
        # STEP 2: Intent Extraction
        # ==========================================
        _log_structured("INFO", run_id, "intent_extraction", "Starting intent extraction")
        await _send_progress(run_id, "intent_extraction", "started")

        try:
            sandbox_data = await sandbox.analyze_sandbox_page(
                url=url,
                domain_summary=domain_summary,  # NEW: Pass domain summary
                model_name=model_name,
            )
            if "error" in sandbox_data:
                raise Exception(f"Intent extraction error: {sandbox_data['error']}")

            _log_structured(
                "INFO",
                run_id,
                "intent_extraction",
                "Intent extracted",
                intent_length=len(sandbox_data.get("intent", "")),
            )
            await _send_progress(
                run_id,
                "intent_extraction",
                "success",
                {"intent": sandbox_data.get("intent", "")[:200]},
            )
        except Exception as exc:
            _log_structured("ERROR", run_id, "intent_extraction", f"Intent extraction failed: {exc}")
            await _send_progress(run_id, "intent_extraction", "failed", {"error": str(exc)})
            raise

        # ==========================================
        # STEP 2.5: Business Category Classification (NEW)
        # ==========================================
        _log_structured("INFO", run_id, "category_classification", "Starting business category classification")
        await _send_progress(run_id, "category_classification", "started")

        business_category = None
        try:
            page_intent = sandbox_data.get("intent", "")
            page_data_for_category = {
                "title": sandbox_data.get("title"),
                "h1": sandbox_data.get("h1"),
                "full_content": sandbox_data.get("full_text", ""),
            }
            business_category = await sandbox.classify_business_category(
                page_data=page_data_for_category,
                page_intent=page_intent,
                domain_summary=domain_summary,  # NEW: Pass domain summary
                model_name=model_name,
            )
            
            _log_structured(
                "INFO",
                run_id,
                "category_classification",
                "Category classified",
                category=business_category,
            )
            await _send_progress(
                run_id,
                "category_classification",
                "success",
                {"category": business_category},
            )
        except Exception as exc:
            _log_structured("ERROR", run_id, "category_classification", f"Category classification failed: {exc}")
            await _send_progress(run_id, "category_classification", "failed", {"error": str(exc)})
            # Don't fail the entire job - continue without category
            logger.warning(f"Category classification failed, continuing without category: {exc}")

        # ==========================================
        # STEP 3: Question Generation (Category-Conditioned)
        # ==========================================
        _log_structured("INFO", run_id, "questions_generation", "Starting question generation")
        await _send_progress(run_id, "questions_generation", "started")

        try:
            page_intent = sandbox_data.get("intent", "")
            page_content = sandbox_data.get("full_text", "")

            if seed.get("questions"):
                # Seeded: Stage 1's queries, or the baseline run's exact questions
                questions = [
                    {"type": q.get("type") or "intent", "q": q["q"]}
                    for q in seed["questions"]
                    if q.get("q") and q.get("type") != "SERP"
                ]
                if not questions:
                    raise Exception("Seed contained no usable questions")
            else:
                questions = await sandbox.generate_user_questions(
                    page_intent=page_intent,
                    business_category=business_category,  # NEW: Pass category
                    domain_summary=domain_summary,  # NEW: Pass domain summary
                    model_name=model_name,
                )

                if not questions or len(questions) < 10:
                    raise Exception(f"Generated only {len(questions)} questions, expected 15")

            _log_structured(
                "INFO",
                run_id,
                "questions_generation",
                f"Generated {len(questions)} questions",
                question_types=list({q.get("type") for q in questions}),  # Convert set to list for JSON serialization
            )
            await _send_progress(
                run_id,
                "questions_generation",
                "success",
                {"count": len(questions), "types": [q.get("type") for q in questions], "seeded": bool(seed.get("questions"))},
            )
        except Exception as exc:
            _log_structured("ERROR", run_id, "questions_generation", f"Question generation failed: {exc}")
            await _send_progress(run_id, "questions_generation", "failed", {"error": str(exc)})
            raise

        # ==========================================
        # STEP 4: Extract Competitors (Hybrid: SERP + LLM)
        # ==========================================
        _log_structured("INFO", run_id, "competitors_extraction", "Starting competitor extraction (hybrid)")
        await _send_progress(run_id, "competitors_extraction", "started")

        try:
            page_intent = sandbox_data.get("intent", "")
# OLD IMPLEMENTATION (kept for reference)
# competitor_result = await sandbox.extract_competitors_from_questions(
#     questions=questions,
#     page_intent=page_intent,
#     business_category=business_category,  # NEW: Pass category for direct competitor extraction
#     sandbox_title=sandbox_data.get("title"),
#     sandbox_url=url,
#     model_name=model_name,
# )
#
# competitor_urls_list = competitor_result.get("competitor_urls", [])
# serp_queries = competitor_result.get("serp_queries", [])

            # NEW IMPLEMENTATION: Parallel execution for SERP extraction
            # Note: This step generates the queries (3 items) + extracts competitors.
            # The actual execution of these 3 queries happens downstream in the crawler/searcher.
            # For this specific step, we keep the call as is because it's a single coordinating
            # function, but we mark it as part of the parallel pipeline being orchestrated.
            if seed_mode == "baseline":
                # "After" run: the baseline's exact pool, no discovery, so the
                # score change reflects the website rather than setup variance
                competitor_urls_list = [dict(entry) for entry in seed.get("source_pool") or []]
                serp_queries = []
                pool_counts = {"reused_from_baseline": len(competitor_urls_list)}
            else:
                competitor_result = await sandbox.extract_competitors_from_questions(
                    questions=questions,
                    page_intent=page_intent,
                    business_category=business_category,
                    sandbox_title=sandbox_data.get("title"),
                    sandbox_url=url,
                    model_name=model_name,
                )
                discovered = competitor_result.get("competitor_urls", [])
                serp_queries = competitor_result.get("serp_queries", [])

                if seed_mode == "stage1":
                    stage1_pool, skipped = stage2_seed.build_stage1_pool(seed.get("stage1_sources") or [], url)
                    competitor_urls_list, pool_counts = stage2_seed.merge_source_pools(
                        stage1_pool, discovered, url, stage2_seed.max_sources_from_env()
                    )
                    pool_counts["stage1_skipped"] = skipped
                else:
                    # Unseeded: same pages as before, just labelled
                    competitor_urls_list = stage2_seed.annotate_discovered(discovered)

            if not competitor_urls_list:
                logger.warning(f"No competitors extracted for run {run_id}")

            # Add SERP queries to questions list with type "SERP"
            for serp_query in serp_queries:
                questions.append({
                    "type": "SERP",
                    "q": serp_query,
                })

            _log_structured(
                "INFO",
                run_id,
                "competitors_extraction",
                f"Extracted {len(competitor_urls_list)} competitor URLs (hybrid approach), {len(serp_queries)} SERP queries",
            )
            await _send_progress(
                run_id,
                "competitors_extraction",
                "success",
                {
                    "count": len(competitor_urls_list),
                    "serp_queries_count": len(serp_queries),
                    "urls": [c.get("url") for c in competitor_urls_list[:5]],  # First 5 for preview
                    "seed_mode": seed_mode,
                    "pool_counts": pool_counts,
                },
            )
        except Exception as exc:
            _log_structured("ERROR", run_id, "competitors_extraction", f"Competitor extraction failed: {exc}")
            await _send_progress(run_id, "competitors_extraction", "failed", {"error": str(exc)})
            raise

        # ==========================================
        # STEP 5: Crawl ALL Competitors
        # ==========================================
        _log_structured("INFO", run_id, "crawling_competitors", "Starting competitor crawling")
        await _send_progress(run_id, "crawling_competitors", "started")

        try:
            crawl_result = await sandbox.crawl_and_store_competitors(
                run_id,
                url,
                sandbox_data,
                competitor_urls_list,  # Now contains dicts with url and source_type
            )

            competitors_data = crawl_result.get("competitors", [])
            chunks_count = crawl_result.get("chunks_count", 0)

            _log_structured(
                "INFO",
                run_id,
                "crawling_competitors",
                f"Crawled {len(competitors_data)} competitors, stored {chunks_count} chunks",
            )
            await _send_progress(
                run_id,
                "crawling_competitors",
                "success",
                {
                    "competitors_count": len(competitors_data),
                    "chunks_count": chunks_count,
                },
            )
        except Exception as exc:
            _log_structured("ERROR", run_id, "crawling_competitors", f"Competitor crawling failed: {exc}")
            await _send_progress(run_id, "crawling_competitors", "failed", {"error": str(exc)})
            raise

        # ==========================================
        # STEP 6: RAG for All Questions
        # ==========================================
        _log_structured("INFO", run_id, "rag_simulation", "Starting RAG simulation")
        await _send_progress(run_id, "rag_simulation", "started")

# OLD IMPLEMENTATION (kept for reference)
# rag_results = []
# try:
#     for i, question_dict in enumerate(questions):
#         try:
#             _log_structured(
#                 "INFO",
#                 run_id,
#                 "rag_simulation",
#                 f"Processing question {i+1}/{len(questions)}",
#                 question=question_dict.get("q", "")[:50],
#             )
#
#             # Process single question (we'll batch this in the actual function)
#             # For now, we'll process all at once and send partial updates
#             pass
#         except Exception as exc:
#             logger.warning(f"Error processing question {i+1}: {exc}")
#
#     # Extract business competitors list for RAG context
#     business_competitors_list = []
#     if competitor_urls_list:
#         # Extract names/domains from business_competitor type
#         for comp in competitor_urls_list:
#             if comp.get("type") == "business_competitor":
#                 comp_url = comp.get("url", "")
#                 if comp_url:
#                     try:
#                         comp_domain = urlparse(comp_url).netloc.lower().replace("www.", "")
#                         business_competitors_list.append(comp_domain.split(".")[0].capitalize())
#                     except Exception:
#                         pass
#
#     # Extract sandbox brand name
#     # Prefer the one extracted by LLM in Step 2 if available
#     sandbox_brand_name = sandbox_data.get("page_brand_name")
#     if not sandbox_brand_name:
#         try:
#             domain = urlparse(url).netloc.lower().replace("www.", "")
#             sandbox_brand_name = domain.split(".")[0].capitalize()
#         except Exception:
#             pass
#
#     # Run RAG simulation for all questions with business context
#     rag_results = await sandbox.run_rag_simulation(
#         run_id=run_id,
#         questions=questions,
#         sandbox_url=url,
#         business_category=business_category,  # NEW: Pass category
#         business_competitors=business_competitors_list if business_competitors_list else None,  # NEW: Pass competitors
#         competitor_urls=competitor_urls_list,  # NEW: Pass full competitor list for classification
#         sandbox_brand_name=sandbox_brand_name,  # NEW: Pass brand name
#         model_name=model_name,
#     )
#
#     _log_structured(
#         "INFO",
#         run_id,
#         "rag_simulation",
#         f"RAG simulation completed for {len(rag_results)} questions",
#         sandbox_appeared=sum(1 for r in rag_results if r.get("did_sandbox_appear", False)),
#     )
#     await _send_progress(
#         run_id,
#         "rag_simulation",
#         "success",
#         {
#             "results_count": len(rag_results),
#             "sandbox_appeared": sum(
#                 1 for r in rag_results if r.get("did_sandbox_appear", False)
#             ),
#         },
#     )

# NEW IMPLEMENTATION: Parallel execution using asyncio.gather
        # Prepare context data needed for each question
        business_competitors_list = []
        if competitor_urls_list:
            for comp in competitor_urls_list:
                if comp.get("type") == "business_competitor":
                    comp_url = comp.get("url", "")
                    if comp_url:
                        try:
                            comp_domain = urlparse(comp_url).netloc.lower().replace("www.", "")
                            business_competitors_list.append(comp_domain.split(".")[0].capitalize())
                        except Exception:
                            pass

        sandbox_brand_name = sandbox_data.get("page_brand_name")
        if not sandbox_brand_name:
            try:
                domain = urlparse(url).netloc.lower().replace("www.", "")
                sandbox_brand_name = domain.split(".")[0].capitalize()
            except Exception:
                pass

        # domain -> (category, group) for tagging every cited and retrieved source
        source_categories = stage2_seed.category_lookup(competitor_urls_list or [])

        # Create list of coroutines for parallel execution
        rag_tasks = []
        total_questions = len(questions)
        
        # Wrapper to add real-time console logging for parallel verification
        async def _logged_rag_simulation(index, q_dict):
            start_ts = datetime.now().strftime("%H:%M:%S")
            # Print to console for immediate user visibility
            print(f"[{start_ts}] START {index}/{total_questions}: {q_dict.get('type', 'unknown')} - {q_dict.get('q', '')[:40]}...")
            
            try:
                result = await sandbox.run_rag_simulation(
                    run_id=run_id,
                    questions=[q_dict],
                    sandbox_url=url,
                    business_category=business_category,
                    business_competitors=business_competitors_list if business_competitors_list else None,
                    competitor_urls=competitor_urls_list,
                    sandbox_brand_name=sandbox_brand_name,
                    model_name=model_name,
                    source_categories=source_categories,
                )

                end_ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{end_ts}] DONE  {index}/{total_questions}: {q_dict.get('type', 'unknown')}")
                return result
            except Exception as e:
                end_ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{end_ts}] FAIL  {index}/{total_questions}: {q_dict.get('type', 'unknown')} - {str(e)}")
                raise e

        rag_sem = asyncio.Semaphore(1)
        async def _bounded_rag_task(index: int, q_dict: dict):
            async with rag_sem:
                res = await _logged_rag_simulation(index, q_dict)
                await asyncio.sleep(1.0)
                return res

        for i, q_dict in enumerate(questions):
            rag_tasks.append(
                _bounded_rag_task(i + 1, q_dict)
            )

        try:
            # Execute all RAG tasks concurrently
            # return_exceptions=True ensures that one failure doesn't crash the whole batch
            raw_results = await asyncio.gather(*rag_tasks, return_exceptions=True)
            
            rag_results = []
            for i, result in enumerate(raw_results):
                question_text = questions[i].get("q", "")
                question_type = questions[i].get("type", "intent")

                if isinstance(result, Exception):
                    # CRITICAL: Replicate original per-question error handling behavior.
                    # In the original loop, a failure on Q1 didn't stop Q2. 
                    # We log the warning and append a structured error result.
                    logger.warning(f"Error processing question {i+1} ('{question_text[:30]}...'): {result}")
                    
                    rag_results.append({
                        "question": question_text,
                        "type": question_type,
                        "answer": f"Error during RAG simulation: {str(result)}",
                        "did_sandbox_appear": False,
                        "chunks_used": 0,
                        "competitor_citations": [],
                        "sandbox_citations": [],
                        "business_competitor_citations": [],
                        "authority_source_citations": [],
                        "metrics": {}
                    })
                elif isinstance(result, list):
                    # run_rag_simulation returns a list of results (for backward compatibility)
                    # We expect exactly one result since we passed a list of one question
                    if result:
                        rag_results.append(result[0])
                    else:
                        # Should technically not happen if logic is correct, but safer to handle
                        logger.warning(f"Empty result list returned for question {i+1}")
                else:
                    # Fallback for unexpected return type
                    logger.error(f"Unexpected return type for question {i+1}: {type(result)}")

            _log_structured(
                "INFO",
                run_id,
                "rag_simulation",
                f"RAG simulation completed for {len(rag_results)} questions",
                sandbox_appeared=sum(1 for r in rag_results if r.get("did_sandbox_appear", False)),
            )
            await _send_progress(
                run_id,
                "rag_simulation",
                "success",
                {
                    "results_count": len(rag_results),
                    "sandbox_appeared": sum(
                        1 for r in rag_results if r.get("did_sandbox_appear", False)
                    ),
                },
            )
        except Exception as exc:
            # This outer catch is for catastrophic failures in the gather orchestration itself,
            # not for individual task failures (which are handled above via return_exceptions=True)
            _log_structured("ERROR", run_id, "rag_simulation", f"RAG simulation failed: {exc}")
            await _send_progress(run_id, "rag_simulation", "failed", {"error": str(exc)})
            raise

        # ==========================================
        # STEP 7: Final Scoring
        # ==========================================
        _log_structured("INFO", run_id, "score_calculation", "Starting final scoring")
        await _send_progress(run_id, "score_calculation", "started")
        try:
            # competitor_urls_list is defined in Step 4, should be in scope here
            scores = await sandbox.calculate_final_scores(
                rag_results=rag_results,
                questions=questions,
                business_category=business_category,  # NEW: Pass category for reality score
                competitor_urls=competitor_urls_list if 'competitor_urls_list' in locals() else None,  # NEW: Pass competitors for reality score
            )
            # Two-stage GEO metrics, alongside (not replacing) the GEO/AEO scores
            scores.update(stage2_seed.summarize_stage2_metrics(rag_results))

            _log_structured(
                "INFO",
                run_id,
                "score_calculation",
                "Scoring completed",
                geo_score=scores.get("geo_score"),
                aeo_score=scores.get("aeo_score"),
            )

            # Prepare final result (sanitize all text fields to remove null bytes)
            from .sandbox import _sanitize_text
            
            final_result = {
                "sandbox_page": {
                    "url": url,
                    "title": _sanitize_text(sandbox_data.get("title")),
                    "h1": _sanitize_text(sandbox_data.get("h1")),
                    "intent": _sanitize_text(sandbox_data.get("intent")),
                    "page_summary": _sanitize_text(sandbox_data.get("page_summary")),
                    "brand_name": _sanitize_text(sandbox_data.get("brand_name")),
                    "ai_summary": _sanitize_text(sandbox_data.get("ai_summary")),
                    "full_text": _sanitize_text(sandbox_data.get("full_text")),
                },
                "questions": [
                    {"type": q.get("type"), "q": _sanitize_text(q.get("q")) or ""}
                    for q in questions
                ],
                "competitors": competitors_data,  # Already sanitized in crawl_and_store_competitors
                # Two-stage GEO: the exact pool crawled (reused by "after" runs) and how it was built
                "source_pool": competitor_urls_list,
                "seed_mode": seed_mode,
                "pool_counts": pool_counts,
                "rag_results": [
                    {
                        **r,
                        "question": _sanitize_text(r.get("question")) or "",
                        "answer": _sanitize_text(r.get("answer")) or "",
                        "metrics": r.get("metrics", {}),
                    }
                    for r in rag_results
                ],
                "scores": {
                    **scores,
                    "recommendations": [
                        _sanitize_text(rec) or "" for rec in scores.get("recommendations", [])
                    ],
                    "missing_content": [
                        _sanitize_text(item) or "" for item in scores.get("missing_content", [])
                    ],
                    "schema_gaps": [
                        _sanitize_text(item) or "" for item in scores.get("schema_gaps", [])
                    ],
                    "structured_data_ops": [
                        _sanitize_text(item) or "" for item in scores.get("structured_data_ops", [])
                    ],
                },
            }

            # Send final result to backend
            await _send_final_result(run_id, final_result)

            _log_structured("INFO", run_id, "job_completed", "Job completed successfully")

        except Exception as exc:
            _log_structured("ERROR", run_id, "score_calculation", f"Final scoring failed: {exc}")
            await _send_progress(run_id, "score_calculation", "failed", {"error": str(exc)})
            raise

    except Exception as exc:
        _log_structured("ERROR", run_id, "job_failed", f"Job failed: {exc}")
        await _send_progress(run_id, "job_failed", "failed", {"error": str(exc)})
        raise

