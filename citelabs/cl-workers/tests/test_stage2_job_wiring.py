"""
Runs the real async_job.run_full_job orchestration in each seed mode with every
LLM, crawl, vector-store and backend call faked, and checks what reaches the
backend. Importing the pipeline takes a few seconds (sandbox.py's model libraries).
"""
import asyncio

import pytest

from app import async_job, crawl, sandbox, stage2_seed

URL = "https://acme.in/payment-gateway"
GENERATED = [{"type": t, "q": f"generated {t} {i}"} for t in ("intent", "experience", "transaction") for i in range(5)]
DISCOVERED = [
    {"url": "https://rival.com/pricing", "type": "business_competitor", "source": "llm"},
    {"url": "https://www.news.com/story", "type": "authority_source", "source": "serp"},
    {"url": "https://en.wikipedia.org/wiki/UPI", "type": "authority_source", "source": "serp"},
]
STAGE1_SOURCES = [
    {"url": "https://news.com/best-gateways", "domain": "news.com", "category": "media", "competitor_group": "ai_visibility", "category_source": "llm", "stability": 1.0},
    {"url": "https://payu.in/fees", "domain": "payu.in", "category": "direct_competitor", "competitor_group": "direct_business", "category_source": "llm", "stability": 0.67},
    {"url": "https://acme.in/blog", "domain": "acme.in", "category": "target_company", "competitor_group": "target", "category_source": "target_domain_match", "stability": 1.0},
]


@pytest.fixture
def fake_pipeline(monkeypatch):
    calls = {"generate_questions": 0, "discovery": 0, "crawled": None, "rag_questions": [], "lookup": None, "final": None}

    async def fake_crawl_url(url):
        return {"title": "Acme", "h1": "Acme", "full_content": "Acme payment gateway page"}

    async def fake_summary(page_data, model_name=None):
        return "summary"

    async def fake_analyze(url, domain_summary=None, model_name=None):
        return {"intent": "payment gateway", "title": "Acme", "h1": "Acme", "full_text": "text", "page_brand_name": "Acme"}

    async def fake_category(page_data, page_intent, domain_summary=None, model_name=None):
        return "Payment gateway"

    async def fake_generate(**kwargs):
        calls["generate_questions"] += 1
        return list(GENERATED)

    async def fake_discovery(**kwargs):
        calls["discovery"] += 1
        return {"competitor_urls": [dict(d) for d in DISCOVERED], "serp_queries": ["serp q"]}

    async def fake_crawl_all(run_id, url, sandbox_data, competitor_urls):
        calls["crawled"] = [dict(c) for c in competitor_urls]
        return {
            "competitors": [{"url": c["url"], "domain": c["url"].split("/")[2], "source_category": c.get("source_category")} for c in competitor_urls],
            "chunks_count": 10,
        }

    async def fake_rag(run_id, questions, sandbox_url, source_categories=None, **kwargs):
        calls["lookup"] = source_categories
        q = questions[0]
        if q["type"] == "SERP":
            return []
        calls["rag_questions"].append(q["q"])
        answer = f"See [Source:https://rival.com/pricing] and [Source:{URL}]."
        return [{
            "question": q["q"], "type": q["type"], "answer": answer, "did_sandbox_appear": True, "chunks_used": 4,
            "competitor_citations": ["https://rival.com/pricing"], "sandbox_citations": [URL],
            "business_competitor_citations": ["https://rival.com/pricing"], "authority_source_citations": [],
            "metrics": {"is_brand_mentioned": True, "is_brand_cited": True},
            "diagnosis_type": None, "diagnosis_detail": None,
            **stage2_seed.tag_rag_result(answer, sandbox_url, [URL, "https://rival.com/pricing"], source_categories or {}),
        }]

    async def fake_scores(**kwargs):
        return {"geo_score": 70.0, "aeo_score": 60.0, "citation_rate": 100.0, "coverage": 40.0, "competitor_dominance": 50.0,
                "recommendations": [], "missing_content": [], "schema_gaps": [], "structured_data_ops": []}

    async def fake_progress(*args, **kwargs):
        return None

    async def fake_final(run_id, result):
        calls["final"] = result

    async def no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(crawl, "crawl_url", fake_crawl_url)
    monkeypatch.setattr(sandbox, "generate_domain_summary", fake_summary)
    monkeypatch.setattr(sandbox, "analyze_sandbox_page", fake_analyze)
    monkeypatch.setattr(sandbox, "classify_business_category", fake_category)
    monkeypatch.setattr(sandbox, "generate_user_questions", fake_generate)
    monkeypatch.setattr(sandbox, "extract_competitors_from_questions", fake_discovery)
    monkeypatch.setattr(sandbox, "crawl_and_store_competitors", fake_crawl_all)
    monkeypatch.setattr(sandbox, "run_rag_simulation", fake_rag)
    monkeypatch.setattr(sandbox, "calculate_final_scores", fake_scores)
    monkeypatch.setattr(async_job, "_send_progress", fake_progress)
    monkeypatch.setattr(async_job, "_send_final_result", fake_final)
    monkeypatch.setattr(async_job.asyncio, "sleep", no_sleep)
    monkeypatch.setenv("STAGE2_MAX_SOURCES", "40")
    return calls


def run(seed=None):
    asyncio.run(async_job.run_full_job("run_test", URL, None, seed))


def test_unseeded_run_keeps_existing_behaviour(fake_pipeline):
    run()
    final = fake_pipeline["final"]
    assert fake_pipeline["generate_questions"] == 1 and fake_pipeline["discovery"] == 1
    # Same pages as discovery returned, now labelled
    assert [c["url"] for c in fake_pipeline["crawled"]] == [d["url"] for d in DISCOVERED]
    assert [c["source_category"] for c in fake_pipeline["crawled"]] == ["direct_competitor", "other", "reference"]
    assert final["seed_mode"] is None
    assert len(fake_pipeline["rag_questions"]) == 15
    # Existing scores untouched, new metrics added alongside
    assert final["scores"]["geo_score"] == 70.0 and final["scores"]["citation_rate"] == 100.0
    assert final["scores"]["strict_citation_rate"] == 100.0 and final["scores"]["mean_citation_position"] == 2.0
    assert final["rag_results"][0]["cited_source_categories"][0]["category"] == "direct_competitor"


def test_stage1_seed_uses_queries_and_merges_pool(fake_pipeline):
    seed = {
        "mode": "stage1",
        "questions": [{"q": "stage1 query A", "type": "intent"}, {"q": "stage1 query B", "type": "provided"}, {"q": ""}],
        "stage1_sources": STAGE1_SOURCES,
    }
    run(seed)
    final = fake_pipeline["final"]
    assert fake_pipeline["generate_questions"] == 0
    assert fake_pipeline["rag_questions"] == ["stage1 query A", "stage1 query B"]
    assert fake_pipeline["discovery"] == 1  # discovery still runs and is merged in
    crawled = [(c["url"], c["source_category"], c["origin"]) for c in fake_pipeline["crawled"]]
    assert crawled == [
        ("https://news.com/best-gateways", "media", "stage1"),
        ("https://payu.in/fees", "direct_competitor", "stage1"),
        ("https://rival.com/pricing", "direct_competitor", "discovery"),
        ("https://en.wikipedia.org/wiki/UPI", "reference", "discovery"),
    ]  # news.com discovered page dropped as same domain; acme.in blog excluded as target
    assert final["seed_mode"] == "stage1"
    assert final["pool_counts"]["discovered_duplicate_domains"] == 1
    assert final["pool_counts"]["stage1_skipped"]["target_domain"] == 1
    assert final["source_pool"] == fake_pipeline["crawled"]
    assert fake_pipeline["lookup"]["payu.in"] == ("direct_competitor", "direct_business")


def test_baseline_seed_reuses_pool_without_discovery(fake_pipeline):
    pool = [
        {"url": "https://news.com/best-gateways", "type": "authority_source", "source_category": "media", "competitor_group": "ai_visibility", "origin": "stage1"},
        {"url": "https://payu.in/fees", "type": "business_competitor", "source_category": "direct_competitor", "competitor_group": "direct_business", "origin": "stage1"},
    ]
    seed = {"mode": "baseline", "questions": [{"q": "stage1 query A", "type": "intent"}, {"q": "old serp", "type": "SERP"}], "source_pool": pool}
    run(seed)
    final = fake_pipeline["final"]
    assert fake_pipeline["generate_questions"] == 0 and fake_pipeline["discovery"] == 0
    assert fake_pipeline["crawled"] == pool  # exactly the baseline's pool
    assert fake_pipeline["rag_questions"] == ["stage1 query A"]  # SERP rows from the baseline are not re-asked
    assert final["seed_mode"] == "baseline" and final["pool_counts"] == {"reused_from_baseline": 2}
    assert final["source_pool"] == pool
