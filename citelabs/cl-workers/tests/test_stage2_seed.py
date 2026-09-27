import pytest

from app.stage2_seed import (
    annotate_discovered,
    build_stage1_pool,
    category_lookup,
    extract_cited_urls,
    is_error_result,
    merge_source_pools,
    pipeline_type,
    summarize_stage2_metrics,
    tag_rag_result,
)

SANDBOX = "https://www.acme.in/payment-gateway/"


def src(url, category, stability, domain=None, group=None):
    return {
        "url": url,
        "domain": domain,
        "category": category,
        "competitor_group": group,
        "category_source": "llm",
        "stability": stability,
    }


class TestBuildStage1Pool:
    def test_dedupes_by_domain_keeping_most_stable_page(self):
        pool, skipped = build_stage1_pool(
            [
                src("https://rival.com/a", "direct_competitor", 0.33),
                src("https://rival.com/b", "direct_competitor", 1.0),
                src("https://www.rival.com/c", "direct_competitor", 0.67),
                src("https://news.com/x", "media", 0.67),
            ],
            SANDBOX,
        )
        assert [p["url"] for p in pool] == ["https://rival.com/b", "https://news.com/x"]
        assert skipped["same_domain_duplicate"] == 2

    def test_tie_keeps_first_seen(self):
        pool, _ = build_stage1_pool(
            [src("https://a.com/first", "media", 0.5), src("https://a.com/second", "media", 0.5)], SANDBOX
        )
        assert pool[0]["url"] == "https://a.com/first"

    def test_excludes_target_and_unresolved(self):
        pool, skipped = build_stage1_pool(
            [
                src("https://acme.in/blog", "target_company", 1.0),
                src("https://docs.acme.in/api", "other", 1.0),  # target subdomain, even if mislabelled
                src(None, "media", 1.0, domain="unresolved.com"),
                src("https://ok.org", "industry_organization", 0.3),
            ],
            SANDBOX,
        )
        assert [p["url"] for p in pool] == ["https://ok.org"]
        assert skipped == {"unresolved_url": 1, "target_domain": 2, "same_domain_duplicate": 0}

    def test_preserves_stage1_labels_and_maps_types(self):
        pool, _ = build_stage1_pool(
            [src("https://rival.com", "direct_competitor", 1.0), src("https://rbi.org.in", "government", 0.5)], SANDBOX
        )
        rival, rbi = pool
        assert (rival["type"], rival["source_category"], rival["competitor_group"], rival["origin"]) == (
            "business_competitor", "direct_competitor", "direct_business", "stage1",
        )
        assert (rbi["type"], rbi["source_category"], rbi["competitor_group"]) == ("authority_source", "government", "ai_visibility")

    def test_pipeline_type(self):
        assert pipeline_type("direct_competitor") == "business_competitor"
        assert pipeline_type("indirect_competitor") == "authority_source"


class TestMergeSourcePools:
    STAGE1 = [
        {"url": "https://rival.com/b", "type": "business_competitor", "source_category": "direct_competitor", "competitor_group": "direct_business", "origin": "stage1"},
        {"url": "https://news.com/x", "type": "authority_source", "source_category": "media", "competitor_group": "ai_visibility", "origin": "stage1"},
    ]

    def test_stage1_first_and_its_labels_win(self):
        discovered = [
            {"url": "https://www.news.com/other", "type": "business_competitor", "source": "llm"},  # same domain as Stage 1
            {"url": "https://newrival.com", "type": "business_competitor", "source": "llm"},
            {"url": "https://en.wikipedia.org/wiki/UPI", "type": "authority_source", "source": "serp"},
            {"url": "https://acme.in/pricing", "type": "authority_source", "source": "serp"},  # target
        ]
        merged, counts = merge_source_pools(self.STAGE1, discovered, SANDBOX, max_sources=10)
        assert [m["url"] for m in merged] == [
            "https://rival.com/b", "https://news.com/x", "https://newrival.com", "https://en.wikipedia.org/wiki/UPI",
        ]
        assert merged[1]["source_category"] == "media"  # not overwritten by discovery's business_competitor
        assert (merged[2]["source_category"], merged[2]["category_source"], merged[2]["origin"]) == ("direct_competitor", "discovery", "discovery")
        assert (merged[3]["source_category"], merged[3]["category_source"]) == ("reference", "heuristic")
        assert counts == {"stage1": 2, "discovered_added": 2, "discovered_duplicate_domains": 1, "dropped_over_cap": 0}

    def test_cap_trims_discovered_before_stage1(self):
        discovered = [{"url": f"https://d{i}.com", "type": "authority_source"} for i in range(5)]
        merged, counts = merge_source_pools(self.STAGE1, discovered, SANDBOX, max_sources=3)
        assert [m["url"] for m in merged] == ["https://rival.com/b", "https://news.com/x", "https://d0.com"]
        assert counts["dropped_over_cap"] == 4

    def test_invalid_cap(self):
        with pytest.raises(ValueError):
            merge_source_pools([], [], SANDBOX, max_sources=0)

    def test_annotate_keeps_every_page(self):
        discovered = [
            {"url": "https://a.com/1", "type": "business_competitor"},
            {"url": "https://a.com/2", "type": "business_competitor"},  # same domain: kept
            {"url": "https://blog.example/x", "type": "authority_source"},
        ]
        annotated = annotate_discovered(discovered)
        assert [a["url"] for a in annotated] == [d["url"] for d in discovered]
        assert [a["source_category"] for a in annotated] == ["direct_competitor", "direct_competitor", "other"]


class TestTagging:
    LOOKUP = category_lookup(
        [
            {"url": "https://rival.com/b", "source_category": "direct_competitor", "competitor_group": "direct_business"},
            {"url": "https://news.com/x", "source_category": "media", "competitor_group": "ai_visibility"},
        ]
    )

    def test_extract_cited_urls_in_order(self):
        answer = "Fees [Source:https://news.com/x]. Also [Source:https://acme.in/p] and www.rival.com/b."
        assert extract_cited_urls(answer) == ["https://news.com/x", "https://acme.in/p", "https://www.rival.com/b."]

    def test_tags(self):
        answer = (
            "Rival is cheapest [Source:https://www.rival.com/b]. News agrees [Source:https://news.com/x]. "
            "Acme settles fastest [Source:https://acme.in/payment-gateway]. Rival again [Source:https://www.rival.com/b]."
        )
        tags = tag_rag_result(
            answer, SANDBOX,
            retrieved_chunk_urls=["https://acme.in/p", "https://rival.com/b", "https://rival.com/b", "https://unknown.io"],
            lookup=self.LOOKUP,
        )
        assert tags["sandbox_citation_position"] == 3
        assert [(c["domain"], c["category"]) for c in tags["cited_source_categories"]] == [
            ("rival.com", "direct_competitor"), ("news.com", "media"), ("acme.in", "target_company"),
        ]
        assert tags["retrieved_source_categories"] == {"target_company": 1, "direct_competitor": 2, "other": 1}

    def test_no_citations(self):
        tags = tag_rag_result("The context does not contain this.", SANDBOX, [], {})
        assert tags == {"sandbox_citation_position": None, "cited_source_categories": [], "retrieved_source_categories": {}}


def rag(mentioned, sandbox_cited, position=None, categories=(), answer="ok", chunks=5):
    return {
        "answer": answer,
        "chunks_used": chunks,
        "sandbox_citations": ["https://acme.in"] if sandbox_cited else [],
        "metrics": {"is_brand_mentioned": mentioned},
        "sandbox_citation_position": position,
        "cited_source_categories": [
            {"category": c, "competitor_group": g} for c, g in categories
        ],
    }


class TestSummary:
    def test_metrics(self):
        results = [
            rag(True, True, 2, [("direct_competitor", "direct_business"), ("target_company", "target")]),
            rag(True, False, None, [("media", "ai_visibility"), ("media", "ai_visibility")]),
            rag(False, True, 1, [("target_company", "target")]),
            rag(False, False, None, answer="Error: 429", chunks=0),  # excluded
        ]
        m = summarize_stage2_metrics(results, taxonomy=("target_company", "direct_competitor", "media", "other"))
        assert m["responses_evaluated"] == 3
        assert m["mention_rate"] == pytest.approx(200 / 3)
        assert m["strict_citation_rate"] == pytest.approx(200 / 3)
        assert (m["mean_citation_position"], m["median_citation_position"]) == (1.5, 1.5)
        assert m["category_breakdown"] == {"target_company": 40.0, "direct_competitor": 20.0, "media": 40.0, "other": 0.0}
        assert m["competitor_group_breakdown"] == {"target": 40.0, "direct_business": 20.0, "ai_visibility": 40.0}

    def test_position_ignored_when_not_strictly_cited(self):
        # position from a subdomain match without an exact-domain sandbox citation
        m = summarize_stage2_metrics([rag(False, False, 1)], taxonomy=("other",))
        assert m["mean_citation_position"] is None and m["strict_citation_rate"] == 0.0

    def test_empty(self):
        m = summarize_stage2_metrics([], taxonomy=("other",))
        assert (m["responses_evaluated"], m["mention_rate"], m["mean_citation_position"]) == (0, 0.0, None)

    def test_error_detection(self):
        assert is_error_result({"answer": "Error: boom", "chunks_used": 0})
        assert not is_error_result({"answer": "Error codes are explained here", "chunks_used": 4})
