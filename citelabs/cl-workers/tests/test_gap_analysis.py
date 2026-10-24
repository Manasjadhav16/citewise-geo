import asyncio

import pytest

from app import gap_analysis as ga
from app.gap_analysis import (
    EvidenceItem,
    apply_judgements,
    count_claims,
    parse_evidence_items,
    parse_json_reply,
    quote_in_text,
    rank_gaps,
    run_gap_analysis,
    select_excerpt,
    select_sources,
    source_token_budget,
)

SANDBOX = "https://acme.in/payment-gateway/"


def run(coro):
    return asyncio.run(coro)


def src(url, appearances, group="ai_visibility", category="media", stability=None, position=None):
    return {"url": url, "domain": url.split("/")[2], "category": category, "competitor_group": group,
            "appearances": appearances, "stability": stability if stability is not None else appearances / 5,
            "best_position": position}


def item(item_id, text="t", quote="q"):
    return EvidenceItem(item_id, item_id.split(".")[0], "fact", text, quote)


class TestSelection:
    def test_top_cited_by_appearances_then_position_and_targets(self):
        sources = [
            src("https://a.com/1", 3, position=4),
            src("https://b.com/1", 3, position=1),  # same appearances, cited higher
            src("https://c.com/1", 2),
            src("https://acme.in/blog/fees", 2, group="target", category="target_company"),
            src("https://acme.in/payment-gateway", 1, group="target", category="target_company"),  # the submitted page
            src("https://d.com/1", 1),
            {"url": None, "domain": "x.com", "appearances": 9, "competitor_group": "ai_visibility"},
        ]
        cited, targets = select_sources(sources, SANDBOX, top_n=3)
        assert [c["url"] for c in cited] == ["https://b.com/1", "https://a.com/1", "https://c.com/1"]
        assert [t["url"] for t in targets] == [SANDBOX, "https://acme.in/blog/fees"]
        assert targets[0]["submitted"] and targets[0]["appearances"] == 1

    def test_budget_follows_stage1_rule(self):
        assert source_token_budget(7) == 1000  # min(cap 1000, 20000 // 7)
        assert source_token_budget(40) == 500
        assert source_token_budget(500) == 200  # floor


class TestQuotes:
    def test_quote_match_ignores_case_punctuation_and_spacing(self):
        page = "Razorpay charges 2% per transaction—no setup fee.\n  UPI is free."
        assert quote_in_text("razorpay charges 2%  per transaction - no setup fee", page)
        assert not quote_in_text("charges 3% per transaction", page)
        assert not quote_in_text("UPI", page)  # too short to mean anything

    def test_reply_shapes(self):
        assert ga.reply_entries({"items": [1]}, "items") == ([1], "object:items")
        assert ga.reply_entries([1, 2], "items") == ([1, 2], "list")
        assert ga.reply_entries({"evidence": [3], "note": "x"}, "items") == ([3], "object:evidence")
        assert ga.reply_entries({"a": 1}, "items") == ([], "object:a")
        page = "UPI payments carry zero MDR under current rules."
        items, _ = parse_evidence_items([{"type": "fact", "text": "UPI zero MDR", "quote": "UPI payments carry zero MDR"}], "S1", page, 1000)
        assert len(items) == 1  # a bare list reply is not silently lost

    def test_parse_json_reply_tolerates_fences(self):
        assert parse_json_reply('```json\n{"items": []}\n```') == {"items": []}
        assert parse_json_reply('Here you go: {"a": 1} thanks') == {"a": 1}

    def test_evidence_items_drop_unverified_invalid_and_over_budget(self):
        page = "The MDR on credit cards is 2% for most merchants. UPI payments carry zero MDR under current rules."
        reply = {"items": [
            {"type": "figure", "text": "Card MDR is 2%", "quote": "The MDR on credit cards is 2%"},
            {"type": "figure", "text": "Card MDR is 3%", "quote": "card MDR is 3%"},  # not on the page
            {"type": "opinion", "text": "x", "quote": "UPI payments carry zero MDR"},  # unknown type
            {"type": "fact", "text": "UPI has zero MDR", "quote": "UPI payments carry zero MDR"},
        ]}
        items, dropped = parse_evidence_items(reply, "S2", page, token_budget=1000)
        assert [(i.item_id, i.text) for i in items] == [("S2.1", "Card MDR is 2%"), ("S2.2", "UPI has zero MDR")]
        assert dropped == {"unverified_quote": 1, "invalid": 1, "over_budget": 0}
        items, dropped = parse_evidence_items(reply, "S2", page, token_budget=12)
        assert len(items) == 1 and dropped["over_budget"] == 1


class TestClaimCounting:
    ITEMS = [item("S1.1"), item("S1.2"), item("S2.1"), item("S3.1"), item("S3.2", text="Only S3 says this")]

    def test_counts_distinct_sources_not_items(self):
        reply = {"claims": [
            {"claim": "Card MDR is about 2%", "item_ids": ["S1.1", "S1.2", "S2.1"]},  # two items from S1
            {"claim": "UPI is free", "item_ids": ["S3.1"]},
        ]}
        claims = count_claims(reply, self.ITEMS, {"S1": 3, "S2": 2, "S3": 1})
        assert [(c.claim, c.source_count, c.sources) for c in claims] == [
            ("Card MDR is about 2%", 2, ["S1", "S2"]),
            ("UPI is free", 1, ["S3"]),
            ("Only S3 says this", 1, ["S3"]),  # left out by the model: kept as its own claim
        ]
        assert [c.claim_id for c in claims] == ["C1", "C2", "C3"]

    def test_ignores_unknown_and_reused_ids(self):
        reply = {"claims": [
            {"claim": "A", "item_ids": ["S1.1", "S9.9"]},
            {"claim": "B", "item_ids": ["S1.1", "S2.1"]},  # S1.1 already belongs to A
            {"claim": "C", "item_ids": ["S7.1"]},  # nothing valid: dropped
        ]}
        claims = count_claims(reply, self.ITEMS[:3], {})
        assert [(c.claim, c.item_ids) for c in claims if c.claim in ("A", "B")] == [("A", ["S1.1"]), ("B", ["S2.1"])]
        assert "C" not in [c.claim for c in claims]

    def test_ties_broken_by_appearances_then_order(self):
        reply = {"claims": [{"claim": "low", "item_ids": ["S1.1"]}, {"claim": "high", "item_ids": ["S2.1"]}]}
        claims = count_claims(reply, [item("S1.1"), item("S2.1")], {"S1": 1, "S2": 3})
        assert [c.claim for c in claims] == ["high", "low"]

    def test_bad_reply_falls_back_to_one_claim_per_item(self):
        claims = count_claims("not json", [item("S1.1", "x"), item("S2.1", "y")], {})
        assert [(c.claim, c.source_count) for c in claims] == [("x", 1), ("y", 1)]


class TestJudgements:
    TARGET = {"https://acme.in/p": "Acme charges 2% per card payment on all standard plans for Indian merchants. "
                                   "Settlement happens in T+2 days."}

    def claims(self):
        return count_claims({"claims": [
            {"claim": "Card MDR is 2%", "item_ids": ["S1.1", "S2.1"]},
            {"claim": "Instant settlement is offered", "item_ids": ["S1.2"]},
            {"claim": "UPI MDR is zero", "item_ids": ["S2.2"]},
        ]}, [item("S1.1"), item("S1.2"), item("S2.1"), item("S2.2")], {})

    def test_statuses_quotes_and_flags(self):
        reply = {"judgements": [
            {"claim_id": "C1", "status": "covered", "target_quote": "Acme charges 2% per card payment on all standard plans", "suggestion": None},
            {"claim_id": "C2", "status": "covered", "target_quote": "Acme settles instantly"},  # quote not on the page
            {"claim_id": "C3", "status": "missing", "target_quote": None,
             "suggestion": "State your UPI MDR; competitors say 0.3%. [add your real figure]"},
        ]}
        rows = {r["claim_id"]: r for r in apply_judgements(self.claims(), reply, self.TARGET, {"S1": "https://a.com", "S2": "https://b.com"})}
        assert rows["C1"]["target_status"] == "covered" and rows["C1"]["target_url"] == "https://acme.in/p"
        assert rows["C1"]["suggestion"] is None and rows["C1"]["sources"] == ["https://a.com", "https://b.com"]
        assert rows["C2"]["target_status"] == "partly"
        assert rows["C2"]["flags"] == ["target_quote_not_found", "covered_without_verified_quote"]
        assert rows["C3"]["target_status"] == "missing" and rows["C3"]["target_quote"] is None
        assert "suggestion_figures_not_on_target_page:0.3" in rows["C3"]["flags"]
        assert rows["C3"]["suggestion"] == "State your UPI MDR; competitors say [add your real figure]. [add your real figure]"
        assert rows["C1"]["target_quote_quality"] == "ok"

    def test_missing_verdict_defaults_to_missing_with_safe_suggestion(self):
        rows = apply_judgements(self.claims()[:1], {}, self.TARGET, {})
        assert rows[0]["target_status"] == "missing" and rows[0]["flags"] == ["no_verdict"]
        assert "add your real figures" in rows[0]["suggestion"]

    def test_rank_puts_support_first_then_missing(self):
        rows = [
            {"source_count": 1, "target_status": "missing"},
            {"source_count": 3, "target_status": "covered"},
            {"source_count": 3, "target_status": "partly"},
        ]
        ranked = rank_gaps(rows)
        assert [(r["rank"], r["source_count"], r["target_status"]) for r in ranked] == [
            (1, 3, "partly"), (2, 3, "covered"), (3, 1, "missing"),
        ]

    def test_excerpt_keeps_relevant_passages_within_budget_in_page_order(self):
        page = "Welcome to Acme.\nOur card MDR is 2% per payment.\nCareers at Acme.\nUPI MDR is zero for merchants."
        claims = count_claims({}, [item("S1.1", "UPI MDR zero"), item("S2.1", "card MDR 2%")], {})
        excerpt = select_excerpt(page, "MDR for UPI and cards", claims, token_budget=100)
        assert excerpt == "Our card MDR is 2% per payment.\nUPI MDR is zero for merchants."
        # both passages match 3 terms: the tie goes to page order, and only one fits in 9 tokens
        assert select_excerpt(page, "MDR for UPI and cards", claims, token_budget=9) == "Our card MDR is 2% per payment."


class FakeModel:
    """Replies by stage; records calls like GeminiJSONModel."""

    model_name = "fake-model"

    def __init__(self, extract_replies, cluster_reply, judge_reply):
        self.extract_replies, self.cluster_reply, self.judge_reply = extract_replies, cluster_reply, judge_reply
        self.calls, self.prompts = [], []

    async def generate_json(self, stage, prompt, max_tokens=4096):
        self.calls.append({"stage": stage, "attempt": 1, "ok": True})
        self.prompts.append((stage, prompt))
        if stage == "extract":
            url = prompt.split("Source: ", 1)[1].split()[0]
            return self.extract_replies[url]
        return self.cluster_reply if stage == "cluster" else self.judge_reply


PAGES = {
    "https://a.com/fees": "Card MDR in India is 2% on most gateways. UPI transactions have zero MDR for merchants.",
    "https://b.com/fees": "Most gateways charge 2% card MDR. Setup fees range from nothing upward.",
    "https://acme.in/payment-gateway/": "Acme Payment Gateway. Accept cards and UPI. "
                                        "Pricing: 2% per card payment on every standard plan for Indian merchants.",
}


class TestReport:
    def payload(self):
        return {
            "observation_id": "obs_1", "sandbox_url": SANDBOX, "brand_name": "Acme",
            "queries": [{"query_index": 0, "query": "What MDR do gateways charge for UPI and cards?", "sources": [
                src("https://a.com/fees", 3), src("https://b.com/fees", 2), src("https://dead.com/x", 1),
            ]}],
            "cached_texts": {"https://a.com/fees": PAGES["https://a.com/fees"]},
        }

    def model(self):
        return FakeModel(
            extract_replies={
                "https://a.com/fees": {"items": [
                    {"type": "figure", "text": "Card MDR is 2%", "quote": "Card MDR in India is 2%"},
                    {"type": "fact", "text": "UPI has zero MDR", "quote": "UPI transactions have zero MDR"},
                ]},
                "https://b.com/fees": {"items": [{"type": "figure", "text": "Card MDR is 2%", "quote": "Most gateways charge 2% card MDR"}]},
            },
            cluster_reply={"claims": [
                {"claim": "Card MDR is about 2%", "item_ids": ["S1.1", "S2.1"]},
                {"claim": "UPI has zero MDR", "item_ids": ["S1.2"]},
            ]},
            judge_reply={"judgements": [
                {"claim_id": "C1", "status": "covered", "target_quote": "Pricing: 2% per card payment on every standard plan", "suggestion": None},
                {"claim_id": "C2", "status": "missing", "target_quote": None, "suggestion": "State your UPI MDR: [add your real figure]."},
            ]},
        )

    def test_report_format_end_to_end(self):
        crawled = []

        async def crawler(url):
            crawled.append(url)
            if url == "https://dead.com/x":
                return {"error": "Failed to crawl: HTTP 404"}
            return {"full_content": PAGES[url]}

        model = self.model()
        report = run(run_gap_analysis(self.payload(), model=model, crawler=crawler))
        assert crawled == ["https://b.com/fees", "https://dead.com/x", SANDBOX]  # cached page not re-crawled
        assert set(report) == {"version", "observation_id", "sandbox_url", "brand_name", "generated_at", "model",
                               "settings", "model_calls", "seconds", "queries", "deadline_reached"}
        assert report["deadline_reached"] is False and report["settings"]["run_time_limit_seconds"] == 600
        assert report["version"] == "gap_v1" and report["model"] == "fake-model"
        assert report["model_calls"]["total"] == 4  # 2 extractions (dead page skipped), 1 grouping, 1 judgement
        assert report["model_calls"]["by_stage"] == {"extract": 2, "cluster": 1, "judge": 1}

        q = report["queries"][0]
        assert q["per_source_token_budget"] == 1000
        assert [(s["label"], s["text_source"], s["items"]) for s in q["sources"]] == [
            ("S1", "cache", 2), ("S2", "crawled", 1), ("S3", "unavailable", 0),
        ]
        assert q["sources"][2]["error"] == "Failed to crawl: HTTP 404"
        assert [(t["label"], t["submitted"], t["text_source"]) for t in q["target_pages"]] == [("T1", True, "crawled")]
        gaps = q["gaps"]
        assert [(g["rank"], g["claim"], g["source_count"], g["target_status"]) for g in gaps] == [
            (1, "Card MDR is about 2%", 2, "covered"), (2, "UPI has zero MDR", 1, "missing"),
        ]
        assert gaps[0]["sources"] == ["https://a.com/fees", "https://b.com/fees"]
        assert gaps[0]["target_quote"] == "Pricing: 2% per card payment on every standard plan" and gaps[0]["suggestion"] is None
        assert gaps[1]["suggestion"] == "State your UPI MDR: [add your real figure]." and gaps[1]["flags"] == []
        assert {"source", "quote"} <= set(gaps[0]["source_quotes"][0])
        assert q["other_claims"] == [] and q["errors"] == []

    def test_no_evidence_skips_grouping_and_judgement(self):
        model = self.model()
        model.extract_replies = {u: {"items": []} for u in model.extract_replies}

        async def crawler(url):
            return {"full_content": PAGES.get(url, "")}

        report = run(run_gap_analysis(self.payload(), model=model, crawler=crawler))
        assert report["model_calls"]["by_stage"] == {"extract": 2}
        assert report["queries"][0]["errors"] == ["no verified evidence from the cited sources"]


class TestModelRetries:
    def make(self, monkeypatch, outcomes, **kwargs):
        class Response:
            def __init__(self, text):
                self.text = text

        class Models:
            def generate_content(self, model, contents, config):
                outcome = outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return Response(outcome)

        class Limiter:
            async def wait_for_slot(self, model):
                return None

        class Provider:
            client = type("C", (), {"models": Models()})()
            rate_limiter = Limiter()

        monkeypatch.setattr("app.llm_providers.factory.get_llm_provider", lambda name=None: Provider())
        sleeps = []

        async def sleep(seconds):
            sleeps.append(seconds)

        return ga.GeminiJSONModel(model_name="gemini-test", sleep=sleep, **kwargs), sleeps

    def test_retries_503_with_backoff_and_counts_every_attempt(self, monkeypatch):
        model, sleeps = self.make(monkeypatch, [RuntimeError("503 UNAVAILABLE"), RuntimeError("503 UNAVAILABLE"), '{"items": []}'])
        assert run(model.generate_json("extract", "p")) == {"items": []}
        assert sleeps == [4.0, 8.0]
        assert ga.summarize_calls(model.calls) == {"total": 3, "succeeded": 1, "retries": 2, "failed": 2, "by_stage": {"extract": 3}}

    def test_retries_are_capped_and_other_errors_are_not_retried(self, monkeypatch):
        model, sleeps = self.make(monkeypatch, [RuntimeError("503 UNAVAILABLE")] * 4, max_retries=3)
        with pytest.raises(RuntimeError, match="503"):
            run(model.generate_json("judge", "p"))
        assert len(model.calls) == 4 and sleeps == [4.0, 8.0, 16.0]
        model, sleeps = self.make(monkeypatch, [ValueError("400 INVALID_ARGUMENT")])
        with pytest.raises(ValueError):
            run(model.generate_json("judge", "p"))
        assert len(model.calls) == 1 and sleeps == []

    def test_invalid_json_is_retried_once(self, monkeypatch):
        model, sleeps = self.make(monkeypatch, ['{"items": [', '{"items": []}'])
        assert run(model.generate_json("cluster", "p")) == {"items": []}
        assert [c["ok"] for c in model.calls] == [False, True] and sleeps == []
        model, _ = self.make(monkeypatch, ['{"items": [', '{"items": ['])
        with pytest.raises(ValueError):
            run(model.generate_json("cluster", "p"))
        assert len(model.calls) == 2

    def test_daily_quota_is_not_retried_and_stops_later_calls(self, monkeypatch):
        daily = RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        model, sleeps = self.make(monkeypatch, [daily, '{"a": 1}'])
        with pytest.raises(RuntimeError, match="PerDay"):
            run(model.generate_json("extract", "p"))
        with pytest.raises(ga.ModelQuotaExhausted):
            run(model.generate_json("extract", "p"))
        assert len(model.calls) == 1 and sleeps == [] and "daily quota" in model.quota_exhausted

    def test_per_minute_429_is_retried(self, monkeypatch):
        model, sleeps = self.make(monkeypatch, [RuntimeError("429 RESOURCE_EXHAUSTED PerMinute"), '{"a": 1}'])
        assert run(model.generate_json("extract", "p")) == {"a": 1} and sleeps == [4.0]

    def test_total_attempt_cap(self, monkeypatch):
        model, _ = self.make(monkeypatch, ['{"a": 1}'] * 3, max_attempts=2)
        run(model.generate_json("extract", "p"))
        run(model.generate_json("extract", "p"))
        with pytest.raises(ga.ModelCallBudgetExceeded):
            run(model.generate_json("extract", "p"))


# ---------- honesty, weak quotes, preflight and time limits ----------

class TestSuggestionHonesty:
    def test_superlatives_removed_unless_the_target_uses_them(self):
        text, flags = ga.check_suggestion(
            "Mention that Acme offers instant settlements alongside industry-leading competitive pricing and the "
            "best, seamless checkout.", "Acme Payment Gateway. Instant settlements for merchants.")
        assert text == "Mention that Acme offers instant settlements alongside competitive pricing and the checkout."
        assert flags == ["promotional_wording_removed:industry-leading,seamless,best"]
        kept, flags = ga.check_suggestion("Explain why merchants call it the fastest checkout.", "The fastest checkout in India.")
        assert kept == "Explain why merchants call it the fastest checkout." and flags == []

    def test_article_and_spacing_fixed_after_removal(self):
        text, _ = ga.check_suggestion("Add an industry-leading uptime section.", "Acme")
        assert text == "Add an uptime section."
        text, _ = ga.check_suggestion("Describe a top UPI integration.", "Acme")
        assert text == "Describe a UPI integration."
        text, _ = ga.check_suggestion("Offer a world-class API.", "Acme")
        assert text == "Offer an API."
        for ordinary in ("Add a top-up FAQ.", "List best practices for retries.", "Explain what is leading to failures.",
                         "Put the fees at the top of the page.", "Keep a summary on top of the pricing table."):
            assert ga.check_suggestion(ordinary, "Acme") == (ordinary, [])

    def test_numbers_become_placeholders_and_competitors_are_flagged(self):
        text, flags = ga.check_suggestion("Say your MDR is 1.9% like Cashfree, settled in 2 days.", "Settled in 2 days.", ["cashfree"])
        assert text == "Say your MDR is [add your real figure] like Cashfree, settled in 2 days."
        assert flags == ["suggestion_figures_not_on_target_page:1.9", "mentions_cited_competitor:cashfree"]

    def test_superlative_in_a_judged_suggestion(self):
        claims = count_claims({}, [item("S1.1", "Instant settlements")], {})
        rows = apply_judgements(claims, {"judgements": [{"claim_id": "C1", "status": "missing",
                                "suggestion": "Highlight Acme's world-class, unmatched instant settlements."}]},
                                {"https://acme.in": "Acme accepts cards and UPI from customers across India."}, {})
        assert rows[0]["suggestion"] == "Highlight Acme's instant settlements."
        assert rows[0]["suggestion_original"] == "Highlight Acme's world-class, unmatched instant settlements."
        assert "promotional_wording_removed:world-class,unmatched" in rows[0]["flags"]

    def test_prompt_states_the_honesty_rules(self):
        claims = count_claims({}, [item("S1.1", "x")], {})
        prompt = ga.build_judge_prompt("q", "Acme", claims, [{"label": "T1", "url": "u", "excerpt": "e"}])
        assert "only verifiable facts" in prompt and "no superlatives" in prompt
        assert "no claims about competitors" in prompt and "[add your real figure]\" in place of any number" in prompt


NAV = "For Startups & SMEs For Enterprises Pricing Blog Learn Events White papers Customer Stories Developer Docs"
MAIN = "Razorpay lets businesses accept payments through cards, UPI, netbanking and wallets on a single integration."


class TestWeakCoveredQuotes:
    def test_quote_quality_rules(self):
        page = ga.PageText(NAV + " " + MAIN, main=MAIN, chrome=NAV)
        assert ga.quote_quality("accept payments through cards", page) == "too_short"
        assert ga.quote_quality(NAV, ga.PageText(NAV + " " + MAIN)) == "menu_like"
        assert ga.quote_quality("Home | Pricing | Docs | Careers | Partners | Contact us today for help", ga.PageText("x")) == "menu_like"
        footer = "Copyright 2026 Razorpay Software Private Limited, all rights reserved here."
        page = ga.PageText(MAIN + " " + footer, main=MAIN, chrome=footer)
        assert ga.quote_quality(footer, page) == "page_chrome"
        assert ga.quote_quality(MAIN, page) is None
        assert ga.quote_quality(MAIN, ga.PageText(MAIN), other_pages=["Intro. " + MAIN]) == "repeated_on_other_target_pages"

    def test_navigation_quote_downgrades_covered_to_partly(self):
        claims = count_claims({}, [item("S1.1", "Razorpay serves startups and enterprises")], {})
        reply = {"judgements": [{"claim_id": "C1", "status": "covered", "target_quote": NAV[:90]}]}
        rows = apply_judgements(claims, reply, {"https://razorpay.com/pg": ga.PageText(NAV + " " + MAIN, MAIN, NAV)}, {})
        assert rows[0]["target_status"] == "partly"
        assert rows[0]["flags"][0] in ("weak_covered_quote:menu_like", "weak_covered_quote:page_chrome")
        assert rows[0]["target_quote_quality"] in ("menu_like", "page_chrome")
        reply = {"judgements": [{"claim_id": "C1", "status": "covered", "target_quote": MAIN}]}
        rows = apply_judgements(claims, reply, {"https://razorpay.com/pg": ga.PageText(NAV + " " + MAIN, MAIN, NAV)}, {})
        assert rows[0]["target_status"] == "covered" and rows[0]["flags"] == []

    def test_target_pages_crawled_for_sections_and_excerpts_skip_menus(self):
        html_sections = {"full_content": NAV + " " + MAIN, "main_content": MAIN, "chrome_content": NAV}
        crawled = []

        async def crawler(url):
            crawled.append(url)
            return html_sections

        store = ga.TextStore({SANDBOX: NAV + " " + MAIN}, crawler)
        page, origin, error = run(store.get_page(SANDBOX))
        assert (page.main, page.chrome, origin, error) == (MAIN, NAV, "crawled+sections", None)
        run(store.get_page(SANDBOX))
        assert crawled == [SANDBOX]  # once per report

        async def failing(url):
            return {"error": "Failed to crawl: HTTP 403"}

        store = ga.TextStore({SANDBOX: MAIN}, failing)
        page, origin, _ = run(store.get_page(SANDBOX))
        assert (page.text, page.main, origin) == (MAIN, None, "cache")  # falls back to the cached text


class TestModelReplyShapes:
    def payload(self):
        return TestReport().payload()

    def test_bare_list_extraction_reply_is_used(self):
        model = TestReport().model()
        model.extract_replies["https://a.com/fees"] = [
            {"type": "figure", "text": "Card MDR is 2%", "quote": "Card MDR in India is 2%"},
        ]

        async def crawler(url):
            return {"full_content": PAGES.get(url, "")}

        report = run(run_gap_analysis(self.payload(), model=model, crawler=crawler))
        s1 = report["queries"][0]["sources"][0]
        assert (s1["items"], s1["reply_shape"]) == (1, "list")

    def test_truncated_grouping_reply_falls_back_to_item_claims(self):
        model = TestReport().model()

        async def broken_cluster(stage, prompt, max_tokens=4096, _original=model.generate_json):
            if stage == "cluster":
                model.calls.append({"stage": stage, "attempt": 1, "ok": False})
                return ga.parse_json_reply('{"claims": [{"claim": "Card MDR is about 2%", "item_ids": ["S1.1"')
            return await _original(stage, prompt, max_tokens)

        model.generate_json = broken_cluster

        async def crawler(url):
            return {"full_content": PAGES.get(url, "")}

        report = run(run_gap_analysis(self.payload(), model=model, crawler=crawler))
        q = report["queries"][0]
        assert q["errors"][0].startswith("claim grouping failed, items counted one by one")
        assert [g["source_count"] for g in q["gaps"]] == [1, 1, 1]


class TestPreflightAndLimits(TestModelRetries):
    def test_daily_quota_preflight_aborts_without_starting(self, monkeypatch):
        daily = RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        model, _ = self.make(monkeypatch, [daily])
        launched = []
        with pytest.raises(ga.GapPreflightError) as info:
            run(ga.preflight_and_start("gap_1", {"sandbox_url": SANDBOX, "queries": [{}]}, model=model,
                                       launch=launched.append))
        assert info.value.status_code == 429 and "nothing saved" in str(info.value)
        assert launched == [] and [c["stage"] for c in model.calls] == ["preflight"]

    def test_preflight_success_launches_the_job(self, monkeypatch):
        model, _ = self.make(monkeypatch, ["not even json"])  # any reply passes the preflight
        launched = []

        def launch(coro):
            launched.append(coro)
            coro.close()

        run(ga.preflight_and_start("gap_1", {"sandbox_url": SANDBOX, "queries": [{}]}, model=model, launch=launch))
        assert len(launched) == 1 and model.calls[0]["ok"] is True

    def test_other_preflight_failure_is_503(self, monkeypatch):
        model, _ = self.make(monkeypatch, [RuntimeError("503 UNAVAILABLE")])
        with pytest.raises(ga.GapPreflightError) as info:
            run(ga.preflight_and_start("gap_1", {}, model=model, launch=lambda c: c.close()))
        assert info.value.status_code == 503

    def test_call_timeout_is_not_retried(self, monkeypatch):
        import time as _time

        model, sleeps = self.make(monkeypatch, [])
        model.call_timeout = 0.05

        def slow(model, contents, config):
            _time.sleep(0.3)

        monkeypatch.setattr(model._provider.client.models, "generate_content", slow)
        monkeypatch.setattr(ga.GeminiJSONModel, "_request", ga.GeminiJSONModel._request)
        real_wait = asyncio.wait_for

        async def short_wait(aw, timeout):
            return await real_wait(aw, timeout=0.05)

        monkeypatch.setattr(ga.asyncio, "wait_for", short_wait)
        with pytest.raises(ga.ModelCallTimeout):
            run(model.generate_json("extract", "p"))
        assert len(model.calls) == 1 and sleeps == []

    def test_no_call_after_the_run_deadline(self, monkeypatch):
        model, _ = self.make(monkeypatch, ['{"a": 1}'])
        model.set_time_limit(-1)
        with pytest.raises(ga.RunDeadlineExceeded):
            run(model.generate_json("extract", "p"))
        assert model.calls == []

    def test_run_limit_skips_remaining_queries(self):
        payload = TestReport().payload()
        payload["queries"].append({"query_index": 1, "query": "Second query", "sources": []})

        async def crawler(url):
            return {"full_content": PAGES.get(url, "")}

        report = run(run_gap_analysis(payload, model=TestReport().model(), crawler=crawler, time_limit=0))
        assert report["deadline_reached"] is True
        assert all(q["errors"] == ["not analyzed: the 0-second run limit was reached"] for q in report["queries"])
