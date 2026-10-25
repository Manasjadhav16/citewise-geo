import asyncio

import pytest

from app import gap_analysis as ga
from app.gap_analysis import (
    EvidenceItem,
    apply_judgements,
    build_claims,
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


def topics(groups, items, appearances=None):
    """Claims from (label, item ids) groups, as the topic step builds them."""
    return build_claims(groups, items, appearances or {})


def singles(items):
    return build_claims([(i.text, [i.item_id]) for i in items], items, {})


def topic_reply(topics_, assignments):
    return {"topics": [{"topic_id": t, "label": l} for t, l in topics_],
            "assignments": [{"item_id": i, "topic_id": t} for i, t in assignments]}


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


class TestTopicCounting:
    ITEMS = [item("S1.1"), item("S1.2"), item("S2.1"), item("S3.1"), item("S3.2", text="Only S3 says this")]

    def book(self, *replies_and_batches):
        book = ga.TopicBook()
        for reply, batch in replies_and_batches:
            book.add_reply(reply, batch)
        return book

    def test_counts_distinct_sources_not_items(self):
        book = self.book((topic_reply([("K1", "card MDR"), ("K2", "UPI MDR fee")],
                                      [("S1.1", "K1"), ("S1.2", "K1"), ("S2.1", "K1"), ("S3.1", "K2")]), self.ITEMS))
        claims = build_claims(book.groups(self.ITEMS), self.ITEMS, {"S1": 3, "S2": 2, "S3": 1})
        assert [(c.claim, c.source_count, c.sources) for c in claims] == [
            ("card MDR", 2, ["S1", "S2"]),  # two items from S1 count once
            ("UPI MDR fee", 1, ["S3"]),
            ("Only S3 says this", 1, ["S3"]),  # left out by the model: kept as its own topic
        ]
        assert [c.claim_id for c in claims] == ["C1", "C2", "C3"]

    def test_ignores_unknown_ids_unlabelled_topics_and_reassignment(self):
        book = self.book((topic_reply([("K1", "A")], [("S1.1", "K1"), ("S9.9", "K1"), ("S2.1", "K7"), ("S1.1", "K1")]),
                          self.ITEMS[:3]))
        groups = book.groups(self.ITEMS[:3])
        assert groups[0] == ("A", ["S1.1"])  # K7 has no label, S9.9 is not in the batch
        assert [g[1] for g in groups[1:]] == [["S1.2"], ["S2.1"]]

    def test_ties_broken_by_appearances_then_order(self):
        items = [item("S1.1"), item("S2.1")]
        claims = topics([("low", ["S1.1"]), ("high", ["S2.1"])], items, {"S1": 1, "S2": 3})
        assert [c.claim for c in claims] == ["high", "low"]

    def test_bad_reply_falls_back_to_one_topic_per_item(self):
        items = [item("S1.1", "x"), item("S2.1", "y")]
        book = self.book(("not json", items))
        assert [(c.claim, c.source_count) for c in build_claims(book.groups(items), items, {})] == [("x", 1), ("y", 1)]


class TestJudgements:
    TARGET = {"https://acme.in/p": "Acme charges 2% per card payment on all standard plans for Indian merchants. "
                                   "Settlement happens in T+2 days."}

    def claims(self):
        return topics([("Card MDR is 2%", ["S1.1", "S2.1"]), ("Instant settlement is offered", ["S1.2"]),
                       ("UPI MDR is zero", ["S2.2"])], [item("S1.1"), item("S1.2"), item("S2.1"), item("S2.2")])

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
        assert rows[0]["suggestion"] == "Add a section on Card MDR is 2% and state your real figures."

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
        claims = singles([item("S1.1", "UPI MDR zero"), item("S2.1", "card MDR 2%")])
        excerpt = select_excerpt(page, "MDR for UPI and cards", claims, token_budget=100)
        assert excerpt == "Our card MDR is 2% per payment.\nUPI MDR is zero for merchants."
        # both passages match 3 terms: the tie goes to page order, and only one fits in 9 tokens
        assert select_excerpt(page, "MDR for UPI and cards", claims, token_budget=9) == "Our card MDR is 2% per payment."


class FakeModel:
    """Replies by stage; records calls like GeminiJSONModel."""

    model_name = "fake-model"

    def __init__(self, extract_replies, topics_reply, judge_reply):
        # topics_reply: one reply, or a list used one per topic call (the last one repeats)
        self.extract_replies, self.topics_reply, self.judge_reply = extract_replies, topics_reply, judge_reply
        self.calls, self.prompts, self.fallbacks = [], [], 0
        self._topic_calls = 0

    def fall_back(self):
        self.fallbacks += 1
        self.model_name = f"fake-model-{self.fallbacks + 1}"
        return self.model_name

    async def generate_json(self, stage, prompt, max_tokens=4096):
        self.calls.append({"stage": stage, "attempt": 1, "ok": True, "model": self.model_name})
        self.prompts.append((stage, prompt))
        if stage == "extract":
            url = prompt.split("Source: ", 1)[1].split()[0]
            return self.extract_replies[url]
        if stage == "topics":
            replies = self.topics_reply if isinstance(self.topics_reply, list) else [self.topics_reply]
            reply = replies[min(self._topic_calls, len(replies) - 1)]
            self._topic_calls += 1
            if isinstance(reply, Exception):
                raise reply
            return reply
        return self.judge_reply


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
            topics_reply=topic_reply([("K1", "card MDR"), ("K2", "UPI MDR")],
                                     [("S1.1", "K1"), ("S2.1", "K1"), ("S1.2", "K2")]),
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
                               "settings", "model_calls", "seconds", "queries", "deadline_reached",
                               "ranking_unreliable_queries", "target_pages", "judge_agreement"}
        assert report["judge_agreement"] == {"compared": 2, "agreed": 2, "share": 1.0}
        assert report["target_pages"] == [{"url": SANDBOX, "submitted": True, "appearances": 0, "seen_for_queries": []}]
        assert report["queries"][0]["judged_against"] == [SANDBOX]
        assert report["ranking_unreliable_queries"] == []
        assert report["deadline_reached"] is False and report["settings"]["run_time_limit_seconds"] == 600
        assert report["version"] == "gap_v2" and report["model"] == "fake-model"
        assert report["model_calls"]["total"] == 5  # 2 extractions (dead page skipped), 1 topic batch, 2 judging passes
        assert report["model_calls"]["by_stage"] == {"extract": 2, "topics": 1, "judge": 2}

        q = report["queries"][0]
        assert q["per_source_token_budget"] == 1000
        assert [(s["label"], s["text_source"], s["items"]) for s in q["sources"]] == [
            ("S1", "cache", 2), ("S2", "crawled", 1), ("S3", "unavailable", 0),
        ]
        assert q["sources"][2]["error"] == "Failed to crawl: HTTP 404"
        assert [(t["label"], t["submitted"], t["text_source"]) for t in q["target_pages"]] == [("T1", True, "crawled")]
        gaps = q["gaps"]
        assert [(g["rank"], g["claim"], g["source_count"], g["target_status"]) for g in gaps] == [
            (1, "card MDR", 2, "covered"), (2, "UPI MDR", 1, "missing"),
        ]
        assert q["clustering"]["method"] == "model" and q["clustering"]["ranking"] == "ok"
        assert q["source_attempts"] == {"tried": 3, "productive": 2, "candidates": 3}
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
        claims = singles([item("S1.1", "Instant settlements")])
        rows = apply_judgements(claims, {"judgements": [{"claim_id": "C1", "status": "missing",
                                "suggestion": "Highlight Acme's world-class, unmatched instant settlements."}]},
                                {"https://acme.in": "Acme accepts cards and UPI from customers across India."}, {})
        assert rows[0]["suggestion"] == "Highlight Acme's instant settlements."
        assert rows[0]["suggestion_original"] == "Highlight Acme's world-class, unmatched instant settlements."
        assert "promotional_wording_removed:world-class,unmatched" in rows[0]["flags"]

    def test_prompt_states_the_honesty_rules(self):
        claims = singles([item("S1.1", "x")])
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
        claims = singles([item("S1.1", "Razorpay serves startups and enterprises")])
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

    def test_truncated_topic_reply_falls_back_to_local_topics(self):
        model = TestReport().model()
        truncated = '{"topics": [{"topic_id": "K1", "label": "card MDR"}], "assignments": [{"item_id": "S1.1"'
        try:
            ga.parse_json_reply(truncated)
        except ValueError as exc:
            model.topics_reply = exc  # every topic call fails like a cut-off reply

        async def crawler(url):
            return {"full_content": PAGES.get(url, "")}

        report = run(run_gap_analysis(self.payload(), model=model, crawler=crawler))
        q = report["queries"][0]
        assert [a["method"] for a in q["clustering"]["attempts"]] == ["model", "model", "local"]
        assert all(a.get("errors") for a in q["clustering"]["attempts"][:2])
        assert q["clustering"]["method"] == "local" and q["clustering"]["ranking"] == "ok"
        assert [g["source_count"] for g in q["gaps"]] == [2, 1]  # the two card-MDR items share content words


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
        model, _ = self.make(monkeypatch, [RuntimeError("503 UNAVAILABLE")] * 4)  # every fallback model too
        with pytest.raises(ga.GapPreflightError) as info:
            run(ga.preflight_and_start("gap_1", {}, model=model, launch=lambda c: c.close()))
        assert info.value.status_code == 503 and len(model.calls) == 4

    def test_call_timeout_falls_back_to_the_next_model(self, monkeypatch):
        import time as _time

        model, sleeps = self.make(monkeypatch, [])
        model.call_timeout = 0.05
        replies = iter([None, '{"a": 1}'])  # first call hangs, the retry answers

        def generate(model, contents, config):
            reply = next(replies)
            if reply is None:
                _time.sleep(0.3)
            return type("R", (), {"text": reply, "candidates": []})()

        monkeypatch.setattr(model._provider.client.models, "generate_content", generate)
        real_wait = asyncio.wait_for

        async def short_wait(aw, timeout):
            return await real_wait(aw, timeout=0.05)

        monkeypatch.setattr(ga.asyncio, "wait_for", short_wait)
        assert run(model.generate_json("extract", "p")) == {"a": 1}
        assert [(c["model"], c["ok"]) for c in model.calls] == [("gemini-test", False), ("gemini-3.7-flash", True)]
        assert "timed out" in model.calls[0]["error"] and sleeps == [4.0]

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


class TestModelFallback(TestModelRetries):
    POOL = ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]

    def test_pool_is_the_existing_gemini_pool_with_the_primary_first(self):
        from app.llm_providers.gemini.gemini_2_5_flash import FALLBACK_MODELS

        assert ga.model_pool("gemini-3.7-flash", FALLBACK_MODELS) == FALLBACK_MODELS
        assert ga.model_pool("gemini-3.8-flash", FALLBACK_MODELS)[:3] == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"]
        assert ga.model_pool("gemini-3.7-flash") == FALLBACK_MODELS  # default: read from the provider module

    def test_503_moves_to_the_next_model_and_stays_there(self, monkeypatch):
        model, sleeps = self.make(monkeypatch, [RuntimeError("503 UNAVAILABLE high demand"), '{"a": 1}', '{"b": 2}'])
        model.model_pool = self.POOL
        model.model_name = model.primary_model = self.POOL[0]
        assert run(model.generate_json("extract", "p")) == {"a": 1}
        assert run(model.generate_json("judge", "p")) == {"b": 2}
        assert [c["model"] for c in model.calls] == ["gemini-3.7-flash", "gemini-3.8-flash", "gemini-3.8-flash"]
        assert sleeps == [4.0] and ga.models_used(model.calls) == {"gemini-3.7-flash": 1, "gemini-3.8-flash": 2}

    def test_per_minute_429_stays_on_the_same_model(self, monkeypatch):
        model, _ = self.make(monkeypatch, [RuntimeError("429 RESOURCE_EXHAUSTED PerMinute"), '{"a": 1}'])
        run(model.generate_json("extract", "p"))
        assert [c["model"] for c in model.calls] == ["gemini-test", "gemini-test"]

    def test_fallback_retries_are_capped(self, monkeypatch):
        model, sleeps = self.make(monkeypatch, [RuntimeError("503 UNAVAILABLE")] * 5)
        model.model_pool = self.POOL
        model.model_name = self.POOL[0]
        with pytest.raises(RuntimeError, match="503"):
            run(model.generate_json("extract", "p"))
        assert [c["model"] for c in model.calls] == self.POOL[:4] and sleeps == [4.0, 8.0, 16.0]

    def test_preflight_falls_back_and_the_run_continues_on_that_model(self, monkeypatch):
        model, sleeps = self.make(monkeypatch, [TimeoutError("Read timed out"), RuntimeError("503 UNAVAILABLE"), "ok", '{"a": 1}'])
        model.model_pool = self.POOL
        model.model_name = self.POOL[0]
        launched = []

        def launch(coro):
            launched.append(coro)
            coro.close()

        run(ga.preflight_and_start("gap_1", {"sandbox_url": SANDBOX, "queries": [{}]}, model=model, launch=launch))
        assert launched and [(c["stage"], c["model"], c["ok"]) for c in model.calls] == [
            ("preflight", "gemini-3.7-flash", False), ("preflight", "gemini-3.8-flash", False),
            ("preflight", "gemini-3.5-flash-lite", True),
        ]
        assert sleeps == [4.0, 8.0]
        run(model.generate_json("extract", "p"))
        assert model.calls[-1]["model"] == "gemini-3.5-flash-lite"

    def test_preflight_daily_quota_still_stops_without_fallback(self, monkeypatch):
        daily = RuntimeError("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        model, sleeps = self.make(monkeypatch, [daily, "ok"])
        with pytest.raises(ga.GapPreflightError) as info:
            run(ga.preflight_and_start("gap_1", {}, model=model, launch=lambda c: c.close()))
        assert info.value.status_code == 429 and len(model.calls) == 1 and sleeps == []

    def test_report_lists_the_pool_and_models_used(self):
        report = run(run_gap_analysis(TestReport().payload(), model=TestReport().model(),
                                      crawler=lambda url: _page(url)))
        assert "models_used" in report["model_calls"] and "model_pool" in report["settings"]


async def _page(url):
    return {"full_content": PAGES.get(url, "")}


# ---------- topics, degenerate grouping, capability claims, source skipping ----------

TOPIC_ITEMS = [
    item("S1.1", "Paytm charges 0% MDR on UPI payments"), item("S1.2", "Paytm supports 100+ payment methods"),
    item("S2.1", "Cashfree UPI MDR is zero for merchants"), item("S2.2", "Cashfree accepts 180+ payment methods"),
    item("S3.1", "PayU settles funds in T+2 days"),
]


class TestTopicMerging:
    def test_topics_merge_across_sources_and_batches(self, monkeypatch):
        monkeypatch.setattr(ga, "TOPIC_BATCH_SIZE", 2)  # three batches; topics carry over
        replies = [
            topic_reply([("K1", "UPI MDR fee"), ("K2", "payment methods supported")], [("S1.1", "K1"), ("S1.2", "K2")]),
            # a new id with an existing label merges into it; K1 is reused by id
            topic_reply([("K3", "Payment methods supported")], [("S2.1", "K1"), ("S2.2", "K3")]),
            topic_reply([("K4", "settlement time")], [("S3.1", "K4")]),
        ]
        model = FakeModel({}, replies, {})
        claims, clustering = run(ga.cluster_topics(model, "q", TOPIC_ITEMS, {"S1": 3, "S2": 2, "S3": 1}))
        assert [(c.claim, c.source_count, c.sources) for c in claims] == [
            ("UPI MDR fee", 2, ["S1", "S2"]), ("payment methods supported", 2, ["S1", "S2"]), ("settlement time", 1, ["S3"]),
        ]
        assert clustering["method"] == "model" and clustering["ranking"] == "ok"
        assert clustering["attempts"][0]["batches"] == 3 and clustering["shared_topics"] == 2
        known_in_last_prompt = model.prompts[-1][1].split("EXISTING TOPICS:")[1]
        assert "K1: UPI MDR fee" in known_in_last_prompt and "K3" not in known_in_last_prompt

    def test_topic_prompt_asks_for_ids_only(self):
        prompt = ga.build_topic_prompt("q", TOPIC_ITEMS[:2], [("K1", "UPI MDR fee")])
        assert '"assignments"' in prompt and "never rewritten claims" in prompt and "K1: UPI MDR fee" in prompt


class TestDegenerateGrouping:
    ONE_PER_ITEM = topic_reply([(f"K{n}", f"topic {n}") for n in range(1, 6)],
                               [(i.item_id, f"K{n}") for n, i in enumerate(TOPIC_ITEMS, start=1)])
    MERGED = topic_reply([("K1", "UPI MDR fee"), ("K2", "payment methods supported"), ("K3", "settlement time")],
                         [("S1.1", "K1"), ("S2.1", "K1"), ("S1.2", "K2"), ("S2.2", "K2"), ("S3.1", "K3")])

    def test_detection(self):
        assert ga.is_degenerate(singles(TOPIC_ITEMS), TOPIC_ITEMS)
        assert not ga.is_degenerate(singles(TOPIC_ITEMS[:2]), TOPIC_ITEMS[:2])  # one source only

    def test_retry_on_next_model_fixes_it(self):
        model = FakeModel({}, [self.ONE_PER_ITEM, self.MERGED], {})
        claims, clustering = run(ga.cluster_topics(model, "q", TOPIC_ITEMS, {}))
        assert model.fallbacks == 1 and clustering["method"] == "model_retry" and clustering["ranking"] == "ok"
        assert [a["model"] for a in clustering["attempts"]] == ["fake-model", "fake-model-2"]
        assert [c.source_count for c in claims] == [2, 2, 1]

    def test_local_clustering_after_two_degenerate_replies(self):
        model = FakeModel({}, [self.ONE_PER_ITEM, self.ONE_PER_ITEM], {})
        claims, clustering = run(ga.cluster_topics(model, "q", TOPIC_ITEMS, {}))
        assert [a["method"] for a in clustering["attempts"]] == ["model", "model", "local"]
        assert clustering["method"] == "local" and clustering["ranking"] == "ok"
        shared = {c.claim: c.sources for c in claims if c.source_count == 2}
        assert sorted(shared.values()) == [["S1", "S2"], ["S1", "S2"]]  # UPI MDR and payment methods merged
        assert any("upi" in label and "mdr" in label for label in shared)

    def test_local_clusters_need_shared_content_words(self):
        groups = ga.local_topic_groups([item("S1.1", "UPI MDR is zero"), item("S2.1", "Settlement takes two days")])
        assert [ids for _, ids in groups] == [["S1.1"], ["S2.1"]]

    def test_unreliable_marker_when_nothing_merges(self):
        unrelated = [item("S1.1", "Paytm onboarding needs PAN"), item("S2.1", "Refunds reach cards within a week"),
                     item("S3.1", "Webhooks retry failed deliveries")]
        reply = topic_reply([("K1", "a"), ("K2", "b"), ("K3", "c")], [("S1.1", "K1"), ("S2.1", "K2"), ("S3.1", "K3")])
        claims, clustering = run(ga.cluster_topics(FakeModel({}, reply, {}), "q", unrelated, {}))
        assert clustering["method"] == "local" and clustering["ranking"] == "unreliable"
        assert all(c.source_count == 1 for c in claims)

    def test_report_lists_unreliable_queries(self):
        payload = TestReport().payload()
        model = TestReport().model()
        model.extract_replies["https://b.com/fees"] = {"items": [
            {"type": "fact", "text": "Setup fees range from nothing upward", "quote": "Setup fees range from nothing upward"}]}
        model.topics_reply = topic_reply([("K1", "a"), ("K2", "b"), ("K3", "c")],
                                         [("S1.1", "K1"), ("S1.2", "K2"), ("S2.1", "K3")])
        report = run(run_gap_analysis(payload, model=model, crawler=_page))
        assert report["ranking_unreliable_queries"] == [0]
        assert report["queries"][0]["clustering"]["ranking"] == "unreliable"


class TestCapabilityClaims:
    def test_assertions_are_detected(self):
        for sentence in (
            "Razorpay offers instant settlements to every merchant.",
            "Razorpay should add details stating that it supports over [add your real figure] payment methods.",
            "Razorpay also provides zero setup fees for new merchants.",
            "We support recurring billing in 25 currencies.",
            "Your page includes a fee calculator.",
            "Razorpay's dashboard has built-in fraud detection.",
        ):
            assert ga.asserts_target_capability(sentence, "Razorpay"), sentence
        for sentence in (
            "Add a section on UPI MDR fees and state your real figures.",
            "If Razorpay offers instant settlements, state the settlement time.",
            "Publish your card MDR as [add your real figure].",
            # the subject is not the brand
            "Razorpay could add details that new merchants can receive zero platform fees under its promotional offer.",
            "Cashfree offers 180+ payment methods.",
            "Add a section explaining that it supports UPI.",  # "it" without the brand in the sentence
        ):
            assert not ga.asserts_target_capability(sentence, "Razorpay"), sentence

    def test_unsupported_claim_rewritten_to_conditional(self):
        text, flags = ga.check_target_claims(
            "Add a section on new-merchant pricing. Razorpay offers zero platform fees for sales up to "
            "[add your real figure] under a festive offer.", "Razorpay", None)
        assert text == ("Add a section on new-merchant pricing. If Razorpay offers this, Razorpay offers zero platform "
                        "fees for sales up to [add your real figure] under a festive offer.")
        assert flags == ["unsupported_target_claim"]

    def test_claim_backed_by_the_target_quote_is_kept(self):
        quote = "Onboard and retain global users with subscriptions, recurring billing and multi-currency support."
        sentence = "Razorpay should add details stating that it supports recurring billing and subscriptions for global users."
        assert ga.check_target_claims(sentence, "Razorpay", quote) == (sentence, [])
        assert ga.check_target_claims(sentence, "Razorpay", None)[1] == ["unsupported_target_claim"]

    def test_judged_suggestion_is_checked_with_its_quote(self):
        claims = topics([("fraud detection", ["S1.1", "S2.1"])], [item("S1.1"), item("S2.1")])
        reply = {"judgements": [{"claim_id": "C1", "status": "missing", "target_quote": None,
                                 "suggestion": "Razorpay should add details stating that it supports built-in fraud detection."}]}
        rows = apply_judgements(claims, reply, {"https://razorpay.com/pg": MAIN}, {}, (), "Razorpay")
        assert rows[0]["suggestion"].startswith("If Razorpay offers this, Razorpay should add details")
        assert "unsupported_target_claim" in rows[0]["flags"]
        assert rows[0]["suggestion_original"] == "Razorpay should add details stating that it supports built-in fraud detection."

    def test_prompt_requires_topic_level_conditional_suggestions(self):
        prompt = ga.build_judge_prompt("q", "Razorpay", singles([item("S1.1", "x")]), [])
        assert "Add a section on" in prompt and "never copy a competitor-specific offer" in prompt
        assert "If Razorpay offers" in prompt


class TestSourceSkipping:
    def payload(self, n):
        return {"observation_id": "o", "sandbox_url": SANDBOX, "brand_name": "Acme",
                "queries": [{"query_index": 0, "query": "card MDR", "sources": [src(f"https://s{i}.com/p", 20 - i) for i in range(n)]}],
                "cached_texts": {}}

    def model_for(self, productive_urls):
        page = "Card MDR is 2% on most gateways in India."
        good = {"items": [{"type": "figure", "text": "Card MDR is 2%", "quote": "Card MDR is 2% on most gateways"}]}

        class Replies(dict):
            def __missing__(self, url):
                return good if url in productive_urls else {"items": []}

        model = FakeModel(Replies(), topic_reply([("K1", "card MDR")], []), {"judgements": []})
        return model, page

    def test_zero_item_sources_are_replaced_by_the_next_cited(self):
        productive = {f"https://s{i}.com/p" for i in (0, 2, 3, 5, 6, 8)}
        model, page = self.model_for(productive)

        async def crawler(url):
            return {"full_content": page} if url != SANDBOX else {"full_content": "Acme gateway."}

        q = run(run_gap_analysis(self.payload(12), model=model, crawler=crawler))["queries"][0]
        assert [s["url"] for s in q["sources"] if s["productive"]] == [f"https://s{i}.com/p" for i in (0, 2, 3, 5, 6)]
        assert q["source_attempts"] == {"tried": 7, "productive": 5, "candidates": 12}
        assert [s["label"] for s in q["sources"]] == [f"S{n}" for n in range(1, 8)]

    def test_attempts_are_capped_at_ten(self):
        model, page = self.model_for({"https://s0.com/p"})

        async def crawler(url):
            return {"full_content": page}

        q = run(run_gap_analysis(self.payload(15), model=model, crawler=crawler))["queries"][0]
        assert q["source_attempts"] == {"tried": 10, "productive": 1, "candidates": 15}
        assert sum(1 for c in model.calls if c["stage"] == "extract") == 10

    def test_crawl_failures_count_as_attempts_without_model_calls(self):
        model, page = self.model_for({f"https://s{i}.com/p" for i in range(10)})

        async def crawler(url):
            return {"error": "Failed to crawl: HTTP 403"} if url in ("https://s0.com/p", "https://s1.com/p") else {"full_content": page}

        q = run(run_gap_analysis(self.payload(10), model=model, crawler=crawler))["queries"][0]
        assert q["source_attempts"] == {"tried": 7, "productive": 5, "candidates": 10}
        assert sum(1 for c in model.calls if c["stage"] == "extract") == 5


# ---------- target pages across the observation, short quotes with figures ----------

class TestObservationTargetPages:
    def payload(self):
        payload = TestReport().payload()
        payload["queries"][0]["sources"].append(src("https://acme.in/blog/upi", 1, group="target", category="target_company"))
        payload["queries"].append({"query_index": 1, "query": "Card fees in India", "sources": [
            src("https://acme.in/pricing", 3, group="target", category="target_company"),
            src("https://acme.in/blog/upi", 2, group="target", category="target_company"),
            src("https://acme.in/payment-gateway", 1, group="target", category="target_company"),  # the submitted page
        ]})
        return payload

    def test_union_of_target_pages(self):
        pages = ga.observation_target_pages(self.payload()["queries"], SANDBOX)
        assert [(p["url"], p["appearances"], p["seen_for_queries"]) for p in pages] == [
            (SANDBOX, 1, [1]), ("https://acme.in/blog/upi", 3, [0, 1]), ("https://acme.in/pricing", 3, [1]),
        ]

    def test_every_query_is_judged_against_all_target_pages(self):
        texts = {SANDBOX: PAGES[SANDBOX], "https://acme.in/pricing": "Acme pricing: card MDR is 2% on every plan.",
                 "https://acme.in/blog/upi": "UPI payments on Acme carry 0% MDR for all merchants."}

        async def crawler(url):
            if url in texts:
                return {"full_content": texts[url], "main_content": texts[url], "chrome_content": ""}
            return {"full_content": PAGES.get(url, "")}

        model = TestReport().model()
        report = run(run_gap_analysis(self.payload(), model=model, crawler=crawler))
        q0 = report["queries"][0]
        assert q0["judged_against"] == [SANDBOX, "https://acme.in/blog/upi", "https://acme.in/pricing"]
        assert [(t["url"], t["seen_for_query"]) for t in q0["target_pages"]] == [
            (SANDBOX, False), ("https://acme.in/blog/upi", True), ("https://acme.in/pricing", False),
        ]
        judge_prompt = [p for stage, p in model.prompts if stage == "judge"][0]
        assert "https://acme.in/pricing" in judge_prompt  # a page seen only for query 1 is used for query 0
        assert [t["url"] for t in report["target_pages"]] == q0["judged_against"]

    def test_target_page_without_text_is_not_judged_against(self):
        async def crawler(url):
            if url == "https://acme.in/pricing":
                return {"error": "Failed to crawl: HTTP 404"}
            return {"full_content": PAGES.get(url, "Acme blog about UPI fees for merchants.")}

        report = run(run_gap_analysis(self.payload(), model=TestReport().model(), crawler=crawler))
        assert "https://acme.in/pricing" not in report["queries"][0]["judged_against"]
        assert any(t["url"] == "https://acme.in/pricing" and t["error"] for t in report["queries"][0]["target_pages"])


class TestShortFigureQuotes:
    MAIN = "Pricing for merchants. UPI MDR 0% on regular UPI. Card payments cost 2% each."

    def test_short_quote_with_a_figure_from_main_content_is_accepted(self):
        page = ga.PageText(self.MAIN, main=self.MAIN, chrome="")
        assert ga.quote_quality("UPI MDR 0% on regular UPI", page) is None
        assert ga.quote_quality("Card payments cost 2%", page) is None  # exactly 4 words

    def test_still_too_short_without_a_figure_too_few_words_or_main_content(self):
        page = ga.PageText(self.MAIN, main=self.MAIN, chrome="")
        assert ga.quote_quality("Pricing for merchants", page) == "too_short"  # no number
        assert ga.quote_quality("costs 2% each", page) == "too_short"  # 3 words
        assert ga.quote_quality("UPI MDR 0% on regular UPI", ga.PageText(self.MAIN)) == "too_short"  # sections unknown
        chrome = "Plans from 0% MDR today"
        assert ga.quote_quality(chrome, ga.PageText(self.MAIN + " " + chrome, main=self.MAIN, chrome=chrome)) == "too_short"

    def test_menu_and_repeat_checks_still_apply_to_short_figure_quotes(self):
        menu = "Plans | 2% | Pricing | Docs"
        assert ga.quote_quality(menu, ga.PageText(menu, main=menu, chrome="")) == "menu_like"
        page = ga.PageText(self.MAIN, main=self.MAIN, chrome="")
        assert ga.quote_quality("UPI MDR 0% on regular UPI", page, other_pages=["Footer: UPI MDR 0% on regular UPI."]) \
            == "repeated_on_other_target_pages"

    def test_short_figure_quote_keeps_a_covered_verdict(self):
        claims = topics([("UPI MDR fee", ["S1.1", "S2.1"])], [item("S1.1"), item("S2.1")])
        reply = {"judgements": [{"claim_id": "C1", "status": "covered", "target_quote": "UPI MDR 0% on regular UPI"}]}
        rows = apply_judgements(claims, reply, {"https://acme.in/p": ga.PageText(self.MAIN, self.MAIN, "")}, {})
        assert rows[0]["target_status"] == "covered" and rows[0]["target_quote_quality"] == "ok" and rows[0]["flags"] == []


# ---------- two judging passes, deterministic missing check, definitions ----------

def judge(*verdicts):
    return {"judgements": [{"claim_id": c, "status": st, "target_quote": q, "suggestion": sg} for c, st, q, sg in verdicts]}


class TwoPassModel(FakeModel):
    """Judge replies per pass; everything else as FakeModel."""

    def __init__(self, base, judge_replies):
        super().__init__(base.extract_replies, base.topics_reply, None)
        self.judge_replies = list(judge_replies)

    async def generate_json(self, stage, prompt, max_tokens=4096):
        if stage != "judge":
            return await super().generate_json(stage, prompt, max_tokens)
        self.calls.append({"stage": stage, "attempt": 1, "ok": True, "model": self.model_name})
        self.prompts.append((stage, prompt))
        reply = self.judge_replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


COVERED_QUOTE = "Pricing: 2% per card payment on every standard plan"


class TestTwoPassJudging:
    def run_with(self, *judge_replies, crawler=None):
        model = TwoPassModel(TestReport().model(), judge_replies)
        return run(run_gap_analysis(TestReport().payload(), model=model, crawler=crawler or _page)), model

    def test_agreeing_passes_keep_the_verdict(self):
        reply = judge(("C1", "covered", COVERED_QUOTE, None), ("C2", "missing", None, "Add a section on UPI MDR."))
        report, model = self.run_with(reply, reply)
        q = report["queries"][0]
        assert [(g["claim"], g["target_status"], g["judge_agreement"]) for g in q["gaps"]] == [
            ("card MDR", "covered", True), ("UPI MDR", "missing", True)]
        assert q["judge"] == {"passes": 2, "topics": 2, "compared": 2, "agreed": 2, "agreement": 1.0}
        assert [len(g["verdicts"]) for g in q["gaps"]] == [2, 2]
        judge_prompts = [p for stage, p in model.prompts if stage == "judge"]
        assert judge_prompts[0].index("C1 topic") < judge_prompts[0].index("C2 topic")
        assert judge_prompts[1].index("C2 topic") < judge_prompts[1].index("C1 topic")  # second pass reversed

    def test_disagreeing_passes_give_uncertain_with_both_verdicts(self):
        first = judge(("C1", "covered", COVERED_QUOTE, None), ("C2", "missing", None, "Add a section on UPI MDR."))
        second = judge(("C1", "missing", None, "Add a section on card MDR and state your real figures."),
                       ("C2", "missing", None, "Add a section on UPI MDR."))
        report, _ = self.run_with(first, second)
        gaps = {g["claim"]: g for g in report["queries"][0]["gaps"]}
        card = gaps["card MDR"]
        assert card["target_status"] == "uncertain" and card["judge_agreement"] is False
        assert [(v["pass"], v["target_status"], v["target_quote"]) for v in card["verdicts"]] == [
            (1, "covered", COVERED_QUOTE), (2, "missing", None)]
        assert card["suggestion"] == "Add a section on card MDR and state your real figures."
        assert "judge_disagreement" in card["flags"]
        assert report["judge_agreement"] == {"compared": 2, "agreed": 1, "share": 0.5}
        assert report["queries"][0]["gaps"][0]["claim"] == "card MDR"  # still ranked by distinct sources

    def test_one_failed_pass_keeps_the_other_as_single_pass(self):
        reply = judge(("C1", "covered", COVERED_QUOTE, None), ("C2", "missing", None, "Add a section on UPI MDR."))
        report, _ = self.run_with(reply, RuntimeError("400 INVALID_ARGUMENT"))
        q = report["queries"][0]
        assert q["judge"]["passes"] == 1 and q["judge"]["compared"] == 0 and q["judge"]["agreement"] is None
        assert all("single_judge_pass" in g["flags"] for g in q["gaps"])
        assert report["judge_agreement"]["share"] is None

    def test_combine_passes_unit(self):
        row = lambda status, quote=None: {"target_status": status, "target_quote": quote, "target_url": "u" if quote else None,
                                          "target_quote_quality": "ok" if quote else None, "suggestion": None, "flags": []}
        agreed, = ga.combine_passes([row("partly")], [row("partly", "a quote with 8 words is long enough here")])
        assert agreed["target_status"] == "partly" and agreed["target_quote"].startswith("a quote")  # quote from pass 2


class TestDeterministicMissingCheck:
    def test_missing_verdict_with_a_matching_passage_becomes_uncertain(self):
        target = "Acme pricing. UPI MDR is 0% on all regular UPI payments for merchants. Card payments cost 2%."
        claims = topics([("UPI MDR fee", ["S1.1", "S2.1"])],
                        [item("S1.1", "Paytm charges 0% UPI MDR"), item("S2.1", "UPI MDR is zero at Cashfree")])
        rows = ga.combine_passes(*(apply_judgements(claims, judge(("C1", "missing", None, None)), {"https://acme.in/p": target}, {})
                                   for _ in range(2)))
        rows = ga.check_missing_verdicts(rows, claims, {"https://acme.in/p": target})
        assert rows[0]["target_status"] == "uncertain" and "missing_but_passage_found" in rows[0]["flags"]
        found = rows[0]["missing_check"]
        assert found["url"] == "https://acme.in/p" and "UPI MDR is 0%" in found["passage"]
        assert found["matched_terms"] == ["mdr", "upi"] and found["matched_numbers"] == ["0"]

    def test_missing_stays_missing_when_nothing_matches(self):
        claims = topics([("settlement time", ["S1.1"])], [item("S1.1", "PayU settles in T+2 days")])
        rows = ga.check_missing_verdicts(
            apply_judgements(claims, judge(("C1", "missing", None, None)), {"u": "Acme accepts UPI and cards."}, {}),
            claims, {"u": "Acme accepts UPI and cards."})
        assert rows[0]["target_status"] == "missing" and rows[0]["missing_check"] is None

    def test_one_keyword_needs_a_source_number(self):
        claim = topics([("settlement time", ["S1.1"])], [item("S1.1", "PayU settles funds in 2 days")])[0]
        assert ga.find_supporting_passage(claim, {"u": "Settlement happens quickly."}) is None
        found = ga.find_supporting_passage(claim, {"u": "Settlement happens in 2 working days."})
        assert found["matched_terms"] == ["settlement"] and found["matched_numbers"] == ["2"]

    def test_uses_main_content_not_navigation(self):
        claim = topics([("UPI MDR fee", ["S1.1"])], [item("S1.1", "UPI MDR is 0%")])[0]
        page = ga.PageText("Nav: UPI MDR pricing docs. Welcome to Acme.", main="Welcome to Acme.", chrome="Nav: UPI MDR pricing docs.")
        assert ga.find_supporting_passage(claim, {"u": page}) is None


class TestDefinitionsAreNotClaims:
    def test_definition_sentence_is_not_flagged_or_rewritten(self):
        sentence = ("Add a section on definition of MDR and state your real figures, explaining that transaction fees for "
                    "payment gateways are also known as the merchant discount rate.")
        assert not ga.asserts_target_capability(sentence, "Razorpay")
        assert ga.check_target_claims(sentence, "Razorpay", None) == (sentence, [])
        for definition in ("MDR is defined as the fee a merchant pays per transaction.",
                           "The platform fee refers to the gateway's own charge on top of MDR."):
            assert not ga.asserts_target_capability(definition, "Razorpay")

    def test_real_brand_claim_is_still_flagged(self):
        text, flags = ga.check_target_claims("Razorpay supports instant settlement for all merchants.", "Razorpay", None)
        assert flags == ["unsupported_target_claim"]
        assert text == "If Razorpay offers this, Razorpay supports instant settlement for all merchants."


class TestPromotionCheck:
    COPIED = ("Razorpay could add details that new merchants can receive zero platform fees for sales up to "
              "[add your real figure] under its promotional offer.")

    def test_copied_offer_is_replaced_by_a_neutral_sentence(self):
        text, flags = ga.check_promotions(self.COPIED, "new merchant pricing", "Razorpay pricing: 2% per transaction.")
        assert text == "Add a section on new merchant pricing and state your real figures."
        assert flags == ["unsupported_promotion:zero platform fees,under its promotional offer"]

    def test_offer_kept_when_the_target_page_uses_that_wording(self):
        text, flags = ga.check_promotions("Explain the festive offer and when it ends.", "festive pricing",
                                          "Our festive offer: lower fees until 31 October.")
        assert (text, flags) == ("Explain the festive offer and when it ends.", [])

    def test_only_the_promotion_sentence_is_replaced(self):
        text, flags = ga.check_promotions("Add a section on setup fees. Mention any cashback for new merchants.",
                                          "setup fee", "Razorpay setup fees.")
        assert text == "Add a section on setup fees. Add a section on setup fee and state your real figures."
        assert flags == ["unsupported_promotion:cashback"]

    def test_fee_types_and_offer_as_a_verb_are_not_promotions(self):
        for sentence in ("Add a section on platform fee and state your real figures.",
                         "State your UPI MDR and the zero MDR rule for RuPay debit cards.",
                         "If Razorpay offers this, state the settlement time.",
                         "List the payment methods your gateway offers.",
                         "State your international card fee as [add your real figure]."):
            assert ga.check_promotions(sentence, "fees", "") == (sentence, []), sentence

    def test_flagged_in_judged_suggestions(self):
        claims = topics([("platform fee", ["S1.1", "S2.1"])], [item("S1.1"), item("S2.1")])
        reply = {"judgements": [{"claim_id": "C1", "status": "missing", "target_quote": None, "suggestion": self.COPIED}]}
        rows = apply_judgements(claims, reply, {"https://razorpay.com/p": "Razorpay pricing: 2% per transaction."}, {},
                                (), "Razorpay")
        assert rows[0]["suggestion"] == "Add a section on platform fee and state your real figures."
        assert rows[0]["flags"][0].startswith("unsupported_promotion:")
        assert rows[0]["suggestion_original"] == self.COPIED
