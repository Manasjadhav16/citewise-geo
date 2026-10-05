import asyncio

import pytest

from app.geo_metrics import ResponseVisibility, canonical_url
from app.llm_providers.base import order_cited_chunks
from app.observation.categorize import (
    SOURCE_FALLBACK,
    SOURCE_HEURISTIC,
    SOURCE_LLM,
    SOURCE_TARGET_MATCH,
    categorize_domains,
    parse_classification_response,
    resolve_category,
)
from app.observation.config import Stage1Config
from app.observation.evidence import (
    build_evidence_prompt,
    compress_query_sources,
    enforce_token_budget,
    estimate_tokens,
    is_no_evidence_reply,
)
from app.observation.fetchers import build_citations, domain_from_title, is_utility_result
from app.observation.stage1_job import (
    aggregate_query_runs,
    brand_terms,
    detect_visibility,
    query_source_metrics,
    select_balanced_queries,
)

TAXONOMY = ("target_company", "direct_competitor", "media", "government", "reference", "other")
REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"


class FakeLLM:
    """Replays canned replies; a reply that is an Exception is raised instead."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def chat(self, messages, max_tokens=None, temperature=None):
        self.prompts.append(messages[0]["content"])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


# ---------- canonical URLs ----------

class TestCanonicalUrl:
    def test_same_page_variants_collapse(self):
        variants = [
            "https://www.example.com/guide/#:~:text=UPI%20fees",
            "http://example.com/guide?utm_source=gemini&utm_medium=ai",
            "https://example.com/guide/",
        ]
        assert {canonical_url(v) for v in variants} == {"https://example.com/guide"}

    def test_keeps_meaningful_query(self):
        assert canonical_url("https://example.com/p?id=7&utm_source=x") == "https://example.com/p?id=7"

    def test_empty(self):
        assert canonical_url("") == ""


# ---------- grounding chunks -> citations ----------

class TestCitations:
    def test_order_by_first_appearance(self):
        spans = [(120, [2]), (0, [1, 0]), (60, [0, 3]), (None, [])]
        assert order_cited_chunks(spans) == [1, 0, 3, 2]

    def test_build_citations_resolves_orders_and_flags(self):
        uris = [REDIRECT + "a", REDIRECT + "b", "https://www.google.com/search?q=time+in+India", REDIRECT + "c"]
        titles = ["a.com", "medium.com", "Current time information in India.", "c.org"]
        resolved = {
            REDIRECT + "a": "https://www.a.com/x#:~:text=hi",
            REDIRECT + "b": "https://someone.medium.com/post",
            REDIRECT + "c": None,  # could not resolve
        }
        citations = build_citations(uris, titles, cited_chunk_order=[1, 0], resolved=resolved)

        assert [c.raw_uri for c in citations] == [uris[1], uris[0], uris[2], uris[3]]
        assert [c.order for c in citations] == [1, 2, 3, 4]
        assert [c.cited_in_answer for c in citations] == [True, True, False, False]
        # Resolved URL beats the title (title said medium.com)
        assert citations[0].domain == "someone.medium.com"
        # Utility result kept but flagged
        assert citations[2].is_web_source is False
        # Unresolved link falls back to the title's domain
        assert citations[3].url is None and citations[3].domain == "c.org" and citations[3].is_web_source

    def test_utility_and_title_helpers(self):
        assert is_utility_result("https://www.google.com/search?q=weather")
        assert not is_utility_result("https://google.com/maps")
        assert domain_from_title("Www.Example.co.in") == "example.co.in"
        assert domain_from_title("Current time information in India.") == ""


# ---------- query selection and visibility ----------

class TestQueriesAndVisibility:
    def test_balanced_selection_round_robins_types(self):
        questions = [{"type": "intent", "q": f"i{n}"} for n in range(5)] + [
            {"type": "experience", "q": f"e{n}"} for n in range(5)
        ] + [{"type": "transaction", "q": f"t{n}"} for n in range(5)]
        selected = select_balanced_queries(questions, 5)
        assert [q["q"] for q in selected] == ["i0", "e0", "t0", "i1", "e1"]

    def test_selection_skips_blanks_and_duplicates(self):
        questions = [{"type": "intent", "q": "Same?"}, {"type": "intent", "q": " same? "}, {"type": "intent", "q": ""}]
        assert [q["q"] for q in select_balanced_queries(questions, 10)] == ["Same?"]

    def test_brand_terms_ignore_unknown(self):
        assert brand_terms("Unknown", "acme.in") == ["acme.in"]
        assert brand_terms("Acme Pay", "acme.in") == ["Acme Pay", "acme.in"]

    def test_mentioned_but_not_cited(self):
        v = detect_visibility("Try Acme Pay for UPI.", ["a.com", "b.com"], "acme.in", ["Acme Pay", "acme.in"])
        assert v == ResponseVisibility(mentioned=True, cited=False, citation_position=None)

    def test_cited_and_position(self):
        v = detect_visibility("Options include Razorpay.", ["a.com", "acme.in"], "acme.in", ["Acme"])
        assert v == ResponseVisibility(mentioned=False, cited=True, citation_position=2)

    def test_mention_needs_whole_word(self):
        assert not detect_visibility("Acmeology is different", [], "acme.in", ["Acme"]).mentioned

    def test_query_with_no_successful_runs_has_no_data(self):
        # Every run failed (e.g. quota errors): rates must be "no data", not 0%
        metrics = query_source_metrics(run_url_sets=[], run_domain_sets=[], visibilities=[])
        assert metrics["successful_runs"] == 0
        assert metrics["mention_rate"] is None and metrics["citation_rate"] is None
        assert metrics["mean_citation_position"] is None and metrics["source_set_stability"] is None
        assert metrics["source_set_diversity"] == 0  # a count: nothing was observed

    def test_query_metrics(self):
        metrics = query_source_metrics(
            run_url_sets=[["u1", "u2"], ["u1"]],
            run_domain_sets=[["a.com"], ["a.com"]],
            visibilities=[ResponseVisibility(True, True, 1), ResponseVisibility(True, False)],
        )
        assert metrics["successful_runs"] == 2
        assert metrics["source_set_diversity"] == 2
        assert metrics["source_set_stability"] == 0.5
        assert metrics["domain_set_stability"] == 1.0
        assert metrics["mention_rate"] == 100.0
        assert metrics["citation_rate"] == 50.0
        assert metrics["mean_citation_position"] == 1.0


# ---------- categorisation ----------

class TestCategorization:
    def test_parse_tolerates_fences_and_drops_unknown(self):
        reply = 'Sure:\n```json\n{"www.A.com": "media", "b.gov": "Government", "c.com": "podcast", "d.com": 3}\n```'
        assert parse_classification_response(reply, TAXONOMY) == {"a.com": "media", "b.gov": "government"}

    def test_parse_garbage(self):
        assert parse_classification_response("no json here", TAXONOMY) == {}
        assert parse_classification_response("{broken", TAXONOMY) == {}

    def test_precedence(self):
        assert resolve_category("acme.in", "media", None, "acme.in").category_source == SOURCE_TARGET_MATCH
        # The LLM cannot make another domain the target
        other = resolve_category("x.com", "target_company", None, "acme.in")
        assert (other.category, other.category_source) == ("other", SOURCE_FALLBACK)
        llm = resolve_category("rbi.gov.in", "reference", "government", "acme.in")
        assert (llm.category, llm.category_source, llm.heuristic_agrees) == ("reference", SOURCE_LLM, False)
        heuristic = resolve_category("rbi.gov.in", None, "government", "acme.in")
        assert (heuristic.category, heuristic.category_source) == ("government", SOURCE_HEURISTIC)
        assert resolve_category("rival.com", "direct_competitor", None, "acme.in").competitor_group == "direct_business"

    def test_failed_batch_falls_back_without_raising(self):
        llm = FakeLLM([RuntimeError("quota"), '{"b.com": "media"}'])
        result = asyncio.run(
            categorize_domains(
                ["https://www.acme.in/x", "en.wikipedia.org", "a.com", "b.com"],
                target_domain="acme.in",
                taxonomy=TAXONOMY,
                batch_size=2,
                llm=llm,
            )
        )
        assert result["acme.in"].category == "target_company"  # never sent to the LLM
        assert (result["en.wikipedia.org"].category, result["en.wikipedia.org"].category_source) == ("reference", SOURCE_HEURISTIC)
        assert (result["a.com"].category, result["a.com"].category_source) == ("other", SOURCE_FALLBACK)
        assert (result["b.com"].category, result["b.com"].category_source) == ("media", SOURCE_LLM)
        assert len(llm.prompts) == 2 and "acme.in" not in llm.prompts[0].split("Domains:")[1]


# ---------- evidence ----------

class TestEvidence:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("NO_RELEVANT_EVIDENCE", True),
            ("NO_Relevant_EVIDENCE", True),  # seen from a live model
            ("No relevant evidence.", True),
            ("  **NO RELEVANT EVIDENCE**\n", True),
            ("- Fact: no relevant evidence of fees was found, but UPI share is 80%", False),
            ("- UPI success rate: 97%", False),
            ("", False),
        ],
    )
    def test_no_evidence_detection(self, reply, expected):
        assert is_no_evidence_reply(reply) is expected

    def test_prompt_mentions_citation_only_for_cited_pages(self):
        cited = build_evidence_prompt("q", "https://a.com", None, "text", 400)
        own = build_evidence_prompt("q", "https://a.com", None, "text", 400, cited_for_query=False)
        assert "cited this page" in cited and "cited this page" not in own
        assert "Partial relevance counts" in cited and "Stay under 300 words" in cited

    def test_token_helpers(self):
        assert estimate_tokens("abcd" * 10) == 10
        assert estimate_tokens("abcde") == 2
        assert enforce_token_budget("short", 10) == "short"
        capped = enforce_token_budget("word " * 100, 10)
        assert len(capped) <= 10 * 4 + 2 and capped.endswith("…")

    def test_batch_survives_failures_and_uses_budget(self):
        sources = [
            {"url": f"https://s{i}.com", "title": None, "page_text": "page text"} for i in range(6)
        ]
        sources[2]["page_text"] = None
        sources[2]["crawl_error"] = "HTTP 403"
        llm = FakeLLM(["- fact A", RuntimeError("timeout"), "- fact C", "NO_RELEVANT_EVIDENCE", "x" * 5000])
        results = asyncio.run(
            compress_query_sources(
                "best upi gateway", sources, total_budget=1000, per_source_cap=400, min_floor=100,
                batch_size=5, max_page_chars=1000, llm=llm,
            )
        )
        assert [r.url for r in results] == [s["url"] for s in sources]
        assert all(r.token_budget == 166 for r in results)  # 1000 // 6
        assert results[0].evidence == "- fact A" and results[0].has_relevant_evidence
        assert results[1].error == "timeout" and results[1].evidence is None
        assert results[2].error == "HTTP 403"  # no LLM call without page text
        assert results[3].evidence == "- fact C"
        assert results[4].has_relevant_evidence is False and results[4].evidence is None
        assert results[5].token_estimate <= 166  # hard-capped, ellipsis included
        assert results[0].latency_seconds is not None
        assert len(llm.prompts) == 5
        assert "compact representation containing the evidence" in llm.prompts[0]

    def test_rolling_window_limits_calls_in_flight(self):
        class SlowLLM:
            def __init__(self):
                self.in_flight = self.peak = self.calls = 0

            async def chat(self, messages, max_tokens=None, temperature=None):
                self.in_flight += 1
                self.calls += 1
                self.peak = max(self.peak, self.in_flight)
                await asyncio.sleep(0.01 * (self.calls % 3))  # uneven durations
                self.in_flight -= 1
                return "- fact"

        llm = SlowLLM()
        sources = [{"url": f"https://s{i}.com", "title": None, "page_text": "text"} for i in range(12)]
        results = asyncio.run(
            compress_query_sources("q", sources, total_budget=5000, per_source_cap=500, min_floor=100,
                                   batch_size=3, max_page_chars=100, llm=llm)
        )
        assert llm.calls == 12 and llm.peak == 3
        assert [r.url for r in results] == [s["url"] for s in sources]  # order preserved


# ---------- config ----------

class TestConfig:
    def test_env_and_overrides(self, monkeypatch):
        monkeypatch.setenv("STAGE1_RUNS_PER_QUERY", "7")
        monkeypatch.setenv("STAGE1_EVIDENCE_TOKEN_BUDGET", "10000")
        config = Stage1Config.from_env({"query_count": 3, "runs_per_query": None})
        assert (config.runs_per_query, config.query_count, config.evidence_token_budget) == (7, 3, 10000)
        assert config.max_sources_per_query == 50  # 10000 // 200

    def test_invalid(self, monkeypatch):
        with pytest.raises(ValueError):
            Stage1Config.from_env({"runs_per_query": 0})
        monkeypatch.setenv("STAGE1_MIN_SOURCE_TOKENS", "5000")
        with pytest.raises(ValueError):
            Stage1Config.from_env()


# ---------- mentions excluding absence-of-information statements ----------

from app.observation.stage1_job import detect_mentions, split_sentences  # noqa: E402

TERMS = ["Razorpay", "razorpay.com"]


class TestMentionExclAbsence:
    @pytest.mark.parametrize(
        "sentence",
        [
            "The provided context does not contain information about Razorpay.",
            "The context doesn't contain any information on Razorpay's pricing.",
            "There is no mention of Razorpay in the sources.",
            "The sources do not mention Razorpay.",
            "The answer doesn’t mention Razorpay at all.",  # curly apostrophe
            "Razorpay is not mentioned in the retrieved pages.",
            "There is no information about Razorpay settlement times.",
            "No information regarding razorpay.com was found.",
            "There is not enough information to compare Razorpay with PayU.",
            "I cannot find details on Razorpay's fees.",
            "I can't answer whether Razorpay supports RuPay credit cards.",
        ],
    )
    def test_absence_statements_are_not_mentions(self, sentence):
        check = detect_mentions(sentence, TERMS)
        assert check.mentioned is True  # the existing string match still fires
        assert check.mentioned_excl_absence is False
        assert check.absence_sentences == [sentence] and check.mention_sentences == []

    @pytest.mark.parametrize(
        "sentence",
        [
            "Razorpay does not charge setup fees.",  # ordinary negation: a real mention
            "Razorpay isn't the cheapest option for international cards.",
            "Unlike Razorpay, PayU does not offer instant settlements.",
            "Razorpay never charges annual maintenance fees.",
            "Razorpay is mentioned by most reviewers as the top choice.",
            "Payment Gateway | Best For\nRazorpay | Full-stack tech & SaaS",  # table row line
        ],
    )
    def test_real_mentions_count(self, sentence):
        check = detect_mentions(sentence, TERMS)
        assert check.mentioned is True and check.mentioned_excl_absence is True
        assert check.absence_sentences == []

    def test_mixed_answer_keeps_both_lists(self):
        text = (
            "The context does not contain information about Razorpay's settlement times. "
            "However, Razorpay does not charge setup fees. Cashfree is also popular."
        )
        check = detect_mentions(text, TERMS)
        assert check.mentioned_excl_absence is True
        assert check.absence_sentences == ["The context does not contain information about Razorpay's settlement times."]
        assert check.mention_sentences == ["However, Razorpay does not charge setup fees."]

    def test_only_absence_sentences(self):
        text = "No mention of Razorpay here. Cashfree and PayU are covered instead."
        check = detect_mentions(text, TERMS)
        assert (check.mentioned, check.mentioned_excl_absence) == (True, False)
        assert check.mention_sentences == []

    def test_absence_statement_about_another_brand_is_ignored(self):
        # Absence wording without a brand term is not a brand sentence at all
        text = "There is no information about PayU. Razorpay offers UPI."
        check = detect_mentions(text, TERMS)
        assert check.absence_sentences == [] and check.mention_sentences == ["Razorpay offers UPI."]

    def test_no_brand(self):
        check = detect_mentions("Cashfree and PayU are popular. There is no mention of fees.", TERMS)
        assert (check.mentioned, check.mentioned_excl_absence, check.mention_sentences, check.absence_sentences) == (False, False, [], [])

    def test_whole_word_rules_match_detect_visibility(self):
        samples = [
            "Razorpayments is a different company.",
            "Try RAZORPAY today.",
            "Visit razorpay.com for details.",
            "Use app.razorpay.com to log in.",
            "There is no mention of Razorpay.",
            "Razorpay-backed merchants grew.",
        ]
        for text in samples:
            assert detect_mentions(text, TERMS).mentioned == detect_visibility(text, [], "razorpay.com", TERMS).mentioned, text

    def test_split_sentences(self):
        assert split_sentences("One. Two? Three!\nFour | five\n\nSix") == ["One.", "Two?", "Three!", "Four | five", "Six"]

    def test_rates_over_answered_runs(self):
        from app.observation.fetchers import OUTCOME_ANSWER, Observation, ObservedCitation

        def run(text):
            cite = ObservedCitation("https://a.com", "https://a.com", "a.com", None, 1, True, True)
            return {"run_index": 0, "ok": True, "observation": Observation(answer_text=text, model="m", citations=[cite], outcome=OUTCOME_ANSWER)}

        agg = aggregate_query_runs(
            [run("Razorpay does not charge setup fees."), run("There is no mention of Razorpay."), run("Only Cashfree.")],
            "razorpay.com", TERMS, reports_activation=False,
        )
        m = agg["metrics"]
        assert m["mention_rate"] == pytest.approx(200 / 3)  # unchanged string match
        assert m["mention_rate_excl_absence"] == pytest.approx(100 / 3)
        stored = agg["serialized"]
        assert [r["mentioned_excl_absence"] for r in stored] == [True, False, False]
        assert stored[1]["absence_sentences"] == ["There is no mention of Razorpay."]
        assert stored[0]["mention_sentences"] == ["Razorpay does not charge setup fees."]
        assert [r["target"]["mentioned"] for r in stored] == [True, True, False]

    def test_no_answered_runs_is_no_data(self):
        assert aggregate_query_runs([], "razorpay.com", TERMS, False)["metrics"]["mention_rate_excl_absence"] is None


# ---------- skip_evidence (observation-only runs) ----------

class TestSkipEvidence:
    def test_config_env_and_override(self, monkeypatch):
        assert Stage1Config.from_env().skip_evidence is False
        monkeypatch.setenv("STAGE1_SKIP_EVIDENCE", "true")
        assert Stage1Config.from_env().skip_evidence is True
        assert Stage1Config.from_env({"skip_evidence": False}).skip_evidence is False
        monkeypatch.setenv("STAGE1_SKIP_EVIDENCE", "maybe")
        with pytest.raises(ValueError, match="STAGE1_SKIP_EVIDENCE"):
            Stage1Config.from_env()

    @staticmethod
    def _run_job(monkeypatch, skip):
        from app.observation import stage1_job
        from app.observation.categorize import categorize_domains
        from app.observation.evidence import EvidenceResult
        from app.observation.fetchers import OUTCOME_ANSWER, Observation, ObservedCitation

        calls = {"categorize": 0, "crawl": 0, "compress": 0, "posted": None}

        class Fetcher:
            name, data_disclaimer, reports_activation, settings = "fake", "fake data", False, {}

            async def preflight(self, planned):
                return {}

            async def observe(self, query):
                cites = [ObservedCitation(f"https://s{i}.com/p", f"https://s{i}.com/p", f"s{i}.com", None, i + 1, True, True) for i in range(3)]
                return Observation(answer_text="Acme is great.", model="m", citations=cites, outcome=OUTCOME_ANSWER)

        async def context(url):
            return {"page_title": "Acme", "page_text": "Acme page text", "domain_summary": None,
                    "page_intent": "x", "brand_name": "Acme", "business_category": "y"}

        class NoLLM:
            async def chat(self, *a, **k):
                return "{}"

        async def categorize(domains, **kwargs):
            calls["categorize"] += 1
            return await categorize_domains(domains, llm=NoLLM(), **kwargs)

        async def crawl(urls, concurrency):
            calls["crawl"] += 1
            return {u: {"page_text": "text", "title": "T", "crawl_error": None} for u in urls}

        async def compress(query, sources, **kwargs):
            calls["compress"] += 1
            return [EvidenceResult(s["url"], "- fact", 100, 3, True) for s in sources]

        async def quiet(*args, **kwargs):
            return None

        async def post(path, payload, timeout):
            if path.endswith("/result"):
                calls["posted"] = payload

        for name, value in [("_build_context", context), ("categorize_domains", categorize), ("_crawl_sources", crawl),
                            ("compress_query_sources", compress), ("_send_progress", quiet), ("_post", post)]:
            monkeypatch.setattr(stage1_job, name, value)
        result = asyncio.run(stage1_job.run_stage1_job("obs", "https://acme.in", ["q"], {"runs_per_query": 2, "skip_evidence": skip}, fetcher=Fetcher()))
        return result, calls

    def test_skip_evidence_skips_crawl_and_evidence_but_keeps_categorisation(self, monkeypatch):
        result, calls = self._run_job(monkeypatch, skip=True)
        assert result["status"] == "completed" and result["evidence_skipped"] is True
        assert calls == {"categorize": 1, "crawl": 0, "compress": 0, "posted": result}
        sources = result["queries"][0]["sources"]
        assert len(sources) == 3
        assert all(s["evidence"] is None and s["evidence_error"] is None and s["crawl_error"] is None for s in sources)
        assert not any(s["over_source_cap"] for s in sources)
        assert result["queries"][0]["target_evidence"] is None
        assert result["metrics"]["mention_rate"] == 100.0  # observation metrics unaffected

    def test_evidence_runs_by_default(self, monkeypatch):
        result, calls = self._run_job(monkeypatch, skip=False)
        assert result["evidence_skipped"] is False
        assert calls["crawl"] == 1 and calls["compress"] >= 1
        assert all(s["evidence"] == "- fact" for s in result["queries"][0]["sources"])
