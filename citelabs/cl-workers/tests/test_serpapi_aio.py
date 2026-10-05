"""SerpApiAIOFetcher with faked HTTP (httpx.MockTransport): no network, no credits."""
import asyncio
import json
import os

import httpx
import pytest

from app.observation.config import Stage1Config
from app.observation.fetchers import (
    OUTCOME_ANSWER,
    OUTCOME_NO_AI_ANSWER,
    OUTCOME_PARSE_ERROR,
    GeminiGroundingFetcher,
    Observation,
    ObservedCitation,
)
from app.observation.serpapi_aio import (
    SerpApiAIOFetcher,
    SerpApiCreditsError,
    SerpApiError,
    SerpApiFormatError,
    parse_ai_overview,
    parse_raw_responses,
    reparse_stored_response,
    sanitize_for_storage,
)
from app.observation import stage1_job
from app.observation.stage1_job import aggregate_query_runs

FIXTURE = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "serpapi_smoke_2026-10-05.json")))
RUN0, RUN1 = FIXTURE["runs"]  # each: [engine=google response, engine=google_ai_overview response]

KEY = "k" * 64

FULL_AIO = {
    "text_blocks": [
        {"type": "paragraph", "snippet": "Razorpay and Cashfree are popular gateways in India.", "reference_indexes": [0, 1]},
        {
            "type": "list",
            "list": [
                {"title": "Razorpay:", "snippet": "2% per transaction.", "reference_indexes": [0]},
                {"title": "PayU:", "snippet": "Wide bank coverage.", "reference_indexes": [2]},
            ],
        },
        {"type": "expandable", "title": "Fees", "text_blocks": [{"type": "paragraph", "snippet": "UPI is often free."}]},
    ],
    "references": [
        {"title": "Razorpay pricing", "link": "https://razorpay.com/pricing/", "source": "Razorpay", "index": 0},
        {"title": "Cashfree", "link": "https://www.cashfree.com/payment-gateway-india/", "source": "Cashfree", "index": 1},
        {"title": "PayU", "link": "https://payu.in/", "source": "PayU", "index": 2},
        {"title": "Unused", "link": "https://example.org/x", "source": "Example", "index": 3},
    ],
}


def ok(body, search_id="s1"):
    return {"search_metadata": {"id": search_id, "status": "Success"}, **body}


class FakeSerpApi:
    """Replays responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.responses.pop(0)
        return httpx.Response(status, json=body)

    def params(self, i):
        return dict(request_params(self.requests[i]))


def request_params(request):
    return list(httpx.QueryParams(request.url.query).multi_items())


def fetcher(fake, **kwargs):
    return SerpApiAIOFetcher(api_key=KEY, transport=httpx.MockTransport(fake), **kwargs)


def run(coro):
    return asyncio.run(coro)


# ---------- the four response cases ----------

class TestObserve:
    def test_full_ai_overview(self):
        fake = FakeSerpApi([(200, ok({"ai_overview": FULL_AIO, "organic_results": []}))])
        obs = run(fetcher(fake).observe("best payment gateway india"))

        sent = fake.params(0)
        assert sent["engine"] == "google" and sent["q"] == "best payment gateway india"
        assert (sent["gl"], sent["hl"], sent["device"], sent["no_cache"]) == ("in", "en", "desktop", "true")
        assert sent["api_key"] == KEY

        assert obs.outcome == OUTCOME_ANSWER and obs.credits_used == 1
        assert [c.url for c in obs.citations] == [r["link"] for r in FULL_AIO["references"]]  # reference order
        assert [c.order for c in obs.citations] == [1, 2, 3, 4]
        assert [c.domain for c in obs.citations] == ["razorpay.com", "cashfree.com", "payu.in", "example.org"]
        assert all(c.cited_in_answer and c.is_web_source for c in obs.citations)
        for text in ("Razorpay and Cashfree", "Razorpay:", "2% per transaction.", "PayU:", "Fees", "UPI is often free."):
            assert text in obs.answer_text
        meta = obs.fetch_meta
        assert meta["engines_called"] == ["google"] and meta["page_token_used"] is False
        assert meta["reference_count"] == 4 and meta["reference_indexes_used_in_text"] == [0, 1, 2]
        assert (meta["gl"], meta["hl"], meta["device"], meta["fetcher"]) == ("in", "en", "desktop", "serpapi_aio")

    def test_page_token_flow(self):
        fake = FakeSerpApi([
            (200, ok({"ai_overview": {"page_token": "TOKEN123", "serpapi_link": "https://serpapi.com/search.json?engine=google_ai_overview&page_token=TOKEN123"}}, "s1")),
            (200, ok({"ai_overview": FULL_AIO}, "s2")),
        ])
        obs = run(fetcher(fake).observe("q"))
        second = fake.params(1)
        assert second["engine"] == "google_ai_overview" and second["page_token"] == "TOKEN123"
        assert second["no_cache"] == "true" and "q" not in second
        assert obs.outcome == OUTCOME_ANSWER and obs.credits_used == 2
        assert obs.fetch_meta["page_token_used"] is True
        assert obs.fetch_meta["engines_called"] == ["google", "google_ai_overview"]
        assert obs.fetch_meta["search_ids"] == ["s1", "s2"]
        assert len(obs.citations) == 4

    def test_no_ai_overview_is_an_outcome_not_a_failure(self):
        fake = FakeSerpApi([(200, ok({"organic_results": [{"link": "https://a.com"}]}))])
        obs = run(fetcher(fake).observe("q"))
        assert obs.outcome == OUTCOME_NO_AI_ANSWER
        assert obs.citations == [] and obs.answer_text == "" and obs.credits_used == 1

    def test_api_error(self):
        fake = FakeSerpApi([(401, {"error": "Invalid API key. Your API key should be here: https://serpapi.com/manage-api-key"})])
        with pytest.raises(SerpApiError, match="HTTP 401.*Invalid API key") as info:
            run(fetcher(fake).observe("q"))
        assert info.value.credits_used == 0

    def test_error_status_in_200_body(self):
        fake = FakeSerpApi([(200, {"search_metadata": {"id": "s1", "status": "Error"}, "error": "Google hasn't returned any results for this query."})])
        with pytest.raises(SerpApiError, match="hasn't returned any results"):
            run(fetcher(fake).observe("q"))

    def test_expired_page_token_keeps_first_credit(self):
        fake = FakeSerpApi([
            (200, ok({"ai_overview": {"page_token": "OLD"}})),
            (400, {"error": "Invalid page_token or page_token expired."}),
        ])
        with pytest.raises(SerpApiError, match="expired") as info:
            run(fetcher(fake).observe("q"))
        assert info.value.credits_used == 1  # the first search was spent
        assert len(info.value.raw_responses) == 1  # and its response is kept


# ---------- strict parsing: unknown shapes are errors, never guesses ----------

class TestStrictParsing:
    @pytest.mark.parametrize(
        "ai_overview,message",
        [
            ({"references": []}, "no text_blocks"),
            ({"text_blocks": []}, "no references"),
            ({"text_blocks": [], "references": [{"title": "no link"}]}, "reference #1 has no http link"),
            ({"text_blocks": [], "references": [{"link": "/relative"}]}, "reference #1 has no http link"),
            ({"text_blocks": "a string", "references": []}, "no text_blocks"),
            ({"text_blocks": [{"type": "list", "list": "not a list"}], "references": []}, "'list' is not a list"),
            ({"text_blocks": [{"type": "video", "snippet": "x"}], "references": []}, "unknown text block type 'video'"),
            ({"text_blocks": [{"snippet": "no type"}], "references": []}, "unknown text block type None"),
            ({"text_blocks": [{"type": "table"}], "references": []}, "table block has no 'detailed' rows"),
            ({"text_blocks": [{"type": "table", "detailed": [[{"text": "x"}]]}], "references": []}, "table cell without a 'snippet' string"),
            ({"text_blocks": [{"type": "paragraph", "snippet_links": [{"text": "x"}]}], "references": []}, "inline link without a 'link' string"),
            ("just text", "expected an object"),
        ],
    )
    def test_unexpected_shapes_raise(self, ai_overview, message):
        with pytest.raises(SerpApiFormatError, match=message):
            parse_ai_overview(ai_overview)

    def test_unknown_shape_from_api_is_a_parse_error_observation(self):
        # Credits were spent, so the observation is kept (as parse_error) instead of raising
        fake = FakeSerpApi([(200, ok({"ai_overview": {"something_new": True}}))])
        obs = run(fetcher(fake).observe("q"))
        assert obs.outcome == OUTCOME_PARSE_ERROR and obs.credits_used == 1
        assert "no text_blocks" in obs.fetch_meta["parse_error"]
        assert obs.citations == [] and len(obs.raw_responses) == 1

    def test_unknown_block_type_is_a_parse_error_observation(self):
        aio = {"text_blocks": [{"type": "paragraph", "snippet": "ok"}, {"type": "carousel", "items": []}], "references": []}
        fake = FakeSerpApi([(200, ok({"ai_overview": aio}))])
        obs = run(fetcher(fake).observe("q"))
        assert obs.outcome == OUTCOME_PARSE_ERROR
        assert "text_blocks[1]: unknown text block type 'carousel'" in obs.fetch_meta["parse_error"]

    def test_page_token_response_without_ai_overview(self):
        fake = FakeSerpApi([(200, ok({"ai_overview": {"page_token": "T"}})), (200, ok({"organic_results": []}))])
        obs = run(fetcher(fake).observe("q"))
        assert obs.outcome == OUTCOME_PARSE_ERROR and obs.credits_used == 2
        assert "google_ai_overview response has no ai_overview" in obs.fetch_meta["parse_error"]


# ---------- credits ----------

def account(left):
    return (200, {"account_id": "acct_secret", "account_email": "me@example.com", "plan_name": "Free Plan", "total_searches_left": left})


class TestCredits:
    def test_preflight_refuses_unaffordable_run_without_searching(self):
        fake = FakeSerpApi([account(3)])
        with pytest.raises(SerpApiCreditsError, match="up to 4 SerpApi credits .* only 3 remain"):
            run(fetcher(fake).preflight(planned_observations=2))
        assert len(fake.requests) == 1 and fake.requests[0].url.path == "/account.json"

    def test_budget_stops_requests_at_remaining(self):
        # 1 observation planned (worst case 2) with exactly 2 left; the observation then
        # needs a third request, which must not be sent
        no_aio = (200, ok({"organic_results": []}))
        fake = FakeSerpApi([account(2), no_aio, no_aio])
        f = fetcher(fake)
        info = run(f.preflight(planned_observations=1))
        assert info == {"serpapi_searches_left_before": 2, "serpapi_worst_case_credits": 2}

        async def three():
            await f.observe("a")
            await f.observe("b")
            await f.observe("c")

        with pytest.raises(SerpApiCreditsError, match="request not sent"):
            run(three())
        assert len(fake.requests) == 3  # account + 2 searches; the third was never sent

    def test_failed_requests_do_not_count_against_budget(self):
        fake = FakeSerpApi([account(2), (500, {"error": "Internal"}), (200, ok({})), (200, ok({}))])
        f = fetcher(fake)
        run(f.preflight(1))
        with pytest.raises(SerpApiError):
            run(f.observe("a"))
        assert run(f.observe("b")).credits_used == 1
        assert run(f.observe("c")).credits_used == 1

    def test_account_format_change_is_an_error(self):
        fake = FakeSerpApi([(200, {"plan_name": "Free Plan"})])
        with pytest.raises(SerpApiFormatError, match="total_searches_left"):
            run(fetcher(fake).preflight(1))


# ---------- provenance, disclaimers, storage safety ----------

class TestProvenance:
    def test_disclaimers_are_fetcher_specific(self):
        serp = SerpApiAIOFetcher(api_key=KEY, gl="us", hl="en", device="mobile")
        assert serp.data_disclaimer == (
            "Google AI Overview results as returned by SerpApi for country (gl) = us, language (hl) = en, device = mobile."
        )
        assert "proxy" not in serp.data_disclaimer
        gemini = GeminiGroundingFetcher(model="gemini-2.5-flash", provider=object())
        assert "approximate proxy" in gemini.data_disclaimer and "not real AI Overview data" in gemini.data_disclaimer
        assert serp.settings == {"fetcher": "serpapi_aio", "gl": "us", "hl": "en", "device": "mobile", "no_cache": True}
        assert serp.reports_activation and not gemini.reports_activation

    def test_sanitize_removes_key_and_account_identifiers(self):
        raw = {
            "search_metadata": {"id": "s1", "json_endpoint": "https://serpapi.com/searches/abc.json", "markdown_endpoint": "https://serpapi.com/searches/acct-hash/s1.md"},
            "search_parameters": {"engine": "google", "q": "x", "api_key": KEY},
            "ai_overview": {"serpapi_link": f"https://serpapi.com/search.json?engine=google_ai_overview&page_token=T&api_key={KEY}"},
            "note": f"echo {KEY}",
            "account": {"account_id": "acct_1", "account_email": "me@example.com", "plan_name": "Free"},
        }
        clean = sanitize_for_storage(raw, (KEY,))
        text = json.dumps(clean)
        assert KEY not in text and "api_key" not in text
        assert "acct_1" not in text and "me@example.com" not in text and clean["account"] == {"plan_name": "Free"}
        assert clean["search_parameters"] == {"engine": "google", "q": "x"}
        assert clean["search_metadata"] == {"id": "s1"}  # archive link dropped
        assert "page_token=T" in clean["ai_overview"]["serpapi_link"]

    def test_config_validation(self):
        assert Stage1Config.from_env({"fetcher": "serpapi_aio", "serpapi_gl": "us"}).serpapi_settings == {"gl": "us", "hl": "en", "device": "desktop"}
        with pytest.raises(ValueError, match="unknown fetcher"):
            Stage1Config.from_env({"fetcher": "bing"})
        with pytest.raises(ValueError, match="serpapi_device"):
            Stage1Config.from_env({"serpapi_device": "watch"})

    def test_device_validated_by_fetcher(self):
        with pytest.raises(ValueError):
            SerpApiAIOFetcher(api_key=KEY, device="watch")


# ---------- rate rules with no-AI-Overview runs ----------

def answered(cited_target, credits=1, mention=True):
    cites = [ObservedCitation("https://rival.com/a", "https://rival.com/a", "rival.com", None, 1, True, True)]
    if cited_target:
        cites.append(ObservedCitation("https://acme.in/x", "https://acme.in/x", "acme.in", None, 2, True, True))
    return {"run_index": 0, "ok": True, "observation": Observation(
        answer_text="Acme is good" if mention else "Rival is good", model="m", citations=cites, outcome=OUTCOME_ANSWER, credits_used=credits)}


def no_aio(credits=1):
    return {"run_index": 0, "ok": True, "observation": Observation(answer_text="", model="m", outcome=OUTCOME_NO_AI_ANSWER, credits_used=credits)}


class TestRateRules:
    def test_rates_over_ai_overview_runs_with_activation_and_overall(self):
        runs = [answered(True, credits=2), answered(False), no_aio(), {"run_index": 3, "ok": False, "error": "x", "credits_used": 1}]
        m = aggregate_query_runs(runs, "acme.in", ["Acme"], reports_activation=True)["metrics"]
        assert m["successful_runs"] == 3  # 2 answered + 1 no-AIO; the failed run is excluded
        assert m["rates_based_on_runs"] == 2 and m["aio_runs"] == 2
        assert m["aio_activation_rate"] == pytest.approx(200 / 3)
        assert m["citation_rate"] == 50.0  # 1 of 2 AI Overviews cited the target
        assert m["overall_citation_rate"] == pytest.approx(100 / 3)  # 1 of 3 successful runs
        assert m["mention_rate"] == 100.0
        assert m["mean_citation_position"] == 2.0
        assert m["credits_used"] == 2 + 1 + 1 + 1

    def test_only_no_aio_runs_is_no_data_with_zero_activation(self):
        m = aggregate_query_runs([no_aio(), no_aio()], "acme.in", ["Acme"], reports_activation=True)["metrics"]
        assert (m["successful_runs"], m["rates_based_on_runs"], m["aio_activation_rate"]) == (2, 0, 0.0)
        assert m["citation_rate"] is None and m["mention_rate"] is None and m["overall_citation_rate"] == 0.0

    def test_gemini_fetcher_has_no_activation_fields(self):
        runs = [answered(True, credits=None), answered(False, credits=None)]
        m = aggregate_query_runs(runs, "acme.in", ["Acme"], reports_activation=False)["metrics"]
        assert m["aio_runs"] is None and m["aio_activation_rate"] is None and m["credits_used"] is None
        assert m["successful_runs"] == m["rates_based_on_runs"] == 2
        assert m["citation_rate"] == m["overall_citation_rate"] == 50.0


# ---------- tables, ragged rows, inline links (real responses) ----------

class TestTablesAndInlineLinks:
    def test_table_cells_are_in_the_answer_text(self):
        parsed = parse_raw_responses(RUN0)
        assert parsed.details["tables_seen"] == 1
        assert "Gateway | Best For | Standard Pricing | Key Advantages" in parsed.answer_text
        # Razorpay appears in this table's gateway column
        assert "Razorpay | Full-stack tech & SaaS | ~2% per transaction" in parsed.answer_text

    def test_ragged_rows_are_flagged(self):
        assert (parse_raw_responses(RUN0).details["ragged_table"], parse_raw_responses(RUN0).details["ragged_rows"]) == (True, 2)
        assert (parse_raw_responses(RUN1).details["ragged_table"], parse_raw_responses(RUN1).details["ragged_rows"]) == (True, 3)

    def test_complete_table_is_not_ragged(self):
        aio = {"text_blocks": [{"type": "table", "detailed": [[{"snippet": "A"}, {"snippet": "B"}], [{"snippet": "1"}, {"snippet": "2"}]]}], "references": []}
        _, _, details = parse_ai_overview(aio)
        assert (details["tables_seen"], details["ragged_table"], details["ragged_rows"]) == (1, False, 0)

    def test_inline_links_recorded_but_not_citations(self):
        parsed = parse_raw_responses(RUN1)
        assert parsed.details["inline_links"] == [{"text": "Razorpay", "link": "https://razorpay.com"}]
        # citations are references only: razorpay.com is not among them
        assert [c.url for c in parsed.citations] == [r["link"] for r in RUN1[1]["ai_overview"]["references"]]
        assert "razorpay.com" not in {c.domain for c in parsed.citations}

    def test_target_inline_linked_is_secondary_metric(self):
        def answer_run(raws):
            parsed = parse_raw_responses(raws)
            return {"run_index": 0, "ok": True, "observation": Observation(
                answer_text=parsed.answer_text, model="m", citations=parsed.citations, outcome=parsed.outcome,
                credits_used=2, fetch_meta=parsed.details, raw_responses=raws)}
        agg = aggregate_query_runs([answer_run(RUN0), answer_run(RUN1)], "razorpay.com", ["Razorpay", "razorpay.com"], True)
        m = agg["metrics"]
        assert [r["target_inline_linked"] for r in agg["serialized"]] == [False, True]
        assert (m["inline_linked_runs"], m["inline_link_rate"]) == (1, 50.0)
        assert m["citation_rate"] == 0.0  # inline links never count as citations
        assert m["mention_rate"] == 100.0


# ---------- parse_error flow ----------

class TestParseErrorFlow:
    def test_parse_errors_are_counted_not_successful(self):
        parse_error = {"run_index": 1, "ok": True, "observation": Observation(
            answer_text="", model="m", outcome=OUTCOME_PARSE_ERROR, credits_used=2, fetch_meta={"parse_error": "bad"})}
        m = aggregate_query_runs([answered(True), parse_error, no_aio()], "acme.in", ["Acme"], True)["metrics"]
        assert m["parse_errors"] == 1
        assert m["successful_runs"] == 2 and m["rates_based_on_runs"] == 1
        assert m["aio_activation_rate"] == 50.0
        assert m["credits_used"] == 1 + 2 + 1  # parse-error credits still count

    @staticmethod
    def _run_observe_all(outcomes, monkeypatch):
        calls = []

        class ScriptedFetcher:
            async def observe(self, query):
                outcome = outcomes[len(calls)]
                calls.append(outcome)
                return Observation(answer_text="", model="m", outcome=outcome, credits_used=1)

        async def no_progress(*args, **kwargs):
            return None

        monkeypatch.setattr(stage1_job, "_send_progress", no_progress)
        config = Stage1Config.from_env({"runs_per_query": len(outcomes), "observation_concurrency": 1})
        runs_by_query, stopped = run(stage1_job._observe_all([{"q": "q", "type": "intent"}], ScriptedFetcher(), config, "obs"))
        return calls, runs_by_query[0], stopped

    def test_three_consecutive_parse_errors_stop_the_run(self, monkeypatch):
        P, A = OUTCOME_PARSE_ERROR, OUTCOME_ANSWER
        calls, runs, stopped = self._run_observe_all([A, P, P, P, A, A], monkeypatch)
        assert stopped and len(calls) == 4  # the last two were never sent (no credits)
        skipped = [r for r in runs if not r["ok"]]
        assert len(skipped) == 2 and all("stopped after 3 consecutive parse errors" in r["error"] for r in skipped)
        assert all(r["credits_used"] is None for r in skipped)

    def test_streak_resets_on_any_other_outcome(self, monkeypatch):
        P, A, N = OUTCOME_PARSE_ERROR, OUTCOME_ANSWER, OUTCOME_NO_AI_ANSWER
        calls, runs, stopped = self._run_observe_all([P, P, A, P, P, N, P, P], monkeypatch)
        assert not stopped and len(calls) == 8


# ---------- re-parsing stored raw responses ----------

def replay(raws):
    """FakeSerpApi responses that reproduce a stored observation."""
    return [(200, raw) for raw in raws]


class TestReparse:
    def test_reparse_reproduces_live_parse(self):
        for raws in (RUN0, RUN1):
            live = run(fetcher(FakeSerpApi(replay(raws))).observe(FIXTURE["query"]))
            rebuilt = reparse_stored_response(live.raw_responses)
            assert rebuilt["outcome"] == live.outcome == OUTCOME_ANSWER
            assert rebuilt["answer_text"] == live.answer_text
            assert [c["url"] for c in rebuilt["citations"]] == [c.url for c in live.citations]
            for key in ("tables_seen", "ragged_table", "ragged_rows", "inline_links", "reference_count"):
                assert rebuilt["details"][key] == live.fetch_meta[key]

    def test_stored_raw_responses_are_sanitized(self):
        page_token = (200, ok({"ai_overview": {"page_token": "T", "serpapi_link": f"https://serpapi.com/search.json?page_token=T&api_key={KEY}"}}, "s1"))
        aio = (200, {"search_metadata": {"id": "s2", "status": "Success", "json_endpoint": "https://serpapi.com/searches/x.json",
                                         "markdown_endpoint": "https://serpapi.com/searches/acct/s2.md"},
                     "search_parameters": {"engine": "google_ai_overview", "api_key": KEY}, "ai_overview": FULL_AIO})
        obs = run(fetcher(FakeSerpApi([page_token, aio])).observe("q"))
        stored = json.dumps(obs.raw_responses)
        assert len(obs.raw_responses) == 2
        assert KEY not in stored and "api_key" not in stored and "serpapi.com/searches" not in stored

    def test_reparse_reports_parse_errors(self):
        rebuilt = reparse_stored_response([ok({"ai_overview": {"text_blocks": [{"type": "video"}], "references": []}})])
        assert rebuilt["outcome"] == OUTCOME_PARSE_ERROR and "unknown text block type 'video'" in rebuilt["parse_error"]
        assert reparse_stored_response([ok({})])["outcome"] == OUTCOME_NO_AI_ANSWER
