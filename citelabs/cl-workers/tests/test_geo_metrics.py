import pytest

from app.geo_metrics import (
    DEFAULT_SOURCE_CATEGORIES,
    GROUP_AI_VISIBILITY,
    GROUP_DIRECT_BUSINESS,
    GROUP_TARGET,
    ResponseVisibility,
    category_breakdown,
    citation_position,
    citation_rate,
    compare_run_metrics,
    competitor_group,
    compute_per_source_token_budget,
    heuristic_source_category,
    jaccard_similarity,
    load_source_categories,
    mean_citation_position,
    mean_pairwise_jaccard,
    median_citation_position,
    mention_rate,
    normalize_domain,
    source_set_diversity,
    source_stability,
    union_sources,
)


# ---------- token budget ----------

class TestTokenBudget:
    def test_few_sources_hit_per_source_cap(self):
        # 20000 / 5 = 4000, capped at 1000
        assert compute_per_source_token_budget(5) == 1000

    def test_exactly_at_cap(self):
        assert compute_per_source_token_budget(20) == 1000

    def test_budget_split_between_cap_and_floor(self):
        # 20000 / 40 = 500
        assert compute_per_source_token_budget(40) == 500

    def test_integer_division(self):
        # 20000 / 30 = 666.67 -> 666
        assert compute_per_source_token_budget(30) == 666

    def test_floor_wins_over_total_budget(self):
        # 20000 / 150 = 133 < floor
        assert compute_per_source_token_budget(150) == 200

    def test_zero_sources(self):
        assert compute_per_source_token_budget(0) == 0

    def test_custom_parameters(self):
        assert compute_per_source_token_budget(10, total_budget=5000, per_source_cap=800, min_floor=150) == 500
        assert compute_per_source_token_budget(100, total_budget=5000, per_source_cap=800, min_floor=150) == 150

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_sources": -1},
            {"num_sources": 5, "total_budget": 0},
            {"num_sources": 5, "per_source_cap": 0},
            {"num_sources": 5, "min_floor": 0},
            {"num_sources": 5, "per_source_cap": 100, "min_floor": 200},
        ],
    )
    def test_invalid_arguments(self, kwargs):
        with pytest.raises(ValueError):
            compute_per_source_token_budget(**kwargs)


# ---------- mention vs citation rate ----------

class TestMentionVsCitationRate:
    def test_mention_without_citation_counts_only_for_mention(self):
        responses = [
            ResponseVisibility(mentioned=True, cited=True, citation_position=1),
            ResponseVisibility(mentioned=True, cited=False),
            ResponseVisibility(mentioned=False, cited=False),
            ResponseVisibility(mentioned=False, cited=False),
        ]
        assert mention_rate(responses) == 50.0
        assert citation_rate(responses) == 25.0

    def test_citation_without_mention_counts_only_for_citation(self):
        responses = [
            ResponseVisibility(mentioned=False, cited=True, citation_position=3),
            ResponseVisibility(mentioned=False, cited=False),
        ]
        assert mention_rate(responses) == 0.0
        assert citation_rate(responses) == 50.0

    def test_no_responses_is_no_data_not_zero(self):
        # Zero responses means nothing was measured; 0% would claim the brand was never mentioned/cited
        assert mention_rate([]) is None
        assert citation_rate([]) is None

    def test_measured_zero_is_still_zero(self):
        responses = [ResponseVisibility(mentioned=False, cited=False)]
        assert mention_rate(responses) == 0.0
        assert citation_rate(responses) == 0.0


# ---------- citation position ----------

class TestCitationPosition:
    def test_position_counts_distinct_domains(self):
        cited = ["https://a.com/x", "https://www.a.com/y", "https://b.org", "https://target.in/page"]
        assert citation_position(cited, "target.in") == 3

    def test_first_position(self):
        assert citation_position(["https://www.target.in", "https://a.com"], "https://target.in/") == 1

    def test_subdomain_matches_target(self):
        assert citation_position(["https://a.com", "https://blog.target.in/p"], "target.in") == 2

    def test_not_cited(self):
        assert citation_position(["https://a.com"], "target.in") is None
        assert citation_position([], "target.in") is None

    def test_mean_and_median_skip_uncited(self):
        positions = [1, None, 3, 2, None, 6]
        assert mean_citation_position(positions) == 3.0
        assert median_citation_position(positions) == 2.5

    def test_median_odd_count(self):
        assert median_citation_position([4, 1, 2]) == 2.0

    def test_never_cited(self):
        assert mean_citation_position([None, None]) is None
        assert median_citation_position([]) is None

    def test_rejects_zero_based_positions(self):
        with pytest.raises(ValueError):
            mean_citation_position([0, 1])


# ---------- source sets: union, stability, diversity, Jaccard ----------

RUNS = [
    {"a.com", "b.com", "c.com"},
    {"a.com", "b.com"},
    {"a.com", "d.com"},
]


class TestSourceSets:
    def test_union_keeps_every_source_ordered_by_frequency(self):
        runs = [["b.com", "a.com"], ["a.com"], ["c.com", "a.com", "b.com"]]
        assert union_sources(runs) == ["a.com", "b.com", "c.com"]

    def test_union_counts_a_source_once_per_run(self):
        assert union_sources([["a.com", "a.com"], ["b.com"], ["b.com"]]) == ["b.com", "a.com"]

    def test_stability(self):
        assert source_stability(RUNS) == pytest.approx(
            {"a.com": 1.0, "b.com": 2 / 3, "c.com": 1 / 3, "d.com": 1 / 3}
        )

    def test_stability_counts_empty_runs(self):
        assert source_stability([{"a.com"}, set()]) == {"a.com": 0.5}

    def test_diversity(self):
        assert source_set_diversity(RUNS) == 4
        assert source_set_diversity([]) == 0

    def test_jaccard(self):
        assert jaccard_similarity({"a", "b", "c"}, {"a", "b"}) == pytest.approx(2 / 3)
        assert jaccard_similarity({"a"}, {"b"}) == 0.0
        assert jaccard_similarity({"a", "b"}, {"b", "a"}) == 1.0

    def test_jaccard_both_empty_is_identical(self):
        assert jaccard_similarity(set(), set()) == 1.0

    def test_mean_pairwise_jaccard(self):
        # pairs: (r1,r2)=2/3, (r1,r3)=1/4, (r2,r3)=1/3
        assert mean_pairwise_jaccard(RUNS) == pytest.approx((2 / 3 + 1 / 4 + 1 / 3) / 3)

    def test_mean_pairwise_jaccard_needs_two_runs(self):
        assert mean_pairwise_jaccard([{"a.com"}]) is None
        assert mean_pairwise_jaccard([]) is None


# ---------- category breakdown ----------

class TestCategoryBreakdown:
    def test_percentages(self):
        breakdown = category_breakdown(["media", "media", "government", "direct_competitor"])
        assert breakdown == {"media": 50.0, "government": 25.0, "direct_competitor": 25.0}

    def test_sums_to_100(self):
        breakdown = category_breakdown(["media", "reference", "community"])
        assert sum(breakdown.values()) == pytest.approx(100.0)
        assert breakdown["media"] == pytest.approx(100 / 3)

    def test_taxonomy_includes_unused_categories(self):
        breakdown = category_breakdown(["media"], taxonomy=DEFAULT_SOURCE_CATEGORIES)
        assert set(breakdown) == set(DEFAULT_SOURCE_CATEGORIES)
        assert breakdown["media"] == 100.0
        assert breakdown["academic"] == 0.0

    def test_unknown_category_is_kept(self):
        breakdown = category_breakdown(["media", "podcast"], taxonomy=["media"])
        assert breakdown == {"media": 50.0, "podcast": 50.0}

    def test_no_citations(self):
        assert category_breakdown([]) == {}
        assert all(v == 0.0 for v in category_breakdown([], taxonomy=["media", "other"]).values())


# ---------- taxonomy and competitor groups ----------

class TestTaxonomy:
    def test_default_when_unset(self):
        assert load_source_categories("") == DEFAULT_SOURCE_CATEGORIES

    def test_env_list_keeps_required_categories(self):
        categories = load_source_categories(" Media, government ,media")
        assert categories == ("media", "government", "target_company", "direct_competitor", "other")

    def test_competitor_groups(self):
        assert competitor_group("target_company") == GROUP_TARGET
        assert competitor_group("direct_competitor") == GROUP_DIRECT_BUSINESS
        for category in ("government", "reference", "media", "community", "indirect_competitor", "other"):
            assert competitor_group(category) == GROUP_AI_VISIBILITY


# ---------- domains and heuristics ----------

class TestDomains:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("https://www.Example.com/a?b=1", "example.com"),
            ("example.com/path", "example.com"),
            ("http://sub.example.co.in:8080", "sub.example.co.in"),
            ("", ""),
        ],
    )
    def test_normalize_domain(self, value, expected):
        assert normalize_domain(value) == expected

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.acme.in/pricing", "target_company"),
            ("https://rival.com/x", "direct_competitor"),
            ("https://www.rbi.gov.in/x", "government"),
            ("https://data.gov", "government"),
            ("https://portal.nic.in", "government"),
            ("https://mit.edu/x", "academic"),
            ("https://www.iitb.ac.in", "academic"),
            ("https://en.wikipedia.org/wiki/X", "reference"),
            ("https://www.reddit.com/r/x", "community"),
            ("https://economictimes.indiatimes.com/x", "media"),
            ("https://some-blog.com/post", None),
        ],
    )
    def test_heuristic_category(self, url, expected):
        assert heuristic_source_category(url, target_domain="acme.in", direct_competitor_domains=["rival.com"]) == expected

    def test_heuristic_does_not_match_lookalike_domain(self):
        # "notreddit.com" must not be treated as reddit.com
        assert heuristic_source_category("https://notreddit.com") is None


# ---------- before/after comparison ----------

class TestCompareRunMetrics:
    def test_improvements_and_deltas(self):
        before = {"geo_score": 40.0, "aeo_score": 35.5, "mention_rate": 20.0, "mean_citation_position": None}
        after = {"geo_score": 52.5, "aeo_score": 30.0, "mention_rate": 45.0, "mean_citation_position": 2.0}
        result = compare_run_metrics(before, after)
        assert result["geo_improvement"] == 12.5
        assert result["aeo_improvement"] == -5.5
        assert result["deltas"]["mention_rate"] == {"before": 20.0, "after": 45.0, "delta": 25.0}
        # Not cited before, so there's no position delta
        assert result["deltas"]["mean_citation_position"]["delta"] is None

    def test_metric_only_in_one_run(self):
        result = compare_run_metrics({"geo_score": 10.0}, {"geo_score": 10.0, "new_metric": 1.0})
        assert result["geo_improvement"] == 0.0
        assert result["deltas"]["new_metric"] == {"before": None, "after": 1.0, "delta": None}
        assert result["aeo_improvement"] is None
