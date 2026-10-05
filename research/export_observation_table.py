"""
Export a stored Stage 1 observation as a per-query table (CSV + Markdown).

Reads only stored values from the backend API (GET /api/observation/<id>?raw=true).
Nothing is estimated: a value the observation did not store is written as "n/a".

Usage:
    python research/export_observation_table.py <observation_id> [--api http://localhost:4000]
"""
import argparse
import csv
import json
import os
import sys
import urllib.request
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "citelabs", "cl-workers"))
from app.geo_metrics import canonical_url, normalize_domain  # noqa: E402  same rules as the pipeline

NA = "n/a"


def fmt(value, digits=None):
    if value is None:
        return NA
    if digits is not None and isinstance(value, float):
        return f"{value:.{digits}f}"
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("observation_id")
    parser.add_argument("--api", default="http://localhost:4000")
    args = parser.parse_args()

    with urllib.request.urlopen(f"{args.api}/api/observation/{args.observation_id}?raw=true") as response:
        obs = json.load(response)
    if obs.get("status") != "completed":
        sys.exit(f"Observation status is {obs.get('status')!r}, not completed")

    analyzed_url = obs["sandboxUrl"]
    analyzed_key = canonical_url(analyzed_url)
    target_domain = normalize_domain(analyzed_url)
    # Stored per-domain category labels (observation_runs.source_categories)
    domain_category = {d: c["category"] for d, c in (obs.get("sourceCategories") or {}).items()}
    taxonomy = (obs.get("config") or {}).get("source_categories") or sorted(set(domain_category.values()))
    # Real AI Overview data (serpapi_aio) has activation, parse errors, credits and inline links
    is_aio = obs.get("fetcher") == "serpapi_aio"
    evidence_skipped = obs.get("evidenceSkipped")  # None for observations made before the option existed

    rows = []
    for q in obs["queries"]:
        responses = q["responses"]
        # Runs whose answer was parsed (for AI Overviews: runs that showed one)
        ok_runs = [r for r in responses if r["ok"] and r.get("outcome") in (None, "answer")]
        web_citations = [
            (r["runIndex"], c) for r in ok_runs for c in r["citations"] if c["isWebSource"] and c["domain"]
        ]

        # Category counts: distinct sources in the query's pool, and citations across runs
        pool_counts = Counter(s["category"] for s in q["sources"])
        citation_counts = Counter(domain_category.get(c["domain"], "uncategorized") for _, c in web_citations)

        # Every cited URL on the target's domain, with how many runs cited it
        target_urls = {}
        for run_index, c in web_citations:
            domain = c["domain"]
            if domain == target_domain or domain.endswith("." + target_domain):
                url = c["url"] or c["rawUri"]
                target_urls.setdefault(url, set()).add(run_index)

        row = {
            "query_index": q["queryIndex"] + 1,
            "query": q["query"],
            "runs_succeeded": q["successfulRuns"],
            "runs_attempted": len(responses),
            "rates_based_on_runs": q["ratesBasedOnRuns"] if q.get("ratesBasedOnRuns") is not None else q["successfulRuns"],
            "unique_sources": q["sourceSetDiversity"],
            "unique_domains": q["domainDiversity"],
            "mean_pairwise_jaccard_urls": q["sourceSetStability"],
            "mean_pairwise_jaccard_domains": q["domainSetStability"],
            "mention_pct": q["mentionRate"],
            # Mentions excluding absence-of-information statements (not stored before 2026-10-06: n/a)
            "mention_excl_absence_pct": q.get("mentionRateExclAbsence"),
            "cited_pct": q["citationRate"],
            "mean_citation_position": q["meanCitationPosition"],
            "median_citation_position": q["medianCitationPosition"],
            "web_citations_total": len(web_citations),
        }
        if is_aio:
            answered = [r for r in responses if r.get("outcome") == "answer"]
            row.update({
                "aio_runs": q.get("aioRuns"),
                "aio_activation_pct": q.get("aioActivationRate"),
                "overall_cited_pct": q.get("overallCitationRate"),
                "no_aio_runs": sum(1 for r in responses if r.get("outcome") == "no_aio"),
                "parse_errors": q.get("parseErrors"),
                "failed_runs": sum(1 for r in responses if not r["ok"]),
                "credits_used": q.get("creditsUsed"),
                # Secondary metric: inline links in the answer text, never counted as citations
                "target_inline_linked_runs": q.get("inlineLinkedRuns"),
                "target_inline_linked_pct": q.get("inlineLinkRate"),
                "responses_with_tables": sum(1 for r in answered if (r.get("tablesSeen") or 0) > 0),
                "responses_with_ragged_table": sum(1 for r in answered if r.get("raggedTable")),
            })
        for category in taxonomy:
            row[f"pool_sources_{category}"] = pool_counts.get(category, 0)
        for category in taxonomy:
            row[f"citations_{category}"] = citation_counts.get(category, 0)
        if citation_counts.get("uncategorized"):
            row["citations_uncategorized"] = citation_counts["uncategorized"]
        row["target_domain_urls_cited"] = " | ".join(
            f"{url} (runs: {len(runs)}/{q['successfulRuns']}; analyzed page: "
            f"{'YES' if canonical_url(url) == analyzed_key else 'no'})"
            for url, runs in sorted(target_urls.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ) or "none"
        row["analyzed_page_cited"] = "YES" if any(canonical_url(u) == analyzed_key for u in target_urls) else "no"

        # Both mention fields per run, and every absence-of-information sentence (for manual checking)
        stored_excl = [r.get("targetMentionedExclAbsence") for r in ok_runs]
        row["runs_mentioned"] = sum(1 for r in ok_runs if r.get("targetMentioned"))
        row["runs_mentioned_excl_absence"] = None if any(v is None for v in stored_excl) else sum(stored_excl)
        absence = [
            (r["runIndex"], sentence)
            for r in ok_runs
            for sentence in (r.get("absenceSentences") or [])
        ]
        row["absence_sentences"] = (
            NA if row["runs_mentioned_excl_absence"] is None
            else " | ".join(f"run {i}: {sentence}" for i, sentence in absence) or "none"
        )
        row["evidence_skipped"] = NA if evidence_skipped is None else ("yes" if evidence_skipped else "no")
        row["_absence"] = absence  # for the Markdown table; not a CSV column
        rows.append((row, target_urls, pool_counts, citation_counts))

    out_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(out_dir, f"{args.observation_id}_per_query")

    # CSV: full stored precision
    fieldnames = []
    for row, *_ in rows:
        for key in row:
            if key not in fieldnames and not key.startswith("_"):
                fieldnames.append(key)
    with open(base + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0)
        writer.writeheader()
        for row, *_ in rows:
            writer.writerow({k: (NA if v is None else v) for k, v in row.items() if not k.startswith("_")})

    # Markdown: same values, rounded for reading
    lines: list = [
        f"# Per-query results: observation `{args.observation_id}`",
        "",
        f"- **Analyzed page:** {analyzed_url}",
        f"- **Observed:** {obs['createdAt']}",
        f"- **Fetcher:** `{obs.get('fetcher')}`",
        f"- **Data source:** {obs.get('dataDisclaimer') or NA}",
        f"- **Evidence extraction:** {'skipped (observation-only run)' if evidence_skipped else 'run' if evidence_skipped is False else 'n/a (not recorded)'}",
        f"- **Runs:** {obs.get('runsSucceeded')}/{obs.get('runsAttempted')} succeeded "
        f"({(obs.get('config') or {}).get('runs_per_query', NA)} per query)",
    ]
    if is_aio:
        settings = obs.get("fetcherSettings") or {}
        lines += [
            f"- **SerpApi settings:** gl={settings.get('gl')}, hl={settings.get('hl')}, device={settings.get('device')}, no_cache={settings.get('no_cache')}",
            f"- **AI Overview runs:** {fmt(obs.get('aioRuns'))} (activation {fmt(obs.get('aioActivationRate'), 1)}%) · "
            f"**parse errors:** {fmt(obs.get('parseErrors'))} · **SerpApi credits:** {fmt(obs.get('creditsUsed'))}",
        ]
    lines += [
        "",
        "All values are stored values from the observation; nothing is estimated. `n/a` means the value "
        "was not stored or cannot be computed (e.g. Jaccard needs at least 2 successful runs). Rates and "
        "positions are over the successful runs shown in *Runs*. Citation counts include web sources only "
        "(search-engine utility results are excluded), and each citation's category is the stored label of "
        "its domain. Full precision is in the accompanying CSV.",
        "",
        *(
            [
                "For AI Overview data, mention %, cited %, positions and Jaccard are over runs that showed an AI Overview "
                "(*Rates over* below). *Overall cited %* is cited runs over all successful runs, including runs with no AI "
                "Overview. Mentions are a case-insensitive whole-word match of the brand name or domain in the answer text, "
                "including table cells; a brand that appears only in a table cell SerpApi dropped is not counted, so the "
                "mention rate can undercount (see *Responses with tables (ragged)*). *Inline-linked* counts answers whose text links to the "
                "target (snippet_links); these are not citations.",
                "",
            ]
            if is_aio
            else []
        ),
        "## Summary",
        "",
        "| # | Query | Runs (ok/attempted) | Unique sources (domains) | Mean Jaccard, URLs / domains | Mention % | Mention % excl. absence | Cited % | Citation position, mean / median | Analyzed page cited? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row, *_ in rows:
        lines.append(
            f"| {row['query_index']} | {row['query']} | {row['runs_succeeded']}/{row['runs_attempted']} | "
            f"{fmt(row['unique_sources'])} ({fmt(row['unique_domains'])}) | "
            f"{fmt(row['mean_pairwise_jaccard_urls'], 3)} / {fmt(row['mean_pairwise_jaccard_domains'], 3)} | "
            f"{fmt(row['mention_pct'], 1)} | {fmt(row['mention_excl_absence_pct'], 1)} | {fmt(row['cited_pct'], 1)} | "
            f"{fmt(row['mean_citation_position'], 2)} / {fmt(row['median_citation_position'], 2)} | "
            f"{row['analyzed_page_cited']} |"
        )

    if is_aio:
        lines += [
            "",
            "## AI Overview activation, parse errors and secondary metrics",
            "",
            "| # | Runs: AIO / no AIO / parse error / failed | Activation % | Rates over (runs) | Overall cited % | Inline-linked runs (%) | Responses with tables (ragged) | Credits |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row, *_ in rows:
            lines.append(
                f"| {row['query_index']} | {fmt(row['aio_runs'])} / {row['no_aio_runs']} / {fmt(row['parse_errors'])} / {row['failed_runs']} | "
                f"{fmt(row['aio_activation_pct'], 1)} | {row['rates_based_on_runs']} | {fmt(row['overall_cited_pct'], 1)} | "
                f"{fmt(row['target_inline_linked_runs'])} ({fmt(row['target_inline_linked_pct'], 1)}) | "
                f"{row['responses_with_tables']} ({row['responses_with_ragged_table']}) | {fmt(row['credits_used'])} |"
            )
    def absence_cell(row):
        if row["runs_mentioned_excl_absence"] is None:
            return NA
        return "<br>".join(f"run {i}: {sentence}".replace("|", "\|") for i, sentence in row["_absence"]) or "none"

    lines += [
        "",
        "## Mentions and absence-of-information sentences",
        "",
        "*Mention %* is the string match of the brand name or domain anywhere in the answer. *Mention % excl. absence* counts "
        "a run only if at least one sentence containing a brand term is not an absence-of-information statement (e.g. \"no "
        "mention of\", \"does not contain information about\", \"cannot find\"); ordinary negation such as \"does not "
        "charge setup fees\" is still a mention. Both are over the same runs. Below are every brand-term sentence classified "
        "as an absence statement, for manual checking. `n/a`: not stored for this observation (the field was added on 2026-10-06).",
        "",
        "| # | Runs mentioned | Runs mentioned excl. absence | Absence-of-information sentences |",
        "|---|---|---|---|",
    ]
    for row, *_ in rows:
        lines.append(
            f"| {row['query_index']} | {row['runs_mentioned']}/{row['rates_based_on_runs']} | "
            f"{fmt(row['runs_mentioned_excl_absence'])}{'' if row['runs_mentioned_excl_absence'] is None else '/' + str(row['rates_based_on_runs'])} | "
            f"{absence_cell(row)} |"
        )
    lines += ["", "## Category counts", ""]
    lines.append("*Citations across runs* counts every web citation in every successful run (a source cited in 3 runs counts 3 times); "
                 "use it for share-of-citations claims. *Distinct sources in pool* counts each source in the query's union pool once.")
    lines.append("")
    header = "| Category | " + " | ".join(
        f"Q{row['query_index']} citations across runs | Q{row['query_index']} distinct sources in pool" for row, *_ in rows
    ) + " |"
    lines += [header, "|---|" + "---|" * (2 * len(rows))]
    categories = list(taxonomy) + (["uncategorized"] if any(cc.get("uncategorized") for *_, cc in rows) else [])
    for category in categories:
        cells = " | ".join(f"{cc.get(category, 0)} | {pc.get(category, 0)}" for _, _, pc, cc in rows)
        lines.append(f"| {category} | {cells} |")
    totals = " | ".join(f"{row['web_citations_total']} | {sum(pc.values())}" for row, _, pc, _ in rows)
    lines.append(f"| **total** | {totals} |")

    lines += ["", f"## {target_domain} URLs cited", "", f"Analyzed page: `{analyzed_url}` (compared after URL canonicalization: scheme, `www.`, trailing slash, fragment and tracking parameters ignored).", ""]
    for row, target_urls, *_ in rows:
        lines.append(f"**Q{row['query_index']}** ({row['runs_succeeded']} successful runs)")
        lines.append("")
        if not target_urls:
            lines.append("- none")
        for url, runs in sorted(target_urls.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            flag = "**analyzed page**" if canonical_url(url) == analyzed_key else "other page"
            lines.append(f"- {url} — cited in {len(runs)}/{row['runs_succeeded']} runs — {flag}")
        lines.append("")

    with open(base + ".md", "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {base}.csv and {base}.md")


if __name__ == "__main__":
    main()
