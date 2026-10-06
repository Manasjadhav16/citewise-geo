"""Combined table for the GitHub real-AIO pilot (step 1, batch A, batch B).

One row per (observation, Table II question). Every figure is read from the
stored observation (GET /api/observation/:id?raw=true); nothing is estimated.

Usage: python build_github_pilot_table.py OUT_PREFIX OBS_JSON [OBS_JSON ...]
"""
import csv
import json
import sys
from collections import Counter

TARGET_HOST = "github.com"  # target = github.com and its subdomains
# Other GitHub-owned domains, listed separately and never counted as the target.
OTHER_GITHUB_OWNED = ("github.blog", "githubusercontent.com", "githubassets.com", "github.io",
                      "githubstatus.com", "githubapp.com", "github.dev", "ghcr.io", "npmjs.com", "npmjs.org")
RETRIEVED = ("answer", "answer_no_references")
# Account balances logged by the run scripts (SerpApi account endpoint, before/after each observation).
BALANCES = {
    "obs_github_pilot_1791297109": (207, 189),
    "obs_github_pilot_A_1791298362": (189, 147),
    "obs_github_pilot_B_1791300066": (147, 127),
}
OUTCOMES = ("answer", "answer_no_references", "no_aio", "followup_empty", "timeout", "parse_error", "error")


def host_matches(domain, base):
    domain = (domain or "").lower()
    return domain == base or domain.endswith("." + base)


def run_notes_for(responses):
    """Explain stored outcomes that the current fetcher would classify differently."""
    notes = []
    for r in responses:
        err, run = r.get("error") or "", f"run {r['runIndex']}"
        if r["outcome"] == "error" and "hasn't returned any results" in err:
            notes.append(f"{run}: stored as error by the pre-fix fetcher; an empty AI Overview follow-up "
                         "(now followup_empty); its follow-up credit was billed but not counted")
        elif r["outcome"] == "error" and "ReadTimeout" in err:
            notes.append(f"{run}: client-side timeout stored as error with 0 credits by the pre-fix fetcher "
                         "(now timeout, 1 credit assumed)")
        elif r["outcome"] == "error" and "HTTP 503" in err:
            notes.append(f"{run}: SerpApi HTTP 503 \"couldn't get valid results\"; not billed, nothing stored")
        elif r["outcome"] == "parse_error" and "no references list" in (r.get("parseError") or ""):
            notes.append(f"{run}: stored as parse_error by the pre-fix fetcher; an overview without a "
                         "references key (now answer_no_references); not in retrieved_overviews")
    return "; ".join(notes)


def rows_for(obs):
    notes = (obs.get("config") or {}).get("run_notes") or {}
    to_table = {int(k): v for k, v in (notes.get("query_index_to_table_ii") or {}).items()}
    excluded = notes.get("excluded_from_mention_analysis") or {}
    brand = (obs.get("context") or {}).get("brand_name")
    for q in sorted(obs["queries"], key=lambda q: q["queryIndex"]):
        responses = sorted(q["responses"], key=lambda r: r["runIndex"])
        if not responses:
            continue
        table_no = to_table[q["queryIndex"]]
        outcomes = Counter(r["outcome"] or "error" for r in responses)
        retrieved = [r for r in responses if r["outcome"] in RETRIEVED]
        target_urls, other_owned = Counter(), Counter()
        for r in responses:
            seen_target, seen_other = set(), set()
            for c in r["citations"]:
                if host_matches(c["domain"], TARGET_HOST):
                    seen_target.add(c["url"])
                elif any(host_matches(c["domain"], d) for d in OTHER_GITHUB_OWNED):
                    seen_other.add(c["url"])
            target_urls.update(seen_target)
            other_owned.update(seen_other)
        yield {
            "observation_id": obs["id"],
            "table_ii_q": table_no,
            "question": q["query"],
            "reconstructed": bool(notes.get("questions_reconstructed")),
            "excluded_from_mention_analysis": str(table_no) in excluded,
            "runs_attempted": len(responses),
            **{f"n_{o}": outcomes.get(o, 0) for o in OUTCOMES},
            "runs_with_references": sum(1 for r in responses if r["outcome"] == "answer"),
            "runs_github_com_cited": sum(1 for r in responses if r["targetCited"]),
            "github_com_urls_cited": "; ".join(f"{u} ({n} run{'s' if n > 1 else ''})" for u, n in sorted(target_urls.items())),
            "other_github_owned_cited": "; ".join(f"{u} ({n} run{'s' if n > 1 else ''})" for u, n in sorted(other_owned.items())),
            "retrieved_overviews": len(retrieved),
            "mentioned_string": sum(1 for r in retrieved if r["targetMentioned"]),
            "mentioned_excl_absence": sum(1 for r in retrieved if r["targetMentionedExclAbsence"]),
            "absence_sentences": " | ".join(s for r in retrieved for s in (r.get("absenceSentences") or [])),
            "credits_counted": sum(r["creditsUsed"] or 0 for r in responses),
            # blank = column did not exist when the observation was stored
            "credits_assumed": (sum(r["creditsAssumed"] or 0 for r in responses)
                                if any(r.get("creditsAssumed") is not None for r in responses) else ""),
            "stored_brand": brand,
            "notes": run_notes_for(responses),
        }


def cell(v):
    return str(v).replace("|", "\\|") if v not in ("", None) else "–"


def write_markdown(path, rows, observations):
    lines = [
        "# GitHub real-AIO pilot: combined per-question table",
        "",
        "Google AI Overviews fetched through SerpApi (`serpapi_aio`, gl=in, hl=en, desktop, no_cache=true, "
        "skip_evidence). Target = https://github.com/ (github.com and its subdomains). "
        "Brand terms: \"GitHub\", \"github.com\".",
        "",
        "- **Questions are reconstructed** from the abridged wording in the paper's Table II; "
        "they are not the original question strings.",
        "- One row per question per observation. Rows are never merged across observations "
        "(Q11 appears twice).",
        "- Q8 (question text names GitHub) is excluded from mention analysis and was not run.",
        "- Every figure is read from the stored observation; nothing is estimated. Mention counts "
        "are over retrieved overviews (`answer` + `answer_no_references`).",
        "- Other GitHub-owned domains checked: " + ", ".join(OTHER_GITHUB_OWNED) + ". None were cited.",
        "- Step 1 and batch A were stored by an earlier fetcher version; the Notes column explains "
        "outcomes it classified differently. Stored values are shown as they are.",
        "",
        "## Credits per observation",
        "",
        "| Observation | Credits counted (stored) | Balance before → after | Balance change | Difference |",
        "|---|---|---|---|---|",
    ]
    for o in observations:
        before, after = BALANCES[o["id"]]
        lines.append(f"| {o['id']} | {o['creditsUsed']} | {before} → {after} | {before - after} | "
                     f"{before - after - o['creditsUsed']} |")
    lines += [
        "",
        "## Per question",
        "",
        "| Observation | Q | Question | Runs | answer | ans. no refs | no_aio | followup_empty | timeout "
        "| parse_error | error | Runs with refs | Runs citing github.com | github.com URLs cited (runs) "
        "| Other GitHub-owned | Mentioned (string) / retrieved | Mentioned (excl. absence) / retrieved "
        "| Credits counted | Credits assumed | Notes |",
        "|" + "---|" * 20,
    ]
    for r in rows:
        q = f"Q{r['table_ii_q']}" + (" (excluded from mention analysis)" if r["excluded_from_mention_analysis"] else "")
        lines.append("| " + " | ".join(cell(v) for v in [
            r["observation_id"], q, r["question"] + (" *(reconstructed)*" if r["reconstructed"] else ""),
            r["runs_attempted"], *[r[f"n_{o}"] for o in OUTCOMES], r["runs_with_references"],
            r["runs_github_com_cited"], r["github_com_urls_cited"], r["other_github_owned_cited"],
            f"{r['mentioned_string']}/{r['retrieved_overviews']}",
            f"{r['mentioned_excl_absence']}/{r['retrieved_overviews']}",
            r["credits_counted"], r["credits_assumed"], r["notes"]]) + " |")
    lines += ["", "No absence sentences were recorded in any retrieved overview."
              if not any(r["absence_sentences"] for r in rows) else "Absence sentences are in the CSV."]
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    rows = [row for p in paths for row in rows_for(json.load(open(p)))]
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_markdown(out + ".md", rows, [json.load(open(p)) for p in paths])
    print(json.dumps(rows, indent=1))


main()
