"""
Rebuild the parsed fields of SerpApi AI Overview observations from their stored
raw responses, without calling SerpApi (no credits).

Uses the same parser as the live fetcher (parse_raw_responses in
citelabs/cl-workers/app/observation/serpapi_aio.py), so re-parsing after a parser
change shows exactly what would change in the stored data.

Usage:
    # every response of a stored observation (compares with what was stored)
    python research/reparse_serpapi_observation.py --observation <id> [--api http://localhost:4000] [--out rebuilt.json]

    # raw response files: one observation = its engine=google file, then its
    # engine=google_ai_overview file (when a page token was followed)
    python research/reparse_serpapi_observation.py --target https://example.com/page --files google.json [aio.json]
"""
import argparse
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "citelabs", "cl-workers"))
from app.geo_metrics import normalize_domain  # noqa: E402
from app.observation.serpapi_aio import reparse_stored_response  # noqa: E402


def target_inline_linked(inline_links, target_domain):
    return any(
        (d := normalize_domain(link.get("link", ""))) == target_domain or d.endswith("." + target_domain)
        for link in inline_links or []
    )


def summarize(rebuilt, target_domain):
    details = rebuilt["details"] or {}
    domains = [normalize_domain(c["url"]) for c in rebuilt["citations"]]
    return {
        "outcome": rebuilt["outcome"],
        "parse_error": rebuilt["parse_error"],
        "references": [c["url"] for c in rebuilt["citations"]],
        "target_cited": any(d == target_domain or d.endswith("." + target_domain) for d in domains),
        "tables_seen": details.get("tables_seen"),
        "ragged_table": details.get("ragged_table"),
        "ragged_rows": details.get("ragged_rows"),
        "inline_links": details.get("inline_links"),
        "target_inline_linked": target_inline_linked(details.get("inline_links"), target_domain) if rebuilt["outcome"] == "answer" else None,
        "answer_text_chars": len(rebuilt["answer_text"]),
    }


def from_observation(observation_id, api):
    with urllib.request.urlopen(f"{api}/api/observation/{observation_id}?raw=true") as response:
        obs = json.load(response)
    target = normalize_domain(obs["sandboxUrl"])
    results, mismatches = [], 0
    for q in obs["queries"]:
        for r in q["responses"]:
            raws = r.get("rawResponses")
            if not raws:
                results.append({"query": q["query"], "run_index": r["runIndex"], "skipped": "no stored raw responses"})
                continue
            rebuilt = summarize(reparse_stored_response(raws), target)
            stored = {
                "outcome": r.get("outcome"),
                "references": [c["url"] for c in sorted(r.get("citations", []), key=lambda c: c["position"])],
                "tables_seen": r.get("tablesSeen"),
                "ragged_table": r.get("raggedTable"),
                "target_inline_linked": r.get("targetInlineLinked"),
            }
            diff = {k: {"stored": v, "rebuilt": rebuilt[k]} for k, v in stored.items() if v != rebuilt[k]}
            mismatches += bool(diff)
            results.append({"query": q["query"], "run_index": r["runIndex"], "rebuilt": rebuilt, "differs_from_stored": diff})
    return {"observation_id": observation_id, "target": obs["sandboxUrl"], "responses": results, "responses_differing": mismatches}


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--observation")
    source.add_argument("--files", nargs="+")
    parser.add_argument("--target", help="analyzed page URL (with --files)")
    parser.add_argument("--api", default="http://localhost:4000")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.observation:
        report = from_observation(args.observation, args.api)
    else:
        if not args.target:
            parser.error("--target is required with --files")
        raws = [json.load(open(path)) for path in args.files]
        report = {"files": args.files, "rebuilt": summarize(reparse_stored_response(raws), normalize_domain(args.target))}

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
