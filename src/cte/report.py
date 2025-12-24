# src/cte/report.py
from pathlib import Path
import json, datetime

def save_report(result: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = outdir / f"cte_verdict_{ts}.json"
    md_path   = outdir / f"cte_verdict_{ts}.md"

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    rows = result.get("requirements", [])
    md = []
    md.append(f"# CTE Character-Fit Verdict ({result.get('candidate','Candidate')})")
    md.append(f"**Verdict:** {result['verdict']}  |  **Overall:** {result['overall']}  |  **Match score:** {result['match_score']}")
    md.append(f"**Risk:** {result['risk_band']}\n")
    if rows:
        md.append("## Required traits & match")
        md.append("| trait | level | score | met |")
        md.append("|---|---|---:|:--:|")
        for r in rows:
            md.append(f"| {r['trait']} | {r['required_level']} | {r['candidate_score']} | {'✅' if r['met'] else '❌'} |")
    if result.get("unmet_high_criticals"):
        md.append("\n**Unmet high-critical traits:** " + ", ".join(result["unmet_high_criticals"]))
    with open(md_path, "w") as f:
        f.write("\n".join(md))

    return json_path, md_path
