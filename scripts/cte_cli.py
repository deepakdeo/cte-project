#!/usr/bin/env python
import argparse, json, sys
from pathlib import Path

from cte.persona import load_persona
from cte.requirements import (
    extract_requirements_llm,
    extract_requirements_hybrid,
    union_requirements,
)
from cte.scoring import score_requirements


def main():
    ap = argparse.ArgumentParser(description="CTE: character-fit verdict for a JD")
    ap.add_argument("--persona", required=True, help="path to 06_profile_persona_llm.json")
    ap.add_argument("--jd", required=True, help="path to JD text file")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model for JD parsing")
    ap.add_argument("--json-out", default=None, help="write final verdict JSON here")
    ap.add_argument("--positives-only", action="store_true", help="print only met traits")

    # Calibrations (thresholds, weights, penalty)
    ap.add_argument("--low",  type=float, default=0.50, help="threshold for 'low' requirement")
    ap.add_argument("--med",  type=float, default=0.60, help="threshold for 'medium' requirement")
    ap.add_argument("--high", type=float, default=0.70, help="threshold for 'high' requirement")

    ap.add_argument("--wlow",  type=float, default=1.0, help="weight for 'low'")
    ap.add_argument("--wmed",  type=float, default=1.2, help="weight for 'medium'")
    ap.add_argument("--whigh", type=float, default=1.5, help="weight for 'high'")

    ap.add_argument("--unmet-high-penalty", type=float, default=0.10,
                    help="extra penalty if a HIGH requirement is unmet (0..1)")

    args = ap.parse_args()

    # 1) Load persona and JD text
    per_trait, persona_full = load_persona(args.persona)
    jd_text = Path(args.jd).read_text()

    # 2) Extract JD requirements (LLM + heuristic), then union
    req_llm    = extract_requirements_llm(jd_text, model=args.model)
    req_hybrid = extract_requirements_hybrid(jd_text)
    req_union  = union_requirements(req_llm, req_hybrid)

    # 3) Score with caller-provided thresholds/weights/penalty
    thresholds = {"low": args.low, "medium": args.med, "high": args.high}
    weights    = {"low": args.wlow, "medium": args.wmed, "high": args.whigh}

    overall, match_ratio, risk, rows, criticals = score_requirements(
        per_trait,
        req_union,
        thresholds=thresholds,
        weights=weights,
        unmet_high_penalty=args.unmet_high_penalty,
    )

    result = {
        "candidate": "Deo",
        "overall": overall,
        "verdict": "Likely successful" if overall in ("Strong fit", "Possible fit") else "Likely to struggle",
        "match_score": round(match_ratio, 2),
        "risk_band": risk,
        "requirements": rows,
        "unmet_high_criticals": criticals,
    }

    # 4) Print (optionally positives-only)
    if args.positives_only:
        met = [r for r in rows if r.get("met")]
        print(f"CTE Character-Fit Verdict (positives-only): {result['verdict']} | score={result['match_score']}")
        for r in met:
            print(f" - {r['trait']} ({r['required_level']}, score={r['candidate_score']:.2f})")
    else:
        print(json.dumps(result, indent=2))

    # 5) Save JSON (and a small Markdown card next to it)
    if args.json_out:
        outp = Path(args.json_out)
        outp.write_text(json.dumps(result, indent=2))

        # Tiny MD card
        md_lines = []
        md_lines.append(f"# CTE Character-Fit Verdict — {result['candidate']}")
        md_lines.append(f"**Verdict:** {result['verdict']}  |  **Overall:** {result['overall']}  |  "
                        f"**Match score:** {result['match_score']:.2f}  |  **Risk:** {result['risk_band']}\n")
        if rows:
            md_lines.append("## Required traits & match")
            md_lines.append("| trait | level | score | met |")
            md_lines.append("|---|---|---:|:--:|")
            for r in rows:
                md_lines.append(f"| {r['trait']} | {r['required_level']} | {r['candidate_score']:.2f} | "
                                f"{'✅' if r['met'] else '❌'} |")
        if result.get("unmet_high_criticals"):
            md_lines.append("\n**Unmet high-critical traits:** " + ", ".join(result["unmet_high_criticals"]))
        md_path = outp.with_suffix(".md")
        md_path.write_text("\n".join(md_lines))
        print("Markdown ->", md_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
