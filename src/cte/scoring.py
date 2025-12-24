# src/cte/scoring.py

from typing import Dict, List, Tuple

_DEFAULT_THRESHOLDS = {"low": 0.50, "medium": 0.60, "high": 0.70}
_DEFAULT_WEIGHTS    = {"low": 1.0,  "medium": 1.2,  "high": 1.5}

def _norm_level(s: str) -> str:
    return (s or "").strip().lower()

def score_requirements(
    per_trait: Dict[str, Dict],                # persona["per_trait"]
    requirements: List[Dict],                  # [{"trait","required_level"}, ...]
    thresholds: Dict[str, float] = None,       # {"low":..,"medium":..,"high":..}
    weights: Dict[str, float] = None,          # {"low":..,"medium":..,"high":..}
    unmet_high_penalty: float = 0.10,          # 0..1 (gentle)
) -> Tuple[str, float, str, List[Dict], List[str]]:
    """
    Returns:
      overall: "Strong fit" | "Possible fit" | "Leaning no" | "Not a fit"
      match_ratio: float in [0,1]
      risk_band: "low-risk" | "moderate-risk" | "elevated-risk" | "high-risk"
      rows: list of per-requirement dicts used by the CLI
      criticals: list of unmet 'high' traits
    """
    thr = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    wts = {**_DEFAULT_WEIGHTS,    **(weights or {})}

    rows: List[Dict] = []
    total_w = 0.0
    met_w   = 0.0
    unmet_high: List[str] = []

    # Compare each JD requirement with persona scores
    for req in requirements or []:
        trait = (req.get("trait") or "").strip()
        lvl   = _norm_level(req.get("required_level") or "low")
        if not trait or lvl not in thr:
            continue

        # Persona score (0 if trait missing)
        score = float(per_trait.get(trait, {}).get("score", 0.0))
        tval  = float(thr[lvl])
        w     = float(wts.get(lvl, 1.0))
        met   = score >= tval

        rows.append({
            "trait": trait,
            "required_level": lvl,
            "candidate_score": round(score, 2),
            "threshold": tval,
            "met": met,
        })

        total_w += w
        if met:
            met_w += w
        elif lvl == "high":
            unmet_high.append(trait)

    if total_w == 0:
        # fallback to avoid div-by-zero; means JD parse produced nothing useful
        total_w = 1.0

    match_ratio = met_w / total_w

    # Gentle penalty for any unmet high-critical traits
    if unmet_high and unmet_high_penalty > 0:
        match_ratio = max(0.0, match_ratio - unmet_high_penalty)

    # Clamp
    match_ratio = max(0.0, min(1.0, match_ratio))

    # Map to overall/risk
    if match_ratio >= 0.75:
        overall, risk = "Strong fit", "low-risk"
    elif match_ratio >= 0.50:
        overall, risk = "Possible fit", "moderate-risk"
    elif match_ratio >= 0.35:
        overall, risk = "Leaning no", "elevated-risk"
    else:
        overall, risk = "Not a fit", "high-risk"

    # Sort rows for nicer display: high→medium→low, met first, higher scores first
    level_rank = {"high": 2, "medium": 1, "low": 0}
    rows.sort(key=lambda r: (level_rank.get(r["required_level"], 0), r["met"], r["candidate_score"]), reverse=True)

    return overall, match_ratio, risk, rows, unmet_high
