# scripts/cte_app.py
import json, datetime
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---- Local imports (cte package must be importable; run with PYTHONPATH=src) ----
from cte.persona import load_persona
from cte.requirements import extract_requirements_llm, extract_requirements_hybrid, union_requirements
from cte.scoring import score_requirements
from cte.report import save_report
from cte.nlp import analyze_sentiment  # sentiment analysis util

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CTE — Character Trait Evaluator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CONSTANTS / PATHS
# =========================
UPD_PATH = Path("data/daily_updates.parquet")

SAMPLE_JD = """Senior Data Scientist

We're seeking an experienced Data Scientist to join our AI team. You'll work on cutting-edge machine learning projects.

Responsibilities:
- Design and implement ML models for production
- Collaborate with cross-functional teams
- Communicate findings to stakeholders
- Mentor junior team members

Requirements:
- 5+ years experience in data science
- Strong Python and ML framework skills
- Excellent communication skills
- Proven track record of delivering projects
- Strategic thinking and problem-solving
- Ability to work independently and in teams
"""

# =========================
# SESSION STATE INIT
# =========================
if "persona_path" not in st.session_state:
    st.session_state["persona_path"] = "notebooks/reports/06_profile_persona_llm.json"
if "last_evaluation" not in st.session_state:
    st.session_state["last_evaluation"] = None
if "evaluation_history" not in st.session_state:
    st.session_state["evaluation_history"] = []
if "show_tutorial" not in st.session_state:
    st.session_state["show_tutorial"] = False
if "jd_text" not in st.session_state:
    st.session_state["jd_text"] = ""
if "auto_evidence_from_reflection" not in st.session_state:
    st.session_state["auto_evidence_from_reflection"] = False

# =========================
# CACHING HELPERS
# =========================
@st.cache_data(show_spinner=False)
def read_persona_cached(path: str):
    """Return (per_trait: dict, persona_full: dict|any)."""
    return load_persona(path)

@st.cache_data(show_spinner=False)
def read_updates_cached(upd_path: str):
    p = Path(upd_path)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()

# =========================
# VIS HELPERS
# =========================
def create_radar_chart(per_trait: dict, title="Character Trait Profile"):
    traits, scores, confidences = [], [], []
    for trait_name, trait_info in per_trait.items():
        traits.append(trait_name.replace("_", " ").title())
        scores.append(float(trait_info.get("score", 0)) * 100)
        confidences.append(float(trait_info.get("confidence", 0)) * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=scores, theta=traits, fill='toself',
                                  name='Trait Score', line_color='#1f77b4'))
    fig.add_trace(go.Scatterpolar(r=confidences, theta=traits, fill='toself',
                                  name='Confidence', line_color='#ff7f0e', opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      showlegend=True, title=title, height=500)
    return fig

def create_trait_bar_chart(per_trait: dict):
    data = [{
        'Trait': t.replace("_", " ").title(),
        'Score': float(info.get("score", 0)) * 100,
        'Confidence': float(info.get("confidence", 0)) * 100
    } for t, info in per_trait.items()]
    if not data:
        return go.Figure()
    df = pd.DataFrame(data).sort_values('Score', ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df['Trait'], x=df['Score'], name='Score', orientation='h', marker=dict(color='#1f77b4')))
    fig.add_trace(go.Bar(y=df['Trait'], x=df['Confidence'], name='Confidence', orientation='h', marker=dict(color='#ff7f0e')))
    fig.update_layout(barmode='group', title='Trait Scores vs Confidence',
                      xaxis_title='Percentage', yaxis_title='', height=max(400, len(df) * 30),
                      showlegend=True)
    return fig

def create_daily_trend_chart(updates_df: pd.DataFrame):
    if updates_df is None or updates_df.empty or len(updates_df) < 2:
        return None
    df = updates_df.copy()
    df['date'] = pd.to_datetime(df['ts']).dt.date

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['prod'], mode='lines+markers',
                             name='Productivity %', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['focus_mins'] / 6.0, mode='lines+markers',
                             name='Focus Minutes (scaled)', line=dict(color='#ff7f0e', width=2)))

    # Sentiment trend if available
    if "sentiment_score" in df.columns and df["sentiment_score"].notna().any():
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['sentiment_score'] * 100.0,
            mode='lines+markers',
            name='Sentiment (0–100)',
            line=dict(color='#2ca02c', width=2, dash='dot')
        ))

    fig.update_layout(title='Daily Performance Trends', xaxis_title='Date', yaxis_title='Score/Level',
                      hovermode='x unified', height=400)
    return fig

def create_match_gauge(match_ratio: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=float(match_ratio) * 100.0,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Match Score"},
        delta={'reference': 70, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
        }
    ))
    fig.update_layout(height=300)
    return fig

def export_to_md_report(evaluation_data: dict) -> str:
    md = f"""# Job Fit Evaluation Report

**Candidate**: {evaluation_data.get('candidate', 'Unknown')}
**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
**Overall Verdict**: {evaluation_data.get('overall', 'N/A')}

---

## Summary

- **Match Score**: {evaluation_data.get('match_score', 0):.0%}
- **Risk Level**: {evaluation_data.get('risk_band', 'N/A')}
- **Recommendation**: {evaluation_data.get('verdict', 'N/A')}

---

## Requirements Analysis
"""
    for req in evaluation_data.get('requirements', []):
        status = "✅" if req.get('met') else "❌"
        md += f"- {status} **{req.get('trait','').replace('_',' ').title()}** "
        md += f"(Required: {req.get('required_level','').upper()}, "
        md += f"Score: {req.get('candidate_score',0):.0%}, "
        md += f"Threshold: {req.get('threshold',0):.0%})\n"

    crits = evaluation_data.get('unmet_high_criticals', [])
    if crits:
        md += "\n## ⚠️ Critical Gaps\n\n"
        for c in crits:
            md += f"- {str(c).replace('_',' ').title()}\n"
    return md

def _z01(x, lo, hi):
    if hi == lo: return 0.0
    v = max(lo, min(hi, x))
    return (v - lo) / (hi - lo)

def _save_persona_and_refresh(ppath: Path, persona_obj: dict) -> Path:
    """Write updated persona next to original, refresh caches, return new path."""
    updated_path = ppath.with_name(ppath.stem + "_updated.json")
    updated_path.write_text(json.dumps(persona_obj, indent=2))
    st.session_state["persona_path"] = str(updated_path)
    read_persona_cached.clear()
    return updated_path

def _seed_persona_from_onboarding(answers: dict) -> dict:
    """
    Build a low-confidence persona from onboarding responses.
    Scores are normalized to 0..1 and intentionally conservative.
    """
    def _z(x, lo=1, hi=5):
        return (max(lo, min(hi, x)) - lo) / (hi - lo)

    focus = _z(answers["focus"])
    planning = _z(answers["planning"])
    adaptability = _z(answers["adaptability"])
    communication = _z(answers["communication"])
    learning = _z(answers["learning"])
    resilience = _z(answers["resilience"])

    # Blend to keep scores near center for cold start
    def _blend(v):
        return round(0.4 + 0.6 * v, 3)

    per_trait = {
        "focus": {"score": _blend(focus), "confidence": 0.35, "evidence": []},
        "planning": {"score": _blend(planning), "confidence": 0.35, "evidence": []},
        "adaptability": {"score": _blend(adaptability), "confidence": 0.35, "evidence": []},
        "communication": {"score": _blend(communication), "confidence": 0.35, "evidence": []},
        "learning_mindset": {"score": _blend(learning), "confidence": 0.35, "evidence": []},
        "resilience": {"score": _blend(resilience), "confidence": 0.35, "evidence": []},
        "reliability": {"score": _blend((focus + planning) / 2), "confidence": 0.35, "evidence": []},
        "impact": {"score": _blend((focus + communication) / 2), "confidence": 0.35, "evidence": []},
        "teamwork": {"score": _blend(communication), "confidence": 0.35, "evidence": []},
        "independence": {"score": _blend((focus + resilience) / 2), "confidence": 0.35, "evidence": []},
    }

    return {"name": "Starter Persona", "per_trait": per_trait}

# =========================
# HEADER / TUTORIAL
# =========================
col_logo, col_title, col_actions = st.columns([1, 6, 2])
with col_logo:
    st.markdown("# 🧭")
with col_title:
    st.markdown("# Character Trait Evaluator")
    st.caption("AI-powered job fit analysis based on your ongoing character profile")
with col_actions:
    if st.button("📚 Tutorial", use_container_width=True):
        st.session_state["show_tutorial"] = not st.session_state["show_tutorial"]
st.divider()

if st.session_state["show_tutorial"]:
    with st.expander("📚 Quick Start Tutorial", expanded=True):
        st.markdown("""
**Step 1: Load Your Persona** (Sidebar)
- Point to your `persona.json` (must include a `per_trait` dict of trait scores + confidence)

**Step 2: Build Evidence** (Tab: Daily Update + Add Evidence)
- Daily signals adjust scores & confidence
- Add Evidence records concrete proof items per trait

**Step 3: Evaluate a JD** (Tab: Evaluate Job)
- Paste a JD → extract requirements → compare persona
- See strengths, risks, match gauge, and recommendation

**Step 4: Explore** (Dashboard + View Persona)
- Radar/bar charts; trends (productivity/focus/sentiment)
- Review evaluation history & evidence
        """)

# =========================
# SIDEBAR CONFIG
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")

    # Persona
    with st.expander("📋 Persona Settings", expanded=True):
        if "pending_persona_path" in st.session_state:
            st.session_state["persona_path_input"] = st.session_state["pending_persona_path"]
            st.session_state["persona_path"] = st.session_state["pending_persona_path"]
            del st.session_state["pending_persona_path"]
        persona_path = st.text_input(
            "Persona JSON path",
            value=st.session_state["persona_path"],
            key="persona_path_input",
            help="Path to your character persona JSON file"
        )
        st.session_state["persona_path"] = persona_path
        # Load persona (cached)
        try:
            per_trait, persona_full = read_persona_cached(persona_path)
            persona_ready = True
            name = persona_full.get('name', 'Persona') if isinstance(persona_full, dict) else 'Persona'
            st.success(f"✓ Loaded: {name}")

            n = max(len(per_trait or {}), 1)
            avg_conf = sum(float(t.get("confidence", 0)) for t in (per_trait or {}).values()) / n
            avg_score = sum(float(t.get("score", 0)) for t in (per_trait or {}).values()) / n

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Avg Score", f"{avg_score:.0%}", help="Average capability across all traits")
            with c2:
                st.metric("Confidence", f"{avg_conf:.0%}", help="Data reliability - higher = more evidence")
            if avg_conf < 0.6:
                st.warning("⚠️ Low confidence — log Daily Updates to improve reliability")
            elif avg_conf < 0.75:
                st.info("🟡 Medium confidence — keep logging to improve")
            else:
                st.success("🟢 High confidence — good evidence base")
        except Exception as e:
            persona_ready = False
            per_trait, persona_full = None, None
            st.error("❌ Could not load persona")
            with st.expander("Error details"):
                st.code(str(e))

    # Starter Mode (cold start)
    with st.expander("🚀 Quick Start (No Data Required)", expanded=False):
        st.markdown("""
        **Create a persona in 2 minutes!**

        Answer these questions honestly (1 = Rarely, 5 = Always).
        You'll get a baseline persona you can refine over time.
        """)
        st.divider()
        q1 = st.slider("🎯 I can sustain deep focus when needed", 1, 5, 3, help="Deep work without distractions")
        q2 = st.slider("📋 I plan my work and follow through", 1, 5, 3, help="Setting goals and achieving them")
        q3 = st.slider("🔄 I adapt well to changes and ambiguity", 1, 5, 3, help="Flexibility with shifting priorities")
        q4 = st.slider("💬 I communicate clearly with others", 1, 5, 3, help="Written and verbal clarity")
        q5 = st.slider("📚 I actively learn and improve", 1, 5, 3, help="Seeking growth opportunities")
        q6 = st.slider("💪 I recover quickly from setbacks", 1, 5, 3, help="Bouncing back from challenges")

        st.divider()
        if st.button("✨ Create My Persona", use_container_width=True, type="primary"):
            answers = {
                "focus": q1,
                "planning": q2,
                "adaptability": q3,
                "communication": q4,
                "learning": q5,
                "resilience": q6,
            }
            persona = _seed_persona_from_onboarding(answers)
            outdir = Path("data/personas")
            outdir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            outpath = outdir / f"persona_seeded_{ts}.json"
            outpath.write_text(json.dumps(persona, indent=2))
            st.session_state["pending_persona_path"] = str(outpath)
            read_persona_cached.clear()
            st.success("Persona created! Now try 'Evaluate Job' tab.")
            st.rerun()
        st.caption("💡 Tip: Log daily updates to increase confidence over time.")

    # Analysis
    with st.expander("🤖 Analysis Settings", expanded=False):
        model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4.1-mini"],
            index=0,
            help="Model used to extract requirements from JD"
        )

        st.markdown("**Scoring Thresholds**  (score ≥ threshold ⇒ trait met)")
        c1, c2 = st.columns(2)
        with c1:
            low = st.slider("Low", 0.0, 1.0, 0.50, 0.05)
            med = st.slider("Medium", 0.0, 1.0, 0.60, 0.05)
        with c2:
            high = st.slider("High", 0.0, 1.0, 0.70, 0.05)

        st.markdown("**Importance Weights**")
        wlow = st.slider("Low weight", 0.5, 2.0, 1.0, 0.1)
        wmed = st.slider("Medium weight", 0.5, 2.0, 1.2, 0.1)
        whigh = st.slider("High weight", 0.5, 2.0, 1.5, 0.1)

        unmet_high_penalty = st.slider(
            "High trait penalty", 0.0, 0.5, 0.10, 0.01,
            help="Extra penalty applied if any HIGH-importance trait is unmet"
        )

    # Evidence settings
    with st.expander("🧾 Evidence Settings", expanded=False):
        st.session_state["auto_evidence_from_reflection"] = st.checkbox(
            "Auto-create evidence from reflections",
            value=st.session_state["auto_evidence_from_reflection"],
            help="If on, saving a reflection will attach a brief evidence item to a few relevant traits."
        )

    # Output
    with st.expander("💾 Output Settings", expanded=False):
        outdir = st.text_input("Save reports to", value="notebooks/reports")
        save_history = st.checkbox("Save evaluation to history (session)", value=True)

    # Demo mode
    with st.expander("🧪 Demo Mode", expanded=False):
        st.caption("Load a complete demo with 90 days of synthetic data.")
        st.markdown("""
        **What you get:**
        - Pre-generated persona from 90 days of synthetic behavioral data
        - Sample job description for evaluation
        - Full pipeline demo experience
        """)
        if st.button("🚀 Load Demo Assets", use_container_width=True):
            # Prefer the full demo persona if available
            demo_persona = Path("data/sample/demo_persona.json")
            if demo_persona.exists():
                st.session_state["pending_persona_path"] = str(demo_persona)
            else:
                st.session_state["pending_persona_path"] = "data/sample/sample_persona.json"
            try:
                st.session_state["jd_text"] = Path("data/sample/sample_jd.txt").read_text()
            except Exception:
                st.session_state["jd_text"] = ""
            read_persona_cached.clear()
            st.success("Demo persona and JD loaded! Go to 'Evaluate Job' tab.")
            st.rerun()

# =========================
# MAIN TABS
# =========================
tab_dash, tab_eval, tab_daily, tab_persona = st.tabs(
    ["📊 Dashboard", "📄 Evaluate Job", "📈 Daily Update", "👤 View Persona"]
)

# =========================
# TAB: DASHBOARD
# =========================
with tab_dash:
    st.markdown("### 📊 Character Profile Dashboard")
    if persona_ready and per_trait:
        c1, c2, c3, c4 = st.columns(4)
        n = max(len(per_trait or {}), 1)
        avg_score = sum(float(t.get("score", 0)) for t in per_trait.values()) / n
        avg_conf  = sum(float(t.get("confidence", 0)) for t in per_trait.values()) / n
        total_evidence = sum(len(t.get("evidence", [])) for t in per_trait.values())
        strong_traits = sum(1 for t in per_trait.values() if float(t.get("score", 0)) >= 0.5)

        with c1: st.metric("Avg Score", f"{avg_score:.1%}")
        with c2: st.metric("Confidence", f"{avg_conf:.1%}")
        with c3: st.metric("Evidence Points", int(total_evidence))
        with c4: st.metric("Strong Traits", f"{strong_traits}/{len(per_trait)}")

        st.divider()
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("#### 🎯 Trait Profile (Radar)")
            st.plotly_chart(create_radar_chart(per_trait), use_container_width=True)
        with v2:
            st.markdown("#### 📊 Score vs Confidence")
            st.plotly_chart(create_trait_bar_chart(per_trait), use_container_width=True)

        # Trends
        updates_df = read_updates_cached(str(UPD_PATH))
        if not updates_df.empty and len(updates_df) >= 2:
            st.divider()
            st.markdown("#### 📈 Performance Trends Over Time")
            trend = create_daily_trend_chart(updates_df)
            if trend:
                st.plotly_chart(trend, use_container_width=True)

            s_cols = st.columns(4)
            with s_cols[0]: st.metric("Avg Productivity", f"{updates_df['prod'].mean():.0f}%")
            with s_cols[1]: st.metric("Avg Focus Time", f"{updates_df['focus_mins'].mean():.0f} min")
            if "sentiment_score" in updates_df.columns and updates_df["sentiment_score"].notna().any():
                with s_cols[2]: st.metric("Avg Sentiment", f"{(updates_df['sentiment_score'].dropna().mean()*100):.0f}")
                with s_cols[3]: st.metric("# Reflections", int(updates_df['sentiment_score'].notna().sum()))
            else:
                with s_cols[2]: st.metric("Avg Sentiment", "—")
                with s_cols[3]: st.metric("# Reflections", 0)

        # Strengths / Growth areas
        st.divider()
        sw1, sw2 = st.columns(2)
        with sw1:
            st.markdown("#### 💪 Top Strengths")
            strengths = sorted([(k, float(v.get("score", 0))) for k, v in per_trait.items()],
                               key=lambda x: x[1], reverse=True)[:5]
            for name, score in strengths:
                st.markdown(f"🟢 **{name.replace('_',' ').title()}**: {score:.0%}")
        with sw2:
            st.markdown("#### 🎯 Growth Areas")
            weaknesses = sorted([(k, float(v.get("score", 0))) for k, v in per_trait.items()],
                                key=lambda x: x[1])[:5]
            for name, score in weaknesses:
                st.markdown(f"🔴 **{name.replace('_',' ').title()}**: {score:.0%}")
    else:
        st.info("📋 Load a persona in the sidebar to see your dashboard.")

# =========================
# TAB: EVALUATE JOB
# =========================
with tab_eval:
    st.markdown("### 📄 Evaluate Job Description")
    with st.expander("ℹ️ How Job Evaluation Works", expanded=False):
        st.markdown("""
**Process**
1) AI extracts required traits (LOW/MEDIUM/HIGH) from the JD  
2) Your persona is compared to these requirements  
3) Weighted match score is computed  
4) You get strengths, risks, and a recommendation

**Match Score**: 50%+ = Strong | 30–49% = Possible | <30% = Poor  
**Risk**: 🟢 Low | 🟡 Moderate | 🔴 High
        """)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.session_state["jd_text"] = st.text_area(
            "Job Description",
            height=450,
            value=st.session_state.get("jd_text", ""),
            placeholder="Paste the full job description here..."
        )
        b1, b2 = st.columns([2, 1])
        with b1:
            run_btn = st.button("🚀 Analyze Fit", type="primary", use_container_width=True)
        with b2:
            if st.button("📝 Load Sample", use_container_width=True):
                st.session_state["jd_text"] = SAMPLE_JD
                st.rerun()

    with right:
        if run_btn:
            if not persona_ready:
                st.error("❌ Please load a valid persona first (check sidebar).")
            elif not st.session_state["jd_text"].strip():
                st.error("❌ Please paste a Job Description.")
            else:
                jd_text = st.session_state["jd_text"]
                with st.spinner("🔍 Analyzing job requirements..."):
                    try:
                        req_llm = extract_requirements_llm(jd_text, model=model)
                    except Exception as e:
                        st.warning(f"⚠️ LLM extraction failed: {e}")
                        req_llm = []
                    req_hybrid = extract_requirements_hybrid(jd_text)
                    req_union = union_requirements(req_llm, req_hybrid)

                thresholds = {"low": low, "medium": med, "high": high}
                weights    = {"low": wlow, "medium": wmed, "high": whigh}

                overall, match_ratio, risk, rows, criticals = score_requirements(
                    per_trait=per_trait,
                    requirements=req_union,
                    thresholds=thresholds,
                    weights=weights,
                    unmet_high_penalty=unmet_high_penalty,
                )

                # Store evaluation
                eval_data = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "overall": overall,
                    "match_ratio": float(match_ratio),
                    "risk": risk,
                    "rows": rows,
                    "criticals": criticals,
                    "jd_preview": jd_text[:200] + "…"
                }
                st.session_state["last_evaluation"] = eval_data
                if save_history:
                    st.session_state["evaluation_history"].append(eval_data)

                # Display results
                st.markdown("### 🎯 Results")
                st.plotly_chart(create_match_gauge(match_ratio), use_container_width=True)

                if overall in ["Strong fit", "Possible fit"]:
                    st.success(f"✅ **{overall}**")
                else:
                    st.warning(f"⚠️ **{overall}**")

                m1, m2, m3 = st.columns(3)
                with m1: st.metric("Match", f"{match_ratio:.0%}")
                with m2:
                    met_count = sum(1 for r in rows if r.get("met"))
                    st.metric("Traits Met", f"{met_count}/{len(rows)}")
                with m3:
                    risk_norm = str(risk).lower().replace("_", "-")
                    risk_emoji = {"low-risk": "🟢", "moderate-risk": "🟡", "high-risk": "🔴"}.get(risk_norm, "⚪")
                    st.metric("Risk", f"{risk_emoji} {risk_norm}")

                st.divider()
                s_col, r_col = st.columns(2, gap="large")

                # Strengths
                with s_col:
                    st.markdown("#### 💪 Strengths")
                    strengths = [r for r in rows if r.get("met") and r.get("candidate_score", 0) >= r.get("threshold", 0) + 0.15]
                    if not strengths:
                        strengths = [r for r in rows if r.get("met")]
                    strengths.sort(key=lambda x: x.get("candidate_score", 0) - x.get("threshold", 0), reverse=True)
                    if strengths:
                        for s in strengths[:5]:
                            margin = s.get("candidate_score", 0) - s.get("threshold", 0)
                            st.markdown(
                                f"✅ **{s.get('trait','').replace('_',' ').title()}** "
                                f"({s.get('candidate_score',0):.0%}) — exceeds {s.get('required_level','').upper()} by {margin:.0%}"
                            )
                    else:
                        st.info("Meets few/no requirements with margin — no standout strengths.")

                # Risks
                with r_col:
                    st.markdown("#### ⚠️ Risks")
                    risks_list = [r for r in rows if not r.get("met")]
                    if risks_list:
                        risks_list.sort(
                            key=lambda x: (
                                {"high": 2, "medium": 1, "low": 0}.get(x.get("required_level", "low"), 0),
                                x.get("threshold", 0) - x.get("candidate_score", 0)
                            ),
                            reverse=True
                        )
                        for r0 in risks_list[:5]:
                            lvl = r0.get("required_level", "low")
                            severity = "🔴" if lvl == "high" else ("🟡" if lvl == "medium" else "🟠")
                            gap = r0.get("threshold", 0) - r0.get("candidate_score", 0)
                            st.markdown(
                                f"{severity} **{r0.get('trait','').replace('_',' ').title()}** "
                                f"({r0.get('candidate_score',0):.0%}) — below {lvl.upper()} by {gap:.0%}"
                            )
                    else:
                        st.success("No significant risks!")

                # Recommendation
                st.divider()
                st.markdown("#### 🎯 Hiring Recommendation")
                if overall == "Strong fit":
                    st.success("**✅ RECOMMEND HIRING**\n\n- Meets/exceeds most critical traits\n- Low risk\n- Proceed to final rounds; verify strengths via references")
                elif overall == "Possible fit":
                    st.warning("**⚠️ PROCEED WITH CAUTION**")
                    top_risks = ", ".join([r.get("trait","").replace("_"," ") for r in risks_list[:3]]) if risks_list else "n/a"
                    st.markdown(
                        f"- {match_ratio:.0%} match with moderate gaps\n"
                        f"- Deep-dive on: {top_risks}\n"
                        f"- Set 30/60/90-day goals + mentorship\n"
                    )
                else:
                    st.error("**❌ DO NOT RECOMMEND**\n\n- Significant gaps / high risk\n- Consider different role or stronger onboarding if proceeding")

                # Save + download
                st.divider()
                cexp1, cexp2 = st.columns(2)
                with cexp1:
                    result = {
                        "candidate": persona_full.get("name", "Deo") if isinstance(persona_full, dict) else "Deo",
                        "overall": overall,
                        "verdict": "Likely successful" if overall in ["Strong fit", "Possible fit"] else "Likely to struggle",
                        "match_score": round(float(match_ratio), 2),
                        "risk_band": risk,
                        "requirements": rows,
                        "unmet_high_criticals": criticals,
                    }
                    jp, mp = save_report(result, Path(outdir))
                    st.success("💾 Report saved")
                    st.caption(f"JSON: {Path(jp).name} | MD: {Path(mp).name}")
                with cexp2:
                    md_report = export_to_md_report(result)
                    st.download_button(
                        label="📥 Download Report (Markdown)",
                        data=md_report,
                        file_name=f"job_eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
        elif st.session_state["last_evaluation"]:
            st.info("👆 Analyze a new JD or review your last evaluation below.")
            last = st.session_state["last_evaluation"]
            st.plotly_chart(create_match_gauge(last.get("match_ratio", 0.0)), use_container_width=True)
            if last.get("overall") in ["Strong fit", "Possible fit"]:
                st.success(f"✅ {last.get('overall')}")
            else:
                st.warning(f"⚠️ {last.get('overall')}")
        else:
            st.info("👈 Paste a JD and click **Analyze Fit** to see results here.")

# =========================
# TAB: DAILY UPDATE  (with Reflection + Sentiment)
# =========================
with tab_daily:
    st.markdown("### 📈 Daily Performance Tracker")
    lf, rp = st.columns([1, 1], gap="large")

    with lf:
        with st.form("daily_update", clear_on_submit=False):
            st.markdown("#### Log Today's Performance")

            c1, c2 = st.columns(2)
            with c1:  prod = st.slider("Productivity %", 0, 100, 60)
            with c2:  focus_mins = st.number_input("Deep work min", 0, 600, 90, 15)

            st.divider()

            c3, c4 = st.columns(2)
            with c3:
                comm = st.select_slider("Progress sharing", options=["no", "brief", "clear"], value="brief")
            with c4:
                team = st.select_slider("Collaboration", options=["no", "async", "live"], value="async")

            st.divider()

            c5, c6 = st.columns(2)
            with c5:
                plan = st.checkbox("Had & followed plan")
                learned = st.checkbox("Learned something new")
            with c6:
                adapt = st.select_slider("Adapted to change", options=["no", "some", "yes"], value="some")

            st.divider()
            st.markdown("**📝 Reflection (optional)**")
            reflection = st.text_area(
                "Write a short reflection about your day (1–5 sentences)",
                placeholder="E.g., Morning was rough, but I refactored the pipeline and landed 82% F1; good sync with the team.",
                height=100
            )

            alpha = st.slider(
                "Update strength",
                0.05, 0.50, 0.25, 0.05,
                help="How much today influences scores (e.g., 0.25 = 25% weight to today)"
            )
            submitted = st.form_submit_button("💾 Save & Update Persona", type="primary", use_container_width=True)

    with rp:
        st.markdown("#### 📊 Impact Preview")
        signals = {
            "Focus":          0.5*_z01(prod, 0, 100) + 0.5*_z01(focus_mins, 0, 240),
            "Communication":  {"no":0.2,"brief":0.55,"clear":0.8}[comm],
            "Teamwork":       {"no":0.25,"async":0.55,"live":0.8}[team],
            "Planning":       0.75 if plan else 0.35,
            "Adaptability":   {"no":0.35,"some":0.55,"yes":0.75}[adapt],
            "Learning":       0.75 if learned else 0.45,
            "Impact":         0.4 + 0.4*_z01(prod, 0, 100),
            "Reliability":    0.45 + 0.25*(1 if plan else 0) + 0.30*_z01(prod, 0, 100),
        }

        # Live sentiment preview (optional)
        sent_label, sent_score, sent_rationale = None, None, ""
        if (reflection or "").strip():
            try:
                sa = analyze_sentiment(reflection)
                sent_label, sent_score, sent_rationale = sa["label"], sa["score"], sa.get("rationale", "")
                badge = {"positive":"🟢","neutral":"🟡","negative":"🔴"}[sent_label]
                st.markdown(f"**Sentiment:** {badge} {sent_label.title()}  ·  Score: {sent_score:.2f}")
                if sent_rationale:
                    st.caption(sent_rationale)
            except Exception as e:
                st.warning(f"Sentiment analysis failed: {e}")

        # Apply a gentle sentiment boost to a few traits for preview
        sent_boost = (sent_score - 0.5) * 0.20 if sent_score is not None else 0.0  # ±0.10
        resilience_base = 0.55 + 0.30*_z01(prod, 0, 100) + (0.15 if adapt == "yes" else 0.05 if adapt == "some" else 0.0)
        resilience = max(0.0, min(1.0, resilience_base + sent_boost))
        impact_with_sent = max(0.0, min(1.0, signals["Impact"] + 0.5*sent_boost))
        communication_with_sent = max(0.0, min(1.0, signals["Communication"] + 0.5*sent_boost))

        # Show preview as progress bars
        preview_signals = {
            **signals,
            "Communication": communication_with_sent,
            "Impact": impact_with_sent,
            "Resilience": resilience,
        }
        for trait, score in preview_signals.items():
            emoji = "🟢" if score >= 0.7 else ("🟡" if score >= 0.5 else "🔴")
            st.markdown(f"{emoji} **{trait}**")
            st.progress(int(round(score * 100)))
            st.caption(f"{score:.0%}")

        st.info(f"**Blend:** New = {(1-alpha):.0%} × Old + {alpha:.0%} × Today")

    if submitted:
        with st.spinner("💾 Updating persona..."):
            # Append daily log (include reflection + sentiment if any)
            row = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "prod": prod, "focus_mins": focus_mins, "comm": comm,
                "team": team, "plan": plan, "adapt": adapt, "learned": learned,
                "reflection": (reflection or "").strip() or None,
                "sentiment_label": sent_label if sent_label else None,
                "sentiment_score": float(sent_score) if sent_score is not None else None,
            }
            dfu = pd.DataFrame([row])
            if UPD_PATH.exists():
                dfu = pd.concat([pd.read_parquet(UPD_PATH), dfu], ignore_index=True)
            UPD_PATH.parent.mkdir(parents=True, exist_ok=True)
            dfu.to_parquet(UPD_PATH, index=False)
            read_updates_cached.clear()  # refresh cache

            # Update persona (scores + optional auto-evidence)
            ppath = Path(st.session_state["persona_path"])
            per = json.loads(ppath.read_text())
            pt = per.get("per_trait", {})

            signal_map = {
                "focus":            signals["Focus"],
                "communication":    communication_with_sent,
                "teamwork":         signals["Teamwork"],
                "planning":         signals["Planning"],
                "adaptability":     signals["Adaptability"],
                "learning_mindset": signals["Learning"],
                "impact":           impact_with_sent,
                "reliability":      signals["Reliability"],
                "resilience":       resilience,
                "independence":     0.55 + 0.15*_z01(prod, 0, 100),
            }

            changes = []
            for t, sig in signal_map.items():
                if t not in pt:
                    pt[t] = {"score": sig, "confidence": 0.5, "evidence": []}
                old = float(pt[t].get("score", 0.5))
                new = (1 - alpha) * old + alpha * sig
                delta = new - old
                pt[t]["score"] = round(new, 3)
                pt[t]["confidence"] = round(min(1.0, float(pt[t].get("confidence", 0.5)) + 0.02), 3)
                changes.append({"trait": t, "old": old, "new": new, "delta": delta})

            # Optional: create lightweight evidence from reflection
            if st.session_state["auto_evidence_from_reflection"] and (reflection or "").strip():
                ts_now = datetime.datetime.now().isoformat(timespec="seconds")
                auto_items = [
                    ("communication", "Shared progress / thoughtful reflection"),
                    ("impact", "Reflected on results / outcomes"),
                    ("resilience", f"Emotional tone: {sent_label or 'n/a'} (score {sent_score:.2f if sent_score is not None else 'n/a'})")
                ]
                for key, desc in auto_items:
                    if key not in pt:
                        pt[key] = {"score": 0.5, "confidence": 0.5, "evidence": []}
                    evid = pt[key].get("evidence", [])
                    evid.append({"ts": ts_now, "desc": desc, "link": None})
                    pt[key]["evidence"] = evid
                    pt[key]["confidence"] = round(min(1.0, float(pt[key].get("confidence", 0.5)) + 0.01), 3)

            per["per_trait"] = pt
            updated_path = _save_persona_and_refresh(ppath, per)

        st.success(f"✅ Persona updated → `{Path(updated_path).name}`")
        with st.expander("🔄 What Changed"):
            ch = pd.DataFrame(changes)
            if not ch.empty:
                ch["trait"] = ch["trait"].str.replace("_", " ").str.title()
                ch["old"] = (ch["old"] * 100).round(1).astype(str) + "%"
                ch["new"] = (ch["new"] * 100).round(1).astype(str) + "%"
                ch["delta"] = (ch["delta"] * 100).map(lambda x: f"{x:+.1f}%")
                st.dataframe(ch, hide_index=True, use_container_width=True)
        st.balloons()
        st.rerun()

# =========================
# TAB: VIEW PERSONA  (+ Add / View Evidence)
# =========================
with tab_persona:
    st.markdown("### 👤 Character Persona Details")
    if persona_ready and per_trait:
        with st.expander("ℹ️ Understanding Metrics", expanded=False):
            st.markdown("""
- **Score**: Capability level (50%+ = Strong; 30–49% = Moderate; <30% = Developing)  
- **Confidence**: Data reliability (higher means more evidence behind the score)  
- **Evidence**: Count of documented examples (0 means baseline estimate)

Use **Daily Update** to build signals and **Add Evidence** to attach concrete proof.
            """)

        info_col, table_col = st.columns([1, 2])
        with info_col:
            name = persona_full.get("name", "Unknown") if isinstance(persona_full, dict) else "Unknown"
            st.markdown(f"**Name:** {name}")
            st.markdown(f"**Traits:** {len(per_trait)}")
            n = max(len(per_trait), 1)
            avg_score = sum(float(t.get("score", 0)) for t in per_trait.values()) / n
            avg_conf  = sum(float(t.get("confidence", 0)) for t in per_trait.values()) / n
            total_evidence = sum(len(t.get("evidence", [])) for t in per_trait.values())
            st.metric("Avg Score", f"{avg_score:.1%}")
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
            st.metric("Total Evidence", total_evidence)
            st.caption(st.session_state["persona_path"])

            # Updates summary
            updates_df = read_updates_cached(str(UPD_PATH))
            if not updates_df.empty:
                st.metric("Updates Logged", len(updates_df))
                last = pd.to_datetime(updates_df["ts"].iloc[-1])
                st.caption(f"Last update: {last.strftime('%Y-%m-%d %H:%M')}")
            else:
                st.caption("No daily updates yet")

            st.divider()
            # ----- Add Evidence panel -----
            st.markdown("#### ➕ Add Evidence")
            trait_names_sorted = sorted(list(per_trait.keys()))
            ev_trait = st.selectbox("Trait", [t for t in trait_names_sorted], index=0, key="ev_trait_sel")
            ev_desc = st.text_input("Description", placeholder="e.g., Presented model review to stakeholders")
            ev_link = st.text_input("Link (optional)", placeholder="https://…")
            ev_date = st.date_input("Date", value=datetime.date.today())
            add_ev = st.button("Add evidence", use_container_width=True)
            if add_ev:
                if not ev_desc.strip():
                    st.warning("Please enter a short description.")
                else:
                    ppath = Path(st.session_state["persona_path"])
                    persona = json.loads(Path(ppath).read_text())
                    pt = persona.get("per_trait", {})
                    if ev_trait not in pt:
                        pt[ev_trait] = {"score": 0.5, "confidence": 0.5, "evidence": []}
                    evid_list = pt[ev_trait].get("evidence", [])
                    evid_list.append({
                        "ts": datetime.datetime.combine(ev_date, datetime.datetime.now().time()).isoformat(timespec="seconds"),
                        "desc": ev_desc.strip(),
                        "link": ev_link.strip() or None
                    })
                    pt[ev_trait]["evidence"] = evid_list
                    # small confidence bump
                    pt[ev_trait]["confidence"] = round(min(1.0, float(pt[ev_trait].get("confidence", 0.5)) + 0.02), 3)
                    persona["per_trait"] = pt
                    newp = _save_persona_and_refresh(ppath, persona)
                    st.success(f"Evidence added to **{ev_trait.replace('_',' ').title()}** → {Path(newp).name}")
                    st.experimental_rerun()

        with table_col:
            st.markdown("#### Trait Breakdown")
            trait_rows = []
            for trait_name, trait_info in per_trait.items():
                score = float(trait_info.get("score", 0))
                conf  = float(trait_info.get("confidence", 0))
                evidence = len(trait_info.get("evidence", []))
                status = "🟢" if score >= 0.7 else ("🟡" if score >= 0.5 else "🔴")
                cstat  = "✅" if conf  >= 0.7 else ("⚠️" if conf  >= 0.5 else "❓")
                trait_rows.append({
                    "Trait": trait_name.replace("_", " ").title(),
                    "Score": f"{status} {score:.1%}",
                    "Confidence": f"{cstat} {conf:.1%}",
                    "Evidence": evidence if evidence > 0 else "-"
                })
            df = pd.DataFrame(trait_rows)
            st.dataframe(df, use_container_width=True, height=500, hide_index=True)

            # ----- View evidence for a selected trait -----
            st.divider()
            st.markdown("#### 🔎 View Evidence for a Trait")
            view_trait = st.selectbox("Select trait", [t for t in sorted(per_trait.keys())], key="ev_view_sel")
            ev_items = per_trait.get(view_trait, {}).get("evidence", [])
            if ev_items:
                ev_df = pd.DataFrame(ev_items)
                # tidy columns
                ev_df["ts"] = pd.to_datetime(ev_df["ts"])
                ev_df = ev_df.sort_values("ts", ascending=False)
                ev_df["ts"] = ev_df["ts"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(ev_df.rename(columns={"ts":"Timestamp","desc":"Description","link":"Link"}),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No evidence attached to this trait yet.")

        # Evaluation history table (session)
        if st.session_state["evaluation_history"]:
            st.divider()
            st.markdown("#### 📊 Evaluation History (this session)")
            hist = []
            for i, ed in enumerate(st.session_state["evaluation_history"]):
                hist.append({
                    "#": i + 1,
                    "Date": pd.to_datetime(ed.get("timestamp")).strftime("%Y-%m-%d %H:%M"),
                    "Match": f"{float(ed.get('match_ratio', 0))*100:.0f}%",
                    "Result": ed.get("overall", ""),
                    "Risk": ed.get("risk", ""),
                    "JD Preview": ed.get("jd_preview", "")[:60]
                })
            st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear History"):
                st.session_state["evaluation_history"] = []
                st.rerun()
    else:
        st.warning("⚠️ No persona loaded. Set a valid path in the sidebar.")

# =========================
# FOOTER
# =========================
st.divider()
f1, f2, f3 = st.columns([2, 1, 1])
with f1: st.caption("CTE — Character Trait Evaluator v3.3 • Streamlit + Plotly (+ Sentiment & Evidence)")
with f2:
    if st.session_state.get("evaluation_history"):
        st.caption(f"📊 {len(st.session_state['evaluation_history'])} evaluations saved this session")
with f3: st.caption("Tip: Add concrete evidence to raise confidence and credibility")
