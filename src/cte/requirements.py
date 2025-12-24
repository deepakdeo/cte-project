import json
from .openai_util import chat_json, parse_json_maybe

REQ_SYSTEM = (
 "You analyze a job description and extract the *behavioral* traits needed to thrive in the role. "
 "Focus on traits like: focus, reliability, initiative, communication, teamwork, adaptability, "
 "curiosity, impact, independence, planning, resilience, learning_mindset. "
 "Return STRICT JSON {requirements: [{trait, required_level}]}, where required_level ∈ {low, medium, high}. "
 "Limit to 6–10 traits. Be concise; no commentary."
)

def extract_requirements_llm(jd_text: str, model="gpt-4.1-mini"):
    schema = {
      "type":"object",
      "properties":{
        "requirements":{"type":"array","items":{
          "type":"object",
          "properties":{"trait":{"type":"string"},"required_level":{"type":"string"}},
          "required":["trait","required_level"]
        }}
      },
      "required":["requirements"]
    }
    prompt = "JOB DESCRIPTION:\n" + jd_text + "\n\nSchema:\n" + json.dumps(schema)
    raw = chat_json(REQ_SYSTEM, prompt, model=model, max_tokens=600)
    js  = parse_json_maybe(raw)
    return js.get("requirements", [])

# simple hybrid lexicon pass (keeps levels “high” for stakeholder-heavy bits)
LEX = {
    "communication": ["stakeholder", "present", "communicat", "talk", "write", "storytell"],
    "teamwork": ["cross-functional", "team", "collaborat"],
    "focus": ["detail", "attention", "rigor", "deep work"],
    "reliability": ["deadline", "on-time", "ownership"],
    "planning": ["roadmap", "plan", "milestone", "sprint"],
    "initiative": ["proactive", "drive", "ownership", "lead"],
    "adaptability": ["changing", "dynamic", "ambiguity"],
    "curiosity": ["explore", "research", "question", "learn"],
    "impact": ["business value", "outcome", "impact"],
    "independence": ["autonom", "self-directed"],
    "learning_mindset": ["learn", "upskill"],
    "resilience": ["setback", "failure", "iterate"],
}
CRITICAL_HINTS = ["stakeholder", "cross-functional", "present", "communicat", "collaborat"]

def extract_requirements_hybrid(jd_text: str):
    text = jd_text.lower()
    reqs = {}
    def lvl_for(tr):
        # if JD screams stakeholders, treat comms/teamwork/focus/planning as high
        if tr in {"communication","teamwork","focus","planning"} and any(h in text for h in CRITICAL_HINTS):
            return "high"
        return "medium"
    for tr, keys in LEX.items():
        if any(k in text for k in keys):
            reqs[tr] = reqs.get(tr, lvl_for(tr))
    return [{"trait": t, "required_level": lvl} for t, lvl in reqs.items()]

def union_requirements(req_a, req_b):
    rank = {"low":0,"medium":1,"high":2}
    best = {}
    for r in (req_a or []):
        t, l = r.get("trait","").lower().strip(), r.get("required_level","medium").lower().strip()
        if not t: continue
        if t not in best or rank[l] > rank.get(best[t],0):
            best[t] = l
    for r in (req_b or []):
        t, l = r.get("trait","").lower().strip(), r.get("required_level","medium").lower().strip()
        if not t: continue
        if t not in best or rank[l] > rank.get(best[t],0):
            best[t] = l
    return [{"trait": t, "required_level": best[t]} for t in best]
