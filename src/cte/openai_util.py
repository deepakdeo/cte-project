# src/cte/openai_util.py
import os, json, re
from openai import OpenAI

# Load .env for CLI, notebooks, anything
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True), override=True)
except Exception:
    pass

_client = None
_runtime_key = None

def set_api_key(key: str):
    """Set API key at runtime (e.g., from user input in Streamlit)."""
    global _client, _runtime_key
    _runtime_key = key.strip() if key else None
    _client = None  # Reset client to use new key

def get_api_key() -> str:
    """Get current API key (runtime key takes precedence over env)."""
    return _runtime_key or (os.getenv("OPENAI_API_KEY") or "").strip()

def has_api_key() -> bool:
    """Check if an API key is available."""
    return bool(get_api_key())

def client() -> OpenAI:
    """Singleton OpenAI client initialized from runtime key or OPENAI_API_KEY."""
    global _client
    if _client is None:
        key = get_api_key()
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it in the app sidebar or set it in .env."
            )
        _client = OpenAI(api_key=key)
    return _client

def chat_json(system: str, user_text: str, model: str = "gpt-4.1-mini", max_tokens: int = 600, temperature: float = 0):
    """
    JSON-mode Chat Completions. Returns the raw text the model produced
    (which should be JSON); pair with parse_json_maybe below.
    """
    resp = client().chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_text},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

def parse_json_maybe(text: str, fallback=None):
    """
    Best-effort JSON parse. If the string isn't strict JSON,
    attempts to extract the first {...} block. Returns fallback on failure.
    """
    if fallback is None:
        fallback = {}
    if not text:
        return fallback
    s = text.strip()
    # Try straight JSON
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to extract the first JSON object
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return fallback
    return fallback
