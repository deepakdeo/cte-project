import json
from pathlib import Path

def load_persona(persona_path: str | Path):
    p = Path(persona_path)
    with p.open() as f:
        js = json.load(f)
    # expect {"per_trait": {"trait": {"score": float, ...}, ...}}
    return js.get("per_trait", {}), js
