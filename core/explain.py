"""AI-based damage explanation (Gemini), always framed as hypotheses.

The prompt gives Gemini only verifiable inputs: the monument's documented
materials/conservation notes and the *stored* detection numbers. Output is
cached in the `explanations` table and always shown with an "AI-generated
hypothesis — not an expert finding" label.
"""
from __future__ import annotations

from core import database as db
from core import llm
from core.damage_reference import DAMAGE
from core.monuments import Monument

PROMPT = """You are assisting a heritage-conservation monitoring tool. An object detector (YOLO26l,
trained on close-range photos of heritage masonry) flagged the damage category below on {name}
({location}). You have NOT inspected the monument and the detector can be wrong.

Monument facts (from the project knowledge base):
- Materials: {materials}
- Documented conservation issues: {issues}

Summary from the stored monitoring record:
- Category: {label}
- Number of boxes: {count}; {conf_note}; image type: {kind}; image date: {date}

Write in plain English, using markdown with these exact headings:
### What this category means
### Possible causes at {name}
(bullet list; every bullet must start with "Possibly" or "May be"; relate to the materials/climate only when the facts above support it)
### Preventive maintenance
(bullet list)
### Preservation recommendations
(bullet list; recommend assessment by qualified conservation professionals / ASI)
### Uncertainty
(2-3 sentences: detector false positives, lighting, image resolution, that causes are hypotheses)
Keep it under 280 words. Do not invent dates, events or measurements."""


def explain(m: Monument, label: str, count: int, conf: float, kind: str, date: str,
            refresh: bool = False) -> dict:
    """Return {text, source, model}. source = 'gemini' | 'cache' | 'reference'."""
    if not refresh:
        c = db.cached_explanation(m.slug, label)
        if c:
            return {"text": c.text, "source": "cache", "model": c.model}
    try:
        text, model = llm.generate(PROMPT.format(
            name=m.name, location=m.location, materials=", ".join(m.materials) or "not recorded",
            issues=(m.known_conservation_issues or "not recorded")[:1200], label=label, count=count,
            conf_note=("marked by hand, so no detector confidence exists" if conf <= 0
                       else f"highest detector confidence: {conf:.0%}"),
            kind=kind, date=date))
        db.save_explanation(m.slug, label, text, model)
        return {"text": text, "source": "gemini", "model": model}
    except llm.LLMUnavailable as e:
        return {"text": reference_text(label), "source": "reference", "model": str(e)}


def reference_text(label: str) -> str:
    ref = DAMAGE.get(label, {})
    causes = "\n".join(f"- Possibly {c[0].lower() + c[1:]}" for c in ref.get("typical_causes", []))
    prev = "\n".join(f"- {p}" for p in ref.get("prevention", []))
    return (f"### What this category means\n{ref.get('what', '')}\n\n### Possible causes (general reference)\n{causes}"
            f"\n\n### Preventive maintenance\n{prev}\n\n### Uncertainty\nThis is general conservation reference "
            "material, not a diagnosis of this monument. Confirm on site with a qualified conservator.")
