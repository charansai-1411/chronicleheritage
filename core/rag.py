"""Monument-specific RAG chatbot (LangChain + ChromaDB + Sentence-Transformers + Gemini).

* One Chroma collection per monument (`heritage_<slug>`) built from that
  monument's knowledge-base entry plus the damage-category reference.
* The *live* monitoring record (latest detection run, image dates) is read from
  the database at question time and added to the context, so answers about
  detected damage always reflect what is actually stored — never invented.
* Without a Gemini key the bot returns the retrieved passages verbatim.
"""
from __future__ import annotations

from functools import lru_cache

from config import settings
from core import database as db
from core import llm
from core.damage_reference import DAMAGE
from core.monuments import Monument


@lru_cache(maxsize=1)
def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def _documents(m: Monument):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    facts = (f"{m.name} is located in {m.location} (lat {m.lat}, lon {m.lon}). Built by: {m.built_by}. "
             f"Construction period: {m.construction_period}. Architectural style: {m.architectural_style}. "
             f"Main materials: {', '.join(m.materials)}. UNESCO status: {m.unesco}. {m.summary}")
    sections = {"facts": facts, "history": m.history, "architecture": m.architecture,
                "significance": m.significance, "conservation": m.known_conservation_issues}
    docs = [Document(page_content=t, metadata={"monument": m.slug, "section": s}) for s, t in sections.items() if t]
    for label, d in DAMAGE.items():
        docs.append(Document(page_content=(f"Damage category '{label}': {d['what']} Typical causes in heritage "
                                           f"masonry: {'; '.join(d['typical_causes'])}. Preventive measures: "
                                           f"{'; '.join(d['prevention'])}."),
                             metadata={"monument": m.slug, "section": "damage-reference"}))
    return RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(docs)


@lru_cache(maxsize=16)
def store(slug: str):
    from langchain_chroma import Chroma
    from core.monuments import by_slug
    vs = Chroma(collection_name=f"heritage_{slug.replace('-', '_')}", embedding_function=_embeddings(),
                persist_directory=str(settings.CHROMA_DIR))
    if not vs.get(limit=1)["ids"]:
        vs.add_documents(_documents(by_slug(slug)))
    return vs


def rebuild(m: Monument) -> int:
    vs = store(m.slug)
    ids = vs.get()["ids"]
    if ids:
        vs.delete(ids=ids)
    docs = _documents(m)
    vs.add_documents(docs)
    return len(docs)


def monitoring_context(m: Monument) -> str:
    imgs = db.list_images(m.slug)
    run = db.latest_run(m.slug)
    lines = [f"Images on record: {len(imgs)} ("
             + ", ".join(f"{k}: {sum(i.kind == k for i in imgs)}" for k in ("satellite", "drone", "ground")) + ")."]
    if not run:
        lines.append("No YOLO26l detection run has been stored for this monument yet, so no damage has been detected or recorded.")
    else:
        dets = run["detections"]
        when = run["image"]["acquired_on"]
        acq = f"acquired {when:%d %b %Y}" if when else "acquisition date unknown"
        lines.append(f"Latest detection run #{run['id']} on {run['created_at']:%d %b %Y} using {run['model']}, "
                     f"on a {run['image']['kind']} image from {run['image']['source']} ({acq}).")
        if dets:
            by = {}
            for d in dets:
                by.setdefault(d["label"], []).append(d["confidence"])
            lines.append("Detected: " + "; ".join(f"{k} ×{len(v)} (max confidence {max(v):.0%})" for k, v in by.items()) + ".")
        else:
            lines.append("That run detected no damage above the confidence threshold.")
    for label in {d["label"] for d in (run or {}).get("detections", [])}:
        c = db.cached_explanation(m.slug, label)
        if c:
            lines.append(f"Stored AI-generated (Gemini) hypothesis for {label}: {c.text[:600]}")
    return "\n".join(lines)


PROMPT = """You are the {name} AI Assistant inside ChronicleHeritage, a heritage-monitoring platform.
Answer ONLY from the CONTEXT below. If the context does not contain the answer, say you don't have
that information in the knowledge base. Never invent historical facts, dates, people or monitoring results.
When talking about causes of damage, say "possible causes include…" and note they are hypotheses,
not expert-confirmed findings. Be concise (under 180 words) and friendly.

MONITORING RECORD (live from database):
{monitoring}

KNOWLEDGE BASE PASSAGES:
{passages}

CONVERSATION SO FAR:
{history}

QUESTION: {question}"""


def ask(m: Monument, question: str, history: list[dict] | None = None, k: int = 5) -> dict:
    docs = store(m.slug).similarity_search(question, k=k)
    passages = "\n\n".join(f"[{d.metadata.get('section')}] {d.page_content}" for d in docs)
    mon = monitoring_context(m)
    hist = "\n".join(f"{h['role']}: {h['content'][:400]}" for h in (history or [])[-6:])
    sources = sorted({d.metadata.get("section", "") for d in docs})
    try:
        text, model = llm.generate(PROMPT.format(name=m.name, monitoring=mon, passages=passages,
                                                 history=hist or "(none)", question=question), temperature=0.2)
        return {"answer": text, "sources": sources + ["monitoring-record"], "mode": f"Gemini · {model}"}
    except llm.LLMUnavailable as e:
        q = question.lower()
        if any(w in q for w in ("damage", "detect", "crack", "moss", "monitor", "condition")):
            body = mon
        else:
            body = "\n\n".join(d.page_content for d in docs[:2]) if docs else "No matching passage."
        return {"answer": f"{body}\n\n_Retrieval-only answer — {e}. Add a Gemini key for conversational answers._",
                "sources": sources, "mode": "Retrieval only"}
