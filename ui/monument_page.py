"""Individual monument monitoring page — one reusable component for all 14 monuments.

Workflow on the page (top → bottom):
dossier → imagery (acquire/upload) → YOLO26l detection → 3D twin → Gemini explanation
→ historical comparison → monument chatbot → PDF report.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from config import settings
from core import acquisition, compare, database as db, detection, explain, rag, reports, viewer3d
from core.damage_reference import DAMAGE, color as dmg_color
from core.monuments import Monument, condition_of
from ui.components import (before_after, cover_url, credit_line, empty, esc, masthead, md, section, status_pill)

FACADES = ["north", "east", "south", "west", "top"]


def _img_label(i: db.ImageRecord) -> str:
    d = i.acquired_on.strftime("%d %b %Y") if i.acquired_on else "date unknown"
    return f"#{i.id} · {i.kind} · {d} · {i.source}"


def _meta(i: db.ImageRecord) -> str:
    d = i.acquired_on.strftime("%d %B %Y") if i.acquired_on else "Unknown"
    link = f' · <a href="{esc(i.source_url)}" target="_blank">source ↗</a>' if i.source_url else ""
    return (f'<div class="imgmeta"><b>Acquired:</b> {d} &nbsp;·&nbsp; <b>Type:</b> {i.kind} &nbsp;·&nbsp; '
            f'<b>Source:</b> {esc(i.source)}{link}<br>{esc(i.date_note)}<br>'
            f'<span style="color:var(--faint)">{esc(i.attribution)}{" · " + esc(i.license) if i.license else ""}</span></div>')


# =============================================================== page
def render(m: Monument) -> None:
    run = db.latest_run(m.slug)
    cur = {"count": len(run["detections"]), "labels": [d["label"] for d in run["detections"]]} if run else None
    status, tone = condition_of(cur)
    masthead(status_pill(status, tone))

    url = cover_url(m.slug)
    md(f"""<div class="mhero">{f'<img src="{url}" alt="{esc(m.name)}">' if url else ''}<div class="shade"></div>
      <div class="back"><a href="?" target="_self">← ALL MONUMENTS</a></div>
      <div class="in"><div class="kicker">{esc(m.location)} · {m.lat:.4f}°N, {m.lon:.4f}°E</div><h1>{esc(m.name)}</h1>
      <div class="tl">{esc(m.tagline)}</div></div><div class="credit">{credit_line(m.slug)}</div></div>""")
    md(f"""<div class="facts">
      <div><div class="kicker">Built</div><div class="val">{esc(m.construction_period)}</div></div>
      <div><div class="kicker">Commissioned by</div><div class="val">{esc(m.built_by)}</div></div>
      <div><div class="kicker">Style</div><div class="val">{esc(m.architectural_style)}</div></div>
      <div><div class="kicker">Status</div><div class="val">{esc(m.unesco)}</div></div></div>""")
    chapters = [("01", "Dossier", "dossier"), ("02", "Imagery", "imagery"), ("03", "Detection", "detection"),
                ("04", "Digital twin", "twin"), ("05", "Explanation", "explain"), ("06", "Change", "change"),
                ("07", "Assistant", "assistant"), ("08", "Report", "report")]
    md('<div class="chapters">' + "".join(f'<a href="#{a}"><b>{n}</b>{t}</a>' for n, t, a in chapters) + "</div>")

    _dossier(m)
    images = db.list_images(m.slug)
    _imagery(m, images)
    images = db.list_images(m.slug)
    _detection(m, images)
    run = db.latest_run(m.slug)
    _twin(m, run)
    _explanations(m, run)
    _change(m, images)
    _assistant(m)
    _report(m)


# =============================================================== §01
def _dossier(m: Monument) -> None:
    section("01", "Historical dossier", "From the project knowledge base — the same source the assistant retrieves from.", "dossier")
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        paras = "".join(f"<p>{esc(p)}</p>" for p in m.history.split("\n\n") if p.strip())
        md(f'<div class="prose">{paras}</div>')
    with c2:
        md(f"""<div class="panel"><div class="kicker">At a glance</div><h3>{esc(m.name)}</h3><div class="kv">
          <div class="k">Location</div><div>{esc(m.location)}</div>
          <div class="k">Period</div><div>{esc(m.construction_period)}</div>
          <div class="k">Built by</div><div>{esc(m.built_by)}</div>
          <div class="k">Style</div><div>{esc(m.architectural_style)}</div>
          <div class="k">Materials</div><div>{esc(', '.join(m.materials))}</div>
          <div class="k">UNESCO</div><div>{esc(m.unesco)}</div></div></div>""")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        md(f'<div class="panel"><div class="kicker">Architecture</div><div class="body" style="margin-top:10px">'
           + "".join(f"<p>{esc(p)}</p>" for p in m.architecture.split("\n\n")) + "</div></div>")
    with c2:
        md(f'<div class="panel"><div class="kicker">Historical significance</div><div class="body" style="margin-top:10px">'
           + "".join(f"<p>{esc(p)}</p>" for p in m.significance.split("\n\n")) + "</div></div>")


# =============================================================== §02
def _imagery(m: Monument, images: list) -> None:
    section("02", "Image monitoring", "Every image carries its real acquisition date and source. Nothing is presented as live.", "imagery")
    a1, a2 = st.columns([1, 1.4], gap="large")
    with a1:
        if st.button("Acquire latest available imagery", key="acq", use_container_width=True):
            with st.spinner("Querying Esri Wayback releases and Wikimedia Commons…"):
                try:
                    n_sat = acquisition.fetch_wayback_series(m, max_versions=4)
                    n_gr = acquisition.fetch_commons_gallery(m, limit=4)
                    st.success(f"Added {n_sat} dated satellite view(s) and {n_gr} ground photo(s). Duplicates are skipped.")
                except Exception as e:
                    st.error(f"Acquisition failed (network?): {e}")
            images = db.list_images(m.slug)
    with a2:
        with st.expander("Upload drone / field image"):
            with st.form("upl", clear_on_submit=True):
                f = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
                u1, u2, u3 = st.columns(3)
                kind = u1.selectbox("Type", ["drone", "ground", "satellite"])
                facade = u2.selectbox("Facade shown", FACADES, index=2)
                when = u3.date_input("Acquired on", value=None, max_value=date.today())
                src = st.text_input("Source / photographer", placeholder="e.g. DJI Mini 3, field visit")
                if st.form_submit_button("Store image") and f:
                    acquisition.store_upload(m, f.getvalue(), f.name, kind,
                                             datetime.combine(when, datetime.min.time()) if when else None, src, facade)
                    st.success("Stored.")
                    images = db.list_images(m.slug)

    if not images:
        empty("No imagery yet", f"No images stored for {esc(m.name)}.",
              "Use <b>Acquire latest available imagery</b> above (needs internet) or upload a drone / field photo.")
        return
    dated = [i for i in images if i.acquired_on]
    latest = dated[0] if dated else images[0]
    previous = next((i for i in dated[1:] if i.kind == latest.kind), dated[1] if len(dated) > 1 else None)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        md('<div class="kicker">Latest available image</div>')
        st.image(latest.path, use_container_width=True)
        md(_meta(latest))
    with c2:
        md('<div class="kicker">Previous image</div>')
        if previous:
            st.image(previous.path, use_container_width=True)
            md(_meta(previous))
        else:
            empty("—", "No earlier dated image", "Acquire more imagery to build a time series.")
    md(f'<div class="kicker" style="margin:26px 0 10px">Historical gallery · {len(images)} images, newest first</div>')
    cols = st.columns(4)
    for k, i in enumerate(images[:12]):
        with cols[k % 4]:
            st.image(i.path, caption=_img_label(i), use_container_width=True)


# =============================================================== §03
def _detection(m: Monument, images: list) -> None:
    section("03", "AI damage detection", "YOLO26l · 7 classes · trained on 6,317 heritage images (5,097 train / 972 val / 248 test).", "detection")
    if not settings.weights_available():
        empty("Integration point", "Trained YOLO26l weights are not installed",
              f"Copy your trained <code>best.pt</code> to <code>models/weights/yolo26l_heritage.pt</code> "
              f"(or set <code>YOLO_WEIGHTS</code> in <code>.env</code>) and reload. Detection is disabled until then — "
              "ChronicleHeritage never shows placeholder predictions.")
        _history(m)
        return
    if not images:
        empty("Nothing to analyse", "Add an image first", "Acquire or upload imagery in § 02.")
        return
    try:
        if detection.names_are_positional():
            md('<span class="tag-warn">Class names assumed</span> <span class="imgmeta">The weights name their '
               'classes numerically, so ids are mapped by position: '
               + ", ".join(f"{i} → {c}" for i, c in enumerate(settings.DAMAGE_CLASSES))
               + '. Run <code>python scripts/check_model.py</code> and confirm this against your training '
                 'data.yaml before relying on the labels.</span>')
    except detection.DetectorUnavailable:
        pass
    c1, c2, c3 = st.columns([2.2, 1, 1])
    choice = c1.selectbox("Image to analyse", images, format_func=_img_label, key="det_img")
    facade = c2.selectbox("Facade shown (for 3D mapping)", FACADES,
                          index=FACADES.index(choice.facade) if choice.facade in FACADES else 2, key="det_fac")
    conf = c3.slider("Confidence ≥", 0.05, 0.9, 0.25, 0.05, key="det_conf")
    if choice.kind == "satellite":
        md('<span class="tag-warn">Domain warning</span> <span class="imgmeta">The model was trained on close-range photographs. '
           'Satellite pixels (≈0.3–0.6 m) are far coarser than cracks or moss, so results on nadir imagery are unreliable.</span>')
    if st.button("Run YOLO26l detection", key="run_det"):
        with st.spinner("Running inference…"):
            try:
                img = Image.open(choice.path).convert("RGB")
                dets = detection.detect(img, conf=conf)
                ann = detection.annotate(img, dets)
                out = settings.RESULT_DIR / m.slug
                out.mkdir(parents=True, exist_ok=True)
                ap = out / f"run_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                ann.save(ap, quality=90)
                rid = db.save_run(choice.id, m.slug, "YOLO26l", detection.weights_sha(), conf, facade, str(ap), dets)
                st.success(f"Run #{rid} stored · {len(dets)} detection(s).")
            except detection.DetectorUnavailable as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Inference failed: {e}")
    run = db.latest_run(m.slug)
    if run:
        c1, c2 = st.columns([1.5, 1], gap="large")
        with c1:
            if run["annotated_path"] and Path(run["annotated_path"]).exists():
                st.image(run["annotated_path"], use_container_width=True,
                         caption=f"Run #{run['id']} · {run['created_at']:%d %b %Y %H:%M} UTC · conf ≥ {run['conf']:.2f}")
        with c2:
            dets = run["detections"]
            cnt = Counter(d["label"] for d in dets)
            md(f'<div class="kicker">Detections per category · run #{run["id"]}</div>')
            md('<div class="count-row" style="margin:10px 0 16px">' + "".join(
                f'<span class="cc"><span class="dot" style="background:{DAMAGE[k]["color"]}"></span>{esc(k)}<span class="n">{cnt.get(k, 0)}</span></span>'
                for k in DAMAGE) + "</div>")
            if dets:
                st.dataframe(pd.DataFrame([{"#": i, "Category": d["label"], "Confidence": f"{d['confidence']:.1%}",
                                            "Centre (u, v)": f"{d['u']:.2f}, {d['v']:.2f}"} for i, d in enumerate(dets, 1)]),
                             hide_index=True, use_container_width=True)
            else:
                md('<div class="imgmeta">No damage above the confidence threshold in this image.</div>')
    _history(m)


def _history(m: Monument) -> None:
    runs = db.runs_for(m.slug)
    if runs:
        with st.expander(f"Monitoring history · {len(runs)} run(s)"):
            st.dataframe(pd.DataFrame([{
                "Run": r["id"], "Run at (UTC)": f"{r['created_at']:%d %b %Y %H:%M}", "Image": r["image"]["kind"],
                "Image acquired": r["image"]["acquired_on"].strftime("%d %b %Y") if r["image"]["acquired_on"] else "unknown",
                "Detections": len(r["detections"]), "Facade": r["facade"], "Weights": r["weights_sha"]} for r in runs]),
                hide_index=True, use_container_width=True)


# =============================================================== §04
def _twin(m: Monument, run: dict | None) -> None:
    section("04", "3D damage visualisation", "Drag to orbit · scroll to zoom · right-drag to pan · click a marker for details.", "twin")
    dets = []
    notes = {}
    if run:
        for d in run["detections"]:
            x1, y1, x2, y2 = d["box"]
            try:
                W, H = Image.open(run["image"]["path"]).size
                area = (x2 - x1) * (y2 - y1) / (W * H)
            except Exception:
                area = 0.01
            dets.append({**d, "facade": run["facade"], "area": area})
        for lbl in {d["label"] for d in dets}:
            c = db.cached_explanation(m.slug, lbl)
            if c:
                notes[lbl] = esc(c.text[:420]).replace("\n", "<br>")
    viewer3d.render(m, dets, height=640, ai_notes=notes)
    msg = (f"Showing {len(dets)} marker(s) from run #{run['id']} ({run['facade']} facade)." if run
           else "No detection run stored yet — the twin shows the model without damage markers.")
    md(f'<div class="imgmeta">{msg} Marker positions are <b>approximate</b>: each 2D box centre is projected onto the model '
       f'from the facade the image shows (manual spatial correspondence), not reconstructed by photogrammetry.</div>')


# =============================================================== §05
def _explanations(m: Monument, run: dict | None) -> None:
    section("05", "Possible causes & preservation", "Gemini hypotheses grounded in this monument's materials and the stored detections.", "explain")
    if not run or not run["detections"]:
        empty("Waiting on detections", "Nothing to explain yet",
              "Explanations are generated only for damage the model actually detected and stored.")
        return
    dets = run["detections"]
    cnt = Counter(d["label"] for d in dets)
    when = run["image"]["acquired_on"]
    refresh = st.button("Regenerate with Gemini", key="regen", type="secondary")
    cols = st.columns(2, gap="large")
    for k, (label, n) in enumerate(cnt.most_common()):
        best = max(d["confidence"] for d in dets if d["label"] == label)
        with cols[k % 2]:
            with st.spinner(f"Explaining {label}…"):
                r = explain.explain(m, label, n, best, run["image"]["kind"],
                                    when.strftime("%d %b %Y") if when else "unknown", refresh=refresh)
            tag = ('<span class="tag-ai">AI-generated hypothesis · ' + esc(r["model"]) + "</span>") if r["source"] != "reference" \
                else '<span class="tag-ref">General reference · Gemini unavailable</span>'
            md(f'<div class="panel"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
               f'<div class="kicker"><span class="dot" style="background:{dmg_color(label)}"></span> {n} detection(s) · max {best:.0%}</div>{tag}</div>'
               f'<h3>{esc(label)}</h3></div>')
            st.markdown(r["text"])
    md('<div class="imgmeta">These are <b>possible</b> causes, not expert-confirmed findings. Refer active structural issues to qualified conservators / ASI.</div>')


# =============================================================== §06
def _change(m: Monument, images: list) -> None:
    section("06", "Historical comparison", "OpenCV registration (ORB + RANSAC homography), illumination normalisation and SSIM change mapping.", "change")
    if len(images) < 2:
        empty("Need two images", "Comparison needs at least two images",
              "Acquire imagery in § 02 — Esri Wayback gives several dated satellite views of the same footprint.")
        return
    sat = [i for i in images if i.kind == "satellite"]
    pool = sat if len(sat) >= 2 else images
    c1, c2, c3 = st.columns([1.3, 1.3, .8])
    after = c1.selectbox("After (newer)", pool, index=0, format_func=_img_label, key="cmp_a")
    before = c2.selectbox("Before (older)", pool, index=min(1, len(pool) - 1), format_func=_img_label, key="cmp_b")
    sens = c3.slider("Sensitivity", 0.0, 1.0, 0.5, 0.05, key="cmp_s")
    if before.id == after.id:
        st.info("Pick two different images.")
        return
    if before.kind != after.kind:
        md('<span class="tag-warn">Different image types</span> <span class="imgmeta">Viewpoints differ completely; pixel differences will not be meaningful.</span>')
    res = compare.compare(Image.open(before.path), Image.open(after.path), sens)
    lb = before.acquired_on.strftime("%d %b %Y") if before.acquired_on else "undated"
    la = after.acquired_on.strftime("%d %b %Y") if after.acquired_on else "undated"
    before_after(res["before"], res["after"], lb, la, height=int(560 * res["before"].height / res["before"].width) + 20)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registration", "Aligned" if res["aligned"] else "Failed", f"{res['inliers']} inliers", delta_color="off")
    k2.metric("Structural similarity", f"{res['ssim']:.3f}")
    k3.metric("Changed area", f"{res['changed_pct']:.1f}%")
    k4.metric("Change regions", len(res["regions"]))
    c1, c2 = st.columns(2, gap="large")
    c1.image(res["overlay"], caption="SSIM change map over the aligned 'after' image (boxes = regions to review)", use_container_width=True)
    c2.image(res["diff"], caption="Absolute intensity difference after normalisation", use_container_width=True)
    if not res["aligned"]:
        md('<span class="tag-warn">Not registered</span> <span class="imgmeta">The images could not be aligned reliably, so '
           'the difference maps are shown for reference only and should not be interpreted.</span>')
    md('<div class="imgmeta">Differences may come from lighting, sun angle, season, vegetation, shadows, people or vehicles and '
       'are <b>visible changes to review</b> — never confirmed structural damage.</div>')
    if st.button("Save comparison to monitoring record", key="cmp_save", type="secondary"):
        db.save_comparison(monument_slug=m.slug, before_id=before.id, after_id=after.id, aligned=res["aligned"],
                           inliers=res["inliers"], changed_pct=res["changed_pct"], ssim=res["ssim"],
                           regions={"boxes": res["regions"]})
        st.success("Comparison saved — it will appear in the report.")


# =============================================================== §07
def _assistant(m: Monument) -> None:
    section("07", f"{m.name} AI Assistant", "Retrieval-augmented: answers come from this monument's knowledge base and its stored monitoring record.", "assistant")
    key = f"chat_{m.slug}"
    hist = st.session_state.setdefault(key, [])
    sugg = ["Who constructed it?", "When was it constructed?", "What architectural style does it follow?",
            "What damage has been detected?", "What preservation methods are recommended?"]
    cols = st.columns(len(sugg))
    picked = None
    for c, q in zip(cols, sugg):
        if c.button(q, key=f"sg_{m.slug}_{q}", type="secondary", use_container_width=True):
            picked = q
    box = st.container(border=True)
    with box:
        if not hist:
            md(f'<div class="imgmeta" style="padding:6px 4px">Ask anything about {esc(m.name)} — its history, architecture, '
               'detected damage or recommended preservation.</div>')
        for h in hist:
            with st.chat_message(h["role"], avatar="🏛️" if h["role"] == "assistant" else None):
                st.markdown(h["content"])
                if h.get("meta"):
                    st.caption(h["meta"])
        q = st.chat_input(f"Ask the {m.name} assistant…", key=f"ci_{m.slug}") or picked
        if q:
            hist.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant", avatar="🏛️"):
                with st.spinner("Retrieving…"):
                    try:
                        r = rag.ask(m, q, hist[:-1])
                        ans, meta = r["answer"], f"{r['mode']} · sources: {', '.join(r['sources'])}"
                    except Exception as e:
                        ans, meta = f"The assistant could not answer: {e}", ""
                st.markdown(ans)
                st.caption(meta)
            hist.append({"role": "assistant", "content": ans, "meta": meta})


# =============================================================== §08
def _report(m: Monument) -> None:
    section("08", "Monitoring report", "PDF built only from stored records: imagery, detections, explanations and comparisons.", "report")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Generate PDF report", key="rep"):
            with st.spinner("Composing report…"):
                st.session_state[f"rep_{m.slug}"] = str(reports.build(m))
    p = st.session_state.get(f"rep_{m.slug}")
    with c2:
        if p and Path(p).exists():
            st.download_button("Download report", Path(p).read_bytes(), file_name=Path(p).name, mime="application/pdf")
            md(f'<div class="imgmeta">Saved to <code>{esc(Path(p).relative_to(settings.ROOT))}</code></div>')
