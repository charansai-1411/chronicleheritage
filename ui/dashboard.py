"""Home page & dashboard — every number is computed from the database."""
from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from config import settings
from core import database as db
from core.damage_reference import DAMAGE, color as dmg_color
from core.monuments import condition_of, load_monuments
from ui.components import credit_line, cover_url, empty, esc, masthead, md, section, status_pill

FONT = dict(family="Inter Tight, sans-serif", color="#0B1F3A", size=12)


def _ago(t: datetime) -> str:
    s = (datetime.utcnow() - t).total_seconds()
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n:
            return f"{int(s // n)}{unit} ago"
    return "just now"


def render() -> None:
    mons = load_monuments()
    stats = db.global_stats()
    cur = stats["current"]
    names = {m.slug: m.name for m in mons}
    attention = [m for m in mons if condition_of(cur.get(m.slug))[1] == "alert"]
    scanned = sum(1 for m in mons if m.slug in cur)

    sys_pills = (status_pill("YOLO26l weights loaded" if settings.weights_available() else "YOLO26l weights missing",
                             "ok" if settings.weights_available() else "watch")
                 + status_pill("Gemini connected" if settings.gemini_available() else "Gemini key missing",
                               "ok" if settings.gemini_available() else "watch"))
    masthead(sys_pills)

    hero = next((m for m in mons if cover_url(m.slug)), None)
    hero_img = (f'<img src="{cover_url(hero.slug)}" alt="{esc(hero.name)}"><div class="grad"></div>'
                f'<div class="cap"><span class="pill"><span class="dot"></span>{esc(hero.name)} · {esc(hero.city)}</span>'
                f'<span class="credit">{credit_line(hero.slug)}</span></div>') if hero else '<div class="noimg">◇</div>'
    md(f"""<div class="hero"><div>
      <div class="kicker">AI heritage monitoring · digital twins · threat assessment</div>
      <h1>Watching India's monuments <i>age</i>, one image at a time.</h1>
      <p class="lede">ChronicleHeritage gathers dated satellite and ground imagery of fourteen monuments, detects seven
      kinds of masonry damage with a trained YOLO26l model, pins every finding onto a 3D digital twin and explains
      possible causes — always marking what is AI hypothesis and what is record.</p></div>
      <div class="hero-media">{hero_img}</div></div>""")

    by_class = stats["by_class"]
    top_cls = max(by_class, key=by_class.get) if by_class else None
    md(f"""<div class="ledger">
      <div class="cell"><div class="v">{len(mons)}</div><div class="kicker l">Monuments monitored</div><div class="s">{scanned} scanned at least once</div></div>
      <div class="cell"><div class="v">{stats['detections']}</div><div class="kicker l">Damage detections</div><div class="s">across {stats['runs']} model run{'s' if stats['runs'] != 1 else ''}</div></div>
      <div class="cell"><div class="v">{len(by_class)}<small>/ 7</small></div><div class="kicker l">Damage categories seen</div><div class="s">{esc(top_cls + ' most frequent') if top_cls else 'none recorded yet'}</div></div>
      <div class="cell"><div class="v">{len(attention)}</div><div class="kicker l">Need attention</div><div class="s">{esc(', '.join(m.name for m in attention[:2])) or 'no structural flags'}</div></div>
      <div class="cell"><div class="v">{stats['images']}</div><div class="kicker l">Dated images on record</div><div class="s">satellite · ground · drone</div></div>
    </div>""")

    # ------------------------------------------------------------- analytics
    section("01", "Condition overview", "Status comes from each monument's latest stored detection run — "
            "confidence measures detector certainty, not severity.", "overview")
    c1, c2 = st.columns([1.15, 1], gap="large")
    with c1:
        md('<div class="kicker" style="margin-bottom:8px">Damage distribution · all stored detections</div>')
        labels = list(DAMAGE)
        vals = [by_class.get(k, 0) for k in labels]
        fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=[dmg_color(k) for k in labels],
                        line=dict(width=0)), text=[str(v) for v in vals], textposition="outside",
                        hovertemplate="%{y}: %{x}<extra></extra>"))
        fig.update_layout(height=330, margin=dict(l=0, r=30, t=6, b=10), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=FONT, bargap=.42,
                          xaxis=dict(showgrid=True, gridcolor="#ECEAE4", zeroline=False, rangemode="tozero",
                                     range=[0, max(vals + [4]) * 1.15], dtick=max(1, max(vals + [4]) // 4)),
                          yaxis=dict(autorange="reversed", ticks="", showline=False))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if not stats["detections"]:
            md('<div class="imgmeta">No detections are stored yet — bars will fill as YOLO26l runs are saved.</div>')
    with c2:
        md('<div class="kicker" style="margin-bottom:8px">Monument status matrix</div>')
        cells = []
        for i, m in enumerate(mons, 1):
            label, tone = condition_of(cur.get(m.slug))
            n = cur.get(m.slug, {}).get("count")
            cells.append(f'<a class="mx" href="?m={m.slug}" target="_self"><div class="n">{esc(m.name)}</div>'
                         f'<div class="c"><span class="dot {tone}"></span>{"—" if n is None else str(n) + " found"}</div></a>')
        md(f'<div class="matrix" style="grid-template-columns:repeat(4,1fr)">{"".join(cells)}</div>')
        md('<div class="imgmeta" style="margin-top:12px"><span class="dot alert"></span> structural class detected &nbsp; '
           '<span class="dot watch"></span> surface / biological &nbsp; <span class="dot ok"></span> none found &nbsp; '
           '<span class="dot"></span> not scanned</div>')

    # ------------------------------------------------------------- activity
    section("02", "Recent monitoring activity", "Latest model runs and image acquisitions, newest first.", "activity")
    events = ([("run", r["at"], r["slug"], f"YOLO26l run · {r['count']} detection{'s' if r['count'] != 1 else ''} on {r['kind']} image")
               for r in stats["recent_runs"]] +
              [("img", i["at"], i["slug"], f"{i['kind'].title()} image acquired · {i['source']}") for i in stats["recent_images"]])
    events.sort(key=lambda e: e[1], reverse=True)
    if events:
        rows = "".join(f'<div class="act"><span class="t">{_ago(t)}</span><span><b>{esc(names.get(s, s))}</b> — {esc(txt)}</span>'
                       f'<span class="tag-{"ai" if k == "run" else "ref"}">{"model" if k == "run" else "imagery"}</span></div>'
                       for k, t, s, txt in events[:8])
        md(f'<div class="panel" style="padding:6px 24px">{rows}</div>')
    else:
        empty("No activity yet", "The archive is empty.",
              "Run <code>python scripts/fetch_imagery.py</code> to pull dated satellite and Commons imagery, "
              "then open a monument to run detection.")

    # ------------------------------------------------------------- explore
    section("03", "Explore the monuments", "Fourteen sites, one reusable monitoring dossier each.", "explore")
    cards = []
    for i, m in enumerate(mons, 1):
        label, tone = condition_of(cur.get(m.slug))
        url = cover_url(m.slug)
        ph = f'<img src="{url}" loading="lazy" alt="{esc(m.name)}">' if url else f'<div class="noimg">{esc(m.name[0])}</div>'
        n_img = stats["images_by_monument"].get(m.slug, 0)
        cards.append(f"""<a class="mc" href="?m={m.slug}" target="_self"><div class="ph">{ph}<span class="no">No. {i:02d}</span>
          <span class="st">{status_pill(label, tone)}</span></div><div class="tx"><div class="nm">{esc(m.name)}</div>
          <div class="loc">{esc(m.location)} · {m.lat:.2f}°N {m.lon:.2f}°E</div>
          <div class="ft"><span>{n_img} image{'s' if n_img != 1 else ''}{' · 3D twin' if m.has_3d else ''}</span><span class="go">View details <span>→</span></span></div></div></a>""")
    md(f'<div class="grid">{"".join(cards)}</div>')

    md("""<div style="margin-top:70px;padding-top:22px;border-top:1px solid var(--line);display:flex;justify-content:space-between;flex-wrap:wrap;gap:12px">
      <span class="kicker">ChronicleHeritage · CBIT AI&amp;DS · Batch 103</span>
      <span class="imgmeta" style="margin:0">Imagery: Esri World Imagery Wayback, Wikimedia Commons contributors. AI outputs are hypotheses, not expert findings.</span></div>""")
