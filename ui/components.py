"""Reusable UI pieces shared by the dashboard and every monument page."""
from __future__ import annotations

import base64
import html
import io

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from config import settings
from core.acquisition import cover_credit

LOGO = ('<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FBFAF7" stroke-width="1.6" stroke-linecap="round">'
        '<path d="M3 21h18M5 21V10M19 21V10M9 21v-7M15 21v-7M3 10l9-6 9 6"/><circle cx="12" cy="8.2" r="1.1" fill="#A9834B" stroke="none"/></svg>')


def md(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s or ""))


def cover_url(slug: str) -> str | None:
    p = settings.STATIC_DIR / "covers" / f"{slug}.jpg"
    return f"/{settings.STATIC_URL_PREFIX}/covers/{slug}.jpg" if p.exists() else None


def credit_line(slug: str) -> str:
    c = cover_credit(slug)
    if not c:
        return ""
    return f"Photo: {esc(c.get('author', '')[:60])} · {esc(c.get('license', ''))} · Wikimedia Commons"


def masthead(right_html: str = "") -> None:
    md(f"""<div class="mast"><a class="brand" href="?" target="_self"><span class="mark">{LOGO}</span>
    <span class="name">Chronicle<i>Heritage</i></span></a><div class="meta">{right_html}</div></div>""")


def section(no: str, title: str, aside: str = "", anchor: str = "") -> None:
    md(f'<div class="sec" id="{anchor}"><div class="no">§ {no}</div><h2>{title}</h2><div class="aside">{aside}</div></div>')


def empty(kicker: str, title: str, body: str) -> None:
    md(f'<div class="empty"><div class="kicker">{kicker}</div><h4>{title}</h4><p>{body}</p></div>')


def status_pill(label: str, tone: str) -> str:
    return f'<span class="pill"><span class="dot {tone}"></span>{esc(label)}</span>'


def _b64(img: Image.Image, width: int = 1400) -> str:
    img = img.convert("RGB")
    if img.width > width:
        img = img.resize((width, int(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=86)
    return base64.b64encode(buf.getvalue()).decode()


def before_after(before: Image.Image, after: Image.Image, label_a: str, label_b: str, height: int = 520) -> None:
    a, b = _b64(before), _b64(after.resize(before.size))
    ratio = before.height / before.width
    components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>body{{margin:0;font-family:'JetBrains Mono',monospace}}
.w{{position:relative;width:100%;padding-top:{ratio*100:.2f}%;overflow:hidden;border-radius:16px;background:#eee;user-select:none;cursor:ew-resize}}
.w img{{position:absolute;inset:0;width:100%;height:100%;object-fit:cover}}
#top{{clip-path:inset(0 50% 0 0)}}
#bar{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;box-shadow:0 0 0 1px rgba(11,31,58,.25)}}
#knob{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:42px;height:42px;border-radius:50%;background:#0B1F3A;color:#fff;
display:grid;place-items:center;font-size:14px;box-shadow:0 6px 20px rgba(0,0,0,.25)}}
.lb{{position:absolute;top:12px;font-size:10.5px;letter-spacing:.08em;background:rgba(255,255,255,.9);color:#0B1F3A;padding:6px 9px;border-radius:999px}}
</style><div class="w" id="w"><img src="data:image/jpeg;base64,{b}"><img id="top" src="data:image/jpeg;base64,{a}">
<div id="bar"></div><div id="knob">⇆</div><span class="lb" style="left:12px">BEFORE · {esc(label_a)}</span>
<span class="lb" style="right:12px">AFTER · {esc(label_b)}</span></div>
<script>const w=document.getElementById('w'),t=document.getElementById('top'),bar=document.getElementById('bar'),k=document.getElementById('knob');let d=false;
function mv(x){{const r=w.getBoundingClientRect();let p=Math.max(0,Math.min(1,(x-r.left)/r.width))*100;t.style.clipPath=`inset(0 ${{100-p}}% 0 0)`;bar.style.left=k.style.left=p+'%'}}
w.onmousedown=e=>{{d=true;mv(e.clientX)}};onmouseup=()=>d=false;onmousemove=e=>d&&mv(e.clientX);
w.ontouchmove=e=>mv(e.touches[0].clientX);w.onclick=e=>mv(e.clientX);</script>""", height=height)
