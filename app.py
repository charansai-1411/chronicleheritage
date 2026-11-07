"""ChronicleHeritage — entry point.  Run:  streamlit run app.py

Single-page, card-navigated app (no sidebar / nav bar / tabs). Routing uses the
URL query string: `?` → dashboard, `?m=<slug>` → that monument's page.
"""
import streamlit as st

st.set_page_config(page_title="ChronicleHeritage", page_icon="🏛️", layout="wide",
                   initial_sidebar_state="collapsed")

from core import database as db  # noqa: E402
from core.monuments import by_slug  # noqa: E402
from ui import dashboard, monument_page, theme  # noqa: E402

db.init_db()
theme.apply()

slug = st.query_params.get("m")
monument = by_slug(slug) if slug else None
if monument:
    monument_page.render(monument)
else:
    dashboard.render()
