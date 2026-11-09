"""Monument-wise monitoring report (PDF, reportlab). Built only from stored records."""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from config import settings
from core import annotation, database as db
from core.damage_reference import DAMAGE
from core.monuments import Monument

NAVY = colors.HexColor("#0B1F3A")
MUTED = colors.HexColor("#5B6578")
LINE = colors.HexColor("#E3E6EC")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Times-Bold", fontSize=26, textColor=NAVY, alignment=0, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=15, textColor=NAVY, spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, textColor=NAVY, spaceBefore=6)
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8, textColor=MUTED, leading=11)
KICK = ParagraphStyle("k", parent=SMALL, fontName="Courier-Bold", textColor=MUTED, fontSize=8)


def _md(text: str) -> list:
    """Very small markdown → paragraphs converter for Gemini output."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line.replace("&", "&amp;").replace("<", "&lt;"))
        if line.startswith("#"):
            out.append(Paragraph(line.lstrip("# "), H3))
        elif line[:2] in ("- ", "* "):
            out.append(Paragraph("• " + line[2:], BODY))
        else:
            out.append(Paragraph(line, BODY))
    return out


def _table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8.5), ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
               ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY)]
    t.setStyle(TableStyle(st))
    return t


def _img(path: str, max_w: float, max_h: float):
    from PIL import Image
    try:
        w, h = Image.open(path).size
    except Exception:
        return Spacer(1, 1)
    s = min(max_w / w, max_h / h)
    return RLImage(path, width=w * s, height=h * s)


def _footer(c, doc):
    c.saveState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "ChronicleHeritage · AI-assisted monitoring report — AI outputs are hypotheses, not expert findings")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    c.restoreState()


def build(m: Monument) -> Path:
    path = settings.REPORT_DIR / f"{m.slug}_report_{datetime.now():%Y%m%d_%H%M}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm, title=f"{m.name} monitoring report")
    W = A4[0] - 36 * mm
    s = [Paragraph("CHRONICLEHERITAGE · MONITORING REPORT", KICK), Spacer(1, 4), Paragraph(m.name, H1),
         Paragraph(f"{m.location} · generated {datetime.now():%d %B %Y, %H:%M}", SMALL), Spacer(1, 10)]
    cover = settings.IMAGE_DIR / m.slug / "cover.jpg"
    if cover.exists():
        s += [_img(str(cover), W, 80 * mm), Spacer(1, 8)]

    s += [Paragraph("1. Monument details", H2), _table([
        ["Location", Paragraph(f"{m.location} ({m.lat:.4f}, {m.lon:.4f})", BODY)],
        ["Built by", Paragraph(m.built_by, BODY)], ["Period", Paragraph(m.construction_period, BODY)],
        ["Style", Paragraph(m.architectural_style, BODY)], ["Materials", Paragraph(", ".join(m.materials), BODY)],
        ["UNESCO", Paragraph(m.unesco, BODY)]], [32 * mm, W - 32 * mm], header=False),
        Spacer(1, 6), Paragraph(m.summary, BODY)]

    imgs = db.list_images(m.slug)
    s.append(Paragraph("2. Image acquisition record", H2))
    if imgs:
        s.append(_table([["#", "Type", "Source", "Acquired", "Date provenance"]] + [
            [str(i.id), i.kind, Paragraph(i.source, BODY), i.acquired_on.strftime("%d %b %Y") if i.acquired_on else "unknown",
             Paragraph(i.date_note, SMALL)] for i in imgs], [10 * mm, 20 * mm, 40 * mm, 24 * mm, W - 94 * mm]))
    else:
        s.append(Paragraph("No images stored yet.", BODY))

    runs = db.runs_for(m.slug)
    ref = annotation.is_reference(runs[0]) if runs else False
    s.append(Paragraph("3. Damage findings — hand annotation (detector not trained)" if ref
                       else "3. Damage detection (YOLO26l)", H2))
    if ref:
        s.append(Paragraph("<b>These findings are not model output.</b> The YOLO26l weights are not yet installed, "
                           "so the damage below was marked by eye by the project team on the photograph shown, and "
                           "is reported here as a hand annotation. No confidence score exists for a hand mark. "
                           "Everything else in this report — imagery, capture dates, sources and comparisons — is "
                           "recorded fact.", SMALL))
    if not runs:
        s.append(Paragraph("No detection run has been stored for this monument. No damage findings are reported.", BODY))
    else:
        r = runs[0]
        acq = r["image"]["acquired_on"]
        acq_s = f"acquired {acq:%d %b %Y}" if acq else "acquisition date unknown"
        origin = (f"{r['model']}" if ref
                  else f"model {r['model']} (weights {r['weights_sha']}) · confidence &ge; {r['conf']:.2f}")
        s.append(Paragraph(f"Latest record #{r['id']} · {r['created_at']:%d %b %Y %H:%M} UTC · {origin} · "
                           f"image: {r['image']['kind']}, {r['image']['source']}, {acq_s}", SMALL))
        if r["annotated_path"] and Path(r["annotated_path"]).exists():
            s += [Spacer(1, 4), _img(r["annotated_path"], W, 95 * mm)]
        dets = r["detections"]
        if dets:
            notes = annotation.notes_for(m.slug) if ref else []
            conf_head = "Marked as" if ref else "Confidence"
            s += [Spacer(1, 6), _table([["#", "Category", conf_head, "Image location (u, v)", "3D location"]] + [
                [str(i), d["label"],
                 (notes[i - 1] if i <= len(notes) else "hand annotation") if ref else f"{d['confidence']:.1%}",
                 f"{d['u']:.2f}, {d['v']:.2f}", f"{r['facade'] or '—'} facade · approximate"]
                for i, d in enumerate(dets, 1)], [10 * mm, 48 * mm, 40 * mm, 28 * mm, W - 126 * mm])]
            cnt = Counter(d["label"] for d in dets)
            s.append(Paragraph("4. Possible causes & recommendations", H2))
            s.append(Paragraph("Text below marked <b>AI-generated</b> is a Gemini hypothesis; text marked "
                               "<b>reference</b> is general conservation literature. Neither is an expert-confirmed finding.", SMALL))
            for label, n in cnt.most_common():
                c = db.cached_explanation(m.slug, label)
                block = [Paragraph(f"{label} — {n} {'hand annotation' if ref else 'detection'}(s) · {'AI-generated (' + c.model + ')' if c else 'reference'}", H3)]
                if c:
                    block += _md(c.text)
                else:
                    ref = DAMAGE.get(label, {})
                    block += [Paragraph("Possible causes include: " + "; ".join(ref.get("typical_causes", [])) + ".", BODY),
                              Paragraph("Recommended: " + "; ".join(ref.get("prevention", [])) + ".", BODY)]
                s.append(KeepTogether(block))
        else:
            s.append(Paragraph("The latest record contains no damage findings.", BODY))
        if len(runs) > 1:
            s += [Paragraph("Monitoring history", H3), _table([["Run", "Date", "Image", "Detections"]] + [
                [str(x["id"]), f"{x['created_at']:%d %b %Y}", x["image"]["kind"], str(len(x["detections"]))] for x in runs],
                [20 * mm, 40 * mm, 40 * mm, W - 100 * mm])]

    comps = db.comparisons_for(m.slug)
    s.append(Paragraph("5. Historical comparisons", H2))
    if comps:
        s.append(_table([["Before", "After", "Registered", "SSIM", "Changed area", "Regions"]] + [
            [f"img {c.before_id}", f"img {c.after_id}", "yes" if c.aligned else "no (not meaningful)", f"{c.ssim:.3f}",
             f"{c.changed_pct:.1f}%", str(len(c.regions.get('boxes', [])))] for c in comps],
            [22 * mm, 22 * mm, 34 * mm, 20 * mm, 26 * mm, W - 124 * mm]))
        s.append(Paragraph("Visible differences may come from lighting, season, shadows, vegetation or camera angle; "
                           "they are flags for expert review, not confirmed structural damage.", SMALL))
    else:
        s.append(Paragraph("No comparisons stored.", BODY))

    s += [Paragraph("6. Documented conservation context", H2), Paragraph(m.known_conservation_issues or "—", BODY)]
    doc.build(s, onFirstPage=_footer, onLaterPages=_footer)
    return path
