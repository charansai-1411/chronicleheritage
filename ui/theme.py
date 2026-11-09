"""Visual system: 'Conservation Atlas' — ivory paper, navy ink, serif display, mono catalogue labels."""
import streamlit as st

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --paper:#FBFAF7; --paper-2:#F3F1EB; --card:#FFFFFF; --ink:#0B1F3A; --ink-2:#22324F; --muted:#5B6578; --faint:#8C94A3;
  --line:#E4E2DC; --line-2:#D6D3CA; --brass:#A9834B; --ok:#2E7D5B; --watch:#B7791F; --alert:#B03A2E; --idle:#8C94A3;
  --serif:'Instrument Serif',Georgia,serif; --sans:'Inter Tight',system-ui,sans-serif; --mono:'JetBrains Mono',ui-monospace,monospace;
  --shadow:0 1px 2px rgba(11,31,58,.04),0 18px 40px -24px rgba(11,31,58,.28);
}
html,body,.stApp{background:var(--paper)!important;color:var(--ink);font-family:var(--sans)}
.stApp *{font-family:var(--sans)}
header[data-testid="stHeader"],[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stToolbar"],
[data-testid="stDecoration"],footer,#MainMenu{display:none!important}
.block-container{max-width:1320px!important;padding:0 2.2rem 5rem!important}
@media (max-width:700px){.block-container{padding:0 1rem 4rem!important}}
h1,h2,h3,h4{color:var(--ink)}
p,li{color:var(--ink-2)}
a{color:var(--ink)}
[data-testid="stVerticalBlock"]{gap:1rem}

/* ---------- primitives ---------- */
.kicker{font:500 11px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.serif{font-family:var(--serif)!important;font-weight:400;letter-spacing:-.01em}
.rule{height:1px;background:var(--line);margin:8px 0 0}
.pill{display:inline-flex;align-items:center;gap:7px;font:500 11px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;
  padding:7px 11px;border-radius:999px;border:1px solid var(--line-2);background:rgba(255,255,255,.75);color:var(--ink);white-space:nowrap}
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;background:var(--idle)}
.dot.ok{background:var(--ok)}.dot.watch{background:var(--watch)}.dot.alert{background:var(--alert);box-shadow:0 0 0 4px rgba(176,58,46,.14)}
.tag-ai{display:inline-block;font:500 10px var(--mono);letter-spacing:.1em;text-transform:uppercase;padding:4px 8px;border-radius:6px;
  background:#EEF2FA;color:#2B4A86;border:1px solid #D6DFF0}
.tag-ref{display:inline-block;font:500 10px var(--mono);letter-spacing:.1em;text-transform:uppercase;padding:4px 8px;border-radius:6px;
  background:#F5F1E8;color:#7A5C2E;border:1px solid #E6DCC6}
.tag-warn{display:inline-block;font:500 10px var(--mono);letter-spacing:.1em;text-transform:uppercase;padding:4px 8px;border-radius:6px;
  background:#FFF4E5;color:#8A5300;border:1px solid #F3D9B1}
.tag-hand{display:inline-block;font:500 10px var(--mono);letter-spacing:.1em;text-transform:uppercase;padding:4px 8px;border-radius:6px;
  background:#FBECEA;color:#8E2F22;border:1px solid #EFCCC6}

/* a statement the reader must not skim past — used where hand annotations stand in for the model */
.callout-warn{border:1px solid #EFCCC6;border-left:3px solid var(--alert);background:#FDF6F4;color:var(--ink);
  border-radius:12px;padding:14px 16px;font-size:13.5px;line-height:1.62;margin:4px 0 18px}
.callout-warn b{font-weight:600}
.callout-warn code{font:500 12px var(--mono);background:#FFF;border:1px solid var(--line);border-radius:5px;padding:1px 5px}

/* ---------- masthead ---------- */
.mast{display:flex;justify-content:space-between;align-items:center;padding:22px 0 18px;border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:12px;text-decoration:none!important}
.brand .mark{width:34px;height:34px;border-radius:9px;background:var(--ink);display:grid;place-items:center}
.brand .name{font:400 26px/1 var(--serif);color:var(--ink);letter-spacing:-.01em}
.brand .name i{font-style:italic;color:var(--brass)}
.mast .meta{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:flex-end}

/* ---------- hero ---------- */
.hero{display:grid;grid-template-columns:1.05fr .95fr;gap:44px;padding:54px 0 34px;align-items:end}
@media (max-width:980px){.hero{grid-template-columns:1fr;gap:26px;padding-top:34px}}
.hero h1{font:400 clamp(46px,6.4vw,92px)/.94 var(--serif)!important;letter-spacing:-.025em;margin:18px 0 22px;color:var(--ink);padding:0}
.hero h1 i{color:var(--brass)}
.hero p.lede{font-size:17px;line-height:1.6;color:var(--ink-2);max-width:560px;margin:0}
.hero-media{position:relative;height:440px;border-radius:22px;overflow:hidden;box-shadow:var(--shadow);background:var(--paper-2)}
.hero-media img{width:100%;height:100%;object-fit:cover;display:block;filter:saturate(.92) contrast(1.02)}
.hero-media .cap{position:absolute;left:16px;bottom:16px;right:16px;display:flex;justify-content:space-between;gap:10px;align-items:flex-end}
.hero-media .cap .pill{background:rgba(255,255,255,.88);backdrop-filter:blur(8px)}
.hero-media .grad{position:absolute;inset:0;background:linear-gradient(180deg,rgba(11,31,58,0) 55%,rgba(11,31,58,.35))}
.credit{font:400 10px var(--mono);color:rgba(255,255,255,.9);text-align:right;max-width:55%;text-shadow:0 1px 2px rgba(0,0,0,.4)}

/* ---------- stat ledger ---------- */
.ledger{display:grid;grid-template-columns:repeat(5,1fr);border-top:1px solid var(--ink);border-bottom:1px solid var(--line);margin:8px 0 6px}
@media (max-width:900px){.ledger{grid-template-columns:repeat(2,1fr)}}
.ledger .cell{padding:22px 20px 20px;border-right:1px solid var(--line)}
.ledger .cell:last-child{border-right:none}
.ledger .v{font:400 54px/1 var(--serif);color:var(--ink);letter-spacing:-.02em}
.ledger .v small{font:500 13px var(--mono);color:var(--muted);margin-left:6px;letter-spacing:0}
.ledger .l{margin-top:10px}
.ledger .s{font-size:12.5px;color:var(--muted);margin-top:6px;line-height:1.4}

/* ---------- section headers ---------- */
.sec{display:grid;grid-template-columns:120px 1fr auto;gap:24px;align-items:end;padding:64px 0 18px;border-bottom:1px solid var(--line);margin-bottom:22px}
@media (max-width:800px){.sec{grid-template-columns:1fr;gap:6px;padding-top:44px}}
.sec .no{font:400 15px var(--mono);color:var(--brass)}
.sec h2{font:400 clamp(32px,3.6vw,48px)/1 var(--serif)!important;letter-spacing:-.02em;margin:0;padding:0}
.sec .aside{font-size:13.5px;color:var(--muted);max-width:420px;text-align:right}
@media (max-width:800px){.sec .aside{text-align:left}}

/* ---------- panels ---------- */
.panel{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:22px 24px;box-shadow:var(--shadow)}
.panel h3{font:400 26px/1.1 var(--serif)!important;margin:6px 0 10px;padding:0}
.panel .body{font-size:14.5px;line-height:1.65;color:var(--ink-2)}
.empty{border:1px dashed var(--line-2);border-radius:18px;padding:28px;background:repeating-linear-gradient(135deg,transparent 0 10px,rgba(11,31,58,.018) 10px 20px)}
.empty h4{font:400 24px var(--serif);margin:8px 0 6px}
.empty p{font-size:14px;color:var(--muted);margin:0;line-height:1.6}
.empty code{font:500 12px var(--mono);background:var(--paper-2);padding:2px 6px;border-radius:5px;color:var(--ink)}

/* ---------- monument cards ---------- */
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:22px}
@media (max-width:1100px){.grid{grid-template-columns:repeat(3,1fr)}}
@media (max-width:820px){.grid{grid-template-columns:repeat(2,1fr)}}
@media (max-width:520px){.grid{grid-template-columns:1fr}}
a.mc{display:block;text-decoration:none!important;color:var(--ink)!important;background:var(--card);border:1px solid var(--line);
  border-radius:18px;overflow:hidden;box-shadow:0 1px 2px rgba(11,31,58,.04);transition:transform .25s cubic-bezier(.2,.7,.2,1),box-shadow .25s,border-color .25s}
a.mc:hover{transform:translateY(-4px);box-shadow:var(--shadow);border-color:var(--line-2)}
.mc .ph{position:relative;aspect-ratio:4/3;overflow:hidden;background:var(--paper-2)}
.mc .ph img{width:100%;height:100%;object-fit:cover;transition:transform .6s cubic-bezier(.2,.7,.2,1);filter:saturate(.9)}
a.mc:hover .ph img{transform:scale(1.045)}
.mc .no{position:absolute;left:12px;top:12px;font:500 11px var(--mono);color:#fff;background:rgba(11,31,58,.72);backdrop-filter:blur(6px);padding:5px 8px;border-radius:7px;letter-spacing:.06em}
.mc .st{position:absolute;right:12px;top:12px}
.mc .st .pill{background:rgba(255,255,255,.9);font-size:10px;padding:6px 9px}
.mc .tx{padding:16px 18px 18px}
.mc .nm{font:400 25px/1.05 var(--serif);letter-spacing:-.01em}
.mc .loc{font:400 11px var(--mono);color:var(--muted);margin-top:7px;letter-spacing:.04em;text-transform:uppercase}
.mc .ft{display:flex;justify-content:space-between;align-items:center;margin-top:16px;padding-top:13px;border-top:1px solid var(--line);font-size:12.5px;color:var(--muted)}
.mc .go{font-weight:600;color:var(--ink)}
.mc .go span{display:inline-block;transition:transform .25s}
a.mc:hover .go span{transform:translateX(4px)}
.noimg{width:100%;height:100%;display:grid;place-items:center;font:400 40px var(--serif);color:var(--faint)}

/* ---------- condition matrix ---------- */
.matrix{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}
@media (max-width:900px){.matrix{grid-template-columns:repeat(4,1fr)}}
.mx{border:1px solid var(--line);border-radius:12px;padding:10px 11px;background:var(--card);text-decoration:none!important;color:var(--ink)!important;display:block}
.mx:hover{border-color:var(--ink)}
.mx .n{font-size:12.5px;font-weight:600;line-height:1.2;min-height:30px}
.mx .c{font:500 10px var(--mono);color:var(--muted);margin-top:8px;display:flex;align-items:center;gap:6px;text-transform:uppercase;letter-spacing:.06em}

/* ---------- activity ---------- */
.act{display:grid;grid-template-columns:110px 1fr auto;gap:14px;padding:13px 0;border-bottom:1px solid var(--line);font-size:13.5px;align-items:center}
.act:last-child{border-bottom:none}
.act .t{font:400 11px var(--mono);color:var(--muted)}

/* ---------- monument page ---------- */
.mhero{position:relative;height:min(62vh,560px);border-radius:0 0 26px 26px;overflow:hidden;margin:0 -2.2rem;background:var(--ink)}
@media (max-width:700px){.mhero{margin:0 -1rem}}
.mhero img{width:100%;height:100%;object-fit:cover;opacity:.82}
.mhero .shade{position:absolute;inset:0;background:linear-gradient(180deg,rgba(11,31,58,.25) 0%,rgba(11,31,58,0) 30%,rgba(11,31,58,.82) 100%)}
.mhero .in{position:absolute;left:0;right:0;bottom:0;padding:0 2.6rem 34px;max-width:1320px;margin:0 auto}
.mhero .back{position:absolute;left:2.2rem;top:22px}
.mhero .back a{color:#fff!important;text-decoration:none;font:500 12px var(--mono);letter-spacing:.1em;background:rgba(255,255,255,.14);backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,.25);padding:9px 14px;border-radius:999px}
.mhero h1{font:400 clamp(52px,8vw,118px)/.9 var(--serif)!important;color:#fff!important;letter-spacing:-.03em;margin:10px 0 12px;padding:0}
.mhero .tl{color:rgba(255,255,255,.88);font-size:17px;max-width:640px}
.mhero .kicker{color:rgba(255,255,255,.75)}
.mhero .credit{position:absolute;right:2.2rem;bottom:14px}
.facts{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid var(--line)}
@media (max-width:900px){.facts{grid-template-columns:repeat(2,1fr)}}
.facts>div{padding:20px 18px 18px 0;border-right:1px solid var(--line);margin-right:18px}
.facts>div:last-child{border-right:none}
.facts .val{font:400 21px/1.2 var(--serif);margin-top:9px}
.chapters{display:flex;gap:8px;flex-wrap:wrap;padding:20px 0 4px}
.chapters a{text-decoration:none!important;font:500 11.5px var(--mono);letter-spacing:.06em;color:var(--ink)!important;border:1px solid var(--line-2);
  padding:8px 12px;border-radius:999px;background:var(--card)}
.chapters a:hover{background:var(--ink);color:#fff!important;border-color:var(--ink)}
.chapters a b{color:var(--brass);font-weight:500;margin-right:6px}
.prose{font-size:15.5px;line-height:1.75;color:var(--ink-2)}
.prose p{margin:0 0 14px}
.prose p:first-child::first-letter{font:400 64px/.8 var(--serif);float:left;margin:6px 10px 0 0;color:var(--ink)}
.kv{display:grid;grid-template-columns:130px 1fr;gap:10px 16px;font-size:13.5px}
.kv .k{font:500 10.5px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);padding-top:3px}
.imgmeta{font-size:12.5px;color:var(--muted);line-height:1.55;margin-top:10px}
.imgmeta b{color:var(--ink);font-weight:600}
.count-row{display:flex;gap:8px;flex-wrap:wrap}
.cc{display:flex;align-items:center;gap:8px;border:1px solid var(--line);background:var(--card);border-radius:10px;padding:8px 11px;font-size:12.5px}
.cc .n{font:500 13px var(--mono)}

/* ---------- streamlit widgets ---------- */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{background:var(--ink)!important;color:#fff!important;border:1px solid var(--ink)!important;
  border-radius:999px!important;padding:.6rem 1.2rem!important;font-weight:600!important;letter-spacing:.01em;transition:all .2s}
.stButton>button p,.stDownloadButton>button p{color:#fff!important}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--ink-2)!important;transform:translateY(-1px)}
.stButton>button[kind="secondary"]{background:var(--card)!important;color:var(--ink)!important;border:1px solid var(--line-2)!important}
.stButton>button[kind="secondary"] p{color:var(--ink)!important}
[data-baseweb="select"]>div,.stTextInput input,.stDateInput input,.stNumberInput input,textarea{border-radius:12px!important;border-color:var(--line-2)!important;background:var(--card)!important}
[data-testid="stFileUploaderDropzone"]{border-radius:16px;background:var(--card);border:1px dashed var(--line-2)}
[data-testid="stWidgetLabel"] p{font:500 11px var(--mono)!important;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)!important}
[data-testid="stImage"] img{border-radius:14px}
[data-testid="stImageCaption"]{font:400 11px var(--mono)!important;color:var(--muted)!important}
[data-testid="stChatMessage"]{background:var(--card);border:1px solid var(--line);border-radius:16px}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:14px!important;background:var(--card)}
[data-testid="stVerticalBlockBorderWrapper"]:has(>div>[data-testid="stVerticalBlock"]) {border-radius:18px}
.stAlert{border-radius:14px}
[data-testid="stMetricValue"]{font-family:var(--serif)!important}
</style>
"""


def apply() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
