"""Presentation helpers for the Streamlit app: design tokens and small HTML components.

Design rules
  - Red corner / blue corner are the only accent colours, and they always mean
    "fighter A" / "fighter B". Everything else is ink, grey, or a hairline rule.
  - Barlow / Barlow Condensed (a sports-broadcast face) with tabular numbers so
    odds and percentages line up.
  - Hairline rules and alignment instead of cards, shadows and rounded boxes.
"""
from __future__ import annotations

from html import escape

RED = "#c8102e"
BLUE = "#1d4f91"
INK = "#16181d"
MUTED = "#6b7079"
RULE = "#e2e4e8"
BG = "#fbfbf9"
GOOD = "#1e7b4a"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');

:root {{ --cond:'Barlow Condensed','Arial Narrow','Roboto Condensed',sans-serif; --red:{RED}; --blue:{BLUE}; --ink:{INK}; --muted:{MUTED}; --rule:{RULE}; --good:{GOOD}; }}
.stApp {{ background:{BG}; color:var(--ink); }}
html, body, .stApp, .stMarkdown, button, input, textarea, [data-baseweb] {{
  font-family:'Barlow', system-ui, sans-serif; }}
.block-container {{ padding-top:1.6rem; padding-bottom:3rem; max-width:1120px; }}
header[data-testid="stHeader"] {{ background:transparent; }}
h1, h2, h3, h4 {{ font-family:var(--cond) !important; letter-spacing:.01em; color:var(--ink); }}
h3 {{ font-size:1.25rem !important; font-weight:600 !important; margin:1.4rem 0 .4rem !important; padding:0 !important; }}
.num {{ font-variant-numeric:tabular-nums; }}

/* masthead */
.oo-mast {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; flex-wrap:wrap;
  border-bottom:2px solid var(--ink); padding-bottom:.35rem; margin-bottom:.9rem; }}
.oo-mast .brand {{ font-family:var(--cond); font-weight:700; font-size:1.7rem; text-transform:uppercase; letter-spacing:.04em; }}
.oo-mast .meta {{ color:var(--muted); font-size:.85rem; }}

/* widget labels */
[data-testid="stWidgetLabel"] p {{ font-family:var(--cond); text-transform:uppercase; letter-spacing:.06em;
  font-size:.78rem !important; color:var(--muted); font-weight:600; }}

/* head to head */
.oo-h2h {{ margin:.6rem 0 .2rem; }}
.oo-h2h .row {{ display:grid; grid-template-columns:1fr auto 1fr; align-items:end; gap:.75rem; }}
.oo-h2h .side.b {{ text-align:right; }}
.oo-h2h .corner {{ font-family:var(--cond); font-size:.75rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }}
.oo-h2h .a .corner {{ color:var(--red); }} .oo-h2h .b .corner {{ color:var(--blue); }}
.oo-h2h .name {{ font-family:var(--cond); font-weight:700; font-size:clamp(1.25rem, 3.4vw, 2rem); line-height:1.05; }}
.oo-h2h .pct {{ font-family:var(--cond); font-weight:600; font-size:clamp(2.2rem, 7vw, 3.6rem); line-height:1; font-variant-numeric:tabular-nums; }}
.oo-h2h .a .pct {{ color:var(--red); }} .oo-h2h .b .pct {{ color:var(--blue); }}
.oo-h2h .line {{ color:var(--muted); font-size:.9rem; font-variant-numeric:tabular-nums; }}
.oo-h2h .line b {{ color:var(--ink); font-weight:600; }}
.oo-h2h .mid {{ color:var(--muted); font-family:var(--cond); padding-bottom:.6rem; }}
.oo-bar {{ display:flex; height:8px; margin:.7rem 0 .55rem; }}
.oo-bar span {{ display:block; height:100%; }}
.oo-verdict {{ font-size:1.02rem; margin:.1rem 0 .2rem; }}
.oo-note {{ color:var(--muted); font-size:.88rem; margin:.15rem 0; }}

/* tables (override Streamlit's markdown table borders) */
.stMarkdown table.oo, .stMarkdown table.oo th, .stMarkdown table.oo td {{ border-left:none !important; border-right:none !important;
  border-top:none !important; background:transparent !important; }}
.stMarkdown table.oo tr {{ background:transparent !important; }}
table.oo {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:.95rem; margin:.2rem 0 .6rem; }}
table.oo th {{ border-bottom:1px solid var(--ink) !important; font-family:var(--cond); font-weight:600; text-transform:uppercase; letter-spacing:.06em; font-size:.75rem;
  color:var(--muted); text-align:right; padding:.35rem .5rem; border-bottom:1px solid var(--ink); }}
table.oo th:first-child, table.oo td:first-child {{ text-align:left; padding-left:0; }}
table.oo td {{ padding:.42rem .5rem; border-bottom:1px solid var(--rule) !important; text-align:right; }}
table.oo td.lbl {{ color:var(--ink); }}
table.oo .tl {{ text-align:left !important; }}
table.oo td.muted, table.oo .muted {{ color:var(--muted); }}
table.oo td.fav {{ font-weight:600; }}
table.oo tr.sub td {{ color:var(--muted); font-size:.85rem; }}
.oo-cap {{ color:var(--muted); font-size:.82rem; margin:-.3rem 0 .8rem; }}

/* tale of the tape */
table.tape td {{ width:30%; }}
table.tape td.lbl {{ width:40%; text-align:center !important; color:var(--muted); font-size:.88rem; }}
table.tape td.va {{ text-align:left !important; padding-left:0; }}
table.tape td.vb {{ text-align:right !important; }}
table.tape td.win.va {{ color:var(--red); font-weight:600; }}
table.tape td.win.vb {{ color:var(--blue); font-weight:600; }}
table.tape th.ha {{ color:var(--red); text-align:left; padding-left:0; }}
table.tape th.hb {{ color:var(--blue); text-align:right; }}
table.tape th.hl {{ text-align:center; }}

/* inline bars */
.cellbar {{ display:inline-block; height:6px; vertical-align:middle; margin-left:.45rem; }}
.factor {{ display:grid; grid-template-columns:1fr 2.2fr 1fr; align-items:center; gap:.6rem; padding:.32rem 0; border-bottom:1px solid var(--rule); font-size:.93rem; }}
.factor .l {{ text-align:right; color:var(--muted); }}
.factor .track {{ position:relative; height:10px; }}
.factor .track::before {{ content:''; position:absolute; left:50%; top:-4px; bottom:-4px; border-left:1px solid #b9bcc2; }}
.factor .fill {{ position:absolute; top:0; height:100%; }}
.factor .r {{ color:var(--muted); }}
.factor-head {{ display:grid; grid-template-columns:1fr 2.2fr 1fr; gap:.6rem; font-family:var(--cond); font-size:.75rem;
  text-transform:uppercase; letter-spacing:.06em; font-weight:600; padding-bottom:.3rem; border-bottom:1px solid var(--ink); }}
.res-W {{ color:var(--good); font-weight:600; }} .res-L {{ color:var(--muted); font-weight:600; }}

/* page titles */
h2.oo-page {{ font-size:1.9rem !important; font-weight:700 !important; margin:.2rem 0 .2rem !important; padding:0 !important; }}
.oo-lede {{ font-size:1.05rem; max-width:46rem; margin-bottom:.6rem; }}

/* navigation row under the masthead (Streamlit marks the current page with a grey pill) */
.st-key-oo_nav {{ gap:.4rem !important; margin:-.5rem 0 1rem; }}
.st-key-oo_nav [data-testid="stPageLink-NavLink"] {{ border-radius:2px; padding:.2rem .6rem; }}
.st-key-oo_nav [data-testid="stPageLink-NavLink"] p {{ font-family:var(--cond) !important; text-transform:uppercase;
  letter-spacing:.07em; font-weight:600 !important; font-size:.95rem !important; }}

/* about */
.oo-about-name {{ font-family:var(--cond); font-weight:700; font-size:2.2rem; line-height:1; margin-top:1.2rem; }}
.oo-about-role {{ color:var(--muted); font-size:.95rem; margin:.35rem 0 1rem; padding-bottom:.8rem; border-bottom:2px solid var(--ink); }}
.oo-links a {{ color:var(--ink); font-weight:600; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:1.4rem; border-bottom:1px solid var(--rule); }}
.stTabs [data-baseweb="tab"] {{ padding:.5rem 0; }}
.stTabs [role="tab"] p {{ font-family:var(--cond) !important; text-transform:uppercase; letter-spacing:.06em; font-weight:600;
  font-size:1rem; color:var(--muted); }}
.stTabs [role="tab"][aria-selected="true"] p {{ color:var(--ink) !important; }}
.stTabs [aria-selected="true"] {{ color:var(--ink) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--ink) !important; }}
[data-testid="stExpander"] details {{ border:none; border-top:1px solid var(--rule); border-radius:0; }}
[data-testid="stExpander"] summary p {{ font-weight:500; }}

@media (max-width: 640px) {{
  .block-container {{ padding-left:1rem; padding-right:1rem; }}
  .oo-h2h .mid {{ display:none; }}
  .oo-h2h .row {{ grid-template-columns:1fr 1fr; }}
  .factor, .factor-head {{ grid-template-columns:1fr 1.6fr 1fr; font-size:.85rem; }}
  table.oo {{ font-size:.88rem; }}
}}
</style>
"""


def e(x) -> str:
    return escape(str(x))


def masthead(updated: str) -> str:
    return (f'<div class="oo-mast"><span class="brand">Octagon Odds</span>'
            f'<span class="meta">UFC results through {e(updated)}</span></div>')


def head_to_head(a, b, pa, pb, line_a, line_b) -> str:
    return f"""
<div class="oo-h2h">
  <div class="row">
    <div class="side a"><div class="corner">Red corner</div><div class="name">{e(a)}</div>
      <div class="pct">{pa:.0%}</div><div class="line">Line <b>{e(line_a)}</b></div></div>
    <div class="mid">vs</div>
    <div class="side b"><div class="corner">Blue corner</div><div class="name">{e(b)}</div>
      <div class="pct">{pb:.0%}</div><div class="line">Line <b>{e(line_b)}</b></div></div>
  </div>
  <div class="oo-bar"><span style="width:{pa*100:.1f}%;background:{RED}"></span><span style="width:{pb*100:.1f}%;background:{BLUE}"></span></div>
</div>"""


def table(headers: list[str], rows: list[list], cls: str = "oo", row_cls: list[str] | None = None,
          left: int = 1) -> str:
    """rows: lists of cell HTML strings (already escaped) or (html, css_class) tuples.

    The first `left` columns are left-aligned (labels); the rest are right-aligned (numbers).
    """
    th = "".join(f'<th{" class=tl" if j < left else ""}>{h}</th>' for j, h in enumerate(headers))
    body = []
    for i, r in enumerate(rows):
        tds = []
        for j, c in enumerate(r):
            html, klass = (c if isinstance(c, tuple) else (c, "lbl" if j == 0 else ""))
            if j < left:
                klass += " tl"
            tds.append(f'<td class="{klass}">{html}</td>')
        rc = f' class="{row_cls[i]}"' if row_cls and row_cls[i] else ""
        body.append(f"<tr{rc}>{''.join(tds)}</tr>")
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def tape(a: str, b: str, rows: list[tuple[str, str, str, int]]) -> str:
    """Tale-of-the-tape table. rows: (label, value_a, value_b, edge) with edge 1=A, -1=B, 0=none."""
    out = [f'<table class="oo tape"><thead><tr><th class="ha">{e(a)}</th><th class="hl"></th>'
           f'<th class="hb">{e(b)}</th></tr></thead><tbody>']
    for label, va, vb, edge in rows:
        ca = "va win" if edge == 1 else "va"
        cb = "vb win" if edge == -1 else "vb"
        out.append(f'<tr><td class="{ca}">{va}</td><td class="lbl">{e(label)}</td><td class="{cb}">{vb}</td></tr>')
    out.append("</tbody></table>")
    return "".join(out)


def surname(name: str) -> str:
    parts = str(name).split()
    return parts[-1] if parts else str(name)


def factors(a: str, b: str, items: list[tuple[str, float]]) -> str:
    """Diverging bars: positive values push toward A (red, right), negative toward B (blue, left)."""
    if not items:
        return ""
    a, b = surname(a), surname(b)
    peak = max(abs(v) for _, v in items) or 1
    rows = [f'<div class="factor-head"><span></span><span style="display:flex">'
            f'<span style="width:50%;text-align:right;padding-right:.5rem;color:{BLUE}">← {e(b)}</span>'
            f'<span style="width:50%;padding-left:.5rem;color:{RED}">{e(a)} →</span></span><span></span></div>']
    for label, v in items:
        w = abs(v) / peak * 50
        if v >= 0:
            fill = f'<span class="fill" style="left:50%;width:{w:.1f}%;background:{RED}"></span>'
            l, r = "", e(label)
        else:
            fill = f'<span class="fill" style="right:50%;width:{w:.1f}%;background:{BLUE}"></span>'
            l, r = e(label), ""
        rows.append(f'<div class="factor"><span class="l">{l}</span><span class="track">{fill}</span><span class="r">{r}</span></div>')
    return "".join(rows)


def bar(p: float, color: str, scale: float = 1.0) -> str:
    return f'<span class="cellbar" style="width:{max(p * scale, 0) * 70:.0f}px;background:{color}"></span>'


def pct(p: float, digits: int = 0) -> str:
    return f"{p:.{digits}%}"


def height_str(inches) -> str:
    try:
        i = int(round(float(inches)))
    except (TypeError, ValueError):
        return "—"
    return f"{i // 12}′{i % 12}″"


def style_chart(chart):
    """Apply the app's chart styling (works across Altair versions)."""
    return (chart.configure(font="Barlow", background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor=RULE, tickColor=RULE,
                            gridColor="#eef0f2", labelFontSize=12, titleFontSize=12, titleFontWeight=500)
            .configure_legend(labelColor=INK, labelFontSize=12, orient="top", symbolType="square"))
