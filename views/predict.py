"""Predict page: pick two fighters, get the win chance, method, round, prices and résumé."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from mmapred import backend as B
from mmapred import ui
from mmapred.odds import (american_to_prob, expected_value, fmt_american, markets, price_board,
                          prob_to_american, remove_margin)
from views.common import get_predictor, load_metrics, film_for

MARGINS = {"Fair price (no margin)": 0.0, "Sharp book (2.5%)": 0.025,
           "Typical book (4.5%)": 0.045, "Recreational book (7%)": 0.07}
for k, v in {"margin": "Typical book (4.5%)", "odds_fmt": "American", "vig_method": "Power",
             "n_sims": 100_000}.items():
    st.session_state.setdefault(k, v)

pr = get_predictor()
metrics = load_metrics()
idx = pr.index
t = idx.table

# ------------------------------------------------------------------ matchup picker
active_names = idx.names(min_fights=2, active_since="2024-01-01")
with st.container():
    c_div, c_a, c_b, c_r = st.columns([1.1, 1.5, 1.5, 0.9])
    divisions = sorted(t.loc[active_names, "weightclass"].dropna().unique())
    div_opts = ["All divisions"] + divisions
    saved = st.session_state.get("matchup", {})
    div_default = saved.get("division", "Lightweight")
    division = c_div.selectbox("Division", div_opts,
                               index=div_opts.index(div_default) if div_default in div_opts else 0)
    include_old = st.session_state.get("include_old_saved", False)
    names = idx.names(min_fights=1) if include_old else active_names
    pool = [n for n in names if division == "All divisions" or t.loc[n, "weightclass"] == division]
    pool = sorted(pool, key=lambda n: -t.loc[n, "elo"])  # best-rated first
    if len(pool) < 2:
        st.warning("Not enough fighters in this division. Pick another division or include inactive fighters.")
        st.stop()
    a = c_a.selectbox("Red corner", pool, index=pool.index(saved["a"]) if saved.get("a") in pool else 0,
                      help="Type to search. Sorted by current rating.")
    pool_b = [n for n in pool if n != a]
    b = c_b.selectbox("Blue corner", pool_b, index=pool_b.index(saved["b"]) if saved.get("b") in pool_b else 0,
                      help="Type to search.")
    rounds = c_r.radio("Rounds", [3, 5], index=1 if saved.get("rounds") == 5 else 0, horizontal=True,
                       help="Five rounds for main events and title fights.")
    st.session_state["matchup"] = {"division": division, "a": a, "b": b, "rounds": rounds}
    if st.checkbox("Include fighters inactive since 2024", value=include_old) != include_old:
        st.session_state["include_old_saved"] = not include_old
        st.rerun()

# ------------------------------------------------------------------ model + markets
with st.spinner("Simulating the fight…"):
    out, A, Bf = pr.predict(a, b, rounds=rounds, n_sims=st.session_state["n_sims"])
margin = MARGINS[st.session_state["margin"]]
method = st.session_state["vig_method"].lower()
board = price_board(markets(out, a, b, rounds), margin, method)


def price(row: dict) -> str:
    if st.session_state["odds_fmt"] == "Decimal":
        return f"{row['Decimal']:.2f}"
    return row["Book odds"]


ml = {r["Outcome"]: r for r in board["Moneyline"]}
st.markdown(ui.head_to_head(a, b, out["p_a"], out["p_b"], price(ml[a]), price(ml[b])), unsafe_allow_html=True)

# one-line read of the fight
fav, p_fav = (a, out["p_a"]) if out["p_a"] >= out["p_b"] else (b, out["p_b"])
if p_fav < 0.55:
    lean = "Close to a coin flip."
elif p_fav < 0.65:
    lean = f"{fav} is a slight favourite."
elif p_fav < 0.78:
    lean = f"{fav} is the favourite."
else:
    lean = f"{fav} is a heavy favourite."
names_m = {"KO": "KO/TKO", "SUB": "submission", "DEC": "decision"}
best = max(((w, m_, v) for w, d in out["by_method"].items() for m_, v in d.items()), key=lambda x: x[2])
best_name = a if best[0] == "A" else b
st.markdown(
    f'<p class="oo-verdict">{ui.e(lean)} Most likely result: <b>{ui.e(best_name)} by {names_m[best[1]]}</b> '
    f'({best[2]:.0%}). Goes to the judges {out["p_distance"]:.0%} of the time.</p>',
    unsafe_allow_html=True)

notes = []
for name, row in ((a, A), (b, Bf)):
    n = int(row["n_prior"].iloc[0])
    if n < 3:
        notes.append(f"{name} has only {n} UFC fight{'s' if n != 1 else ''}, so their numbers lean on league averages.")
mp = out["models"]
spread = max(mp[k] for k in ("Logistic regression", "XGBoost", "Glicko-2")) - \
    min(mp[k] for k in ("Logistic regression", "XGBoost", "Glicko-2"))
if spread > 0.2:
    notes.append("The models disagree on this one. See Method for each model's call.")
for n_ in notes:
    st.markdown(f'<p class="oo-note">{ui.e(n_)}</p>', unsafe_allow_html=True)

tab_fight, tab_odds, tab_resume, tab_method = st.tabs(["Breakdown", "Odds", "Résumé", "Method"])

# ================================================================== BREAKDOWN
with tab_fight:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("### How it ends")
        rows, tot = [], {"A": 0, "B": 0}
        for key, label in (("KO", "KO/TKO"), ("SUB", "Submission"), ("DEC", "Decision")):
            pa_, pb_ = out["by_method"]["A"][key], out["by_method"]["B"][key]
            rows.append([label, (f"{pa_:.0%}{ui.bar(pa_, ui.RED)}", ""), (f"{pb_:.0%}{ui.bar(pb_, ui.BLUE)}", "")])
        rows.append(["Total", (f"{out['p_a']:.0%}", "fav" if out["p_a"] >= .5 else ""),
                     (f"{out['p_b']:.0%}", "fav" if out["p_b"] > .5 else "")])
        st.markdown(ui.table(["", ui.e(a), ui.e(b)], rows), unsafe_allow_html=True)
    with right:
        st.markdown("### When it ends")
        dec = {w: out["by_method"][w]["DEC"] for w in "AB"}
        rows = []
        for w, name in (("A", a), ("B", b)):
            for i in range(rounds):
                v = out["by_round"][w][i] - (dec[w] if i == rounds - 1 else 0)
                rows.append({"when": f"R{i + 1}", "fighter": name, "p": v, "o": 0 if w == "A" else 1})
            rows.append({"when": "Decision", "fighter": name, "p": dec[w], "o": 0 if w == "A" else 1})
        order = [f"R{i + 1}" for i in range(rounds)] + ["Decision"]
        chart = alt.Chart(pd.DataFrame(rows)).mark_bar(size=34).encode(
            x=alt.X("when:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("p:Q", stack=True, title=None, axis=alt.Axis(format="%", tickCount=4)),
            color=alt.Color("fighter:N", scale=alt.Scale(domain=[a, b], range=[ui.RED, ui.BLUE]), title=None),
            order=alt.Order("o:Q"),
            tooltip=[alt.Tooltip("fighter:N", title="Winner"), alt.Tooltip("when:N", title="When"),
                     alt.Tooltip("p:Q", title="Chance", format=".1%")],
        ).properties(height=230)
        st.altair_chart(ui.style_chart(chart), width="stretch")

    st.markdown("### Why the model leans this way")
    drivers = list(out["drivers"].items())[:7]
    st.markdown(ui.factors(a, b, drivers), unsafe_allow_html=True)

    def edge(va, vb, higher_better=True, eps=1e-9):
        if va is None or vb is None or pd.isna(va) or pd.isna(vb) or abs(va - vb) <= eps:
            return 0
        return 1 if (va > vb) == higher_better else -1

    # ---- film study (Research page)
    fa_, fb_ = film_for(a), film_for(b)
    if fa_ or fb_:
        st.markdown("### From your film study")

        def agg(profiles, key):
            vals = [(p_.get(key), p_.get("seconds") or 0) for p_ in profiles if p_.get(key) is not None]
            w = sum(s_ for _, s_ in vals)
            return sum(v * s_ for v, s_ in vals) / w if w else None

        def stance_of(profiles):
            tot = {}
            for p_ in profiles:
                for k_, v_ in (p_.get("stance") or {}).items():
                    tot[k_] = tot.get(k_, 0) + v_ * (p_.get("seconds") or 0)
            return max(tot, key=tot.get) if tot else "—"

        def fmt(v, f):
            return "—" if v is None else f.format(v)

        film_rows = [("Clips studied", f"{len(fa_)} ({sum(p_.get('seconds') or 0 for p_ in fa_) / 60:.1f} min)",
                      f"{len(fb_)} ({sum(p_.get('seconds') or 0 for p_ in fb_) / 60:.1f} min)", 0),
                     ("Stance on film", stance_of(fa_), stance_of(fb_), 0)]
        for label_, key_, f_ in (("Moving forward", "advancing", "{:.0%}"), ("Hands up", "guard_high", "{:.0%}"),
                                 ("Punch attempts / min", "punches_pm", "{:.1f}"), ("Kick attempts / min", "kicks_pm", "{:.1f}"),
                                 ("Level changes / min", "level_changes_pm", "{:.1f}")):
            va_, vb_ = agg(fa_, key_), agg(fb_, key_)
            film_rows.append((label_, fmt(va_, f_), fmt(vb_, f_), edge(va_, vb_, eps=.01) if va_ is not None and vb_ is not None else 0))
        st.markdown(ui.tape(a, b, film_rows), unsafe_allow_html=True)
        st.markdown('<p class="oo-cap">Measured from the clips you tracked on the Research page. Shown for '
                    'context; the odds above come from the models only.</p>', unsafe_allow_html=True)
    else:
        st.page_link("views/research.py", label=f"Track {ui.surname(a)} and {ui.surname(b)} from fight footage →")

    st.markdown("### Tale of the tape")

    ra, rb = A.iloc[0], Bf.iloc[0]
    wa, la, da = idx.record(a)
    wb, lb, db = idx.record(b)
    fa, fb = idx.fighters.loc[a] if a in idx.fighters.index else None, idx.fighters.loc[b] if b in idx.fighters.index else None
    rec = lambda w, l, d: f"{w}–{l}" + (f"–{d}" if d else "")  # noqa: E731
    form = lambda n: " ".join(f'<span class="res-{r}">{r}</span>' for r in idx.recent_form(n))  # noqa: E731
    num = lambda v, f="{:.1f}": "—" if pd.isna(v) else f.format(v)  # noqa: E731
    tape_rows = [
        ("UFC record", rec(wa, la, da), rec(wb, lb, db), 0),
        ("Last 3, newest first", form(a), form(b), 0),
        ("Rating (Elo)", num(ra["elo"], "{:.0f}"), num(rb["elo"], "{:.0f}"), edge(ra["elo"], rb["elo"], eps=5)),
        ("Age", num(ra["age"], "{:.0f}"), num(rb["age"], "{:.0f}"), 0),
        ("Height", ui.height_str(ra["height"]), ui.height_str(rb["height"]), 0),
        ("Reach", num(ra["reach"], "{:.0f}″"), num(rb["reach"], "{:.0f}″"), edge(ra["reach"], rb["reach"], eps=.5)),
        ("Stance", ui.e(fa["stance"]) if fa is not None else "—", ui.e(fb["stance"]) if fb is not None else "—", 0),
        ("Sig. strikes landed / min", num(ra["slpm"], "{:.2f}"), num(rb["slpm"], "{:.2f}"), edge(ra["slpm"], rb["slpm"], eps=.05)),
        ("Sig. strikes absorbed / min", num(ra["sapm"], "{:.2f}"), num(rb["sapm"], "{:.2f}"),
         edge(ra["sapm"], rb["sapm"], higher_better=False, eps=.05)),
        ("Striking accuracy", ui.pct(ra["sig_acc"]), ui.pct(rb["sig_acc"]), edge(ra["sig_acc"], rb["sig_acc"], eps=.005)),
        ("Striking defence", ui.pct(ra["sig_def"]), ui.pct(rb["sig_def"]), edge(ra["sig_def"], rb["sig_def"], eps=.005)),
        ("Takedowns / 15 min", num(ra["td15"], "{:.2f}"), num(rb["td15"], "{:.2f}"), edge(ra["td15"], rb["td15"], eps=.05)),
        ("Takedown defence", ui.pct(ra["td_def"]), ui.pct(rb["td_def"]), edge(ra["td_def"], rb["td_def"], eps=.005)),
        ("Submission attempts / 15 min", num(ra["sub15"], "{:.2f}"), num(rb["sub15"], "{:.2f}"),
         edge(ra["sub15"], rb["sub15"], eps=.05)),
    ]
    st.markdown(ui.tape(a, b, tape_rows), unsafe_allow_html=True)
    st.markdown('<p class="oo-cap">Rates are per UFC fight time. Fighters with few UFC fights are pulled toward the '
                'league average. The better number is highlighted.</p>', unsafe_allow_html=True)

# ================================================================== ODDS
with tab_odds:
    s1, s2, s3 = st.columns([1, 1.4, 2])
    s1.radio("Odds format", ["American", "Decimal"], key="odds_fmt", horizontal=True)
    s2.selectbox("Pricing", list(MARGINS), key="margin",
                 help="How much margin a sportsbook would add on top of the fair price.")

    def cell(row):
        return f'{price(row)} <span class="muted">{row["Fair %"]:.0%}</span>'

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Moneyline")
        st.markdown(ui.table(["", "Price", "Fair price", "Chance"], [
            [ui.e(r["Outcome"]), (price(r), "fav" if r["Fair %"] >= .5 else ""), (r["Fair odds"], "muted"),
             (f'{r["Fair %"]:.1%}', "muted")] for r in board["Moneyline"]]), unsafe_allow_html=True)

        st.markdown("### Total rounds")
        tot_rows = []
        for k in [k for k in board if k.startswith("Total rounds")]:
            over, under = board[k]
            tot_rows.append([k.replace("Total rounds ", ""), (cell(over), ""), (cell(under), "")])
        st.markdown(ui.table(["Rounds", "Over", "Under"], tot_rows), unsafe_allow_html=True)
        st.markdown('<p class="oo-cap">Over 2.5 means the fight is still going halfway through round 3.</p>',
                    unsafe_allow_html=True)
        dist = {r["Outcome"]: r for r in board["Goes the distance"]}
        st.markdown("### Goes the distance")
        st.markdown(ui.table(["", "Price", "Chance"], [
            [k, (price(dist[k]), ""), (f'{dist[k]["Fair %"]:.0%}', "muted")] for k in ("Yes", "No")]),
            unsafe_allow_html=True)
    with right:
        mov = {r["Outcome"]: r for r in board["Method of victory"]}
        st.markdown("### Method of victory")
        st.markdown(ui.table(["", ui.e(a), ui.e(b)], [
            [lab, (cell(mov[f"{a} by {k}"]), ""), (cell(mov[f"{b} by {k}"]), "")]
            for lab, k in (("KO/TKO", "KO/TKO"), ("Submission", "submission"), ("Decision", "decision"))]),
            unsafe_allow_html=True)
        rb_ = {r["Outcome"]: r for r in board["Round betting"]}
        st.markdown("### Round betting")
        rows = [[f"Round {i}", (cell(rb_[f"{a} in round {i}"]), ""), (cell(rb_[f"{b} in round {i}"]), "")]
                for i in range(1, rounds + 1)]
        rows.append(["Decision", (cell(rb_[f"{a} by decision"]), ""), (cell(rb_[f"{b} by decision"]), "")])
        st.markdown(ui.table(["", ui.e(a), ui.e(b)], rows), unsafe_allow_html=True)
    st.markdown(f'<p class="oo-cap">Grey figures are the model\'s fair chance. '
                f'{"Prices include no margin." if margin == 0 else f"Prices include a {margin:.1%} margin on two-way markets; markets with more outcomes carry more, as they do at real sportsbooks."}'
                f'</p>', unsafe_allow_html=True)
    with st.expander("How margin is applied"):
        st.radio("Margin method", ["Power", "Proportional"], key="vig_method", horizontal=True,
                 help="Power puts more margin on longshots, which is how most books price.")
        st.markdown("A sportsbook turns each fair chance into a price, then shades every price so the chances add up "
                    "to more than 100%. The extra is its margin. **Power** adds proportionally more to longshots, "
                    "matching how real markets price underdogs. **Proportional** adds the same percentage to every "
                    "outcome.")

    st.markdown("### Check a sportsbook's line")
    st.markdown('<p class="oo-note">Paste the moneyline from any book to see how it compares with this model.</p>',
                unsafe_allow_html=True)

    def parse_line(text: str):
        txt = text.strip().replace("−", "-").replace(" ", "")
        if not txt:
            return None, None
        try:
            v = int(float(txt))
        except ValueError:
            return None, "Use American odds, like -150 or +130."
        if -100 < v < 100:
            return None, "American odds are +100 or higher, or -100 or lower."
        return v, None

    i1, i2 = st.columns(2)
    la_txt = i1.text_input(f"{a}", placeholder="e.g. -150", key=f"line_a_{a}")
    lb_txt = i2.text_input(f"{b}", placeholder="e.g. +130", key=f"line_b_{b}")
    la, err_a = parse_line(la_txt)
    lb, err_b = parse_line(lb_txt)
    if err_a:
        i1.error(err_a)
    if err_b:
        i2.error(err_b)
    if la is not None and lb is not None:
        q = np.array([american_to_prob(la), american_to_prob(lb)])
        if q.sum() < 1:
            st.warning("Those prices add up to less than 100%, which a book wouldn't offer. Check the signs (+/-).")
        else:
            fair = remove_margin(q, method if method == "power" else "multiplicative")
            rows = []
            for name, p_mod, f_, line in ((a, out["p_a"], fair[0], la), (b, out["p_b"], fair[1], lb)):
                ev = expected_value(p_mod, line)
                ev_html = f'<span style="color:{ui.GOOD};font-weight:600">{ev:+.1%}</span>' if ev > 0 else f"{ev:+.1%}"
                rows.append([ui.e(name), (fmt_american(line), ""), (f"{f_:.1%}", "muted"), (f"{p_mod:.1%}", ""),
                             (f"{p_mod - f_:+.1%}", ""), (ev_html, "")])
            st.markdown(ui.table(["", "Their price", "Their fair chance", "Model", "Difference",
                                  "Expected return"], rows), unsafe_allow_html=True)
            st.markdown(f'<p class="oo-cap">Their margin on this fight: {q.sum() - 1:.1%}. A positive expected return '
                        f'only means the model disagrees with the market. Books also know about injuries, camps and '
                        f'weight cuts that this model can\'t see.</p>', unsafe_allow_html=True)
    elif not (err_a or err_b):
        st.caption("Enter both prices to compare.")

# ================================================================== RÉSUMÉ
with tab_resume:
    st.markdown('<p class="oo-note">Who they\'ve beaten, and how they did against the people in front of them. '
                'Opponent ratings are from the day of each fight.</p>', unsafe_allow_html=True)

    def best_win_str(name):
        bw = idx.best_win(name)
        if bw is None:
            return "—"
        return f'{ui.e(bw[0])} <span class="muted">({bw[2]:.0f}, {bw[1].year})</span>'

    sos_rows = [
        ("Average opponent rating", num(ra["sos_elo"], "{:.0f}"), num(rb["sos_elo"], "{:.0f}"),
         edge(ra["sos_elo"], rb["sos_elo"], eps=3)),
        ("Average rating of opponents beaten", num(ra["win_q"], "{:.0f}"), num(rb["win_q"], "{:.0f}"),
         edge(ra["win_q"], rb["win_q"], eps=3)),
        ("Wins over top-rated opponents (1600+)", num(ra["top_wins"], "{:.0f}"), num(rb["top_wins"], "{:.0f}"),
         edge(ra["top_wins"], rb["top_wins"], eps=.5)),
        ("Best win", best_win_str(a), best_win_str(b), 0),
        ("Strikes landed vs. what opponents usually allow", num(ra["adj_sl"], "{:+.2f}/min"),
         num(rb["adj_sl"], "{:+.2f}/min"), edge(ra["adj_sl"], rb["adj_sl"], eps=.05)),
        ("Strikes absorbed vs. what opponents usually land", num(ra["adj_sa"], "{:+.2f}/min"),
         num(rb["adj_sa"], "{:+.2f}/min"), edge(ra["adj_sa"], rb["adj_sa"], higher_better=False, eps=.05)),
        ("Takedowns vs. what opponents usually allow", num(ra["adj_td"], "{:+.2f}/15m"),
         num(rb["adj_td"], "{:+.2f}/15m"), edge(ra["adj_td"], rb["adj_td"], eps=.05)),
    ]
    st.markdown(ui.tape(a, b, sos_rows), unsafe_allow_html=True)
    st.markdown('<p class="oo-cap">Ratings start at 1500; title contenders sit around 1700–1850. For strikes absorbed, '
                'lower is better.</p>', unsafe_allow_html=True)

    def fight_list(name, color):
        h = idx.schedule_view(name, 8)
        if h.empty:
            return '<p class="oo-note">No UFC fights on record.</p>'
        rows = []
        for _, r in h.iterrows():
            res = r["Result"]
            klass = "res-W" if res == "W" else "res-L" if res == "L" else "muted"
            meth = {"KO": "KO/TKO", "SUB": "Sub", "DEC": "Dec", "INJ": "Injury"}.get(r["Method"], r["Method"])
            rnd = "" if pd.isna(r["Round"]) else f" R{int(r['Round'])}"
            rows.append([(f'<span class="{klass}">{res}</span> {ui.e(r["Opponent"])}', "lbl"),
                         (f'{ui.e(meth)}{rnd}', "muted"),
                         (num(r["Opp. Elo then"], "{:.0f}"), ""), (r["Date"].strftime("%b %Y"), "muted")])
        return (f'<h3 style="color:{color}">{ui.e(name)}</h3>'
                + ui.table(["Recent fights", "Result", "Opp. rating", "Date"], rows))

    l, r = st.columns(2, gap="large")
    l.markdown(fight_list(a, ui.RED), unsafe_allow_html=True)
    r.markdown(fight_list(b, ui.BLUE), unsafe_allow_html=True)

# ================================================================== METHOD
with tab_method:
    l, r = st.columns([1, 1.2], gap="large")
    with l:
        st.markdown(f"### Each model's call")
        order = ["Stacked ensemble", "Logistic regression", "XGBoost", "Glicko-2", "Elo"]
        label = {"Stacked ensemble": "Ensemble (used for prices)"}
        st.markdown(ui.table(["", f"{ui.e(a)} wins"], [
            [label.get(k, k), (f"{mp[k]:.0%}{ui.bar(mp[k], ui.RED)}", "fav" if k == "Stacked ensemble" else "")]
            for k in order if k in mp]), unsafe_allow_html=True)
        w = metrics.get("stack_weights", {})
        if w:
            st.markdown(f'<p class="oo-cap">The ensemble blends the logistic regression ({w["Logistic regression"]:+.2f}), '
                        f'XGBoost ({w["XGBoost"]:+.2f}) and Glicko-2 ({w["Glicko-2"]:+.2f}). Weights were fit on 2023 '
                        f'fights only.</p>', unsafe_allow_html=True)
        st.markdown("### Finish chance per round")
        hz = out["hazards"]
        st.markdown(ui.table(["", ui.e(a), ui.e(b)], [
            ["By KO/TKO", (f'{hz["A"]["KO"]:.1%}', ""), (f'{hz["B"]["KO"]:.1%}', "")],
            ["By submission", (f'{hz["A"]["SUB"]:.1%}', ""), (f'{hz["B"]["SUB"]:.1%}', "")],
        ]), unsafe_allow_html=True)
        st.markdown(f'<p class="oo-cap">The simulator plays this fight {st.session_state["n_sims"]:,} times, '
                    f'round by round, using these chances.</p>', unsafe_allow_html=True)
        st.selectbox("Simulated fights per matchup", [10_000, 100_000, 500_000, 1_000_000], key="n_sims",
                     format_func=lambda n: f"{n:,}", help="More fights give smoother round-by-round numbers.")
    with r:
        st.markdown("### Track record")
        tb = metrics["test_by_model"]
        n_test = metrics["test_model"]["n"]
        rows = [[label.get(k, k), (f'{tb[k]["accuracy"]:.1%}', "fav" if k == "Stacked ensemble" else ""),
                 (f'{tb[k]["logloss"]:.3f}', ""), (f'{tb[k]["brier"]:.3f}', "")] for k in order if k in tb]
        rows.append(["Coin flip", ("50.0%", "muted"), ("0.693", "muted"), ("0.250", "muted")])
        st.markdown(ui.table(["", "Picked winner", "Log loss", "Brier"], rows), unsafe_allow_html=True)
        st.markdown(f'<p class="oo-cap">Tested on all {n_test:,} UFC fights from January 2024 on, which the models never '
                    f'saw during training. Log loss and Brier score check how good the percentages are, not just '
                    f'the picks; lower is better.</p>', unsafe_allow_html=True)
        st.page_link("views/about.py", label="How the whole pipeline works →")

    with st.expander("Formulas"):
        st.markdown("**Elo**")
        st.latex(r"E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}} \qquad R_A' = R_A + K\,(S_A - E_A)")
        st.markdown("**Glicko-2**: φ is rating uncertainty, which grows while a fighter is inactive.")
        st.latex(r"g(\phi) = \frac{1}{\sqrt{1 + 3\phi^2/\pi^2}} \qquad "
                 r"P(A) = \frac{1}{1 + e^{-g\left(\sqrt{\phi_A^2+\phi_B^2}\right)(\mu_A - \mu_B)}}"
                 r"\qquad \phi^{*} = \sqrt{\phi^2 + \sigma^2 t}")
        st.markdown("**Strength of schedule and opponent-adjusted striking** (opponent numbers from before each fight)")
        st.latex(r"\text{SOS}_A = \frac{\sum_i \text{Elo}_{\text{opp}_i} + 1500\,k}{n + k}"
                 r"\qquad \text{adj SL}_A = \frac{\sum_i \big(\text{SL}_i - \text{SApM}_{\text{opp}_i} m_i\big)}"
                 r"{\sum_i m_i + m_0}")
        st.markdown("**Ensemble** (no intercept, so swapping corners exactly flips the probability)")
        st.latex(r"P(A) = \sigma\big(w_1\,\text{logit}\,p_{\text{Glicko}} + w_2\,\text{logit}\,p_{\text{LR}} "
                 r"+ w_3\,\text{logit}\,p_{\text{XGB}}\big)")
        st.markdown("**Simulator**: per-round finish chance, with a random good-night/bad-night factor f")
        st.latex(r"h^{KO}_A = \frac{\text{KO}^{\text{off}}_A \cdot \text{KO}^{\text{def}}_B}{\overline{\text{KO}}}"
                 r"\qquad P(\text{A finishes in round } r) = h_A f_A (1 + 0.08(r-1))(1 - h_B/2)")
        st.markdown("**Pricing**")
        st.latex(r"\text{American} = \begin{cases} -100\,\frac{p}{1-p} & p \ge 0.5 \\ +100\,\frac{1-p}{p} & p < 0.5 "
                 r"\end{cases} \qquad q_i = p_i^{\,k},\ \sum_i q_i = 1 + \text{margin}"
                 r"\qquad \text{EV} = p\,(d - 1) - (1 - p)")

    st.markdown(f'<p class="oo-cap" style="margin-top:1.4rem">Data: UFCStats via '
                f'<a href="https://github.com/Greco1899/scrape_ufc_stats">scrape_ufc_stats</a>. '
                f'Compute: {ui.e(B.describe())}. Built for fun and learning, not betting advice.</p>',
                unsafe_allow_html=True)

