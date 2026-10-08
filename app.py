"""Streamlit front end.   streamlit run app.py"""
from __future__ import annotations

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from mmapred import backend as B
from mmapred.odds import (american_to_prob, expected_value, fmt_american, markets, price_board,
                          prob_to_american, remove_margin)
from mmapred.predict import Predictor
from mmapred.train import ART

st.set_page_config(page_title="Octagon Odds", page_icon="🥊", layout="wide")
GREEN, GREY = "#76b900", "#8a8f98"


@st.cache_resource(show_spinner="Loading fight history and models…")
def get_predictor():
    return Predictor()


if not (ART / "model.json").exists():
    st.error("No trained model found. Run `python -m mmapred.train` first.")
    st.stop()

pr = get_predictor()
metrics = json.loads((ART / "metrics.json").read_text())
t = pr.index.table

st.title("Octagon Odds")
st.caption(
    f"UFC fight pricing engine · Glicko-2 + logistic regression + XGBoost ensemble · Monte Carlo round simulator · "
    f"data through {pr.data_through.date()} · running on **{B.describe()}**"
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Matchup")
    active = st.toggle("Active fighters only (fought since 2024)", value=True)
    names = pr.index.names(min_fights=2, active_since="2024-01-01" if active else None)
    classes = ["All"] + sorted(t.loc[names, "weightclass"].dropna().unique())
    wc = st.selectbox("Weight class", classes, index=classes.index("Lightweight") if "Lightweight" in classes else 0)
    pool = [n for n in names if wc == "All" or t.loc[n, "weightclass"] == wc]
    pool = sorted(pool, key=lambda n: -t.loc[n, "elo"])  # best-rated first → marquee default
    a = st.selectbox("Fighter A", pool, index=0)
    b = st.selectbox("Fighter B", [n for n in pool if n != a], index=0)
    rounds = st.radio("Scheduled rounds", [3, 5], horizontal=True)
    n_sims = st.select_slider("Simulations", [10_000, 100_000, 500_000, 1_000_000], value=100_000)
    st.divider()
    st.subheader("Book settings")
    vig = st.slider("Margin (overround) on 2-way markets", 0.0, 0.10, 0.045, 0.005, format="%.3f",
                    help="-110/-110 is a 4.8% overround. Multi-outcome markets get proportionally more.")
    vig_method = st.radio("Margin method", ["power", "multiplicative"], horizontal=True,
                          help="Power puts more margin on longshots (favourite-longshot bias).")
    st.divider()
    m = metrics["test_model"]
    st.markdown(
        f"**Held-out test (2024+ fights, n={m['n']:,})**  \n"
        f"Ensemble accuracy {m['accuracy']:.1%} · log loss {m['logloss']:.3f}  \n"
        f"Elo baseline {metrics['test_elo_baseline']['accuracy']:.1%}"
    )

out, A, Bf = pr.predict(a, b, rounds=rounds, n_sims=n_sims)
mk = markets(out, a, b, rounds)
board = price_board(mk, vig, vig_method)

# ------------------------------------------------------------------ header
c1, c2, c3 = st.columns([2, 1, 2])
ml = {r["Outcome"]: r for r in board["Moneyline"]}
c1.metric(a, f"{out['p_a']:.0%}")
c1.caption(f"Moneyline **{ml[a]['Book odds']}** · fair {ml[a]['Fair odds']}")
c2.markdown(f"<div style='text-align:center;padding-top:1.6rem;color:{GREY}'>vs</div>", unsafe_allow_html=True)
c3.metric(b, f"{out['p_b']:.0%}")
c3.caption(f"Moneyline **{ml[b]['Book odds']}** · fair {ml[b]['Fair odds']}")
st.progress(out["p_a"])

tab_pred, tab_odds, tab_sos, tab_models, tab_math = st.tabs(
    ["Prediction", "Odds board", "Strength of schedule", "Model breakdown", "The math"])

# ------------------------------------------------------------------ prediction
with tab_pred:
    left, right = st.columns(2)
    with left:
        st.subheader("How it ends")
        rows = [{"fighter": (a if w == "A" else b), "method": mth, "p": v}
                for w, d in out["by_method"].items() for mth, v in d.items()]
        st.altair_chart(
            alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                x=alt.X("p:Q", axis=alt.Axis(format="%"), title=None),
                y=alt.Y("method:N", sort=["KO", "SUB", "DEC"], title=None),
                color=alt.Color("fighter:N", scale=alt.Scale(domain=[a, b], range=[GREEN, GREY]),
                                legend=alt.Legend(orient="bottom")),
                yOffset="fighter:N",
                tooltip=["fighter", "method", alt.Tooltip("p:Q", format=".1%")],
            ).properties(height=220),
            use_container_width=True,
        )
        st.caption(f"Finish {out['p_finish']:.0%} · goes the distance {out['p_distance']:.0%}")
    with right:
        st.subheader("When it ends")
        rows = [{"fighter": (a if w == "A" else b), "round": f"R{i+1}", "p": v}
                for w, lst in out["by_round"].items() for i, v in enumerate(lst)]
        st.altair_chart(
            alt.Chart(pd.DataFrame(rows)).mark_bar().encode(
                x=alt.X("round:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("p:Q", axis=alt.Axis(format="%"), title=None, stack=True),
                color=alt.Color("fighter:N", scale=alt.Scale(domain=[a, b], range=[GREEN, GREY]),
                                legend=alt.Legend(orient="bottom")),
                tooltip=["fighter", "round", alt.Tooltip("p:Q", format=".1%")],
            ).properties(height=220),
            use_container_width=True,
        )
        st.caption("Last round includes decisions.")

    st.subheader("What drove the prediction")
    d = pd.Series(out["drivers"]).rename("impact").reset_index().rename(columns={"index": "factor"})
    d["favours"] = d["impact"].map(lambda v: a if v > 0 else b)
    st.altair_chart(
        alt.Chart(d).mark_bar().encode(
            x=alt.X("impact:Q", title=f"← favours {b}   |   favours {a} →"),
            y=alt.Y("factor:N", sort=None, title=None),
            color=alt.Color("favours:N", scale=alt.Scale(domain=[a, b], range=[GREEN, GREY]), legend=None),
            tooltip=["factor", "favours", alt.Tooltip("impact:Q", format=".3f")],
        ).properties(height=260),
        use_container_width=True,
    )
    st.caption("XGBoost SHAP contributions in log-odds, averaged over both corner orientations.")

    st.subheader("Tale of the tape")
    cols = {"elo": "Elo", "g_r": "Glicko-2 rating", "g_rd": "Glicko-2 uncertainty (RD)", "n_prior": "UFC fights",
            "age": "Age", "reach": "Reach (in)", "slpm": "Sig. strikes landed / min",
            "sapm": "Sig. strikes absorbed / min", "sig_acc": "Striking accuracy", "sig_def": "Striking defense",
            "td15": "Takedowns / 15 min", "td_def": "Takedown defense", "sub15": "Sub attempts / 15 min",
            "last3": "Recent form"}
    tape = pd.DataFrame({a: A[list(cols)].iloc[0].values, b: Bf[list(cols)].iloc[0].values}, index=list(cols.values()))
    pct = ["Striking accuracy", "Striking defense", "Takedown defense", "Recent form"]
    st.dataframe(
        tape.style.format("{:.0%}", subset=pd.IndexSlice[pct, :]).format(
            "{:.1f}", subset=pd.IndexSlice[[i for i in tape.index if i not in pct], :]),
        use_container_width=True, height=530,
    )

# ------------------------------------------------------------------ odds board
with tab_odds:
    st.markdown(
        f"Prices built the way a sportsbook builds them: **fair probability from the model → add margin "
        f"({vig:.1%} overround on 2-way markets, {vig_method} method) → convert to American odds.**"
    )

    def show(name):
        df = pd.DataFrame(board[name])
        st.markdown(f"**{name}**")
        st.dataframe(
            df.style.format({"Fair %": "{:.1%}", "Book %": "{:.1%}", "Decimal": "{:.2f}"}),
            hide_index=True, use_container_width=True,
        )
        hold = df["Book %"].sum() - 1
        st.caption(f"Overround {hold:.1%} · book's theoretical hold {hold / (1 + hold):.1%}")

    l, r = st.columns(2)
    with l:
        show("Moneyline")
        show("Goes the distance")
        for k in [k for k in board if k.startswith("Total rounds")]:
            show(k)
    with r:
        show("Method of victory")
        show("Round betting")
    st.caption("x.5 totals assume a finish is equally likely at any point within its round.")

    st.divider()
    st.subheader("Compare with a real sportsbook line")
    st.markdown("Enter the moneyline from any book. It's de-vigged to the book's fair probability and compared "
                "with this model.")
    x1, x2 = st.columns(2)
    la = x1.number_input(f"{a} odds (American)", value=int(prob_to_american(min(out['p_a'], .95))) or -110, step=5)
    lb = x2.number_input(f"{b} odds (American)", value=int(prob_to_american(min(out['p_b'], .95))) or -110, step=5)
    if abs(la) >= 100 and abs(lb) >= 100:
        q = np.array([american_to_prob(la), american_to_prob(lb)])
        fair = remove_margin(q, vig_method)
        comp = pd.DataFrame({
            "Fighter": [a, b],
            "Book implied %": q,
            "Book fair % (de-vigged)": fair,
            "Model %": [out["p_a"], out["p_b"]],
            "Edge (model − book fair)": [out["p_a"] - fair[0], out["p_b"] - fair[1]],
            "EV per $1": [expected_value(out["p_a"], la), expected_value(out["p_b"], lb)],
        })
        st.dataframe(comp.style.format({c: "{:.1%}" for c in comp.columns[1:5]} | {"EV per $1": "{:+.3f}"}),
                     hide_index=True, use_container_width=True)
        st.caption(f"Book overround: {q.sum() - 1:.1%}. Positive EV only means the model disagrees with the market; "
                   "markets also price in injuries, camps and weight cuts this model can't see. Not betting advice.")
    else:
        st.info("American odds are ≥ +100 or ≤ −100.")

# ------------------------------------------------------------------ strength of schedule
with tab_sos:
    st.markdown("Who have they actually beaten? Elo ratings are taken **at the time of each fight**, so "
                "beating someone in their prime counts for more than beating them on the way down.")
    sos_cols = {"sos_elo": "Avg. opponent Elo (strength of schedule)", "win_q": "Avg. Elo of opponents beaten",
                "loss_q": "Avg. Elo of opponents lost to", "top_wins": "Wins over 1600+ rated opponents",
                "adj_sl": "Opp.-adjusted strikes landed / min", "adj_sa": "Opp.-adjusted strikes absorbed / min",
                "adj_td": "Opp.-adjusted takedowns / 15 min"}
    sos = pd.DataFrame({a: A[list(sos_cols)].iloc[0].values, b: Bf[list(sos_cols)].iloc[0].values},
                       index=list(sos_cols.values()))
    st.dataframe(sos.style.format("{:+.2f}", subset=pd.IndexSlice[list(sos_cols.values())[4:], :]).format(
        "{:.0f}", subset=pd.IndexSlice[list(sos_cols.values())[:4], :]), use_container_width=True)
    st.caption("Opponent-adjusted stats compare each fight with what that opponent normally allows "
               "(e.g. +1.0 = lands one more strike per minute than that opponent usually absorbs). "
               "Lower is better for strikes absorbed. Averages are shrunk toward a 1500 opponent "
               "for fighters with few fights.")
    l, r = st.columns(2)
    for col, name in ((l, a), (r, b)):
        with col:
            st.markdown(f"**{name}: recent opponents**")
            st.dataframe(pr.index.schedule_view(name, 8), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ model breakdown
with tab_models:
    mp = out["models"]
    st.markdown(f"Each model's probability that **{a}** wins. The ensemble is the number used everywhere else.")
    mdf = pd.DataFrame({"model": list(mp), "p": list(mp.values())})
    base = alt.Chart(mdf).encode(y=alt.Y("model:N", sort=list(mp), title=None))
    st.altair_chart(
        (base.mark_bar(color=GREEN).encode(x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"),
                                                   title=f"P({a} wins)"),
                                           tooltip=["model", alt.Tooltip("p:Q", format=".1%")])
         + base.mark_text(align="left", dx=4).encode(x="p:Q", text=alt.Text("p:Q", format=".1%"))
         + alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color=GREY, strokeDash=[4, 4]).encode(x="x:Q")
         ).properties(height=220),
        use_container_width=True,
    )
    st.subheader("How each model did on unseen 2024+ fights")
    tb = pd.DataFrame(metrics["test_by_model"]).T[["accuracy", "logloss", "brier", "auc"]]
    st.dataframe(tb.style.format({"accuracy": "{:.1%}", "logloss": "{:.4f}", "brier": "{:.4f}", "auc": "{:.3f}"}),
                 use_container_width=True)
    st.caption("Log loss and Brier score measure probability quality (lower is better); a coin flip scores "
               "0.693 and 0.250. Accuracy alone ignores how confident each pick was.")
    w = metrics.get("stack_weights", {})
    if w:
        st.markdown("**Ensemble weights** (fit on 2023 fights only): " +
                    ", ".join(f"{k} {v:+.2f}" for k, v in w.items()))
    st.subheader("Simulator inputs")
    hz = out["hazards"]
    st.dataframe(pd.DataFrame({
        a: [hz["A"]["KO"], hz["A"]["SUB"]], b: [hz["B"]["KO"], hz["B"]["SUB"]],
    }, index=["KO chance per round", "Submission chance per round"]).style.format("{:.1%}"),
        use_container_width=True)

# ------------------------------------------------------------------ the math
with tab_math:
    st.subheader("1. Ratings")
    st.markdown("**Elo**: expected score and update after each fight (K is larger for new fighters and finishes):")
    st.latex(r"E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}} \qquad R_A' = R_A + K\,(S_A - E_A)")
    st.markdown("**Glicko-2** adds a rating deviation φ (uncertainty) that grows with inactivity and shrinks "
                "with every fight. Uncertain ratings pull predictions toward 50%:")
    st.latex(r"g(\phi) = \frac{1}{\sqrt{1 + 3\phi^2/\pi^2}} \qquad "
             r"P(A) = \frac{1}{1 + e^{-g\left(\sqrt{\phi_A^2+\phi_B^2}\right)(\mu_A - \mu_B)}}")
    st.latex(r"\phi^{*} = \sqrt{\phi^2 + \sigma^2 t_{\text{inactive}}} \qquad "
             r"\phi' = \Big(\tfrac{1}{\phi^{*2}} + \tfrac{1}{v}\Big)^{-1/2} \qquad "
             r"\mu' = \mu + \phi'^2 g(\phi_B)(s - E)")

    st.subheader("2. Strength of schedule")
    st.latex(r"\text{SOS}_A = \frac{\sum_{i} \text{Elo}_{\text{opp}_i,\,t_i} + 1500\,k}{n + k}"
             r"\qquad \text{adj\_SL}_A = \frac{\sum_i \big(\text{SL}_i - \text{SApM}_{\text{opp}_i}\cdot m_i\big)}"
             r"{\sum_i m_i + m_0}")
    st.markdown("Opponent ratings and rates are taken from *before* each fight. k and m₀ are shrinkage "
                "pseudo-counts so a two-fight career isn't extreme.")

    st.subheader("3. Win-probability models")
    st.markdown("**Logistic regression** on standardised differences x = features(A) − features(B):")
    st.latex(r"P(A) = \sigma(\beta^\top x) = \frac{1}{1 + e^{-\beta^\top x}}")
    st.markdown("**XGBoost** sums hundreds of shallow trees, each fitted to the previous trees' errors, "
                "to capture interactions such as a striker facing a wrestler with poor takedown defense.")
    st.latex(r"\text{logit}\,P(A) = \sum_{t=1}^{T} \eta\, f_t(x)")
    st.markdown("**Stacked ensemble**: a second logistic regression blends the models' log-odds. It has no "
                "intercept, so swapping corners exactly flips the probability:")
    st.latex(r"P(A) = \sigma\big(w_1\,\text{logit}\,p_{\text{Glicko}} + w_2\,\text{logit}\,p_{\text{LR}} "
             r"+ w_3\,\text{logit}\,p_{\text{XGB}}\big)")
    st.markdown("Every fight appears twice in training (A vs B and B vs A), and predictions average both "
                "orientations.")

    st.subheader("4. Monte Carlo fight simulation")
    st.latex(r"h^{KO}_A = \frac{\text{KO}^{\text{off}}_A \cdot \text{KO}^{\text{def}}_B}{\overline{\text{KO}}}"
             r"\quad \text{(shrunk toward the league rate)}")
    st.latex(r"P(\text{A finishes in round } r) = h_A\, f_A\,(1 + 0.08\,(r-1))\,(1 - h_B/2),"
             r"\quad f_A \sim \text{LogNormal}(-\tfrac{\sigma^2}{2}, \sigma)")
    st.markdown("Fights still going after the last round go to the judges, with a decision probability q chosen "
                "so the simulator's total win probability matches the ensemble:")
    st.latex(r"q = \frac{P_{\text{model}}(A) - P(\text{A finishes})}{P(\text{decision})}")

    st.subheader("5. Pricing")
    st.markdown("Fair odds come straight from the probability. The book then adds margin so the implied "
                "probabilities sum to more than 100%:")
    st.latex(r"\text{American} = \begin{cases} -100\,\frac{p}{1-p} & p \ge 0.5 \\ +100\,\frac{1-p}{p} & p < 0.5 "
             r"\end{cases} \qquad \text{Decimal} = \frac{1}{p}")
    st.latex(r"\textbf{power method: } q_i = p_i^{\,k},\ \ \sum_i q_i = 1 + O"
             r"\qquad \textbf{multiplicative: } q_i = p_i (1 + O)")
    st.latex(r"\text{hold} = \frac{O}{1 + O} \qquad \text{EV} = p\,(d - 1) - (1 - p)")
    st.markdown("De-vigging a real line runs the power step backwards, solving for k with Σ qᵢᵏ = 1.")
    st.caption("Real sportsbooks also move lines on betting volume and sharp money, and use injury and camp news. "
               "This engine models the opening-number part.")

st.caption("For fun and learning, not betting advice.")
