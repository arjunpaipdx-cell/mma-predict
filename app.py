"""Streamlit front end.   streamlit run app.py"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from mmapred import backend as B
from mmapred.predict import Predictor
from mmapred.train import ART

st.set_page_config(page_title="Octagon Odds", page_icon="🥊", layout="wide")
GREEN, GREY = "#76b900", "#8a8f98"


@st.cache_resource(show_spinner="Loading fight history and model…")
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
    f"UFC fight predictions · XGBoost win model + Monte Carlo round simulator · data through "
    f"{pr.data_through.date()} · running on **{B.describe()}**"
)

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
    m = metrics["test_model"]
    st.markdown(
        f"**Held-out test (2024+ fights, n={m['n']:,})**  \n"
        f"Accuracy {m['accuracy']:.1%} · log loss {m['logloss']:.3f}  \n"
        f"Elo baseline {metrics['test_elo_baseline']['accuracy']:.1%}"
    )

out, A, Bf = pr.predict(a, b, rounds=rounds, n_sims=n_sims)

c1, c2, c3 = st.columns([2, 1, 2])
c1.metric(a, f"{out['p_a']:.0%}")
c2.markdown(f"<div style='text-align:center;padding-top:1.6rem;color:{GREY}'>vs</div>", unsafe_allow_html=True)
c3.metric(b, f"{out['p_b']:.0%}")
st.progress(out["p_a"])

left, right = st.columns(2)
with left:
    st.subheader("How it ends")
    rows = [{"fighter": (a if w == "A" else b), "method": mth, "p": v}
            for w, d in out["by_method"].items() for mth, v in d.items()]
    df = pd.DataFrame(rows)
    st.altair_chart(
        alt.Chart(df).mark_bar().encode(
            x=alt.X("p:Q", axis=alt.Axis(format="%"), title=None),
            y=alt.Y("method:N", sort=["KO", "SUB", "DEC"], title=None),
            color=alt.Color("fighter:N", scale=alt.Scale(range=[GREEN, GREY]), legend=alt.Legend(orient="bottom")),
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
            color=alt.Color("fighter:N", scale=alt.Scale(range=[GREEN, GREY]), legend=alt.Legend(orient="bottom")),
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
cols = {"elo": "Elo", "n_prior": "UFC fights", "age": "Age", "reach": "Reach (in)", "slpm": "Sig. strikes landed / min",
        "sapm": "Sig. strikes absorbed / min", "sig_acc": "Striking accuracy", "sig_def": "Striking defense",
        "td15": "Takedowns / 15 min", "td_def": "Takedown defense", "sub15": "Sub attempts / 15 min",
        "last3": "Recent form"}
tape = pd.DataFrame({a: A[list(cols)].iloc[0].values, b: Bf[list(cols)].iloc[0].values}, index=list(cols.values()))
pct = ["Striking accuracy", "Striking defense", "Takedown defense", "Recent form"]
st.dataframe(
    tape.style.format("{:.0%}", subset=pd.IndexSlice[pct, :]).format(
        "{:.1f}", subset=pd.IndexSlice[[i for i in tape.index if i not in pct], :]),
    use_container_width=True,
    height=460,
)
st.caption("Career rates are shrunk toward the league average for fighters with few UFC fights. "
           "For fun and learning, not betting advice.")
