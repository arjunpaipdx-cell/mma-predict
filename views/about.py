"""About page: who built Octagon Odds, what it does, and how."""
from __future__ import annotations

import json

import streamlit as st

from mmapred import ui
from mmapred.train import ART
from views.common import load_metrics

metrics = load_metrics()
me, proj = st.columns([1, 1.9], gap="large")
with me:
    st.markdown(
        '<div class="oo-about-name">Arjun Pai</div>'
        '<div class="oo-about-role">Computer science student · ML and AI infrastructure</div>'
        "<p>I build projects to learn how real systems work end to end: the data pipeline, the models, and the "
        "GPU compute underneath them.</p>"
        "<p>Octagon Odds is how I learned NVIDIA's GPU data science stack. I picked UFC because fights are hard "
        "to predict, the data is public, and a wrong answer is easy to check the next Saturday.</p>"
        '<p class="oo-links"><a href="https://github.com/arjunpaipdx-cell">GitHub</a> · '
        '<a href="https://github.com/arjunpaipdx-cell/mma-predict">Source code for this app</a></p>',
        unsafe_allow_html=True)
with proj:
    st.markdown("### What it does")
    st.markdown(
        "Pick any two UFC fighters and Octagon Odds tells you who's likely to win, how the fight ends "
        "(KO, submission or decision) and in which round. It then prices every outcome the way a sportsbook "
        "would, so you can compare its numbers with a real line. Every prediction comes with its reasoning: "
        "the stats that moved it, each fighter's record against quality opposition, and how each model voted. "
        "The Research page adds film study: upload footage and it measures how each fighter actually moves.")

    st.markdown("### How it works")
    st.markdown(
        "1. **Data.** Every UFC result and round-by-round stat since 1994, about 8,900 fights, refreshed from "
        "UFCStats.\n"
        "2. **Ratings.** Each fighter gets an Elo and a Glicko-2 rating, updated one fight at a time. Glicko-2 "
        "also tracks how sure it is, so long layoffs and short careers count for less.\n"
        "3. **Features.** About 55 per matchup: striking, wrestling and finishing rates, physical edges, "
        "layoffs, and strength of schedule. Every number uses only fights that happened *before* the one being "
        "predicted, so the model can't peek at results.\n"
        "4. **Win chance.** A logistic regression and an XGBoost model each predict the winner. A small "
        "ensemble blends them with Glicko-2, using weights fit on 2023 fights only.\n"
        "5. **Simulation.** A Monte Carlo simulator plays the fight out round by round, 100,000 times by "
        "default, to get the method and round.\n"
        "6. **Pricing.** Fair chances become American or decimal odds, with an optional sportsbook margin.\n"
        "7. **Film study.** On the Research page, a pose model tracks both fighters in uploaded footage and "
        "measures stance, range, pressure, guard, strikes and level changes.")

    tm = metrics["test_model"]
    st.markdown(
        f'<p class="oo-note">On all {tm["n"]:,} UFC fights since January 2024, none of them seen in training, '
        f'it picked the winner {tm["accuracy"]:.1%} of the time. Ratings alone manage '
        f'{metrics["test_elo_baseline"]["accuracy"]:.1%}.</p>', unsafe_allow_html=True)

    st.markdown("### Built with")
    st.markdown(ui.table(["Stage", "Tool", "Why"], [
        ["Feature engineering", ("RAPIDS cuDF", "lbl"), ("Pandas-style group-bys on the GPU", "muted")],
        ["Model training", ("XGBoost (CUDA)", "lbl"), ("Gradient-boosted trees trained on the GPU", "muted")],
        ["Fight simulation", ("CuPy", "lbl"), ("Millions of simulated fights as one batch of GPU arrays", "muted")],
        ["Logistic model", ("scikit-learn", "lbl"), ("Regularised logistic regression", "muted")],
        ["Pose tracking", ("YOLO11-pose + ONNX Runtime", "lbl"),
         ("Skeletons for both fighters; uses TensorRT or CUDA on an NVIDIA GPU", "muted")],
        ["App", ("Streamlit + Altair", "lbl"), ("This interface and its charts", "muted")],
    ], left=3), unsafe_allow_html=True)
    st.markdown('<p class="oo-cap">Every GPU step falls back to the CPU automatically, which is how this '
                'page runs on a free server.</p>', unsafe_allow_html=True)

    bench_path = ART / "benchmarks.json"
    if bench_path.exists():
        bench = json.loads(bench_path.read_text())
        names_b = {"monte carlo (CuPy)": "Simulating 99M fights (CuPy)",
                   "xgboost train": "Training XGBoost",
                   "features (cuDF)": "Building features (cuDF)"}
        rows = []
        for k in ("monte carlo (CuPy)", "xgboost train", "features (cuDF)"):
            v = bench["stages"].get(k)
            if v and "gpu_s" in v:
                rows.append([names_b[k], (f'{v["cpu_s"]:.1f}s', "muted"), (f'{v["gpu_s"]:.1f}s', ""),
                             (f'{v["speedup"]:.1f}×', "fav")])
        if rows:
            st.markdown("### GPU vs CPU")
            st.markdown(ui.table(["", "CPU", "GPU (T4)", "Faster by"], rows), unsafe_allow_html=True)
            st.markdown('<p class="oo-cap">Measured on a free Google Colab NVIDIA T4. The simulator gains most '
                        'because every simulated fight is independent. Feature building barely gains, because '
                        'the table is small enough that moving it to the GPU costs almost as much as it '
                        'saves.</p>', unsafe_allow_html=True)
