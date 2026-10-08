"""Check the simulator against what actually happened in held-out (2024+) fights.

Runs every test fight through the simulator in one batched GPU call, then
compares predicted finish/decision and method probabilities with real outcomes.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import backend as B
from .data import load
from .features import build_training_table
from .simulate import hazards, simulate
from .train import ART, TEST_START, VAL_START


def main(n_sims: int = 20_000):
    fights, long, fighters = load()
    X, meta, sim, P = build_training_table(fights, long, fighters, prior_before=VAL_START)
    preds = pd.read_csv(ART / "test_preds.csv")
    te = (meta["date"] >= TEST_START).values
    m, s = meta[te].reset_index(drop=True), sim[te].reset_index(drop=True)
    m = m.merge(preds, on="fight_id")
    h = hazards(s.a_ko_off, s.a_sub_off, s.a_ko_def, s.a_sub_def,
                s.b_ko_off, s.b_sub_off, s.b_ko_def, s.b_sub_def, P["ko_off"], P["sub_off"])
    rounds = m["sched_rounds"].fillna(3).clip(1, 5).astype(int).values
    t0 = time.perf_counter()
    r = simulate(*[np.asarray(x) for x in h], m["p_a"].values, rounds, n_sims=n_sims)
    dt = time.perf_counter() - t0
    pr = r["probs"]  # [M, 2, 3, 5]

    ok = m["method"].isin(["KO", "SUB", "DEC"]).values
    p_fin = pr[:, :, :2].sum(axis=(1, 2, 3))
    y_fin = m["method"].isin(["KO", "SUB"]).values.astype(int)
    meth = pr.sum(axis=(1, 3))  # [M, 3]
    pred_m = np.array(["KO", "SUB", "DEC"])[meth.argmax(1)]
    base_rate = y_fin[ok].mean()
    rep = {
        "device": B.xp.__name__,
        "fights": int(ok.sum()),
        "sims_per_fight": n_sims,
        "seconds": dt,
        "finish_brier": float(brier_score_loss(y_fin[ok], p_fin[ok])),
        "finish_brier_baseline": float(brier_score_loss(y_fin[ok], np.full(ok.sum(), base_rate))),
        "mean_pred_finish": float(p_fin[ok].mean()),
        "actual_finish_rate": float(base_rate),
        "finish_auc": float(roc_auc_score(y_fin[ok], p_fin[ok])),
        "method_accuracy": float((pred_m[ok] == m["method"].values[ok]).mean()),
        "method_baseline_always_DEC": float((m["method"].values[ok] == "DEC").mean()),
    }
    (ART / "sim_eval.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20_000)
    main(ap.parse_args().sims)
