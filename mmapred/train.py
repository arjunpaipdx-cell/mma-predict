"""Train the win-probability models and the stacked ensemble.

Models (the same family sportsbooks and serious modellers use for MMA):
    1. Elo                 rating system baseline
    2. Glicko-2            rating + uncertainty (RD); no training needed
    3. Logistic regression on ~55 standardised stat/SOS differentials
    4. XGBoost (CUDA)      gradient-boosted trees for non-linear interactions
    5. Stacked ensemble    logistic regression over models 2-4's log-odds,
                           fit on the validation year (blends + recalibrates)

Split is strictly chronological so the test set looks like real future fights:
    train: fights before 2023-01-01
    val:   2023 (hyperparameter search + early stopping)
    test:  2024-01-01 onward (touched once, reported in README)

Usage:
    python -m mmapred.train               # default params
    python -m mmapred.train --sweep 40    # GPU random search over 40 configs
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from . import backend as B
from .data import download, load
from .features import build_training_table, swap

ART = Path(__file__).resolve().parent.parent / "artifacts"
VAL_START, TEST_START = "2023-01-01", "2024-01-01"
MIN_DATE = "2001-01-01"  # pre-unified-rules era is a different sport

BASE_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    learning_rate=0.03, max_depth=3, min_child_weight=20, subsample=0.8,
    colsample_bytree=0.7, reg_lambda=5.0, n_estimators=2000,
)


STACK_INPUTS = ["Glicko-2", "Logistic regression", "XGBoost"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def make_logreg():
    """Standardised, L2-regularised logistic regression (missing values -> median)."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000))


def augment(X, y):
    """Stack each fight with its corner-swapped mirror."""
    return pd.concat([X, swap(X)], ignore_index=True), np.concatenate([y, 1 - y])


def sym_predict(model, X):
    """P(A wins), averaged over both corner orientations so p(A,B) = 1 - p(B,A)."""
    p1 = model.predict_proba(X)[:, 1]
    p2 = model.predict_proba(swap(X))[:, 1]
    return (p1 + (1 - p2)) / 2


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(n=int(len(y)), accuracy=float(accuracy_score(y, p > 0.5)), logloss=float(log_loss(y, p)),
                brier=float(brier_score_loss(y, p)), auc=float(roc_auc_score(y, p)))


def elo_baseline(X):
    return 1 / (1 + 10 ** (-X["d_elo"].values / 400))


def fit(params, Xtr, ytr, Xva, yva):
    m = xgb.XGBClassifier(**params, device=B.xgb_device(), early_stopping_rounds=100)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def sample_params(rng) -> dict:
    p = dict(BASE_PARAMS)
    p.update(
        learning_rate=float(10 ** rng.uniform(-2.3, -1.0)),
        max_depth=int(rng.integers(2, 7)),
        min_child_weight=float(10 ** rng.uniform(0, 2)),
        subsample=float(rng.uniform(0.6, 1.0)),
        colsample_bytree=float(rng.uniform(0.4, 1.0)),
        reg_lambda=float(10 ** rng.uniform(-1, 1.5)),
        gamma=float(rng.uniform(0, 2)),
    )
    return p


def main(sweep: int = 0, seed: int = 0):
    print(B.describe())
    download()
    fights, long, fighters = load()
    t0 = time.perf_counter()
    X, meta, sim, P = build_training_table(fights, long, fighters, prior_before=VAL_START)
    t_feat = time.perf_counter() - t0
    y = meta["a_win"].values.astype(int)
    d = meta["date"]
    tr = (d >= MIN_DATE) & (d < VAL_START)
    va = (d >= VAL_START) & (d < TEST_START)
    te = d >= TEST_START
    print(f"features: {X.shape[1]} cols in {t_feat:.2f}s | train {tr.sum()} val {va.sum()} test {te.sum()}")

    Xtr, ytr = augment(X[tr], y[tr])
    Xva, yva = augment(X[va], y[va])

    best, best_params, t_sweep = None, BASE_PARAMS, 0.0
    if sweep:
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        for i in range(sweep):
            p = sample_params(rng)
            m = fit(p, Xtr, ytr, Xva, yva)
            ll = log_loss(y[va], np.clip(sym_predict(m, X[va]), 1e-6, 1 - 1e-6))
            if best is None or ll < best:
                best, best_params = ll, p
                print(f"  [{i+1}/{sweep}] new best val logloss {ll:.4f}")
        t_sweep = time.perf_counter() - t0
        print(f"sweep of {sweep} configs on {B.xgb_device()}: {t_sweep:.1f}s")

    # ---- base model 1: XGBoost (non-linear interactions)
    model = fit(best_params, Xtr, ytr, Xva, yva)
    n_trees = model.best_iteration + 1

    # ---- base model 2: regularised logistic regression (the classic, interpretable book model)
    lr = make_logreg().fit(Xtr, ytr)

    # ---- base model 3: Glicko-2 ratings alone (no training needed)
    def base_logits(xgb_m, lr_m, Xs):
        return np.column_stack([
            Xs["d_glicko_logit"].values,
            logit(sym_predict(lr_m, Xs)),
            logit(sym_predict(xgb_m, Xs)),
        ])

    # ---- stacked ensemble: logistic regression on the three models' log-odds, fit on 2023 only.
    # No intercept keeps P(A beats B) = 1 - P(B beats A); the fitted weights also recalibrate.
    Lva = base_logits(model, lr, X[va])
    stack = LogisticRegression(fit_intercept=False, C=1.0)
    stack.fit(np.vstack([Lva, -Lva]), np.concatenate([y[va], 1 - y[va]]))
    w = stack.coef_[0]

    Lte = base_logits(model, lr, X[te])
    p_te = sigmoid(Lte @ w)
    report = {
        "device": B.xgb_device(),
        "backend": B.describe(),
        "data_through": str(fights["date"].max().date()),
        "split": {"train": f"{MIN_DATE}..{VAL_START}", "val": f"{VAL_START}..{TEST_START}", "test": f"{TEST_START}.."},
        "test_model": metrics(y[te], p_te),
        "test_by_model": {
            "Elo": metrics(y[te], elo_baseline(X[te])),
            "Glicko-2": metrics(y[te], sigmoid(Lte[:, 0])),
            "Logistic regression": metrics(y[te], sigmoid(Lte[:, 1])),
            "XGBoost": metrics(y[te], sigmoid(Lte[:, 2])),
            "Stacked ensemble": metrics(y[te], p_te),
        },
        "stack_weights": dict(zip(STACK_INPUTS, map(float, w))),
        "n_trees": int(n_trees),
        "params": best_params,
        "feature_seconds": t_feat,
        "sweep_configs": sweep,
        "sweep_seconds": t_sweep,
    }
    report["test_elo_baseline"] = report["test_by_model"]["Elo"]
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(p_te, bins) - 1
    report["calibration"] = [
        {"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "n": int((idx == i).sum()),
         "pred": float(p_te[idx == i].mean()), "actual": float(y[te][idx == i].mean())}
        for i in range(10) if (idx == i).sum() > 0
    ]
    for k, v in report["test_by_model"].items():
        print(f"  {k:<20} acc {v['accuracy']:.3f}  logloss {v['logloss']:.4f}  brier {v['brier']:.4f}")
    print("  stack weights:", {k: round(v, 3) for k, v in report["stack_weights"].items()})

    # ---- final models: refit on everything for live predictions (stack weights stay from 2023)
    allm = d >= MIN_DATE
    Xall, yall = augment(X[allm], y[allm])
    final = xgb.XGBClassifier(**{**best_params, "n_estimators": n_trees}, device=B.xgb_device())
    final.fit(Xall, yall, verbose=False)
    lr_final = make_logreg().fit(Xall, yall)

    ART.mkdir(exist_ok=True)
    final.save_model(ART / "model.json")
    imp, sc, clf = lr_final
    (ART / "logreg.json").write_text(json.dumps({
        "features": list(X.columns), "median": list(map(float, imp.statistics_)),
        "mean": list(map(float, sc.mean_)), "scale": list(map(float, sc.scale_)),
        "coef": list(map(float, clf.coef_[0])), "intercept": float(clf.intercept_[0]),
    }))
    (ART / "metrics.json").write_text(json.dumps(report, indent=2))
    (ART / "meta.json").write_text(json.dumps({
        "features": list(X.columns), "priors": P,
        "stack_inputs": STACK_INPUTS, "stack_weights": list(map(float, w)),
        "logreg_coefs": dict(zip(X.columns, map(float, lr_final[-1].coef_[0]))),
    }, indent=2))
    pd.DataFrame({"fight_id": meta["fight_id"][te].values, "p_a": p_te}).to_csv(ART / "test_preds.csv", index=False)
    print(f"saved models + metrics to {ART}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.sweep, a.seed)
