"""Live predictions: pick two fighters → win probability, method/round breakdown, drivers.

    python -m mmapred.predict "Islam Makhachev" "Arman Tsarukyan" --rounds 5
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from .data import download, load
from .features import FighterIndex, swap
from .ratings import elo_prob, glicko_prob
from .simulate import hazards, simulate, summarize
from .train import ART

PRETTY = {
    "elo": "Elo rating", "age": "Age", "reach": "Reach", "height": "Height", "days_off": "Layoff",
    "n_prior": "UFC experience", "slpm": "Strikes landed/min", "sapm": "Strikes absorbed/min",
    "sig_acc": "Striking accuracy", "sig_def": "Striking defense", "td15": "Takedowns/15m",
    "td_acc": "Takedown accuracy", "td_def": "Takedown defense", "kd15": "Knockdowns/15m",
    "kdd15": "Knocked down/15m", "sub15": "Sub attempts/15m", "ctrl_share": "Control time",
    "opp_ctrl_share": "Controlled by opp.", "win_rate": "Win rate", "last3": "Recent form (last 3)",
    "ko_off": "KO power", "ko_def": "KO losses", "sub_off": "Sub wins", "sub_def": "Sub losses",
    "strike_edge": "Striking matchup", "ko_edge": "KO matchup", "sub_edge": "Submission matchup",
    "grap_edge": "Wrestling matchup", "sos_elo": "Strength of schedule", "win_q": "Quality of wins",
    "loss_q": "Quality of losses", "top_wins": "Wins vs top opponents", "adj_sl": "Opp.-adjusted striking offense",
    "adj_sa": "Opp.-adjusted strikes absorbed", "adj_td": "Opp.-adjusted wrestling", "g_r": "Glicko-2 rating",
    "g_rd": "Rating uncertainty", "glicko_logit": "Glicko-2 rating", "head_share": "Head-hunting", "leg_share": "Leg kicks",
    "ground_share": "Ground striking", "clinch_share": "Clinch work", "dec_share": "Goes to decision",
}


def pretty(col: str) -> str:
    key = col.split("_", 1)[1] if col[:2] in ("d_", "a_", "b_") else col
    return PRETTY.get(key, key)


class JsonLogReg:
    """The trained logistic regression, stored as plain numbers (no pickle/version issues).

    P(A) = sigmoid(intercept + coef · (x_imputed - mean) / scale)
    """

    def __init__(self, d):
        self.cols = d["features"]
        self.median, self.mean, self.scale = (np.array(d[k]) for k in ("median", "mean", "scale"))
        self.coef, self.intercept = np.array(d["coef"]), d["intercept"]

    def predict_proba(self, X):
        x = X[self.cols].to_numpy(dtype=float)
        x = np.where(np.isnan(x), self.median, x)
        z = self.intercept + ((x - self.mean) / self.scale) @ self.coef
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


class Predictor:
    def __init__(self):
        meta = json.loads((ART / "meta.json").read_text())
        self.features = meta["features"]
        self.P = meta["priors"]
        self.stack_inputs = meta.get("stack_inputs", [])
        self.stack_w = np.array(meta.get("stack_weights", []))
        self.logreg_coefs = meta.get("logreg_coefs", {})
        self.booster = xgb.Booster()
        self.booster.load_model(ART / "model.json")
        self.booster.set_param({"device": "cpu"})  # single-row inference is fastest on CPU
        lr_path = ART / "logreg.json"
        self.logreg = JsonLogReg(json.loads(lr_path.read_text())) if lr_path.exists() else None
        download()  # fetch the CSVs on first run (they aren't shipped in the zip)
        fights, long, fighters = load()
        self.index = FighterIndex(fights, long, fighters, self.P)
        self.data_through = fights["date"].max()

    def _xgb(self, X):
        return float(self.booster.predict(xgb.DMatrix(X[self.features]))[0])

    def _sym(self, f, X):
        """Average over both corner orientations so P(A,B) = 1 - P(B,A)."""
        return (f(X) + 1 - f(swap(X))) / 2

    def models(self, X, A, Bf) -> dict:
        """Every model's P(A wins), plus the stacked ensemble."""
        out = {
            "Elo": float(elo_prob(A["elo"].values[0], Bf["elo"].values[0])),
            "Glicko-2": float(glicko_prob(A["g_r"].values[0], A["g_rd"].values[0],
                                          Bf["g_r"].values[0], Bf["g_rd"].values[0])),
            "XGBoost": float(self._sym(self._xgb, X)),
        }
        if self.logreg is not None:
            out["Logistic regression"] = float(self._sym(
                lambda Z: self.logreg.predict_proba(Z[self.features])[0, 1], X))
        if self.logreg is not None and len(self.stack_w):
            z = np.array([_logit(out[k]) for k in self.stack_inputs])
            out["Stacked ensemble"] = float(1 / (1 + np.exp(-(z @ self.stack_w))))
        else:
            out["Stacked ensemble"] = out["XGBoost"]
        return out

    def predict(self, a: str, b: str, rounds: int = 3, title: int = 0, n_sims: int = 200_000, seed: int = 0):
        X, A, Bf = self.index.matchup(a, b, sched_rounds=rounds, title=title)
        mp = self.models(X, A, Bf)
        p = mp["Stacked ensemble"]

        # SHAP-style contributions from XGBoost (log-odds), symmetrised the same way
        c1 = self.booster.predict(xgb.DMatrix(X[self.features]), pred_contribs=True)[0][:-1]
        c2 = self.booster.predict(xgb.DMatrix(swap(X)[self.features]), pred_contribs=True)[0][:-1]
        contrib = pd.Series((c1 - c2) / 2, index=self.features)
        merged = contrib.groupby([pretty(c) for c in self.features]).sum().sort_values(key=np.abs, ascending=False)

        h = hazards(A.ko_off.values, A.sub_off.values, A.ko_def.values, A.sub_def.values,
                    Bf.ko_off.values, Bf.sub_off.values, Bf.ko_def.values, Bf.sub_def.values,
                    self.P["ko_off"], self.P["sub_off"])
        sim = simulate(*h, np.array([p]), np.array([rounds]), n_sims=n_sims, seed=seed)
        out = summarize(sim["probs"][0], rounds)
        out.update(a=a, b=b, rounds=rounds, p_model=p, models=mp, drivers=merged.head(8).to_dict(),
                   hazards={"A": {"KO": float(h[0][0]), "SUB": float(h[1][0])},
                            "B": {"KO": float(h[2][0]), "SUB": float(h[3][0])}})
        return out, A, Bf


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    out, _, _ = Predictor().predict(args.a, args.b, rounds=args.rounds)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
