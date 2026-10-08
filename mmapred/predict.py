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
    "grap_edge": "Wrestling matchup", "head_share": "Head-hunting", "leg_share": "Leg kicks",
    "ground_share": "Ground striking", "clinch_share": "Clinch work", "dec_share": "Goes to decision",
}


def pretty(col: str) -> str:
    key = col.split("_", 1)[1] if col[:2] in ("d_", "a_", "b_") else col
    return PRETTY.get(key, key)


class Predictor:
    def __init__(self):
        meta = json.loads((ART / "meta.json").read_text())
        self.features = meta["features"]
        self.P = meta["priors"]
        self.booster = xgb.Booster()
        self.booster.load_model(ART / "model.json")
        self.booster.set_param({"device": "cpu"})  # single-row inference is fastest on CPU
        download()  # fetch the CSVs on first run
        fights, long, fighters = load()
        self.index = FighterIndex(fights, long, fighters, self.P)
        self.data_through = fights["date"].max()

    def _p(self, X):
        return self.booster.predict(xgb.DMatrix(X[self.features]))

    def predict(self, a: str, b: str, rounds: int = 3, title: int = 0, n_sims: int = 200_000, seed: int = 0):
        X, A, Bf = self.index.matchup(a, b, sched_rounds=rounds, title=title)
        p_ab, p_ba = self._p(X)[0], self._p(swap(X))[0]
        p = float((p_ab + 1 - p_ba) / 2)

        # SHAP-style contributions (log-odds), symmetrised the same way
        c1 = self.booster.predict(xgb.DMatrix(X[self.features]), pred_contribs=True)[0][:-1]
        c2 = self.booster.predict(xgb.DMatrix(swap(X)[self.features]), pred_contribs=True)[0][:-1]
        contrib = pd.Series((c1 - c2) / 2, index=self.features)
        merged = contrib.groupby([pretty(c) for c in self.features]).sum().sort_values(key=np.abs, ascending=False)

        h = hazards(A.ko_off.values, A.sub_off.values, A.ko_def.values, A.sub_def.values,
                    Bf.ko_off.values, Bf.sub_off.values, Bf.ko_def.values, Bf.sub_def.values,
                    self.P["ko_off"], self.P["sub_off"])
        sim = simulate(*h, np.array([p]), np.array([rounds]), n_sims=n_sims, seed=seed)
        out = summarize(sim["probs"][0], rounds)
        out.update(a=a, b=b, rounds=rounds, p_model=p, drivers=merged.head(8).to_dict())
        return out, A, Bf


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
