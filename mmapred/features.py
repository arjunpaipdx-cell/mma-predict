"""Leak-free fighter features.

Every feature for a fight is computed only from that fighter's *earlier* fights.
The heavy part, per-fighter cumulative aggregation over ~18k fighter-fights × 30
stat columns, runs on cuDF when a GPU is present. Elo is a sequential update, so
it stays on the CPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backend as B

SUMS = [
    "sig_l", "sig_a", "td_l", "td_a", "kd", "sub_att", "ctrl", "head_l", "body_l", "leg_l",
    "dist_l", "clinch_l", "ground_l", "opp_sig_l", "opp_sig_a", "opp_td_l", "opp_td_a", "opp_kd",
    "opp_ctrl", "stat_secs", "rounds_fought", "is_win", "is_loss", "ko_win", "sub_win", "dec_win",
    "ko_loss", "sub_loss", "dec_loss", "five_rd",
]

# Pseudo-counts used to shrink small samples toward the league average.
PRIOR_MIN = 15.0      # minutes of "average fighter" added to each career
PRIOR_ROUNDS = 10.0   # rounds (finish rates are noisy: one fluke stoppage shouldn't dominate)
PRIOR_FIGHTS = 3.0    # fights
PRIOR_ATT = 30.0      # strike/td attempts

RATE_COLS = [
    "slpm", "sapm", "sig_acc", "sig_def", "td15", "td_acc", "td_def", "kd15", "kdd15", "sub15",
    "ctrl_share", "opp_ctrl_share", "head_share", "leg_share", "ground_share", "clinch_share",
    "win_rate", "ko_off", "ko_def", "sub_off", "sub_def", "dec_share", "last3",
]


# ---------------------------------------------------------------- Elo (CPU)
def add_elo(long: pd.DataFrame, fights: pd.DataFrame, k: float = 40.0) -> tuple[pd.DataFrame, dict]:
    """Pre-fight Elo for every (fight, fighter); returns current ratings too."""
    elo: dict[str, float] = {}
    n: dict[str, int] = {}
    rows = []
    for fid, a, b, y, m in fights[["fight_id", "a", "b", "a_win", "method"]].itertuples(index=False):
        ra, rb = elo.get(a, 1500.0), elo.get(b, 1500.0)
        rows.append((fid, a, ra))
        rows.append((fid, b, rb))
        if y != y:  # NaN: draw/NC
            if m is not None:
                y = 0.5
            else:
                continue
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        mult = 1.2 if m in ("KO", "SUB") else 1.0
        ka = k * (1.5 if n.get(a, 0) < 5 else 1.0) * mult
        kb = k * (1.5 if n.get(b, 0) < 5 else 1.0) * mult
        elo[a] = ra + ka * (y - ea)
        elo[b] = rb - kb * (y - ea)
        n[a] = n.get(a, 0) + 1
        n[b] = n.get(b, 0) + 1
    pre = pd.DataFrame(rows, columns=["fight_id", "fighter", "elo"])
    return long.merge(pre, on=["fight_id", "fighter"], how="left"), elo


# ------------------------------------------------- cumulative sums (cuDF/pandas)
def _prep(long: pd.DataFrame) -> pd.DataFrame:
    d = long.copy()
    d["stat_secs"] = d["fight_secs"] * d["has_stats"]
    d["rounds_fought"] = d["rounds_fought"] * d["has_stats"]
    d["five_rd"] = (d["sched_rounds"] == 5).astype(int)
    return d


def cumulative(long: pd.DataFrame, df_lib=None):
    """Pre-fight career sums for every row. Runs on cuDF if available.

    Returns a pandas frame with `pre_<col>`, `n_prior`, `days_off`, `last3`.
    """
    xdf = df_lib or B.xdf
    if xdf.__name__ == "cudf" and df_lib is None:
        try:
            return _cumulative(long, xdf)
        except Exception as e:  # keep the pipeline alive if a cuDF op isn't supported
            print(f"[mmapred] cuDF path failed ({e!r}); falling back to pandas")
            return _cumulative(long, pd)
    return _cumulative(long, xdf)


def _cumulative(long: pd.DataFrame, xdf):
    d = _prep(long)[["fight_id", "fighter", "date"] + SUMS].sort_values(["fighter", "date", "fight_id"])
    d = d.reset_index(drop=True)
    if xdf.__name__ == "cudf":
        d = xdf.from_pandas(d)

    g = d.groupby("fighter", sort=False)
    cum = g[SUMS].cumsum()
    out = d[["fight_id", "fighter", "date"]].copy()
    for c in SUMS:
        out[f"pre_{c}"] = cum[c] - d[c]           # exclude the current fight
    out["n_prior"] = g.cumcount()
    prev = g["date"].shift(1)
    out["days_off"] = (d["date"] - prev).dt.days
    # wins / fights over the previous three bouts
    d["cw"] = cum["is_win"]
    d["cf"] = cum["is_win"] + cum["is_loss"]
    g2 = d.groupby("fighter", sort=False)
    w3 = (out["pre_is_win"] - g2["cw"].shift(4).fillna(0)).astype("float64")
    f3 = (out["pre_is_win"] + out["pre_is_loss"] - g2["cf"].shift(4).fillna(0)).astype("float64")
    out["last3"] = (w3 + 1.0) / (f3 + 2.0)
    return B.to_host(out)


def current_sums(long: pd.DataFrame) -> pd.DataFrame:
    """Career sums *including* each fighter's latest fight (for live predictions)."""
    d = _prep(long).sort_values(["fighter", "date", "fight_id"])
    tot = d.groupby("fighter")[SUMS].sum()
    tot.columns = [f"pre_{c}" for c in SUMS]
    tot["n_prior"] = d.groupby("fighter").size()
    tot["last_date"] = d.groupby("fighter")["date"].max()
    tail = d.groupby("fighter").tail(3).groupby("fighter")
    tot["last3"] = (tail["is_win"].sum() + 1.0) / (tail["is_win"].sum() + tail["is_loss"].sum() + 2.0)
    return tot.reset_index()


# ------------------------------------------------------------ rate features
def league_priors(long: pd.DataFrame, before=None) -> dict:
    d = _prep(long)
    if before is not None:
        d = d[d["date"] < pd.Timestamp(before)]
    s = d[SUMS].sum()
    mins = s["stat_secs"] / 60
    rnd = s["rounds_fought"]
    fights = s["is_win"] + s["is_loss"]
    p = {
        "slpm": s["sig_l"] / mins, "sapm": s["opp_sig_l"] / mins,
        "sig_acc": s["sig_l"] / s["sig_a"], "sig_def": 1 - s["opp_sig_l"] / s["opp_sig_a"],
        "td15": s["td_l"] / mins * 15, "td_acc": s["td_l"] / s["td_a"],
        "td_def": 1 - s["opp_td_l"] / s["opp_td_a"], "kd15": s["kd"] / mins * 15,
        "kdd15": s["opp_kd"] / mins * 15, "sub15": s["sub_att"] / mins * 15,
        "ctrl_share": s["ctrl"] / s["stat_secs"], "opp_ctrl_share": s["opp_ctrl"] / s["stat_secs"],
        "head_share": s["head_l"] / s["sig_l"], "leg_share": s["leg_l"] / s["sig_l"],
        "ground_share": s["ground_l"] / s["sig_l"], "clinch_share": s["clinch_l"] / s["sig_l"],
        "ko_off": s["ko_win"] / rnd, "sub_off": s["sub_win"] / rnd,
        "dec_share": (s["dec_win"] + s["dec_loss"]) / fights,
    }
    p["ko_def"], p["sub_def"] = p["ko_off"], p["sub_off"]
    return {k: float(v) for k, v in p.items()}


def rates(pre: pd.DataFrame, P: dict) -> pd.DataFrame:
    """Turn pre_* sums into shrunk per-minute / per-round / percentage rates."""
    g = lambda c: pre[f"pre_{c}"]  # noqa: E731
    mins = g("stat_secs") / 60
    rnd = g("rounds_fought")
    fights = g("is_win") + g("is_loss")
    sh = lambda num, den, prior, w: (num + prior * w) / (den + w)  # noqa: E731
    r = pd.DataFrame(index=pre.index)
    r["slpm"] = sh(g("sig_l"), mins, P["slpm"], PRIOR_MIN)
    r["sapm"] = sh(g("opp_sig_l"), mins, P["sapm"], PRIOR_MIN)
    r["sig_acc"] = sh(g("sig_l"), g("sig_a"), P["sig_acc"], PRIOR_ATT)
    r["sig_def"] = 1 - sh(g("opp_sig_l"), g("opp_sig_a"), 1 - P["sig_def"], PRIOR_ATT)
    r["td15"] = sh(g("td_l") * 15, mins, P["td15"], PRIOR_MIN)
    r["td_acc"] = sh(g("td_l"), g("td_a"), P["td_acc"], PRIOR_ATT / 3)
    r["td_def"] = 1 - sh(g("opp_td_l"), g("opp_td_a"), 1 - P["td_def"], PRIOR_ATT / 3)
    r["kd15"] = sh(g("kd") * 15, mins, P["kd15"], PRIOR_MIN)
    r["kdd15"] = sh(g("opp_kd") * 15, mins, P["kdd15"], PRIOR_MIN)
    r["sub15"] = sh(g("sub_att") * 15, mins, P["sub15"], PRIOR_MIN)
    r["ctrl_share"] = sh(g("ctrl"), g("stat_secs"), P["ctrl_share"], PRIOR_MIN * 60)
    r["opp_ctrl_share"] = sh(g("opp_ctrl"), g("stat_secs"), P["opp_ctrl_share"], PRIOR_MIN * 60)
    for c in ("head", "leg", "ground", "clinch"):
        r[f"{c}_share"] = sh(g(f"{c}_l"), g("sig_l"), P[f"{c}_share"], PRIOR_ATT)
    r["win_rate"] = sh(g("is_win"), fights, 0.5, PRIOR_FIGHTS)
    r["ko_off"] = sh(g("ko_win"), rnd, P["ko_off"], PRIOR_ROUNDS)
    r["ko_def"] = sh(g("ko_loss"), rnd, P["ko_def"], PRIOR_ROUNDS)
    r["sub_off"] = sh(g("sub_win"), rnd, P["sub_off"], PRIOR_ROUNDS)
    r["sub_def"] = sh(g("sub_loss"), rnd, P["sub_def"], PRIOR_ROUNDS)
    r["dec_share"] = sh(g("dec_win") + g("dec_loss"), fights, P["dec_share"], PRIOR_FIGHTS)
    r["last3"] = pre["last3"]
    r["n_prior"] = pre["n_prior"]
    r["five_rd_exp"] = g("five_rd")
    return r


# --------------------------------------------------------- matchup builder
def matchup_features(A: pd.DataFrame, Bf: pd.DataFrame) -> pd.DataFrame:
    """A and B are aligned per-fighter frames (rates + elo + age/reach/height/days_off)."""
    X = pd.DataFrame(index=A.index)
    for c in RATE_COLS + ["elo", "age", "reach", "height", "days_off", "n_prior"]:
        X[f"d_{c}"] = A[c].values - Bf[c].values
    # A's offense vs B's defense interactions (what the simulator also uses)
    X["d_strike_edge"] = (A["slpm"].values - Bf["sapm"].values) - (Bf["slpm"].values - A["sapm"].values)
    X["d_ko_edge"] = A["ko_off"].values * Bf["ko_def"].values - Bf["ko_off"].values * A["ko_def"].values
    X["d_sub_edge"] = A["sub_off"].values * Bf["sub_def"].values - Bf["sub_off"].values * A["sub_def"].values
    X["d_grap_edge"] = (A["td15"].values * (1 - Bf["td_def"].values)) - (Bf["td15"].values * (1 - A["td_def"].values))
    for c in ("elo", "age", "n_prior"):
        X[f"a_{c}"] = A[c].values
        X[f"b_{c}"] = Bf[c].values
    return X


def _physicals(frame: pd.DataFrame, fighters: pd.DataFrame, on_date: pd.Series) -> pd.DataFrame:
    p = frame[["fighter"]].merge(fighters, on="fighter", how="left")
    p.index = frame.index
    out = pd.DataFrame(index=frame.index)
    out["age"] = (pd.to_datetime(on_date).values - p["dob"].values) / np.timedelta64(1, "D") / 365.25
    out["reach"] = p["reach"].values
    out["height"] = p["height"].values
    return out


def build_training_table(fights, long, fighters, prior_before="2023-01-01", df_lib=None):
    """One row per decided fight with A/B features, label a_win, and metadata."""
    long_e, elo_now = add_elo(long, fights)
    P = league_priors(long, before=prior_before)
    pre = cumulative(long, df_lib=df_lib)
    pre = pre.merge(long_e[["fight_id", "fighter", "elo"]], on=["fight_id", "fighter"], how="left")
    R = rates(pre, P)
    per = pd.concat([pre[["fight_id", "fighter", "date", "days_off", "elo"]], R], axis=1)
    per = per.loc[:, ~per.columns.duplicated()]

    fx = fights[fights["a_win"].notna()].copy()
    A = fx[["fight_id", "a", "date"]].rename(columns={"a": "fighter"}).merge(
        per.drop(columns="date"), on=["fight_id", "fighter"], how="left")
    Bf = fx[["fight_id", "b", "date"]].rename(columns={"b": "fighter"}).merge(
        per.drop(columns="date"), on=["fight_id", "fighter"], how="left")
    A = pd.concat([A, _physicals(A, fighters, A["date"])], axis=1)
    Bf = pd.concat([Bf, _physicals(Bf, fighters, Bf["date"])], axis=1)
    for d in (A, Bf):
        d["days_off"] = d["days_off"].fillna(365).clip(upper=1500)

    X = matchup_features(A, Bf)
    X["sched_rounds"] = fx["sched_rounds"].fillna(3).values
    X["title"] = fx["title"].values
    X["women"] = fx["women"].values
    meta = fx[["fight_id", "date", "a", "b", "a_win", "method", "end_round", "sched_rounds",
               "weightclass", "EVENT"]].reset_index(drop=True)
    sim_cols = ["ko_off", "ko_def", "sub_off", "sub_def", "elo"]
    simA = A[sim_cols].add_prefix("a_").reset_index(drop=True)
    simB = Bf[sim_cols].add_prefix("b_").reset_index(drop=True)
    return X.reset_index(drop=True), meta, pd.concat([simA, simB], axis=1), P


def swap(X: pd.DataFrame) -> pd.DataFrame:
    """Same fights seen from the other corner (removes the 'winner listed first' bias)."""
    S = X.copy()
    for c in X.columns:
        if c.startswith("d_"):
            S[c] = -X[c]
    for c in ("elo", "age", "n_prior"):
        S[f"a_{c}"], S[f"b_{c}"] = X[f"b_{c}"], X[f"a_{c}"]
    return S


class FighterIndex:
    """Current (post-last-fight) features for live predictions in the app."""

    def __init__(self, fights, long, fighters, P):
        _, self.elo = add_elo(long, fights)
        cur = current_sums(long)
        R = rates(cur, P)
        self.table = pd.concat([cur[["fighter", "last_date"]], R], axis=1).set_index("fighter")
        self.table["elo"] = self.table.index.map(self.elo).astype(float)
        self.fighters = fighters.set_index("fighter")
        last = long.sort_values("date").groupby("fighter").tail(1).set_index("fighter")
        self.table["weightclass"] = last["weightclass"]
        self.table["women"] = last["women"]
        self.long = long

    def names(self, min_fights: int = 1, active_since: str | None = None):
        t = self.table[self.table["n_prior"] >= min_fights]
        if active_since:
            t = t[t["last_date"] >= pd.Timestamp(active_since)]
        return sorted(t.index)

    def row(self, name: str, on_date) -> pd.DataFrame:
        r = self.table.loc[[name]].copy()
        on = pd.Timestamp(on_date)
        r["days_off"] = min((on - r["last_date"].iloc[0]).days, 1500)
        dob = self.fighters["dob"].get(name, pd.NaT)
        r["age"] = (on - dob).days / 365.25 if pd.notna(dob) else np.nan
        r["reach"] = self.fighters["reach"].get(name, np.nan)
        r["height"] = self.fighters["height"].get(name, np.nan)
        return r.reset_index(drop=True)

    def matchup(self, a: str, b: str, sched_rounds=3, title=0, on_date=None):
        on_date = on_date or pd.Timestamp.today().normalize()
        A, Bf = self.row(a, on_date), self.row(b, on_date)
        X = matchup_features(A, Bf)
        X["sched_rounds"] = sched_rounds
        X["title"] = title
        X["women"] = int(self.table.loc[a, "women"] or 0)
        return X, A, Bf
