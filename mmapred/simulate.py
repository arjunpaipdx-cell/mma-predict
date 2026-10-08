"""Round-by-round Monte Carlo fight simulator (CuPy on GPU, NumPy on CPU).

For each matchup we simulate S fights. In every round, each fighter has a
chance (hazard) of finishing the other by KO or by submission:

    ko_hazard(A)  = A's KO-wins-per-round  × B's KO-losses-per-round / league rate
    sub_hazard(A) = same idea for submissions

Each simulated fight also draws a random "form" multiplier per fighter (good
night / bad night), and hazards rise slightly in later rounds (fatigue,
damage). If nobody is finished, the fight goes to the judges. The judges'
probability is chosen so the simulator's overall win probability matches the
XGBoost model, which keeps the two models consistent while the simulator adds
*how* and *when*.

All matchups × all simulations run as one batch of arrays, which is where the
GPU parallelism pays off (see benchmarks/bench_sim.py).
"""
from __future__ import annotations

import numpy as np

from . import backend as B

METHODS = ("KO", "SUB", "DEC")


def hazards(a_ko_off, a_sub_off, a_ko_def, a_sub_def, b_ko_off, b_sub_off, b_ko_def, b_sub_def,
            league_ko: float, league_sub: float, shrink: float = 0.8, base: float = 0.8,
            cap: float = 0.6):
    """Per-round finish hazards for both fighters (arrays of shape [M]).

    Raw matchup hazards are pulled toward `base` × league rate by `shrink`
    (shrink=1 keeps them raw). shrink/base/form_sigma were tuned on the 2023
    validation fights only, never on the 2024+ test set.
    """
    lk, ls = league_ko * base, league_sub * base
    f = lambda raw, lv: np.clip(lv + (raw - lv) * shrink, 0, cap)  # noqa: E731
    ko_a = f(a_ko_off * b_ko_def / league_ko, lk)
    sub_a = f(a_sub_off * b_sub_def / league_sub, ls)
    ko_b = f(b_ko_off * a_ko_def / league_ko, lk)
    sub_b = f(b_sub_off * a_sub_def / league_sub, ls)
    return ko_a, sub_a, ko_b, sub_b


def simulate(ko_a, sub_a, ko_b, sub_b, p_win, rounds, n_sims: int = 100_000, form_sigma: float = 0.7,
             fatigue: float = 0.08, seed: int = 0, xp=None, chunk_elems: int = 2**24):
    """Simulate M matchups × n_sims fights.

    Args are length-M arrays (host or device). `p_win` is the model's P(A wins);
    `rounds` is scheduled rounds (3 or 5) per matchup.

    Returns dict of host NumPy arrays:
        probs[M, 2, 3, 5]: P(winner ∈ {A,B}, method ∈ {KO,SUB,DEC}, round 1..5)
        p_a_sim[M]:        simulated P(A wins)
    """
    xp = xp or B.xp
    M = len(ko_a)
    R = 5
    out = np.zeros((M, 2, 3, R), dtype=np.float64)
    rows_per_chunk = max(1, chunk_elems // n_sims)
    for s in range(0, M, rows_per_chunk):
        e = min(M, s + rows_per_chunk)
        out[s:e] = _simulate_chunk(
            *[xp.asarray(np.asarray(v[s:e], dtype=np.float32)) for v in (ko_a, sub_a, ko_b, sub_b, p_win)],
            xp.asarray(np.asarray(rounds[s:e], dtype=np.int32)), n_sims, form_sigma, fatigue, seed + s, xp)
    return {"probs": out, "p_a_sim": out[:, 0].sum(axis=(1, 2))}


def _simulate_chunk(ko_a, sub_a, ko_b, sub_b, p_win, rounds, S, sigma, fatigue, seed, xp):
    rng = xp.random.default_rng(seed)
    m = ko_a.shape[0]
    f32 = xp.float32
    # per-fight form: lognormal with mean 1
    form_a = xp.exp(rng.standard_normal((m, S), dtype=f32) * sigma - sigma**2 / 2)
    form_b = xp.exp(rng.standard_normal((m, S), dtype=f32) * sigma - sigma**2 / 2)
    alive = xp.ones((m, S), dtype=xp.bool_)
    # result code: 0 = undecided; winner*15 + method*5 + round + 1 otherwise
    res = xp.zeros((m, S), dtype=xp.int16)
    for r in range(5):
        active = alive & (rounds[:, None] > r)
        grow = 1.0 + fatigue * r
        ha = ((ko_a + sub_a)[:, None] * form_a * grow).clip(0, 0.9)
        hb = ((ko_b + sub_b)[:, None] * form_b * grow).clip(0, 0.9)
        u = rng.random((m, S), dtype=f32)
        # who lands the finish first this round (split ties by relative hazard)
        a_fin = u < ha * (1 - hb / 2)
        b_fin = (~a_fin) & (u < ha * (1 - hb / 2) + hb * (1 - ha / 2))
        v = rng.random((m, S), dtype=f32)
        a_ko = v < (ko_a / xp.maximum(ko_a + sub_a, 1e-9))[:, None]
        b_ko = v < (ko_b / xp.maximum(ko_b + sub_b, 1e-9))[:, None]
        code_a = xp.where(a_ko, 0, 5) + r + 1
        code_b = 15 + xp.where(b_ko, 0, 5) + r + 1
        res = xp.where(active & a_fin, code_a, res)
        res = xp.where(active & b_fin, code_b, res)
        alive = alive & ~(active & (a_fin | b_fin))

    # finish probabilities -> choose judges' probability to match the model's p_win
    counts = xp.zeros((m, 30), dtype=f32)
    for c in range(1, 31):
        counts[:, c - 1] = (res == c).sum(axis=1)
    counts = counts / S
    p_dec = alive.mean(axis=1)
    p_a_fin = counts[:, :15].sum(axis=1)
    q = xp.clip((p_win - p_a_fin) / xp.maximum(p_dec, 1e-6), 0.02, 0.98)

    out = xp.zeros((m, 2, 3, 5), dtype=f32)
    c3 = counts.reshape(m, 2, 3, 5)
    out[:, :, 0:2, :] = c3[:, :, 0:2, :]
    last = (rounds - 1).astype(xp.int64)
    idx = xp.arange(m)
    out[idx, 0, 2, last] = p_dec * q
    out[idx, 1, 2, last] = p_dec * (1 - q)
    return B.to_host(out).astype(np.float64)


def summarize(probs: np.ndarray, rounds: int) -> dict:
    """Readable summary for one matchup (probs: [2, 3, 5])."""
    p = probs[:, :, :rounds]
    return {
        "p_a": float(p[0].sum()),
        "p_b": float(p[1].sum()),
        "by_method": {w: {m: float(p[i, j].sum()) for j, m in enumerate(METHODS)} for i, w in enumerate("AB")},
        "by_round": {w: [float(x) for x in p[i].sum(axis=0)] for i, w in enumerate("AB")},
        "p_finish": float(p[:, :2].sum()),
        "p_distance": float(p[:, 2].sum()),
    }
