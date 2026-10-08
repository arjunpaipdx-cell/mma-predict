"""Sportsbook pricing math: fair odds, margin ("vig"), de-vigging, expected value.

A book starts from a fair probability p for each outcome, then adds a margin so
the implied probabilities sum to more than 100%. That excess is the overround,
e.g. -110 / -110 is 52.4% + 52.4% = 104.8%, so a 4.8% overround.

Ways to spread the margin across outcomes:
  multiplicative  q_i = p_i · (1 + O)           (same % on every outcome)
  power           q_i = p_i^k,  Σ q_i = 1 + O    (more margin on longshots, which
                                                  mirrors the favourite-longshot
                                                  bias seen in real betting markets)
De-vigging runs the same step backwards: given a book's prices, recover the
fair probabilities it implies.
"""
from __future__ import annotations

import numpy as np


# ---------- conversions
def prob_to_american(p: float) -> int:
    p = float(np.clip(p, 1e-4, 1 - 1e-4))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def american_to_prob(a: float) -> float:
    a = float(a)
    return -a / (-a + 100) if a < 0 else 100 / (a + 100)


def prob_to_decimal(p: float) -> float:
    return 1.0 / max(float(p), 1e-6)


def fmt_american(a: int) -> str:
    return f"+{a}" if a > 0 else str(a)


# ---------- adding margin
def _solve_power(p: np.ndarray, target: float) -> float:
    """Find k with Σ p_i^k = target (k < 1 inflates small p more)."""
    lo, hi = 0.3, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.sum(p**mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def add_margin(p, overround: float = 0.045, method: str = "power") -> np.ndarray:
    """Fair probabilities (sum to 1) -> book implied probabilities (sum to 1 + overround)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1)
    p = p / p.sum()
    if method == "multiplicative":
        return p * (1 + overround)
    k = _solve_power(p, 1 + overround)
    return p**k


def remove_margin(q, method: str = "power") -> np.ndarray:
    """Book implied probabilities -> fair probabilities (de-vig)."""
    q = np.asarray(q, dtype=float)
    if method == "multiplicative":
        return q / q.sum()
    lo, hi = 1.0, 3.0  # find k >= 1 with Σ q_i^k = 1
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.sum(q**mid) > 1:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    return q**k


def expected_value(p_model: float, american: float) -> float:
    """EV per $1 staked if the true win chance is p_model."""
    dec = 1 + (american / 100 if american > 0 else 100 / -american)
    return p_model * (dec - 1) - (1 - p_model)


# ---------- markets from the simulator output
def markets(summary: dict, a: str, b: str, rounds: int) -> dict[str, list[tuple[str, float]]]:
    """Fair probabilities for each market, each market's outcomes summing to 1.

    Within a round, finishes are assumed uniform in time when splitting a round
    in half for x.5 totals (a simplification: real finishes skew slightly early).
    """
    m = summary["by_method"]
    r = summary["by_round"]
    fin_by_round = [r["A"][i] + r["B"][i] for i in range(rounds)]
    dec = m["A"]["DEC"] + m["B"]["DEC"]
    fin_by_round[-1] -= dec  # by_round's last entry includes decisions
    out = {
        "Moneyline": [(a, summary["p_a"]), (b, summary["p_b"])],
        "Method of victory": [
            (f"{a} by KO/TKO", m["A"]["KO"]), (f"{a} by submission", m["A"]["SUB"]), (f"{a} by decision", m["A"]["DEC"]),
            (f"{b} by KO/TKO", m["B"]["KO"]), (f"{b} by submission", m["B"]["SUB"]), (f"{b} by decision", m["B"]["DEC"]),
        ],
        "Goes the distance": [("Yes", dec), ("No", 1 - dec)],
    }
    for line in [x + 0.5 for x in range(1, rounds)]:
        whole = int(line)
        under = sum(fin_by_round[:whole]) + 0.5 * fin_by_round[whole]
        out[f"Total rounds {line}"] = [(f"Over {line}", 1 - under), (f"Under {line}", under)]
    rb = []
    for w, name in (("A", a), ("B", b)):
        for i in range(rounds):
            v = r[w][i] - (m[w]["DEC"] if i == rounds - 1 else 0)
            rb.append((f"{name} in round {i + 1}", v))
        rb.append((f"{name} by decision", m[w]["DEC"]))
    out["Round betting"] = rb
    return out


def price_board(mk: dict, overround: float, method: str = "power") -> dict[str, list[dict]]:
    """Attach fair and book prices to every market outcome."""
    board = {}
    for name, outcomes in mk.items():
        labels = [o[0] for o in outcomes]
        p = np.clip(np.array([o[1] for o in outcomes]), 1e-6, 1)
        p = p / p.sum()
        # bigger markets carry more margin at real books; scale overround with outcome count
        ovr = overround * (1 + 0.5 * max(len(p) - 2, 0))
        q = add_margin(p, ovr, method)
        board[name] = [
            {"Outcome": l, "Fair %": pi, "Fair odds": fmt_american(prob_to_american(pi)),
             "Book %": qi, "Book odds": fmt_american(prob_to_american(min(qi, 0.9999))),
             "Decimal": round(prob_to_decimal(qi), 2)}
            for l, pi, qi in zip(labels, p, q)
        ]
    return board
