"""Rating systems: Elo and Glicko-2.

Both are run fight by fight in date order, and every fight stores each
fighter's rating *before* the fight, so features never see the result.

Elo
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    R_A' = R_A + K (S_A - E_A)            (K bigger for new fighters and finishes)

Glicko-2 (Glickman, 2012), one fight = one rating period
    μ = (r - 1500) / 173.7178,  φ = RD / 173.7178
    before a fight, inactivity inflates uncertainty: φ* = sqrt(φ² + σ² · t)
    g(φ) = 1 / sqrt(1 + 3φ²/π²)
    E = 1 / (1 + exp(-g(φ_B)(μ_A - μ_B)))
    v = 1 / (g² E (1 - E)),  Δ = v g (s - E)
    σ' from the Illinois root-finding step, then
    φ' = 1 / sqrt(1/(φ*² + σ'²) + 1/v),  μ' = μ + φ'² g (s - E)

Glicko's advantage over Elo is the rating deviation (RD): a fighter with few or
old fights has a wide RD, so predictions involving them are pulled toward 50%.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SCALE = 173.7178
TAU = 0.5            # volatility constraint
INIT_RD = 250.0      # UFC debutants already have pro records, so not the full 350
INIT_SIGMA = 0.06
PERIOD_DAYS = 180.0  # inactivity unit for RD inflation
ELO_K = 40.0


def _g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / math.pi**2)


def _new_sigma(phi, sigma, v, delta):
    a = math.log(sigma * sigma)
    eps = 1e-6

    def f(x):
        ex = math.exp(x)
        num = ex * (delta * delta - phi * phi - v - ex)
        den = 2.0 * (phi * phi + v + ex) ** 2
        return num / den - (x - a) / (TAU * TAU)

    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU
    fA, fB = f(A), f(B)
    while abs(B - A) > eps:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
    return math.exp(A / 2.0)


def glicko_update(mu, phi, sigma, mu_o, phi_o, s):
    g = _g(phi_o)
    E = 1.0 / (1.0 + math.exp(-g * (mu - mu_o)))
    v = 1.0 / (g * g * E * (1 - E))
    delta = v * g * (s - E)
    sigma2 = _new_sigma(phi, sigma, v, delta)
    phi_star = math.sqrt(phi * phi + sigma2 * sigma2)
    phi2 = 1.0 / math.sqrt(1.0 / (phi_star * phi_star) + 1.0 / v)
    mu2 = mu + phi2 * phi2 * g * (s - E)
    return mu2, phi2, sigma2


def inflate(phi, sigma, days):
    """Uncertainty grows while a fighter is inactive."""
    t = max(days, 0) / PERIOD_DAYS
    return min(math.sqrt(phi * phi + sigma * sigma * t), INIT_RD / SCALE)


def glicko_prob(r_a, rd_a, r_b, rd_b):
    """P(A beats B) under Glicko-2, accounting for both fighters' uncertainty."""
    mu_a, mu_b = (np.asarray(r_a) - 1500) / SCALE, (np.asarray(r_b) - 1500) / SCALE
    phi = np.sqrt((np.asarray(rd_a) / SCALE) ** 2 + (np.asarray(rd_b) / SCALE) ** 2)
    g = 1.0 / np.sqrt(1.0 + 3.0 * phi**2 / np.pi**2)
    return 1.0 / (1.0 + np.exp(-g * (mu_a - mu_b)))


def elo_prob(e_a, e_b):
    return 1.0 / (1.0 + 10 ** ((np.asarray(e_b) - np.asarray(e_a)) / 400.0))


def compute(fights: pd.DataFrame):
    """Run both systems over all fights.

    Returns (pre, state):
      pre:   one row per (fight_id, fighter) with pre-fight elo, g_r, g_rd and the
             opponent's pre-fight elo / g_r (for strength of schedule)
      state: {fighter: dict(elo, mu, phi, sigma, last_date, n)} after the last fight
    """
    st: dict[str, dict] = {}
    rows = []
    for fid, date, a, b, y, m in fights[["fight_id", "date", "a", "b", "a_win", "method"]].itertuples(index=False):
        sa = st.setdefault(a, dict(elo=1500.0, mu=0.0, phi=INIT_RD / SCALE, sigma=INIT_SIGMA, last_date=None, n=0))
        sb = st.setdefault(b, dict(elo=1500.0, mu=0.0, phi=INIT_RD / SCALE, sigma=INIT_SIGMA, last_date=None, n=0))
        for s in (sa, sb):
            if s["last_date"] is not None:
                s["phi"] = inflate(s["phi"], s["sigma"], (date - s["last_date"]).days)
        ga_r, gb_r = 1500 + SCALE * sa["mu"], 1500 + SCALE * sb["mu"]
        rows.append((fid, a, sa["elo"], ga_r, SCALE * sa["phi"], sb["elo"], gb_r))
        rows.append((fid, b, sb["elo"], gb_r, SCALE * sb["phi"], sa["elo"], ga_r))

        if y != y:  # draw or no contest
            if m is None:
                continue
            y = 0.5
        # Elo
        ea = 1.0 / (1.0 + 10 ** ((sb["elo"] - sa["elo"]) / 400.0))
        mult = 1.2 if m in ("KO", "SUB") else 1.0
        ka = ELO_K * (1.5 if sa["n"] < 5 else 1.0) * mult
        kb = ELO_K * (1.5 if sb["n"] < 5 else 1.0) * mult
        new_ea, new_eb = sa["elo"] + ka * (y - ea), sb["elo"] - kb * (y - ea)
        # Glicko-2 (simultaneous: both use pre-fight values)
        mu_a, phi_a, sig_a = glicko_update(sa["mu"], sa["phi"], sa["sigma"], sb["mu"], sb["phi"], y)
        mu_b, phi_b, sig_b = glicko_update(sb["mu"], sb["phi"], sb["sigma"], sa["mu"], sa["phi"], 1 - y)
        sa.update(elo=new_ea, mu=mu_a, phi=phi_a, sigma=sig_a, last_date=date, n=sa["n"] + 1)
        sb.update(elo=new_eb, mu=mu_b, phi=phi_b, sigma=sig_b, last_date=date, n=sb["n"] + 1)

    pre = pd.DataFrame(rows, columns=["fight_id", "fighter", "elo", "g_r", "g_rd", "opp_elo", "opp_g_r"])
    return pre, st


def current(state: dict, name: str, on_date) -> dict:
    """Ratings for a live prediction, with RD inflated for time since the last fight."""
    s = state[name]
    days = (pd.Timestamp(on_date) - s["last_date"]).days if s["last_date"] is not None else 0
    return dict(elo=s["elo"], g_r=1500 + SCALE * s["mu"], g_rd=SCALE * inflate(s["phi"], s["sigma"], days))
