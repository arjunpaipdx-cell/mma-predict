"""Download and clean UFC data.

Source: https://github.com/Greco1899/scrape_ufc_stats — a community scraper
that mirrors UFCStats.com into CSVs and is refreshed regularly.

Output: one row per fight (`fights`) and one row per fighter per fight (`long`).
String parsing happens in pandas; the groupby-heavy feature work in
features.py is where cuDF takes over.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
FILES = [
    "ufc_event_details",
    "ufc_fight_results",
    "ufc_fight_stats",
    "ufc_fighter_tott",
]
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def download(force: bool = False, raw_dir: Path = RAW) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        out = raw_dir / f"{f}.csv"
        if out.exists() and not force:
            continue
        print(f"downloading {f}.csv")
        urllib.request.urlretrieve(BASE + f + ".csv", out)


# ---------- small parsers ----------
def _of(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'16 of 40' -> (16, 40)."""
    parts = s.astype(str).str.extract(r"(\d+)\s+of\s+(\d+)")
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def _mmss(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.extract(r"(\d+):(\d+)")
    return pd.to_numeric(parts[0], errors="coerce") * 60 + pd.to_numeric(parts[1], errors="coerce")


def _inches(s: pd.Series) -> pd.Series:
    """Height `5' 11"` or reach `72"` -> inches."""
    def conv(x):
        if not isinstance(x, str):
            return np.nan
        m = re.match(r"(\d+)'\s*(\d+)", x)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
        m = re.match(r"(\d+(?:\.\d+)?)\"", x.strip())
        return float(m.group(1)) if m else np.nan

    return s.map(conv)


def _method(m: str) -> str | None:
    m = str(m).strip()
    if m.startswith("KO/TKO") or m.startswith("TKO"):
        return "KO"
    if m.startswith("Submission"):
        return "SUB"
    if m.startswith("Decision"):
        return "DEC"
    return None  # DQ, overturned, could not continue, other


# ---------- main loaders ----------
def load(raw_dir: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (fights, long, fighters)."""
    ev = pd.read_csv(raw_dir / "ufc_event_details.csv")
    res = pd.read_csv(raw_dir / "ufc_fight_results.csv")
    st = pd.read_csv(raw_dir / "ufc_fight_stats.csv")
    tott = pd.read_csv(raw_dir / "ufc_fighter_tott.csv")

    for df in (ev, res, st):
        df["EVENT"] = df["EVENT"].astype(str).str.strip()
    for df in (res, st):
        df["BOUT"] = df["BOUT"].astype(str).str.strip()

    ev["date"] = pd.to_datetime(ev["DATE"], format="%B %d, %Y", errors="coerce")
    ev = ev.drop_duplicates("EVENT")

    # ---- fights ----
    f = res.merge(ev[["EVENT", "date"]], on="EVENT", how="left")
    names = f["BOUT"].str.split(r"\s+vs\.\s+", n=1, expand=True)
    f["a"] = names[0].str.strip()
    f["b"] = names[1].str.strip()
    oc = f["OUTCOME"].astype(str).str.strip()
    f["a_win"] = np.select([oc == "W/L", oc == "L/W"], [1.0, 0.0], default=np.nan)
    f["method"] = f["METHOD"].map(_method)
    # A stoppage due to injury (e.g. a knee giving out) isn't a knockout: keep the
    # win/loss but don't let it inflate KO power / KO vulnerability.
    injury = f["DETAILS"].astype(str).str.contains("Injury", case=False) & (f["method"] == "KO")
    f.loc[injury, "method"] = "INJ"
    f["end_round"] = pd.to_numeric(f["ROUND"], errors="coerce")
    f["end_secs"] = _mmss(f["TIME"])
    f["sched_rounds"] = pd.to_numeric(
        f["TIME FORMAT"].astype(str).str.extract(r"^(\d) Rnd \(5")[0], errors="coerce"
    )
    wc = f["WEIGHTCLASS"].astype(str)
    f["title"] = wc.str.contains("Title").astype(int)
    f["women"] = wc.str.contains("Women").astype(int)
    f["weightclass"] = (
        wc.str.replace(r"UFC |Women's |Interim |Title |Bout|Tournament|Ultimate Fighter.*", "", regex=True)
        .str.strip()
        .where(lambda s: s != "", "Open")
    )
    f["weightclass"] = np.where(f["women"] == 1, "W " + f["weightclass"], f["weightclass"])
    f["fight_secs"] = (f["end_round"] - 1) * 300 + f["end_secs"]
    f = f.dropna(subset=["date", "a", "b"]).drop_duplicates(["EVENT", "BOUT"])
    f = f.sort_values(["date", "EVENT", "BOUT"]).reset_index(drop=True)
    f["fight_id"] = np.arange(len(f))

    # ---- per-round stats -> per fighter per fight totals ----
    st = st.dropna(subset=["FIGHTER"]).copy()
    st["FIGHTER"] = st["FIGHTER"].str.strip()
    cols = {}
    for raw, key in [
        ("SIG.STR.", "sig"), ("TOTAL STR.", "tot"), ("TD", "td"), ("HEAD", "head"),
        ("BODY", "body"), ("LEG", "leg"), ("DISTANCE", "dist"), ("CLINCH", "clinch"), ("GROUND", "ground"),
    ]:
        cols[f"{key}_l"], cols[f"{key}_a"] = _of(st[raw])
    agg = pd.DataFrame(cols)
    agg["kd"] = pd.to_numeric(st["KD"], errors="coerce")
    agg["sub_att"] = pd.to_numeric(st["SUB.ATT"], errors="coerce")
    agg["rev"] = pd.to_numeric(st["REV."], errors="coerce")
    agg["ctrl"] = _mmss(st["CTRL"])
    agg[["EVENT", "BOUT", "fighter"]] = st[["EVENT", "BOUT", "FIGHTER"]].values
    tot = agg.groupby(["EVENT", "BOUT", "fighter"], as_index=False).sum(numeric_only=True)

    # ---- long format: one row per (fight, fighter) with opponent stats ----
    base = ["fight_id", "date", "EVENT", "BOUT", "method", "end_round", "fight_secs",
            "sched_rounds", "title", "women", "weightclass"]
    A = f[base + ["a", "b", "a_win"]].rename(columns={"a": "fighter", "b": "opp", "a_win": "win"})
    B = f[base + ["b", "a", "a_win"]].rename(columns={"b": "fighter", "a": "opp"})
    B["win"] = 1 - B["a_win"]
    B = B.drop(columns="a_win")
    long = pd.concat([A, B], ignore_index=True)

    stat_cols = [c for c in tot.columns if c not in ("EVENT", "BOUT", "fighter")]
    long = long.merge(tot, on=["EVENT", "BOUT", "fighter"], how="left")
    opp = tot.rename(columns={"fighter": "opp", **{c: f"opp_{c}" for c in stat_cols}})
    long = long.merge(opp, on=["EVENT", "BOUT", "opp"], how="left")
    long["has_stats"] = long["sig_a"].notna().astype(int)
    long[stat_cols + [f"opp_{c}" for c in stat_cols]] = long[stat_cols + [f"opp_{c}" for c in stat_cols]].fillna(0)
    long["fight_secs"] = long["fight_secs"].fillna(0)

    fin = long["method"].isin(["KO", "SUB"])
    long["ko_win"] = ((long["method"] == "KO") & (long["win"] == 1)).astype(int)
    long["sub_win"] = ((long["method"] == "SUB") & (long["win"] == 1)).astype(int)
    long["ko_loss"] = ((long["method"] == "KO") & (long["win"] == 0)).astype(int)
    long["sub_loss"] = ((long["method"] == "SUB") & (long["win"] == 0)).astype(int)
    long["dec_win"] = ((long["method"] == "DEC") & (long["win"] == 1)).astype(int)
    long["dec_loss"] = ((long["method"] == "DEC") & (long["win"] == 0)).astype(int)
    long["is_win"] = (long["win"] == 1).astype(int)
    long["is_loss"] = (long["win"] == 0).astype(int)
    long["rounds_fought"] = np.where(fin, long["end_round"] - 1 + long["fight_secs"].mod(300).div(300), long["fight_secs"] / 300)
    long["rounds_fought"] = long["rounds_fought"].fillna(0).clip(lower=0)
    long = long.sort_values(["date", "fight_id", "fighter"]).reset_index(drop=True)

    # ---- fighter physicals (name collisions: keep the first record) ----
    tott["FIGHTER"] = tott["FIGHTER"].astype(str).str.strip()
    fighters = pd.DataFrame({
        "fighter": tott["FIGHTER"],
        "height": _inches(tott["HEIGHT"]),
        "reach": _inches(tott["REACH"]),
        "stance": tott["STANCE"].fillna("Unknown"),
        "dob": pd.to_datetime(tott["DOB"], format="%b %d, %Y", errors="coerce"),
    }).drop_duplicates("fighter")

    return f, long, fighters


if __name__ == "__main__":
    download()
    f, long, fighters = load()
    print(f"{len(f):,} fights | {long.fighter.nunique():,} fighters | "
          f"{f.date.min().date()} → {f.date.max().date()}")
    print(f["method"].value_counts(dropna=False))
