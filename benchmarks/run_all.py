"""CPU vs GPU benchmarks for every stage of the pipeline.

    python benchmarks/run_all.py            # on Colab with a T4: CPU and GPU columns
    python benchmarks/run_all.py --quick    # smaller sizes

Writes artifacts/benchmarks.json and artifacts/benchmarks.png.
Each GPU timing does one warm-up run first (CUDA context + kernel compile).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmapred import backend as B  # noqa: E402
from mmapred.data import download, load  # noqa: E402
from mmapred.features import build_training_table, cumulative  # noqa: E402
from mmapred.simulate import simulate  # noqa: E402
from mmapred.train import ART, augment  # noqa: E402


def timeit(fn, warmup=True, sync=None):
    if warmup:
        fn()
        sync and sync()
    t = time.perf_counter()
    fn()
    sync and sync()
    return time.perf_counter() - t


def main(quick=False):
    print(B.describe())
    download()
    fights, long, fighters = load()
    results = {"backend": B.describe(), "stages": {}}
    sync = None
    if B.GPU:
        import cupy as cp

        sync = cp.cuda.Device().synchronize

    # 1. Feature engineering: pandas vs cuDF (data replicated to stress the groupby)
    import pandas as pd

    reps = 5 if quick else 20
    big = pd.concat([long.assign(fighter=long["fighter"] + f"#{i}", fight_id=long["fight_id"] + i * 10**6)
                     for i in range(reps)], ignore_index=True)
    st = {"rows": len(big), "cpu_s": timeit(lambda: cumulative(big, df_lib=pd), warmup=False)}
    if B.CUDF:
        import cudf

        st["gpu_s"] = timeit(lambda: cumulative(big, df_lib=cudf), sync=sync)
    results["stages"]["features (cuDF)"] = st
    print("features", st)

    # 2. XGBoost training: hist on CPU vs CUDA (fixed 500 trees, no early stopping)
    import xgboost as xgb

    X, meta, _, _ = build_training_table(fights, long, fighters)
    y = meta["a_win"].values.astype(int)
    Xa, ya = augment(X, y)
    Xa = pd.concat([Xa] * (2 if quick else 8), ignore_index=True)
    ya = np.tile(ya, 2 if quick else 8)
    kw = dict(n_estimators=500, max_depth=6, learning_rate=0.05, tree_method="hist")
    st = {"rows": len(Xa), "cpu_s": timeit(lambda: xgb.XGBClassifier(**kw, device="cpu").fit(Xa, ya), warmup=False)}
    if B.GPU:
        st["gpu_s"] = timeit(lambda: xgb.XGBClassifier(**kw, device="cuda").fit(Xa, ya))
    results["stages"]["xgboost train"] = st
    print("xgboost", st)

    # 3. Monte Carlo: every pair in a 100-fighter division × 20k sims
    rng = np.random.default_rng(0)
    M = 1000 if quick else 4950
    S = 5_000 if quick else 20_000
    h = [rng.uniform(0.02, 0.15, M) for _ in range(4)]
    p = rng.uniform(0.2, 0.8, M)
    rounds = np.full(M, 3)
    st = {"fights_simulated": M * S,
          "cpu_s": timeit(lambda: simulate(*h, p, rounds, n_sims=S, xp=np), warmup=False)}
    if B.GPU:
        import cupy as cp

        st["gpu_s"] = timeit(lambda: simulate(*h, p, rounds, n_sims=S, xp=cp), sync=sync)
    results["stages"]["monte carlo (CuPy)"] = st
    print("monte carlo", st)

    for st in results["stages"].values():
        if "gpu_s" in st:
            st["speedup"] = st["cpu_s"] / st["gpu_s"]

    ART.mkdir(exist_ok=True)
    (ART / "benchmarks.json").write_text(json.dumps(results, indent=2))
    plot(results)
    print(json.dumps(results, indent=2))


def plot(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results["stages"])
    cpu = [results["stages"][n]["cpu_s"] for n in names]
    gpu = [results["stages"][n].get("gpu_s", np.nan) for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150)
    yy = np.arange(len(names))
    ax.barh(yy + 0.2, cpu, height=0.38, color="#9aa0a6", label="CPU")
    ax.barh(yy - 0.2, gpu, height=0.38, color="#76b900", label="GPU")
    for i, n in enumerate(names):
        s = results["stages"][n].get("speedup")
        if s:
            ax.text(max(cpu[i], gpu[i]) * 1.02, i, f"{s:.1f}× faster", va="center", fontsize=9)
    ax.set_yticks(yy, names)
    ax.invert_yaxis()
    ax.set_xlabel("seconds (lower is better)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(results["backend"], fontsize=8, loc="left", color="#555")
    fig.tight_layout()
    fig.savefig(ART / "benchmarks.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
