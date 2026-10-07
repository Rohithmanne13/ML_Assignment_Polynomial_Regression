"""Model selection: sweep polynomial degree x regulariser (Ridge / Lasso) x alpha
with repeated 5-fold CV and pick the configuration with the lowest CV MSE.

Usage:  python src/select_model.py --var 1
Writes: results/var{v}_sweep.csv, results/var{v}_best.json, figures/var{v}_degree_sweep.png
"""
import argparse, json, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

CFG = {
    1: dict(ridge_deg=range(1, 9), lasso_deg=range(1, 8),
            ridge_alphas=[0.01, 0.1, 1, 3, 10, 30, 100],
            lasso_alphas=[0.005, 0.01, 0.02, 0.03, 0.05]),
    2: dict(ridge_deg=range(2, 21), lasso_deg=range(2, 13),
            ridge_alphas=[0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30],
            lasso_alphas=[0.003, 0.01, 0.03, 0.1]),
}


def sweep(var, X, y, verbose=True):
    c = CFG[var]
    rows, t0 = [], time.time()
    for kind, degs, alphas in [("ridge", c["ridge_deg"], c["ridge_alphas"]),
                               ("lasso", c["lasso_deg"], c["lasso_alphas"])]:
        for deg in degs:
            for a in alphas:
                mk = lambda: make_model(kind, deg, a)
                row = dict(model=kind, degree=deg, alpha=a)
                row["cv_mse"] = repeated_cv_mse(mk, X, y)
                if var == 1:  # extra diagnostics for the covariate-shifted problem
                    row["shift_mse"] = norm_shift_mse(mk, X, y)
                rows.append(row)
            if verbose:
                print(f"  {kind} degree {deg:2d} done  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def best_row(df):
    return df.loc[df["cv_mse"].idxmin()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--var", type=int, required=True)
    var = ap.parse_args().var
    X, y, Xte = load(var)

    # (1) full sweep on all training data
    df = sweep(var, X, y)
    df.to_csv(RESULTS / f"var{var}_sweep.csv", index=False)
    b = best_row(df)
    best = dict(model=b["model"], degree=int(b["degree"]), alpha=float(b["alpha"]), cv_mse=float(b["cv_mse"]))

    # (2) Extra validation step: re-run the full grid search on an 80/20 split
    #     and test the winning model on the hold-out set to verify our CV scores.
    idx = np.random.RandomState(123).permutation(len(X)); cut = int(0.8 * len(X))
    a_, b_ = idx[:cut], idx[cut:]
    df80 = sweep(var, X[a_], y[a_], verbose=False)
    b80 = best_row(df80)
    m = make_model(b80["model"], int(b80["degree"]), float(b80["alpha"])).fit(X[a_], y[a_])
    p = m.predict(X[b_])
    best["holdout"] = dict(
        selected_on_80pct=dict(model=b80["model"], degree=int(b80["degree"]), alpha=float(b80["alpha"])),
        mse=float(((p - y[b_]) ** 2).mean()), r2=float(r2(y[b_], p)))
    json.dump(best, open(RESULTS / f"var{var}_best.json", "w"), indent=2)
    print(json.dumps(best, indent=2))

    # (3) figure: best CV MSE per degree for each regulariser
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for kind, col in [("ridge", "C0"), ("lasso", "C3")]:
        s = df[df.model == kind].groupby("degree")["cv_mse"].min()
        ax.plot(s.index, s.values, "o-", color=col, label=f"{kind.capitalize()} (best alpha per degree)", ms=4)
    ax.axvline(best["degree"], color="gray", ls="--", lw=1)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale("log"); ax.set_xlabel("polynomial degree"); ax.set_ylabel("5-fold CV MSE (log scale)")
    ax.set_title(f"var{var}: CV error vs. polynomial degree"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / f"var{var}_degree_sweep.png", dpi=170)
