"""Fit the selected model on ALL training data and write the prediction file.

Usage:  python src/train_predict.py --var 1
Reads:  results/var{v}_best.json (written by select_model.py)
Writes: predictions/BT2024144_pred_var{v}.csv, results/var{v}_final.json,
        figures/var{v}_diagnostics.png
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from common import *

ap = argparse.ArgumentParser(); ap.add_argument("--var", type=int, required=True)
var = ap.parse_args().var
X, y, Xte = load(var)
best = json.load(open(RESULTS / f"var{var}_best.json"))
kind, deg, alpha = best["model"], best["degree"], best["alpha"]

model = make_model(kind, deg, alpha).fit(X, y)
pred = model.predict(Xte)

# prediction file, same format as sample_submission.csv (single column `y`)
sample = pd.read_csv(DATA / "sample_submission.csv")
assert len(pred) == len(sample) == len(Xte), "row count must match the sample submission"
assert np.isfinite(pred).all()
out = pd.DataFrame({"y": pred})
out.to_csv(PREDS / f"{ROLL}_pred_var{var}.csv", index=False)

# summary stats
oof = cross_val_predict(make_model(kind, deg, alpha), X, y, cv=KFold(5, shuffle=True, random_state=0))
n_terms = model[-1].coef_.size
info = dict(var=var, model=kind, degree=deg, alpha=alpha, n_poly_terms=int(n_terms),
            n_nonzero=int((model[-1].coef_ != 0).sum()),
            train_mse=float(((model.predict(X) - y) ** 2).mean()), train_r2=float(r2(y, model.predict(X))),
            oof_cv_mse=float(((oof - y) ** 2).mean()), oof_cv_r2=float(r2(y, oof)),
            y_train_range=[float(y.min()), float(y.max())], y_pred_range=[float(pred.min()), float(pred.max())],
            y_train_mean_std=[float(y.mean()), float(y.std())], y_pred_mean_std=[float(pred.mean()), float(pred.std())])
json.dump(info, open(RESULTS / f"var{var}_final.json", "w"), indent=2)
print(json.dumps(info, indent=2))

# diagnostics figure: out-of-fold predicted vs actual, residuals, test-pred distribution
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
ax[0].scatter(y, oof, s=5, alpha=.5); lim = [y.min(), y.max()]; ax[0].plot(lim, lim, "r--", lw=1)
ax[0].set_xlabel("actual y"); ax[0].set_ylabel("5-fold out-of-fold prediction"); ax[0].set_title("OOF predicted vs actual")
ax[1].scatter(oof, y - oof, s=5, alpha=.5); ax[1].axhline(0, color="r", lw=1)
ax[1].set_xlabel("OOF prediction"); ax[1].set_ylabel("residual"); ax[1].set_title("Residuals")
ax[2].hist(y, bins=40, alpha=.6, density=True, label="train y"); ax[2].hist(pred, bins=40, alpha=.6, density=True, label="test prediction")
ax[2].set_xlabel("y"); ax[2].set_title("Distribution"); ax[2].legend()
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIGS / f"var{var}_diagnostics.png", dpi=170)
