"""Exploratory checks: train/test covariate shift and basic statistics."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

X, y, Xte = load(1)
fr = lambda A: (np.abs(A) == 1).mean(0)
print("var1 fraction of entries clipped at +-1 (train):", fr(X).round(3), "(test):", fr(Xte).round(3))
print("var1 std train/test:", X.std(0).round(3), Xte.std(0).round(3))
X2, y2, Xte2 = load(2)
print("var2 fraction clipped train/test:", fr(X2).round(3), fr(Xte2).round(3))
print("var2 std train/test:", X2.std(0).round(3), Xte2.std(0).round(3))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
k = np.arange(1, 7); wd = .38
ax[0].bar(k - wd/2, fr(X), wd, label="train"); ax[0].bar(k + wd/2, fr(Xte), wd, label="test")
ax[0].set_xlabel("feature number i (input $x_i$)"); ax[0].set_ylabel("fraction of values exactly at $\\pm1$"); ax[0].legend(); ax[0].set_title("var1: train vs test clipping")
nc = lambda A: (np.abs(A) == 1).sum(1)
ax[1].hist([nc(X), nc(Xte)], bins=np.arange(-.5, 7.5, 1), density=True, label=["train", "test"])
ax[1].set_xlabel("number of features (out of 6) at $\\pm1$ in a row"); ax[1].set_ylabel("fraction of rows"); ax[1].set_title("var1: rows with many saturated features"); ax[1].legend()
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIGS / "var1_shift.png", dpi=170)
