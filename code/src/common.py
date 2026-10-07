"""Shared helpers: data loading, model factory, cross-validation utilities."""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROLL = "BT2024144"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
PREDS = ROOT / "predictions"
for p in (RESULTS, FIGS, PREDS):
    p.mkdir(exist_ok=True)


def load(var):
    """Return X_train, y_train, X_test (numpy) for problem `var` (1 or 2)."""
    tr = pd.read_csv(DATA / f"{ROLL}_train_var{var}.csv")
    te = pd.read_csv(DATA / f"{ROLL}_test_var{var}.csv")
    return tr.drop(columns="y").values, tr["y"].values, te.values


def make_model(kind, degree, alpha):
    """Polynomial features (all terms with total degree <= `degree`) ->
    standardisation -> Ridge (L2) or Lasso (L1)."""
    reg = Ridge(alpha=alpha) if kind == "ridge" else Lasso(alpha=alpha, max_iter=100000, tol=1e-4)
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)


def repeated_cv_mse(make, X, y, n_splits=5, seeds=(0, 1)):
    """Mean held-out MSE over repeated K-fold CV."""
    errs = []
    for s in seeds:
        for tr, va in KFold(n_splits, shuffle=True, random_state=s).split(X):
            m = make().fit(X[tr], y[tr])
            e = (m.predict(X[va]) - y[va]) ** 2
            errs.append(e.mean())
    return float(np.mean(errs))


def norm_shift_mse(make, X, y, q=0.7):
    """Extrapolation-style check: train on the 70% of points closest to the
    origin, validate on the 30% farthest (mimics the heavier-tailed test set)."""
    r = np.linalg.norm(X, axis=1)
    hi = r > np.quantile(r, q)
    m = make().fit(X[~hi], y[~hi])
    return float(((m.predict(X[hi]) - y[hi]) ** 2).mean())




def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
