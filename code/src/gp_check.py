"""Gaussian-process 5-fold CV MSE,
used in the report as a sanity check that polynomials are near the noise floor."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.model_selection import cross_val_score, KFold
from common import *
for v in (2, 1):
    X, y, _ = load(v)
    g = GPR(C(10) * RBF([1.0] * X.shape[1]) + WhiteKernel(0.3), normalize_y=True, random_state=0)
    s = -cross_val_score(g, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error").mean()
    print(f"var{v}: GP CV MSE = {s:.4f}")
