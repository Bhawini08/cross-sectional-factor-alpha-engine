from __future__ import annotations
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


def ols_hac(y: pd.Series, X: pd.DataFrame, maxlags=3):
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags":maxlags})
    return model


def regularized_fit(y, X, kind="ridge", alphas=None):
    alphas = np.logspace(-4, 1, 40) if alphas is None else alphas
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))
    elif kind == "lasso":
        model = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=5, max_iter=20000, random_state=42))
    else: raise ValueError(kind)
    model.fit(X, y)
    return model


def pca_summary(X: pd.DataFrame, n_components=None):
    scaler = StandardScaler(); z = scaler.fit_transform(X)
    pca = PCA(n_components=n_components).fit(z)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns,
                            columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    evr = pd.Series(pca.explained_variance_ratio_, index=loadings.columns, name="explained_variance_ratio")
    return pca, scaler, loadings, evr
