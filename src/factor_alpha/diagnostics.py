from __future__ import annotations
import numpy as np, pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller


def vif_table(X: pd.DataFrame):
    arr = X.dropna().values
    return pd.DataFrame({"factor":X.columns,"VIF":[variance_inflation_factor(arr,i) for i in range(arr.shape[1])]})


def adf_table(df: pd.DataFrame):
    rows=[]
    for c in df.columns:
        s=df[c].dropna()
        stat,p,*_ = adfuller(s, autolag="AIC")
        rows.append({"series":c,"adf_stat":stat,"p_value":p,"stationary_5pct":bool(p<.05)})
    return pd.DataFrame(rows)


def rolling_ols(y, X, window=60):
    import statsmodels.api as sm
    rows=[]
    for end in range(window, len(y)+1):
        ys=y.iloc[end-window:end]; xs=X.iloc[end-window:end]
        fit=sm.OLS(ys, sm.add_constant(xs)).fit()
        row={"date":y.index[end-1],"alpha":fit.params["const"]}
        row.update({c:fit.params[c] for c in X.columns}); rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def stability_metrics(rolling: pd.DataFrame):
    fac=[c for c in rolling.columns if c!="alpha"]
    rows=[]
    for c in fac:
        s=rolling[c].dropna(); rows.append({"factor":c,"mean_beta":s.mean(),"beta_std":s.std(),"range":s.max()-s.min(),"sign_flip_rate":(np.sign(s).diff()!=0).mean()})
    return pd.DataFrame(rows)
