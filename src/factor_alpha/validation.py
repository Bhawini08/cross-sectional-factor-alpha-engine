from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
from .models import regularized_fit


def walk_forward(y, X, train_window=84, test_window=12, step=12, horizon=1):
    """Chronological rolling-origin validation using lagged factors to forecast future excess returns."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    aligned = pd.concat([X.copy(), y.shift(-horizon).rename("target")], axis=1).dropna()
    Xa = aligned[X.columns]
    ya = aligned["target"]

    rows=[]
    for start in range(0, len(ya)-train_window-test_window+1, step):
        tr=slice(start,start+train_window)
        te=slice(start+train_window,start+train_window+test_window)
        ytr,yte=ya.iloc[tr],ya.iloc[te]
        Xtr,Xte=Xa.iloc[tr],Xa.iloc[te]

        import statsmodels.api as sm
        ols=sm.OLS(ytr, sm.add_constant(Xtr)).fit()
        p_ols=ols.predict(sm.add_constant(Xte, has_constant="add"))

        ridge=regularized_fit(ytr,Xtr,"ridge")
        lasso=regularized_fit(ytr,Xtr,"lasso")
        preds={"OLS":p_ols,"Ridge":ridge.predict(Xte),"Lasso":lasso.predict(Xte)}

        baseline=np.repeat(ytr.mean(),len(yte))
        base_mse=mean_squared_error(yte,baseline)

        for name,p in preds.items():
            mse=mean_squared_error(yte,p)
            r2_oos=1-mse/base_mse if base_mse>0 else np.nan
            rows.append({
                "train_start":ytr.index[0],
                "train_end":ytr.index[-1],
                "test_start":yte.index[0],
                "test_end":yte.index[-1],
                "forecast_horizon_months":horizon,
                "model":name,
                "mse":mse,
                "oos_r2":r2_oos,
                "pred_mean":float(np.mean(p)),
                "actual_mean":float(yte.mean()),
            })
    return pd.DataFrame(rows)


def overfit_gap(in_sample_r2, walk_df):
    return float(in_sample_r2 - walk_df.groupby("model")["oos_r2"].mean().get("OLS", np.nan))
