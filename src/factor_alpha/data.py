from __future__ import annotations
import numpy as np
import pandas as pd

ETF_UNIVERSE = ["SPY","QQQ","IWM","IWD","IWF","XLF","XLK","XLE","XLI","XLV","XLP","XLY"]
FACTOR_COLS = ["MKT_RF","SMB","HML","RMW","CMA","MOM"]


def synthetic_panel(start="2000-01-31", periods=300, seed=42):
    """Deterministic offline fixture with time-varying exposures and realistic monthly scale."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="ME")
    # Correlated monthly factors with mild persistence.
    base_cov = np.array([
        [.0020,.0002,-.0001,.0000,-.0001,.0001],
        [.0002,.0010,.0001,-.0001,.0000,.0001],
        [-.0001,.0001,.0011,.0001,.0002,-.0002],
        [.0000,-.0001,.0001,.0007,.0001,.0000],
        [-.0001,.0000,.0002,.0001,.0006,-.0001],
        [.0001,.0001,-.0002,.0000,-.0001,.0012],
    ])
    means = np.array([.005,.0015,.0018,.0012,.0010,.0030])
    shocks = rng.multivariate_normal(means, base_cov, size=periods)
    factors = np.zeros_like(shocks)
    for t in range(periods):
        factors[t] = shocks[t] if t == 0 else 0.12*factors[t-1] + 0.88*shocks[t]
    fac = pd.DataFrame(factors, index=dates, columns=FACTOR_COLS)
    rf = pd.Series(0.0015 + rng.normal(0,0.0004,periods), index=dates, name="RF").clip(lower=0)

    base_betas = {
        "SPY":[1.00,0.00,0.00,0.05,0.00,0.00], "QQQ":[1.12,-0.15,-0.35,0.08,-0.12,0.18],
        "IWM":[1.08,0.75,0.10,-0.05,-0.05,0.02], "IWD":[0.98,0.15,0.48,0.05,0.08,-0.08],
        "IWF":[1.06,-0.10,-0.40,0.10,-0.10,0.12], "XLF":[1.08,0.05,0.22,0.00,0.03,-0.03],
        "XLK":[1.10,-0.08,-0.42,0.12,-0.08,0.20], "XLE":[1.05,0.15,0.32,-0.05,0.12,-0.10],
        "XLI":[1.03,0.18,0.18,0.06,0.04,0.00], "XLV":[0.82,-0.10,0.08,0.18,0.00,0.04],
        "XLP":[0.70,-0.12,0.22,0.20,0.10,-0.05], "XLY":[1.08,0.00,-0.18,0.02,-0.04,0.10],
    }
    # Small true alphas; deliberately not all economically meaningful.
    alpha_annual = {k:0.0 for k in ETF_UNIVERSE}
    alpha_annual.update({"XLK":0.012, "XLP":0.006, "IWM":-0.006})
    rets = {}
    x = fac.values
    for j,ticker in enumerate(ETF_UNIVERSE):
        b0 = np.array(base_betas[ticker], dtype=float)
        # slow beta drift + one regime shift to make rolling analysis meaningful
        drift = np.sin(np.linspace(0, 3*np.pi, periods)+j/3)[:,None] * np.array([.05,.08,.07,.04,.04,.06])
        shift = np.zeros((periods,6)); shift[periods//2:,1] += (j%3-1)*0.05
        betas = b0 + drift + shift
        eps = rng.normal(0, 0.018 + 0.002*(j%4), periods)
        excess = alpha_annual[ticker]/12 + np.sum(betas*x, axis=1) + eps
        rets[ticker] = excess + rf.values
    returns = pd.DataFrame(rets, index=dates)
    factors = pd.concat([fac, rf], axis=1)
    return returns, factors


def live_panel(start="2005-01-01", end=None, tickers=None):
    """Fetch adjusted ETF prices from Yahoo and FF5 + Momentum from Ken French via pandas-datareader."""
    tickers = tickers or ETF_UNIVERSE
    try:
        import yfinance as yf
        from pandas_datareader import data as web
    except ImportError as exc:
        raise RuntimeError("Live mode needs yfinance and pandas-datareader. Install requirements.txt") from exc
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series): px = px.to_frame()
    monthly = px.resample("ME").last().pct_change().dropna(how="all")
    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")[0].copy()/100.0
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench")[0].copy()/100.0
    ff5.index = ff5.index.to_timestamp("M")
    mom.index = mom.index.to_timestamp("M")
    ff5 = ff5.rename(columns={"Mkt-RF":"MKT_RF"})
    mom_col = [c for c in mom.columns if "Mom" in c or "MOM" in c][0]
    factors = ff5.join(mom[[mom_col]].rename(columns={mom_col:"MOM"}), how="inner")
    keep = ["MKT_RF","SMB","HML","RMW","CMA","MOM","RF"]
    factors = factors[keep]
    idx = monthly.index.intersection(factors.index)
    return monthly.loc[idx, tickers].dropna(axis=1, thresh=max(60,int(len(idx)*.8))), factors.loc[idx]
