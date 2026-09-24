from __future__ import annotations
import numpy as np
import pandas as pd

ETF_UNIVERSE = ["SPY","QQQ","IWM","IWD","IWF","XLF","XLK","XLE","XLI","XLV","XLP","XLY"]
FACTOR_COLS = ["MKT_RF","SMB","HML","RMW","CMA","MOM"]


def synthetic_panel(start="2000-01-31", periods=300, seed=42):
    """Deterministic offline fixture with time-varying exposures and realistic monthly scale."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="ME")
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
    alpha_annual = {k:0.0 for k in ETF_UNIVERSE}
    alpha_annual.update({"XLK":0.012, "XLP":0.006, "IWM":-0.006})
    rets = {}
    x = fac.values
    for j,ticker in enumerate(ETF_UNIVERSE):
        b0 = np.array(base_betas[ticker], dtype=float)
        drift = np.sin(np.linspace(0, 3*np.pi, periods)+j/3)[:,None] * np.array([.05,.08,.07,.04,.04,.06])
        shift = np.zeros((periods,6)); shift[periods//2:,1] += (j%3-1)*0.05
        betas = b0 + drift + shift
        eps = rng.normal(0, 0.018 + 0.002*(j%4), periods)
        excess = alpha_annual[ticker]/12 + np.sum(betas*x, axis=1) + eps
        rets[ticker] = excess + rf.values
    returns = pd.DataFrame(rets, index=dates)
    factors = pd.concat([fac, rf], axis=1)
    return returns, factors


def _extract_close_prices(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normalize yfinance's single- and multi-index output into Date x Ticker close prices."""
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no price data.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if "Close" in level0:
            px = raw["Close"].copy()
        elif "Close" in level1:
            px = raw.xs("Close", axis=1, level=1).copy()
        else:
            raise RuntimeError(f"Could not find a Close field in Yahoo output. Column levels: {raw.columns.names}")
    else:
        if "Close" not in raw.columns:
            raise RuntimeError(f"Could not find a Close column in Yahoo output: {list(raw.columns)}")
        px = raw[["Close"]].copy()
        if len(tickers) == 1:
            px.columns = [tickers[0]]

    if isinstance(px, pd.Series):
        px = px.to_frame(name=tickers[0] if len(tickers) == 1 else px.name)

    px.columns = [str(c).strip().upper() for c in px.columns]
    px = px.loc[:, ~px.columns.duplicated()]
    px.index = pd.to_datetime(px.index)
    if getattr(px.index, "tz", None) is not None:
        px.index = px.index.tz_localize(None)
    return px.sort_index()


def live_panel(start="2005-01-01", end=None, tickers=None):
    """Fetch adjusted ETF prices from Yahoo and FF5 + Momentum from Ken French."""
    tickers = [str(t).upper() for t in (tickers or ETF_UNIVERSE)]
    try:
        import yfinance as yf
        from pandas_datareader import data as web
    except ImportError as exc:
        raise RuntimeError("Live mode needs yfinance and pandas-datareader. Install requirements.txt") from exc

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    px = _extract_close_prices(raw, tickers)

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        # Retry missing symbols individually because bulk Yahoo downloads can partially fail.
        for ticker in missing:
            one = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            try:
                one_px = _extract_close_prices(one, [ticker])
                if ticker in one_px.columns:
                    px[ticker] = one_px[ticker]
            except RuntimeError:
                pass

    available_requested = [t for t in tickers if t in px.columns]
    if not available_requested:
        raise RuntimeError("None of the requested ETF tickers were returned by Yahoo Finance.")

    monthly = px[available_requested].resample("ME").last().pct_change(fill_method=None).dropna(how="all")

    ff5 = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench")[0].copy() / 100.0
    mom = web.DataReader("F-F_Momentum_Factor", "famafrench")[0].copy() / 100.0
    ff5 = ff5.rename(columns={"Mkt-RF":"MKT_RF"})
    mom_col = next((c for c in mom.columns if "mom" in str(c).lower()), None)
    if mom_col is None:
        raise RuntimeError(f"Momentum factor column not found. Available columns: {list(mom.columns)}")
    factors = ff5.join(mom[[mom_col]].rename(columns={mom_col:"MOM"}), how="inner")
    factors = factors[["MKT_RF","SMB","HML","RMW","CMA","MOM","RF"]]

    # Align on monthly PeriodIndex so differing month-end timestamp conventions cannot break the join.
    monthly.index = monthly.index.to_period("M")
    factors.index = factors.index.asfreq("M") if isinstance(factors.index, pd.PeriodIndex) else pd.to_datetime(factors.index).to_period("M")
    common = monthly.index.intersection(factors.index)
    if len(common) < 84:
        raise RuntimeError(f"Only {len(common)} overlapping monthly observations were available; at least 84 are required.")

    monthly = monthly.loc[common]
    factors = factors.loc[common]

    # Keep only assets with enough history for rolling/walk-forward research.
    min_obs = max(60, int(len(common) * 0.80))
    coverage = monthly.notna().sum()
    keep_tickers = [t for t in tickers if t in monthly.columns and coverage.get(t, 0) >= min_obs]
    if not keep_tickers:
        raise RuntimeError(f"No ETF passed the minimum-history filter ({min_obs} months). Coverage: {coverage.to_dict()}")

    monthly = monthly[keep_tickers]
    monthly.index = monthly.index.to_timestamp("M")
    factors.index = factors.index.to_timestamp("M")

    # Complete-case rows across retained ETFs and factors keep all regressions on a consistent sample.
    joined = monthly.join(factors, how="inner").dropna()
    monthly = joined[keep_tickers]
    factors = joined[["MKT_RF","SMB","HML","RMW","CMA","MOM","RF"]]
    return monthly, factors
