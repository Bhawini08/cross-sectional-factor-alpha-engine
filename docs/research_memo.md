# Research Memo: Cross-Sectional Factor Research & Alpha Validation Engine

## Objective

Evaluate whether a six-factor equity model explains a diversified ETF cross-section, whether estimated alpha survives robust statistical scrutiny, how stable factor exposures are over time, and whether lagged factor information predicts next-month SPY excess returns.

## Sample

The final live dataset contains **258 monthly observations from February 2005 through July 2026** across 12 U.S. equity ETFs.

## Findings

### Alpha

SPY's estimated annualized alpha is approximately **-0.35%**, with a Newey-West/HAC p-value of **0.059** and an FDR-adjusted q-value of **0.142**. The estimate therefore does not meet a 5% significance threshold.

QQQ, IWM, and IWD have nominal HAC p-values below 5%, but after Benjamini-Hochberg correction across all 12 ETF alpha tests, **no alpha remains significant at the 5% FDR level**. This materially weakens any claim of persistent abnormal returns from the cross-sectional screen.

### Factor structure

The contemporaneous factor regressions show economically intuitive exposures. SPY has a market beta near one, IWM loads strongly on SMB, and growth-oriented ETFs such as QQQ exhibit negative HML exposure.

The maximum VIF is **1.96**, indicating that multicollinearity is not severe in the six-factor specification. ADF tests reject a unit root at 5% for all six factors and SPY excess returns in the sample.

PCA shows that PC1 explains **33.0%** of standardized factor variance, the first two components explain about **59.4%**, and the first three explain about **75.7%**. The factor space therefore contains meaningful shared structure without collapsing into a single dominant latent factor.

### Forecasting

The corrected walk-forward design uses month-t factors to forecast month-t+1 SPY excess returns. Mean OOS R² values are:

- OLS: **-0.162**
- Ridge: **-0.105**
- Lasso: **-0.037**

Lasso reduces forecast error relative to OLS and Ridge, but all three models fail to beat the historical-mean benchmark on average.

## Interpretation

The strongest conclusion is not that the factors generate alpha. It is that they are much more useful for **contemporaneous return attribution** than for **short-horizon prediction**. High explanatory fit should therefore not be confused with forecasting skill.

## Limitations

Results depend on the ETF universe, monthly frequency, factor definitions, sample period, and benchmark construction. Alpha estimates do not account for implementation costs or structural breaks beyond the rolling diagnostics. The study is research infrastructure and empirical evidence, not a trading recommendation.
