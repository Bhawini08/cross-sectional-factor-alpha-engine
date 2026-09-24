# Validation Report

## Final live run

- Mode: **live**
- Sample: **February 2005 to July 2026**
- Observations: **258 monthly rows**
- Assets: **12 ETFs**
- Focus asset: **SPY**
- Local automated tests: **3 passed** on the final code path

## Core checks

SPY annualized alpha is **-0.35%**, with HAC p-value **0.059** and Benjamini-Hochberg q-value **0.142**.

Maximum factor VIF is **1.96**, below levels normally associated with severe multicollinearity.

ADF tests reject a unit root at 5% for MKT_RF, SMB, HML, RMW, CMA, MOM, and SPY excess returns.

PC1 explains **33.0%** of standardized factor variance; the first three principal components explain approximately **75.7%**.

## One-month-ahead walk-forward validation

Mean OOS R² against a training-sample historical-mean benchmark:

- OLS: **-0.162**
- Ridge: **-0.105**
- Lasso: **-0.037**

The negative average OOS values are retained rather than optimized away. They show that the lagged factors do not provide robust next-month SPY forecasting power over the sample.

## Multiple testing

Nominal HAC p-values identify QQQ, IWM, and IWD at the 5% level. After Benjamini-Hochberg FDR correction across the 12 ETF alpha tests, **none remain significant at 5%**.

## Leakage control

The forecasting target is shifted forward one month. Test-period factor realizations are not used to predict the same month's return. Standardization and regularization selection occur only inside each training window.

## Reproducibility

Synthetic mode remains available as a deterministic software fixture. Production conclusions in this repository are based on the live market-data run, not the synthetic fixture.
