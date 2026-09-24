# Methodology

## Research question

Can conventional equity factors explain the cross-section and time variation of ETF excess returns, and do apparent alphas survive robust inference, multiple-testing correction, parameter-stability analysis, and genuine out-of-sample forecasting tests?

## Data

Production mode uses monthly adjusted ETF returns for SPY, QQQ, IWM, IWD, IWF, XLF, XLK, XLE, XLI, XLV, XLP, and XLY, joined to the Fama-French market, size, value, profitability, investment, momentum, and risk-free series. The tracked live sample spans February 2005 through July 2026.

## Model stack

1. **OLS** is the interpretable attribution baseline. Coefficients map directly to factor exposures and the intercept maps to alpha.
2. **Newey-West/HAC** covariance is used for inference because monthly residuals can be heteroskedastic and autocorrelated.
3. **Benjamini-Hochberg FDR correction** is applied across ETF alpha tests so nominal p-values are not mistaken for cross-sectional discoveries.
4. **Ridge/Lasso** are prediction-oriented robustness checks. Ridge stabilizes correlated coefficients; Lasso can shrink weak signals toward zero.
5. **PCA** asks whether the six observed factors occupy a lower-dimensional covariance structure. It is diagnostic rather than a replacement for named economic factors.
6. **VIF** reports multicollinearity. High VIF would warn that individual beta estimates may be unstable even if the joint model fits well.
7. **ADF tests** are applied to factor and excess-return series, where stationarity is relevant to the return model.
8. **60-month rolling regressions** reveal time variation in factor exposures and parameter stability.
9. **Walk-forward forecasting** uses factor information at month t to predict excess return at month t+1. Chronological train/test windows are used, and scaling, regularization, and cross-validation are fit only inside each training sample.

## Attribution versus forecasting

Contemporaneous factor regressions answer an attribution question: how much of the realized return co-movement is associated with known factor returns in the same month? They are not forecasts. The walk-forward section deliberately lags the target so that same-month factor realizations are unavailable when the forecast is formed.

## Statistical versus economic alpha

A statistically significant alpha is an intercept distinguishable from zero under the sampling model. An economically meaningful alpha must also be large enough to matter after implementation frictions, instability, and model risk. The project reports HAC p-values, annualized magnitude flags, and FDR-adjusted q-values separately.

## Overfitting controls

The framework uses chronological splits, one-month-ahead target alignment, rolling-origin evaluation, training-only preprocessing, model parsimony, Ridge/Lasso shrinkage, PCA diagnostics, and a historical-mean benchmark. No test-period information is used to choose parameters or form forecasts.
