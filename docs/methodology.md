# Methodology

## Research question
Can conventional equity factors explain the cross-section and time variation of ETF excess returns, and do apparent alphas survive robust inference and genuinely out-of-sample validation?

## Model stack
1. **OLS** is the interpretable baseline because factor models are linear return decompositions. Coefficients map directly to exposures and the intercept maps to alpha.
2. **Newey-West/HAC** covariance is used for inference because monthly residuals can be heteroskedastic and autocorrelated; plain OLS standard errors can overstate precision.
3. **Ridge/Lasso** are prediction-oriented robustness checks. Ridge stabilizes correlated factor coefficients; Lasso can shrink weak exposures toward zero. They are not used to claim structural economic causality.
4. **PCA** asks whether the six observed factors span a lower-dimensional covariance structure. It is diagnostic, not a replacement for named economic factors.
5. **VIF** reports multicollinearity. High VIF warns that individual coefficient estimates can be unstable even when the joint model explains returns well.
6. **ADF tests** are applied to factor and excess-return series, where stationarity is a relevant modeling assumption. Price levels are deliberately not tested inside the return model.
7. **Rolling regressions** reveal changing exposures and parameter instability.
8. **Walk-forward validation** uses chronological train/test windows. Scaling, regularization, and cross-validation are fitted only on each training sample, preventing look-ahead leakage.

## Statistical vs economic alpha
A statistically significant alpha is an intercept distinguishable from zero given the sampling model. An economically meaningful alpha must also be large enough to matter after implementation frictions and model uncertainty. The project reports both HAC p-values and an explicit annualized magnitude flag (2% by default) instead of conflating them.

## Overfitting controls
Chronological splits, rolling-origin evaluation, training-only preprocessing, model parsimony, Ridge/Lasso shrinkage, PCA diagnostics, and comparison against a training-mean baseline are used together. No test-period information is used to choose parameters.
