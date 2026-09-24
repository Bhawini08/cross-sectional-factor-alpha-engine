# Research Memo: Cross-Sectional Factor Research & Alpha Validation Engine

## Objective
Build an auditable factor-research pipeline that separates explanatory fit, inference quality, parameter stability, and out-of-sample predictive validity.

## Data
Production mode uses a diversified U.S. equity ETF universe: SPY, QQQ, IWM, IWD, IWF, XLF, XLK, XLE, XLI, XLV, XLP, and XLY. Monthly adjusted returns are joined to Fama-French market, size, value, profitability, investment, and momentum factors plus the risk-free rate. Offline mode uses a deterministic simulation designed only for software validation.

## Core findings to inspect after a live run
- Whether estimated alpha remains significant under HAC rather than conventional errors.
- Whether annualized alpha is economically material, not merely statistically nonzero.
- Which factors dominate each ETF's exposures and whether VIF indicates unstable decomposition.
- How much of six-factor variation is captured by the first few principal components.
- Whether rolling betas exhibit economically plausible time variation.
- Whether OLS, Ridge, or Lasso retain positive out-of-sample R² in walk-forward tests.

## Interpretation discipline
A high in-sample R² does not validate alpha. A significant intercept does not guarantee investability. Regularized models can improve prediction while reducing coefficient interpretability. PCA can reveal redundant factor directions while obscuring the named economic meaning of the original factors.

## Bias controls
All time splits are chronological. Regularization hyperparameters and standardization are estimated only inside training windows. Test observations never inform model selection. Returns, not future prices, enter the regressions.
