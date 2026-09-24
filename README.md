# Cross-Sectional Factor Research & Alpha Validation Engine

A portfolio-grade quantitative research pipeline for testing whether equity ETF returns are explained by **Fama-French 5 factors + Momentum**, whether apparent alpha survives **HAC/Newey-West inference**, how exposures change through time, and whether the model generalizes **out of sample**.

## What this project demonstrates

- OLS factor attribution with interpretable alpha/betas
- Newey-West/HAC t-statistics and p-values
- Ridge and Lasso robustness models
- PCA of the factor covariance space
- VIF multicollinearity diagnostics
- ADF stationarity tests on factor/return series
- 60-month rolling regressions and parameter-stability metrics
- Rolling-origin walk-forward validation
- Strict training-only preprocessing and regularization selection
- Static Plotly dashboard + reproducible research tables/figures

## Universe
`SPY QQQ IWM IWD IWF XLF XLK XLE XLI XLV XLP XLY`

## Why each method is here
**OLS:** direct economic interpretation of alpha and factor exposures.  
**Regularization:** coefficient stability and out-of-sample prediction when factors are correlated.  
**Newey-West:** robust inference when residual variance and serial correlation violate classical OLS assumptions.  
**PCA:** measures latent dimensionality/redundancy in factor space.  
**Rolling regressions:** tests whether exposures are stable through time.  
**Walk-forward validation:** evaluates generalization without look-ahead bias.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python scripts/run_research.py --mode live --focus SPY
```
Open `results/dashboard.html`.

For an offline reproducibility check:
```bash
python scripts/run_research.py --mode synthetic --focus SPY
```
Synthetic mode is a deterministic engineering fixture, not investment evidence.

## Outputs
- `alpha_summary.csv`: annualized/monthly alpha, HAC t-stat/p-value, R², significance and economic-materiality flags
- `coefficients_hac.csv`: factor loadings with robust inference
- `vif.csv`, `adf.csv`, `pca_*`: diagnostics
- `rolling_exposures.csv`, `parameter_stability.csv`: time-varying exposures
- `walk_forward.csv`: chronological OOS tests for OLS/Ridge/Lasso
- `dashboard.html`: self-contained analytical dashboard
- `figures/*.png`: portfolio-ready result visuals

## No-look-ahead design
Every walk-forward test trains only on observations dated before the test window. Standardization and Ridge/Lasso hyperparameter selection occur inside each training slice. The test set is never used for fitting or tuning.

## Research note
Statistical significance and economic significance are reported separately. A low p-value says the estimated alpha is statistically distinguishable from zero under the model; it does **not** say the alpha is large enough to monetize after costs, instability, and model risk.

## Data sources
Production factor data are loaded through the Kenneth R. French Data Library interface; ETF adjusted prices are loaded from Yahoo Finance. See `docs/methodology.md` for assumptions and interpretation.
