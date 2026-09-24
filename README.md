# Cross-Sectional Factor Research & Alpha Validation Engine

A portfolio-grade quantitative research pipeline for testing whether U.S. equity ETF returns are explained by **Fama-French 5 factors + Momentum**, whether apparent alpha survives robust inference and multiple-testing correction, how exposures evolve through time, and whether lagged factors provide genuine out-of-sample predictive power.

## Live research snapshot

The tracked empirical results use **258 monthly observations from February 2005 through July 2026** across 12 U.S. equity ETFs.

- SPY annualized alpha: **-0.35%**
- SPY HAC p-value: **0.059**
- SPY Benjamini-Hochberg FDR q-value: **0.142**
- Mean 1-month-ahead OOS R²: **OLS -0.162, Ridge -0.105, Lasso -0.037**
- PC1 explained variance: **33.0%**
- Maximum factor VIF: **1.96**

The main empirical result is a separation between **explanation and prediction**: contemporaneous factor models explain ETF returns well, but lagged factor realizations do not deliver robust one-month-ahead SPY forecasts relative to a historical-mean benchmark.

Nominal HAC tests identify several ETF alphas below a 5% p-value, but **none remain significant at 5% after Benjamini-Hochberg FDR correction across the 12-ETF cross-section**.

## Research stack

- OLS factor attribution
- Newey-West/HAC inference
- Benjamini-Hochberg FDR correction
- Ridge and Lasso robustness models
- PCA factor-space diagnostics
- VIF multicollinearity diagnostics
- ADF stationarity tests
- 60-month rolling regressions
- parameter-stability analysis
- chronological 1-month-ahead walk-forward validation
- training-only preprocessing and regularization selection

## Universe

`SPY QQQ IWM IWD IWF XLF XLK XLE XLI XLV XLP XLY`

## Why each method is here

**OLS** provides directly interpretable alpha and factor exposures.  
**Newey-West/HAC** protects inference against heteroskedasticity and serial correlation.  
**FDR correction** controls the false-discovery problem created by testing alpha across multiple ETFs.  
**Ridge/Lasso** test coefficient stability and out-of-sample prediction under correlated predictors.  
**PCA** measures latent dimensionality and factor redundancy.  
**VIF** diagnoses unstable coefficient decomposition from multicollinearity.  
**Rolling regressions** test whether factor exposures are stable through time.  
**Walk-forward validation** separates contemporaneous attribution from actual forecasting and prevents look-ahead leakage.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
pytest -q
python scripts/run_research.py --mode live --focus SPY
```

Open `results/dashboard.html` after the run.

For an offline engineering check:

```bash
python scripts/run_research.py --mode synthetic --focus SPY
```

Synthetic mode is deterministic and exists only to validate software behavior. It is not investment evidence.

## Tracked live outputs

- `results/alpha_summary.csv`
- `results/coefficients_hac.csv`
- `results/vif.csv`
- `results/adf.csv`
- `results/pca_loadings.csv`
- `results/pca_variance.csv`
- `results/parameter_stability.csv`
- `results/walk_forward.csv`
- `results/metrics.json`
- `results/dashboard.html`

A fresh run also regenerates full rolling-exposure histories and publication-ready figures.

## Research discipline

Statistical significance and economic significance are reported separately. Nominal p-values are not treated as discoveries until cross-sectional multiple testing is considered. High contemporaneous R² is not presented as forecasting skill. Test-period observations are excluded from model fitting, scaling, and hyperparameter selection.

See `docs/methodology.md`, `docs/research_memo.md`, and `docs/validation_report.md` for the full interpretation.
