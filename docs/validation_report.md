# Validation Report

The repository was executed end-to-end in deterministic offline mode before packaging.

## Automated checks
- `pytest`: **3 passed**
- Sample span: **300 monthly observations** from Jan 2000 through Dec 2024
- Universe: **12 equity ETFs**
- Focus asset: **SPY**

## Diagnostic sanity checks
- SPY synthetic-fixture annualized alpha: **0.18%**, HAC p-value **0.898**. The engine correctly avoids falsely labeling a near-zero simulated alpha as significant.
- Maximum factor VIF: **1.14**, consistent with the fixture's intentionally modest factor correlation.
- ADF tests reject a unit root at 5% for all six factors and SPY excess returns in the validation fixture.
- First principal component explains **26.4%** of standardized factor variance, so the factor space is not spuriously collapsed into one dominant component.

## Walk-forward validation
Mean rolling-origin OOS R² against the training-mean baseline:
- OLS: **0.780**
- Ridge: **0.782**
- Lasso: **0.793**

These values are expected to be high because the offline fixture is generated from a known factor structure. Their purpose is to verify that chronological slicing, preprocessing, model fitting, and scoring behave correctly. They are **not investment results**.

## Visual QA
Generated charts were manually inspected for legibility and consistency:
- cross-sectional annualized alpha
- 60-month rolling factor betas
- PCA cumulative explained variance
- walk-forward OOS R²

## Production-mode caveat
`--mode live` uses Yahoo Finance and the Kenneth R. French Data Library and therefore requires internet access. The offline validation environment used here blocks outbound Python network calls, so live market results should be regenerated in GitHub Actions or a normal local environment before treating any empirical findings as research evidence.
