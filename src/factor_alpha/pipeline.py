from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from .data import synthetic_panel, live_panel, FACTOR_COLS
from .models import ols_hac, pca_summary
from .diagnostics import vif_table, adf_table, rolling_ols, stability_metrics
from .validation import walk_forward


def run_research(mode="synthetic", outdir="results", focus="SPY"):
    out=Path(outdir); (out/"figures").mkdir(parents=True,exist_ok=True)
    returns,factors = synthetic_panel() if mode=="synthetic" else live_panel()
    common=returns.index.intersection(factors.index); returns=returns.loc[common]; factors=factors.loc[common]
    X=factors[FACTOR_COLS]
    excess=returns.sub(factors["RF"],axis=0)
    summary=[]; coef=[]
    for ticker in excess.columns:
        fit=ols_hac(excess[ticker],X)
        a=fit.params["const"]; ta=fit.tvalues["const"]; pa=fit.pvalues["const"]
        summary.append({"ticker":ticker,"alpha_monthly":a,"alpha_annualized":a*12,"alpha_t_hac":ta,"alpha_p_hac":pa,"r2":fit.rsquared,"adj_r2":fit.rsquared_adj,"n_obs":int(fit.nobs),"stat_sig_5pct":bool(pa<.05),"econ_meaningful_2pct":bool(abs(a*12)>=.02)})
        for k,v in fit.params.items(): coef.append({"ticker":ticker,"term":k,"coef":v,"t_hac":fit.tvalues[k],"p_hac":fit.pvalues[k]})
    summary_df = pd.DataFrame(summary)
    # Cross-sectional alpha testing creates a multiple-comparisons problem.
    # Report Benjamini-Hochberg FDR-adjusted q-values alongside nominal HAC p-values.
    from statsmodels.stats.multitest import multipletests
    _, qvals, _, _ = multipletests(summary_df["alpha_p_hac"].values, alpha=0.05, method="fdr_bh")
    summary_df["alpha_q_fdr_bh"] = qvals
    summary_df["fdr_sig_5pct"] = summary_df["alpha_q_fdr_bh"] < 0.05
    summary_df.to_csv(out/"alpha_summary.csv",index=False)
    pd.DataFrame(coef).to_csv(out/"coefficients_hac.csv",index=False)
    vif_table(X).to_csv(out/"vif.csv",index=False)
    adf_table(pd.concat([X, excess[[focus]]],axis=1)).to_csv(out/"adf.csv",index=False)
    _,_,load,evr=pca_summary(X); load.to_csv(out/"pca_loadings.csv"); evr.to_csv(out/"pca_variance.csv")
    rolling=rolling_ols(excess[focus],X,60); rolling.to_csv(out/"rolling_exposures.csv")
    stability_metrics(rolling).to_csv(out/"parameter_stability.csv",index=False)
    wf=walk_forward(excess[focus],X,84,12,12,horizon=1); wf.to_csv(out/"walk_forward.csv",index=False)
    _plots(out, summary_df, rolling, evr, wf, focus)
    _dashboard(out, summary_df, rolling, evr, wf, focus, mode)
    metrics={
      "mode":mode,"focus":focus,"start":str(common.min().date()),"end":str(common.max().date()),"n_months":len(common),"n_assets":excess.shape[1],
      "focus_alpha_ann":float(summary_df.set_index("ticker").loc[focus,"alpha_annualized"]),
      "focus_alpha_p":float(summary_df.set_index("ticker").loc[focus,"alpha_p_hac"]),
      "focus_alpha_q_fdr_bh":float(summary_df.set_index("ticker").loc[focus,"alpha_q_fdr_bh"]),
      "mean_oos_r2_ols":float(wf[wf.model=="OLS"].oos_r2.mean()),"mean_oos_r2_ridge":float(wf[wf.model=="Ridge"].oos_r2.mean()),"mean_oos_r2_lasso":float(wf[wf.model=="Lasso"].oos_r2.mean()),
      "pc1_variance":float(evr.iloc[0]),"max_vif":float(vif_table(X).VIF.max())}
    pd.Series(metrics).to_json(out/"metrics.json",indent=2)
    return metrics


def _plots(out, summary, rolling, evr, wf, focus):
    import matplotlib.pyplot as plt
    s=summary.sort_values("alpha_annualized")
    fig,ax=plt.subplots(figsize=(10,5)); ax.barh(s.ticker,s.alpha_annualized*100); ax.axvline(0,lw=1); ax.set_xlabel("Annualized alpha (%)"); ax.set_title("HAC-adjusted factor alpha estimates"); fig.tight_layout(); fig.savefig(out/"figures/alpha_cross_section.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5)); rolling.drop(columns="alpha").plot(ax=ax); ax.set_title(f"60-month rolling factor exposures: {focus}"); ax.set_ylabel("Beta"); fig.tight_layout(); fig.savefig(out/"figures/rolling_betas.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); (evr.cumsum()*100).plot(marker="o",ax=ax); ax.set_ylabel("Cumulative explained variance (%)"); ax.set_title("PCA factor-space compression"); fig.tight_layout(); fig.savefig(out/"figures/pca_variance.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); wf.groupby("model").oos_r2.mean().plot(kind="bar",ax=ax); ax.axhline(0,lw=1); ax.set_ylabel("Mean OOS R²"); ax.set_title(f"1-month-ahead walk-forward forecast: {focus}"); fig.tight_layout(); fig.savefig(out/"figures/oos_r2.png",dpi=160); plt.close(fig)


def _dashboard(out, summary, rolling, evr, wf, focus, mode):
    import plotly.express as px, plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig=make_subplots(rows=2,cols=2,subplot_titles=("Annualized alpha","Rolling betas","PCA cumulative variance","1-month-ahead OOS R²"))
    s=summary.sort_values("alpha_annualized")
    fig.add_trace(go.Bar(x=s.ticker,y=s.alpha_annualized*100,name="Alpha %"),1,1)
    for c in [x for x in rolling.columns if x!="alpha"]: fig.add_trace(go.Scatter(x=rolling.index,y=rolling[c],name=c,showlegend=False),1,2)
    fig.add_trace(go.Scatter(x=evr.index,y=evr.cumsum()*100,mode="lines+markers",name="PCA",showlegend=False),2,1)
    m=wf.groupby("model",as_index=False).oos_r2.mean(); fig.add_trace(go.Bar(x=m.model,y=m.oos_r2,name="OOS R²",showlegend=False),2,2)
    fig.update_layout(height=850,title=f"Cross-Sectional Factor Research & Alpha Validation Engine | {mode.title()} | Focus: {focus}")
    fig.write_html(out/"dashboard.html",include_plotlyjs="cdn")
