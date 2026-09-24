from factor_alpha.data import synthetic_panel, FACTOR_COLS
from factor_alpha.models import ols_hac, pca_summary
from factor_alpha.diagnostics import vif_table, adf_table, rolling_ols
from factor_alpha.validation import walk_forward

def test_shapes_and_inference():
    r,f=synthetic_panel(periods=180); X=f[FACTOR_COLS]; y=r.SPY-f.RF
    fit=ols_hac(y,X); assert fit.nobs==180; assert set(FACTOR_COLS).issubset(fit.params.index); assert 0<=fit.rsquared<=1

def test_pca_vif_adf():
    r,f=synthetic_panel(periods=180); X=f[FACTOR_COLS]
    _,_,load,evr=pca_summary(X); assert abs(evr.sum()-1)<1e-8; assert load.shape==(6,6)
    assert vif_table(X).VIF.notna().all(); assert adf_table(X).p_value.between(0,1).all()

def test_rolling_and_walk_forward_no_lookahead():
    r,f=synthetic_panel(periods=180); X=f[FACTOR_COLS]; y=r.SPY-f.RF
    roll=rolling_ols(y,X,60); assert len(roll)==121
    wf=walk_forward(y,X,84,12,12); assert not wf.empty; assert (wf.test_start>wf.train_end).all(); assert set(wf.model)=={"OLS","Ridge","Lasso"}
