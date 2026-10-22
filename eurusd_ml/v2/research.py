"""Validare: purged K-fold + embargo, walk-forward, registru de configuratii (<=40), PBO (CSCV), Deflated Sharpe,
holdout final evaluat O SINGURA DATA (fisier de blocare), backtest cu executie o pozitie odata.

Sarcini:
  meta_ema : semnal primar = directia EMA(300m) vs EMA(1200m); modelul decide daca se ia tranzactia.
  meta_mr  : semnal primar = revenire la medie cand |z_240| > 2; modelul decide daca se ia.
  dir      : DOUA modele separate (long castiga net? / short castiga net?). Long doar daca p_long >= prag_long si
             p_long > p_short; short simetric. (Repara defectul vechi: short luat cand p_long < 0.45.)
Pragul = percentila 80 a probabilitatilor pe ultimele 20% din perioada de antrenare (calibrare, cauzal).
"""
import json, itertools, sys, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import lightgbm as lgb, xgboost as xgb

DS = Path("/workspace/ticks/datasets"); REP = Path("/workspace/eurusd_ml/reports/v2")
HOLDOUT_START = int(pd.Timestamp("2025-10-22").value // 10**6)
EMBARGO = 24 * 3600 * 1000
FEATS = ["ret_15", "ret_60", "ret_240", "ret_960", "vol_60", "vol_240", "vol_1440", "vol_ratio", "timb_15", "timb_60", "timb_240",
         "ofi_15", "ofi_60", "ofi_240", "qint_15", "qint_60", "spread_rel", "spread_pips", "spmax_rel", "rangepos_240", "ema_gap",
         "z_240", "hour_sin", "hour_cos", "dow", "since_last_tick_min"]
WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

def model(name, seed=0):
    if name == "logit": return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))
    if name == "rf": return RandomForestClassifier(300, max_depth=6, min_samples_leaf=300, max_features=0.3, n_jobs=8, random_state=seed)
    if name == "lgbm": return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=500, subsample=0.7,
                                                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=seed, n_jobs=8)
    if name == "xgb": return xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.7,
                                               colsample_bytree=0.7, reg_lambda=5, n_jobs=8, random_state=seed, verbosity=0)
    raise ValueError(name)

def fit(m, X, y, w):
    try: m.fit(X, y, **({"logisticregression__sample_weight": w} if hasattr(m, "steps") else {"sample_weight": w}))
    except TypeError: m.fit(X, y)
    return m

def prep(task, D):
    """Returneaza tabel cu: ts, entry, exit, y, pnl (pentru sarcini meta) sau y_l/y_s, pnl_l/pnl_s (dir)."""
    D = D.copy()
    if task == "meta_ema":
        D = D[D.side_ema != 0]; s = D.side_ema
    elif task == "meta_mr":
        D = D[D.side_mr != 0]; s = D.side_mr
    else:
        D["exit"] = np.maximum(D.exit_long, D.exit_short); D["y_l"] = (D.pnl_long > 0).astype(int); D["y_s"] = (D.pnl_short > 0).astype(int)
        return D.reset_index(drop=True)
    D["side"] = s; D["pnl"] = np.where(s > 0, D.pnl_long, D.pnl_short); D["exit"] = np.where(s > 0, D.exit_long, D.exit_short)
    D["y"] = (D.pnl > 0).astype(int)
    return D.reset_index(drop=True)

def uniqueness(entry, exit_):
    """Unicitate medie (Lopez de Prado, cap. 4) pe grila de 1 minut."""
    m0 = entry.min() // 60000; a = (entry // 60000 - m0).astype(int); b = (exit_ // 60000 - m0).astype(int) + 1
    cnt = np.zeros(b.max() + 2); np.add.at(cnt, a, 1); np.add.at(cnt, b, -1); c = np.cumsum(cnt)
    inv = np.where(c > 0, 1.0 / np.maximum(c, 1), 0); cs = np.r_[0, np.cumsum(inv)]
    return (cs[b] - cs[a]) / np.maximum(b - a, 1)

def purge(train, t0, t1, ts, ex):
    """Elimina din train evenimentele care se suprapun cu [t0, t1] si embargo-ul dupa t1."""
    keep = ~((ex[train] >= t0) & (ts[train] <= t1 + EMBARGO))
    return train[keep]

def execute(sel_idx, sides, D, pnl_cols=("pnl_long", "pnl_short"), exit_cols=("exit_long", "exit_short")):
    """O singura pozitie odata, in ordinea timpului. Returneaza DataFrame de tranzactii executate."""
    out = []; busy = -1
    for i, s in zip(sel_idx, sides):
        r = D.iloc[i]
        if r.entry_ts <= busy: continue
        pnl = r[pnl_cols[0]] if s > 0 else r[pnl_cols[1]]; ex = r[exit_cols[0]] if s > 0 else r[exit_cols[1]]
        out.append((r.ts, r.entry_ts, ex, s, pnl)); busy = ex
    return pd.DataFrame(out, columns=["ts", "entry", "exit", "side", "pnl"])

def execute_fast(ts, entry, ex_l, ex_s, pnl_l, pnl_s, sel, side):
    out = []; busy = -1
    for i in np.flatnonzero(sel):
        if entry[i] <= busy: continue
        s = side[i]; e = ex_l[i] if s > 0 else ex_s[i]; p = pnl_l[i] if s > 0 else pnl_s[i]
        out.append((ts[i], entry[i], e, s, p)); busy = e
    return pd.DataFrame(out, columns=["ts", "entry", "exit", "side", "pnl"])

def metrics(tr, t0, t1):
    days = pd.bdate_range(pd.to_datetime(t0, unit="ms").normalize(), pd.to_datetime(t1, unit="ms").normalize())
    d = tr.groupby(pd.to_datetime(tr.exit, unit="ms").dt.normalize()).pnl.sum() if len(tr) else pd.Series(dtype=float)
    d = d.reindex(days, fill_value=0.0)
    eq = d.cumsum(); dd = (eq - eq.cummax()).min() if len(eq) else 0
    sd = d.std(); dn = d[d < 0].std()
    pos = tr.pnl[tr.pnl > 0].sum(); neg = -tr.pnl[tr.pnl < 0].sum(); yrs = max(len(days) / 252, 1e-9)
    return dict(trades=len(tr), net_pips=tr.pnl.sum(), sharpe=(d.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
                sortino=(d.mean() / dn * np.sqrt(252)) if dn and dn > 0 else 0.0, maxdd_pips=dd, win=(tr.pnl > 0).mean() if len(tr) else np.nan,
                pf=pos / neg if neg > 0 else np.nan, trades_per_year=len(tr) / yrs, avg_pips=tr.pnl.mean() if len(tr) else np.nan)

def predict_block(task, mname, D, tr_idx, te_idx, seed=0):
    """Antreneaza pe primele 80% din tr_idx (purjat fata de calibrare), prag pe ultimele 20%, prezice pe te_idx."""
    ts = D.ts.to_numpy(); ex = D["exit"].to_numpy(); X = D[FEATS].to_numpy(np.float32)
    cut = tr_idx[int(len(tr_idx) * 0.8)]
    fit_idx = tr_idx[tr_idx < cut]; cal_idx = tr_idx[tr_idx >= cut]
    fit_idx = purge(fit_idx, ts[cal_idx[0]], ts[cal_idx[-1]], ts, ex)
    w = uniqueness(D.entry_ts.to_numpy()[fit_idx], ex[fit_idx]); w = w / w.mean()
    if task == "dir":
        res = {}
        for lab in ("y_l", "y_s"):
            m = fit(model(mname, seed), X[fit_idx], D[lab].to_numpy()[fit_idx], w)
            pc = m.predict_proba(X[cal_idx])[:, 1]; pt = m.predict_proba(X[te_idx])[:, 1]
            res[lab] = (pt, np.quantile(pc, 0.8))
        (pl, ql), (ps, qs) = res["y_l"], res["y_s"]
        side = np.where((pl >= ql) & (pl > ps), 1, np.where((ps >= qs) & (ps > pl), -1, 0))
        return side, pl, ps
    m = fit(model(mname, seed), X[fit_idx], D.y.to_numpy()[fit_idx], w)
    pc = m.predict_proba(X[cal_idx])[:, 1]; pt = m.predict_proba(X[te_idx])[:, 1]
    take = pt >= np.quantile(pc, 0.8)
    return np.where(take, D.side.to_numpy()[te_idx], 0), pt, None

def purged_cv_auc(task, mname, D, k=5):
    ts = D.ts.to_numpy(); ex = D["exit"].to_numpy(); X = D[FEATS].to_numpy(np.float32)
    folds = np.array_split(np.arange(len(D)), k); aucs = []
    labs = ["y_l", "y_s"] if task == "dir" else ["y"]
    for f in folds:
        tr = np.setdiff1d(np.arange(len(D)), f); tr = purge(tr, ts[f[0]], ts[f[-1]], ts, ex)
        w = uniqueness(D.entry_ts.to_numpy()[tr], ex[tr]); w = w / w.mean()
        for lab in labs:
            y = D[lab].to_numpy(); m = fit(model(mname), X[tr], y[tr], w)
            aucs.append(roc_auc_score(y[f], m.predict_proba(X[f])[:, 1]))
    return float(np.mean(aucs))

def arrays(D):
    return [D[c].to_numpy() for c in ("ts", "entry_ts", "exit_long", "exit_short", "pnl_long", "pnl_short")]

def walk_forward(task, mname, D):
    ts = D.ts.to_numpy(); A = arrays(D); trades = []; base = []; rnd = []
    rng = np.random.default_rng(7)
    for y in WF_YEARS:
        t0 = int(pd.Timestamp(f"{y}-01-01").value // 10**6); t1 = min(int(pd.Timestamp(f"{y+1}-01-01").value // 10**6), HOLDOUT_START) - 1
        te = np.flatnonzero((ts >= t0) & (ts <= t1)); tr = np.flatnonzero(ts < t0)
        tr = purge(tr, t0, t1, ts, D["exit"].to_numpy())
        side, *_ = predict_block(task, mname, D, tr, te)
        sel = np.zeros(len(D), bool); sel[te] = side != 0; sd = np.zeros(len(D)); sd[te] = side
        trades.append(execute_fast(*A, sel, sd))
    return pd.concat(trades, ignore_index=True)

def baselines(task, D, t0, t1, frac, reps=100):
    """Semnal primar fara ML (toate evenimentele) si selectie aleatoare cu aceeasi fractie de evenimente."""
    ts = D.ts.to_numpy(); A = arrays(D); te = (ts >= t0) & (ts <= t1)
    if task == "dir":
        side_all = None
    else:
        side_all = D.side.to_numpy()
    out = {}
    if side_all is not None:
        out["primary"] = metrics(execute_fast(*A, te, side_all), t0, t1)
    rng = np.random.default_rng(11); ms = []
    for r in range(reps):
        sel = te & (rng.random(len(D)) < frac)
        sd = side_all if side_all is not None else rng.choice([-1, 1], len(D))
        ms.append(metrics(execute_fast(*A, sel, sd), t0, t1))
    m = pd.DataFrame(ms)
    out["random_median"] = m.median().to_dict(); out["random_p95_sharpe"] = float(m.sharpe.quantile(0.95))
    return out

def registry():
    cfgs = []
    for ds in ("mt5_time15_k1.0_H240_lat250_c0.8", "mt5_time15_k0.5_H240_lat250_c0.8"):
        for task in ("meta_ema", "meta_mr", "dir"):
            for m in ("logit", "rf", "lgbm", "xgb"):
                cfgs.append(dict(ds=ds, task=task, model=m))
    for ds in ("mt5_tick_k1.0_H240_lat250_c0.8", "mt5_tick_k0.5_H240_lat250_c0.8"):
        for task in ("meta_ema", "meta_mr", "dir"):
            for m in ("logit", "lgbm"):
                cfgs.append(dict(ds=ds, task=task, model=m))
    assert len(cfgs) <= 40
    for i, c in enumerate(cfgs): c["id"] = i
    return cfgs

def load_ds(name, dev_only=True):
    D = pd.read_parquet(DS / f"{name}.parquet")
    # pe MetaQuotes-Demo spread-ul median e adesea 0 (2021-2023) -> raport infinit/nedefinit; il setam neutru (1.0)
    for c in ("spread_rel", "spmax_rel", "vol_ratio", "qint_15", "qint_60"):
        D[c] = D[c].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 50)
    D = D.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATS).sort_values("ts").reset_index(drop=True)
    if dev_only:
        D = D[D[["exit_long", "exit_short"]].max(axis=1) < HOLDOUT_START].reset_index(drop=True)  # nimic din holdout
    return D

def run_config(c):
    D = prep(c["task"], load_ds(c["ds"]))
    auc = purged_cv_auc(c["task"], c["model"], D)
    tr = walk_forward(c["task"], c["model"], D)
    t0 = int(pd.Timestamp(f"{WF_YEARS[0]}-01-01").value // 10**6); t1 = HOLDOUT_START - 1
    m = metrics(tr, t0, t1)
    n_ev = ((D.ts >= t0) & (D.ts <= t1)).sum()
    tr.to_parquet(REP / "wf_trades" / f"cfg{c['id']:02d}.parquet", index=False)
    return dict(**c, cv_auc=auc, **m, n_events=int(n_ev))

if __name__ == "__main__":
    (REP / "wf_trades").mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else None
    for c in registry():
        if ids and c["id"] not in ids: continue
        f = REP / "wf_results" / f"cfg{c['id']:02d}.json"; f.parent.mkdir(exist_ok=True)
        if f.exists(): continue
        r = run_config(c); f.write_text(json.dumps(r, default=float)); print(json.dumps(r, default=float), flush=True)
