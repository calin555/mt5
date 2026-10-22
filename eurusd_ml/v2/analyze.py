"""Agregare walk-forward: tabel configuratii, baseline-uri, PBO (CSCV), Deflated Sharpe, grafice.
Apoi (comanda separata 'holdout') evaluarea UNICA pe holdout a configuratiei alese inainte."""
import json, sys, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from .research import (REP, registry, load_ds, prep, baselines, metrics, WF_YEARS, HOLDOUT_START, arrays, execute_fast,
                       predict_block, purge)

T0 = int(pd.Timestamp(f"{WF_YEARS[0]}-01-01").value // 10**6); T1 = HOLDOUT_START - 1

def daily(tr, t0=T0, t1=T1):
    days = pd.bdate_range(pd.to_datetime(t0, unit="ms").normalize(), pd.to_datetime(t1, unit="ms").normalize())
    d = tr.groupby(pd.to_datetime(tr.exit, unit="ms").dt.normalize()).pnl.sum() if len(tr) else pd.Series(dtype=float)
    return d.reindex(days, fill_value=0.0)

def pbo_cscv(M, S=16):
    """Bailey et al. 2017: probabilitatea ca cea mai buna config in-sample sa fie sub mediana out-of-sample."""
    blocks = np.array_split(np.arange(len(M)), S); logits = []
    for comb in itertools.combinations(range(S), S // 2):
        IS = np.concatenate([blocks[i] for i in comb]); OS = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        sr = lambda X: X.mean(0) / (X.std(0) + 1e-12)
        best = np.argmax(sr(M[IS])); r = stats.rankdata(sr(M[OS]))[best] / (M.shape[1] + 1)
        logits.append(math.log(r / (1 - r)))
    return float(np.mean(np.array(logits) <= 0))

def deflated_sharpe(d_best, sr_trials):
    """Bailey & Lopez de Prado 2014. Sharpe-uri zilnice (ne-anualizate)."""
    n = len(d_best); sr = d_best.mean() / d_best.std(); g3 = stats.skew(d_best); g4 = stats.kurtosis(d_best, fisher=False)
    N = len(sr_trials); v = np.var(sr_trials); em = 0.5772156649
    sr0 = math.sqrt(v) * ((1 - em) * stats.norm.ppf(1 - 1 / N) + em * stats.norm.ppf(1 - 1 / (N * math.e)))
    z = (sr - sr0) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(z)), float(sr0 * math.sqrt(252))

def wf_summary():
    R = pd.DataFrame([json.loads(f.read_text()) for f in sorted((REP / "wf_results").glob("*.json"))]).sort_values("id")
    # baseline-uri pe aceeasi perioada walk-forward
    rows = []; cache = {}
    for _, r in R.iterrows():
        key = (r.ds, r.task)
        if key not in cache:
            D = prep(r.task, load_ds(r.ds)); frac = 0.2
            cache[key] = baselines(r.task, D, T0, T1, frac, reps=50)
        b = cache[key]
        rows.append(dict(id=r.id, primary_sharpe=b.get("primary", {}).get("sharpe", np.nan), primary_net=b.get("primary", {}).get("net_pips", np.nan),
                         primary_pf=b.get("primary", {}).get("pf", np.nan), random_sharpe_med=b["random_median"]["sharpe"],
                         random_net_med=b["random_median"]["net_pips"], random_p95_sharpe=b["random_p95_sharpe"]))
    R = R.merge(pd.DataFrame(rows), on="id")
    D = {i: daily(pd.read_parquet(REP / "wf_trades" / f"cfg{i:02d}.parquet")) for i in R.id}
    M = np.column_stack([D[i].to_numpy() for i in R.id])
    # CSCV pe randamente lunare (zilele sunt prea zgomotoase)
    Mm = pd.DataFrame(M, index=D[R.id.iloc[0]].index).resample("ME").sum().to_numpy()
    pbo = pbo_cscv(Mm)
    best = R.sharpe.idxmax(); bi = R.loc[best, "id"]
    dsr, sr0 = deflated_sharpe(D[bi].to_numpy(), (M.mean(0) / M.std(0)))
    R.to_csv(REP / "wf_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in R.id: ax.plot(D[i].cumsum(), lw=0.6, color="grey", alpha=0.5)
    ax.plot(D[bi].cumsum(), color="C0", lw=1.6, label=f"cea mai buna WF: cfg{bi}")
    ax.set_title(f"Walk-forward 2020-01 .. 2025-10 (pips, 1 unitate) - {len(R)} configuratii; PBO={pbo:.2f}"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(REP / "wf_equity_all.png", dpi=120, bbox_inches="tight")
    out = dict(n_configs=len(R), pbo=pbo, best_id=int(bi), best_sharpe=float(R.loc[best, "sharpe"]), dsr_prob=dsr, sr0_annual=sr0)
    (REP / "wf_stats.json").write_text(json.dumps(out, indent=1)); return R, out

def holdout(cfg_id):
    lock = REP / "HOLDOUT_USED.json"
    if lock.exists(): print("Holdout deja evaluat:", lock.read_text()); return
    c = [x for x in registry() if x["id"] == cfg_id][0]
    full = load_ds(c["ds"], dev_only=False); D = prep(c["task"], full)
    ts = D.ts.to_numpy(); ex = D["exit"].to_numpy()
    te = np.flatnonzero(ts >= HOLDOUT_START); tr = purge(np.flatnonzero(ts < HOLDOUT_START), HOLDOUT_START, ts.max(), ts, ex)
    side, *_ = predict_block(c["task"], c["model"], D, tr, te)
    A = arrays(D); sel = np.zeros(len(D), bool); sel[te] = side != 0; sd = np.zeros(len(D)); sd[te] = side
    t0, t1 = HOLDOUT_START, int(ts.max())
    trd = execute_fast(*A, sel, sd); m = metrics(trd, t0, t1)
    frac = float(sel[te].mean()); b = baselines(c["task"], D, t0, t1, frac, reps=200)
    res = dict(config=c, ml=m, selected_frac=frac, **b, period=[str(pd.to_datetime(t0, unit="ms")), str(pd.to_datetime(t1, unit="ms"))])
    lock.write_text(json.dumps(res, indent=1, default=float)); trd.to_parquet(REP / "holdout_trades.parquet", index=False)
    fig, ax = plt.subplots(figsize=(11, 5)); ax.plot(daily(trd, t0, t1).cumsum(), label=f"ML cfg{cfg_id}")
    if c["task"] != "dir":
        ax.plot(daily(execute_fast(*A, (ts >= t0), D.side.to_numpy()), t0, t1).cumsum(), label="semnal primar fara ML")
    ax.set_title("Holdout final (evaluat o singura data), pips"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(REP / "holdout_equity.png", dpi=120, bbox_inches="tight")
    print(json.dumps(res, indent=1, default=float))

if __name__ == "__main__":
    if sys.argv[1] == "wf":
        R, out = wf_summary(); pd.set_option("display.width", 250)
        print(R[["id", "ds", "task", "model", "cv_auc", "trades", "net_pips", "sharpe", "pf", "maxdd_pips", "primary_sharpe", "primary_net", "random_sharpe_med", "random_p95_sharpe"]].round(3).to_string()); print(out)
    else:
        holdout(int(sys.argv[2]))
