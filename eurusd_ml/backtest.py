"""Backtest simplu pe trade-uri (rezultatele vin din triple_barrier, deja cu costuri).

Reguli: o singură poziție deodată (un eveniment nou e ignorat dacă poziția anterioară
e încă deschisă), 1 lot standard (100.000 EUR) => 1 pip ≈ 10 USD, capital nominal 100.000 USD
(fără levier efectiv peste 1:1). Metricile sunt calculate pe PnL zilnic.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAPITAL = 100_000.0
USD_PER_PIP = 10.0


def non_overlapping(trades: pd.DataFrame) -> pd.DataFrame:
    keep, busy_until = [], pd.Timestamp.min
    for ev, r in trades.sort_index().iterrows():
        if r["entry_time"] > busy_until:
            keep.append(ev)
            busy_until = r["t1"]
    return trades.loc[keep]


def metrics(trades: pd.DataFrame, bar_index: pd.DatetimeIndex) -> dict:
    if len(trades) == 0:
        return {"trades": 0}
    usd = trades["net_pips"] * USD_PER_PIP
    daily = usd.groupby(trades["t1"].dt.floor("D")).sum()
    days = pd.date_range(bar_index.min().floor("D"), bar_index.max().floor("D"), freq="B")
    r = (daily.reindex(days, fill_value=0.0) / CAPITAL)
    eq = CAPITAL + usd.cumsum()
    dd = (eq - np.maximum.accumulate(np.r_[CAPITAL, eq.to_numpy()])[1:])
    held = ((trades["t1"] - trades["entry_time"]).sum() / (bar_index.max() - bar_index.min()))
    downside = r[r < 0].std()
    win, loss = usd[usd > 0].sum(), -usd[usd < 0].sum()
    return {
        "trades": int(len(trades)),
        "net_pips": round(float(trades["net_pips"].sum()), 1),
        "gross_pips": round(float(trades["gross"].sum() / 1e-4), 1),
        "net_usd_1lot": round(float(usd.sum()), 0),
        "avg_pips": round(float(trades["net_pips"].mean()), 2),
        "win_rate": round(float((usd > 0).mean()), 3),
        "profit_factor": round(float(win / loss), 3) if loss > 0 else np.inf,
        "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else 0.0,
        "sortino": round(float(r.mean() / downside * np.sqrt(252)), 2) if downside and downside > 0 else 0.0,
        "max_dd_usd": round(float(dd.min()), 0),
        "max_dd_pct": round(float(dd.min() / CAPITAL * 100), 2),
        "exposure": round(float(held), 3),
    }


def random_baseline(pool: pd.DataFrame, n_pick: int, n_iter: int, seed: int) -> dict:
    """Alege aleator n_pick trade-uri din pool (aceleași evenimente candidate) de n_iter ori."""
    rng = np.random.default_rng(seed)
    if n_pick == 0 or len(pool) == 0:
        return {"rand_mean_pips": 0.0, "rand_p95_pips": 0.0}
    v = pool["net_pips"].to_numpy()
    sums = np.array([v[rng.choice(len(v), min(n_pick, len(v)), replace=False)].sum() for _ in range(n_iter)])
    return {"rand_mean_pips": round(float(sums.mean()), 1), "rand_p95_pips": round(float(np.percentile(sums, 95)), 1)}


def plot_equity(curves: dict, path: str, title: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, t in curves.items():
        if len(t):
            ax.plot(t["t1"], (t["net_pips"] * USD_PER_PIP).cumsum(), label=f"{name} ({len(t)} tr)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title); ax.set_ylabel("PnL net cumulat, USD (1 lot)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
