"""Caracteristici (strict cauzale) + evenimente + etichete pe tick-uri -> dataset Parquet.

Regula anti look-ahead: o decizie la momentul t vede DOAR barele de 1 minut cu ts_end <= t
(bara care se inchide la t e completa la t). Etichetele incep cu primul tick dupa t + latenta.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from .clean_bars import BARS, CLEAN
from .labels import simulate

OUT = Path("/workspace/ticks/datasets")
START = "2018-01"   # inainte de 2018 fluxul MetaQuotes e eterogen (multe spike-uri/duplicate, vezi raportul de calitate)

def load_minutes(src="mt5", start=START):
    fs = [f for f in sorted((BARS / src).glob("*.parquet")) if f.stem >= start]
    b = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).sort_values("ts_end").reset_index(drop=True)
    return b

def features(b):
    """Toate ferestrele sunt in minute de tranzactionare (randuri de 1m), deci se termina la randul curent."""
    c = b.c; lc = np.log(c); f = pd.DataFrame({"ts": b.ts_end})
    for w in (15, 60, 240, 960):
        f[f"ret_{w}"] = lc - lc.shift(w)
    rv = b.rv
    for w in (60, 240, 1440):
        f[f"vol_{w}"] = np.sqrt(rv.rolling(w, min_periods=w // 2).sum() / w)
    f["vol_ratio"] = f.vol_60 / f.vol_1440
    for w in (15, 60, 240):
        up = b.up.rolling(w).sum(); dn = b.dn.rolling(w).sum()
        f[f"timb_{w}"] = (up - dn) / (up + dn).replace(0, np.nan)
        f[f"ofi_{w}"] = b.ofi.rolling(w).sum() / b.n.rolling(w).sum().replace(0, np.nan)
    nmean = b.n.rolling(1440, min_periods=500).mean()
    f["qint_15"] = b.n.rolling(15).mean() / nmean
    f["qint_60"] = b.n.rolling(60).mean() / nmean
    spm = b.sp.rolling(1440, min_periods=500).median()
    f["spread_rel"] = b.sp.rolling(15).mean() / spm
    f["spread_pips"] = b.sp.rolling(15).mean() * 1e4
    f["spmax_rel"] = b.spmax.rolling(15).max() / spm
    hh = b.h.rolling(240).max(); ll = b.l.rolling(240).min()
    f["rangepos_240"] = (c - ll) / (hh - ll).replace(0, np.nan)
    e1 = c.ewm(span=300, adjust=False).mean(); e2 = c.ewm(span=1200, adjust=False).mean()
    f["ema_gap"] = (e1 - e2) / c / f.vol_1440 / np.sqrt(240)
    m = c.rolling(240).mean(); s = c.rolling(240).std()
    f["z_240"] = (c - m) / s
    t = pd.to_datetime(b.ts_end, unit="ms")
    f["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24); f["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    f["dow"] = t.dt.day_of_week
    f["since_last_tick_min"] = (b.ts_end.diff() / 60000)
    # semnale primare simple
    f["side_ema"] = np.sign(e1 - e2)
    f["side_mr"] = np.where(f.z_240 > 2, -1, np.where(f.z_240 < -2, 1, 0))
    f["sig_rv1440"] = np.sqrt(rv.rolling(1440, min_periods=720).sum())  # vol log pe ~1 zi de tranzactionare
    return f

FEATS = None  # completat la import din coloanele de mai sus (fara ts, side_*, sig_*)

def events(b, f, kind, n_tick=None):
    """kind='time15': inchiderea fiecarei bare de 15m; kind='tick': o bara la fiecare n_tick tick-uri
    (aproximat la rezolutie de 1 minut: evenimentul e la sfarsitul minutului in care se completeaza bara)."""
    if kind == "time15":
        ok = (b.ts_end % 900000 == 0)
    else:
        cum = b.n.cumsum().to_numpy(); k = cum // n_tick
        ok = pd.Series(np.r_[False, k[1:] > k[:-1]])
    t = pd.to_datetime(b.ts_end, unit="ms")
    ok = ok & (b.ts_end.diff() <= 5 * 60000)            # piata activa
    ok = ok & ~((t.dt.day_of_week == 4) & (t.dt.hour >= 19)) & ~((t.dt.day_of_week == 6))  # fara vineri seara / duminica
    return np.flatnonzero(ok.to_numpy())

def build(src="mt5", kind="time15", k=1.0, H_min=240, lat=250, floor=0.3, cost=0.8, tag=None):
    b = load_minutes(src); f = features(b)
    n_tick = None
    if kind == "tick":
        dev = b[b.ts_end < pd.Timestamp("2025-10-22").value // 10**6]
        n_tick = int(round(dev.n.sum() / (len(dev) / 15)))  # tick-uri medii la 15 minute de tranzactionare
    idx = events(b, f, kind, n_tick)
    E = f.iloc[idx].dropna().copy()
    sig_h = E.sig_rv1440 * np.sqrt(H_min / 1440)
    E["sigma_px"] = sig_h * b.c.iloc[E.index].to_numpy()
    E["month"] = pd.to_datetime(E.ts, unit="ms").dt.strftime("%Y-%m")
    res = []
    months = sorted(E.month.unique())
    for i, mon in enumerate(months):
        part = [CLEAN / src / f"{mon}.parquet"]
        if i + 1 < len(months): part.append(CLEAN / src / f"{months[i+1]}.parquet")
        tk = pd.concat([pd.read_parquet(p) for p in part if p.exists()])
        e = E[E.month == mon]
        o = simulate(tk.ts.to_numpy(), tk.bid.to_numpy(), tk.ask.to_numpy(), e.ts.to_numpy(), e.sigma_px.to_numpy(), k, H_min * 60000, lat, floor, cost)
        res.append(pd.DataFrame(o, index=e.index, columns=["pnl_long", "exit_long", "pnl_short", "exit_short", "entry_ts", "reason"]))
        print(mon, len(e), flush=True)
    L = pd.concat(res); D = E.join(L).dropna(subset=["pnl_long", "pnl_short"])
    D["n_tick"] = n_tick
    OUT.mkdir(parents=True, exist_ok=True)
    tag = tag or f"{src}_{kind}_k{k}_H{H_min}_lat{lat}_c{cost}"
    D.reset_index(drop=True).to_parquet(OUT / f"{tag}.parquet", index=False)
    return D

if __name__ == "__main__":
    a = sys.argv[1:]
    build(kind=a[0], k=float(a[1]), H_min=int(a[2]), lat=int(a[3]) if len(a) > 3 else 250, cost=float(a[4]) if len(a) > 4 else 0.8)
