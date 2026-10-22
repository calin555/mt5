"""Raport de calitate a datelor tick, luna cu luna (numaratori reale)."""
import sys
import numpy as np, pandas as pd
from .ticks_store import ROOT

def month_stats(f):
    d = pd.read_parquet(f, columns=["ts", "bid", "ask"])
    ts = d.ts.to_numpy(); sp = (d.ask - d.bid).to_numpy() * 1e4
    mid = (d.ask + d.bid).to_numpy() / 2
    dt = np.diff(ts) / 1000
    t = pd.to_datetime(ts[1:], unit="ms")
    # gol "anormal": >30 min, fara sa traverseze weekendul (vineri 21h - duminica 22h UTC)
    prev = pd.to_datetime(ts[:-1], unit="ms")
    wk = (prev.weekday >= 4) & (t.weekday.isin([6, 0])) | (prev.weekday == 4) & (prev.hour >= 20)
    gaps = int(((dt > 1800) & ~np.asarray(wk)).sum())
    r = np.abs(np.diff(np.log(mid))) * 1e4
    wd = pd.to_datetime(ts, unit="ms").weekday
    return dict(month=f.stem, ticks=len(d), first=pd.to_datetime(ts[0], unit="ms"), last=pd.to_datetime(ts[-1], unit="ms"),
                dup_ts_price=int(d.duplicated().sum()), non_increasing_ts=int((dt < 0).sum()),
                bid_ge_ask=int((d.bid >= d.ask).sum()), weekend_ticks=int((wd == 5).sum()),
                gaps_30m=gaps, spikes_20bp=int((r > 20).sum()),
                spread_p10=np.percentile(sp, 10), spread_med=np.median(sp), spread_p90=np.percentile(sp, 90), spread_p99=np.percentile(sp, 99))

def report(src):
    rows = [month_stats(f) for f in sorted((ROOT / src).glob("*.parquet"))]
    df = pd.DataFrame(rows); df.to_csv(f"/workspace/eurusd_ml/reports/v2/quality_{src}.csv", index=False)
    return df

if __name__ == "__main__":
    import os; os.makedirs("/workspace/eurusd_ml/reports/v2", exist_ok=True)
    df = report(sys.argv[1] if len(sys.argv) > 1 else "mt5")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(df.round(2).to_string())
    print("TOTAL", df.ticks.sum(), df[["dup_ts_price", "bid_ge_ask", "weekend_ticks", "gaps_30m", "spikes_20bp"]].sum().to_dict())
