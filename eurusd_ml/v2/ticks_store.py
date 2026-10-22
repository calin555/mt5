"""Stocare tick-uri in Parquet partitionat pe luni (in afara git-ului).
Schema comuna: ts (int64, ms UTC), bid, ask (float64), src.
- MT5: fisiere .npy exportate de tick_dump.py (structura MetaTrader5: time_msc, bid, ask, flags ...).
- Dukascopy: fisiere .bi5 (LZMA, inregistrari de 20 octeti: ms, ask, bid, askvol, bidvol; pret in 1e-5).
"""
import lzma, glob, os, struct
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd

ROOT = Path("/workspace/ticks/parquet")

def _write(df, src):
    df = df.sort_values("ts").drop_duplicates()
    m = pd.to_datetime(df.ts, unit="ms").dt.strftime("%Y-%m")
    for mon, g in df.groupby(m):
        d = ROOT / src; d.mkdir(parents=True, exist_ok=True)
        f = d / f"{mon}.parquet"
        if f.exists():
            g = pd.concat([pd.read_parquet(f), g]).drop_duplicates().sort_values("ts")
        g.reset_index(drop=True).to_parquet(f, index=False)

def ingest_mt5(npy_dir="/home/box/.mt5/drive_c/xmdemo/ticks"):
    files = sorted(glob.glob(f"{npy_dir}/EURUSD_*.npy"))
    buf = []
    for i, f in enumerate(files):
        r = np.load(f)
        buf.append(pd.DataFrame({"ts": r["time_msc"].astype("int64"), "bid": r["bid"], "ask": r["ask"], "flags": r["flags"].astype("int32")}))
        if len(buf) >= 20 or i == len(files) - 1:
            _write(pd.concat(buf), "mt5"); buf = []

def decode_bi5(path, hour_start_ms):
    raw = lzma.decompress(Path(path).read_bytes())
    a = np.frombuffer(raw, dtype=">i4,>i4,>i4,>f4,>f4")
    return pd.DataFrame({"ts": hour_start_ms + a["f0"].astype("int64"), "ask": a["f1"] / 1e5, "bid": a["f2"] / 1e5,
                         "askv": a["f3"].astype("float32"), "bidv": a["f4"].astype("float32")})

def ingest_duka(raw="/workspace/ticks/duka_raw/EURUSD"):
    by_month = {}
    for f in glob.glob(f"{raw}/*/*/*/*h.bi5"):
        y, m, d, h = f.split("/")[-4:]; h = h[:2]
        by_month.setdefault(f"{y}-{m}", []).append((f, datetime(int(y), int(m), int(d), int(h), tzinfo=timezone.utc)))
    for mon, items in by_month.items():
        dfs = [decode_bi5(f, int(t.timestamp() * 1000)) for f, t in items if os.path.getsize(f) > 0]
        if dfs:
            out = ROOT / "duka" / f"{mon}.parquet"; out.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(dfs).sort_values("ts").reset_index(drop=True).to_parquet(out, index=False)

def load(src="mt5", start=None, end=None, cols=("ts", "bid", "ask")):
    fs = sorted((ROOT / src).glob("*.parquet"))
    fs = [f for f in fs if (start is None or f.stem >= start[:7]) and (end is None or f.stem <= end[:7])]
    return pd.concat([pd.read_parquet(f, columns=list(cols)) for f in fs], ignore_index=True)
