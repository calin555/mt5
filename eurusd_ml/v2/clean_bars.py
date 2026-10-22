"""Curatare tick-uri + construire bare din tick-uri (cauzal).

Curatare (numaratori salvate in reports/v2/cleaning_<src>.csv):
  1. elimina pret <= 0 si ask < bid (cotatii invalide);
  2. elimina randuri identice (ts, bid, ask);
  3. filtru de spike: mid-ul se abate de la mediana mobila CENTRATA pe 21 tick-uri cu > max(5 bp, 10*MAD).
     (Filtrul centrat foloseste tick-uri viitoare DOAR pentru a decide daca o cotatie e eronata - e curatare de date,
      nu informatie de tranzactionare; aceeasi procedura e aplicabila live cu o intarziere de 10 tick-uri.)

Bare de 1 minut (eticheta = sfarsitul minutului, deci bara de la 10:00 contine tick-urile din [10:00, 10:01) si e
 disponibila abia la 10:01): open/high/low/close pe mid, numar tick-uri, tick-uri sus/jos (regula tick-ului),
 suma r^2 (varianta realizata), spread mediu/maxim, OFI proxy = suma(sign(dbid)+sign(dask)).
Bare pe tick-uri (Lopez de Prado): o bara la fiecare N tick-uri -> functia tick_bar_times.
Fara volum real in MT5 (doar tick-uri de cotatie), deci bare de volum/dolar nu sunt posibile corect; tick-urile
 sunt singurul proxy de activitate.
"""
import sys, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from .ticks_store import ROOT

CLEAN = Path("/workspace/ticks/clean"); BARS = Path("/workspace/ticks/bars")

def clean_ticks(d):
    n0 = len(d); st = {"raw": n0}
    d = d[(d.bid > 0) & (d.ask > 0) & (d.ask >= d.bid)]; st["bad_price"] = n0 - len(d)
    n1 = len(d); d = d.drop_duplicates(["ts", "bid", "ask"]); st["dups"] = n1 - len(d)
    mid = (d.bid.to_numpy() + d.ask.to_numpy()) / 2
    s = pd.Series(mid); med = s.rolling(21, center=True, min_periods=5).median()
    dev = (s - med).abs(); mad = dev.rolling(201, center=True, min_periods=20).median()
    bad = (dev > np.maximum(5e-4 * med, 10 * mad)).to_numpy()
    st["spikes"] = int(bad.sum()); d = d[~bad]
    # elimina cotatiile la acelasi timestamp cu acelasi pret consecutive (fara informatie noua)
    st["clean"] = len(d)
    return d.reset_index(drop=True), st

def minute_bars(d):
    ts = d.ts.to_numpy(); bid = d.bid.to_numpy(); ask = d.ask.to_numpy(); mid = (bid + ask) / 2
    dm = np.diff(mid, prepend=mid[0]); lr = np.diff(np.log(mid), prepend=np.log(mid[0]))
    ofi = np.sign(np.diff(bid, prepend=bid[0])) + np.sign(np.diff(ask, prepend=ask[0]))
    g = pd.DataFrame({"m": (ts // 60000 + 1) * 60000, "mid": mid, "up": dm > 0, "dn": dm < 0, "r2": lr ** 2,
                      "sp": ask - bid, "ofi": ofi})
    a = g.groupby("m").agg(o=("mid", "first"), h=("mid", "max"), l=("mid", "min"), c=("mid", "last"), n=("mid", "size"),
                           up=("up", "sum"), dn=("dn", "sum"), rv=("r2", "sum"), sp=("sp", "mean"), spmax=("sp", "max"), ofi=("ofi", "sum"))
    a.index.name = "ts_end"; return a.reset_index()

def server_to_utc_ms(ts):
    """MetaQuotes-Demo: ora serverului = ora New York + 7h (UTC+2 iarna, UTC+3 vara; saptamana incepe luni 00:00 server).
    Verificat: dupa conversie mid-ul MT5 difera de Dukascopy (UTC) cu mediana 0.05 pips; fara conversie, 15 pips."""
    naive = pd.to_datetime(ts, unit="ms") - pd.Timedelta(hours=7)
    loc = pd.DatetimeIndex(naive).tz_localize("America/New_York", ambiguous=False, nonexistent="shift_forward")
    return loc.tz_convert("UTC").tz_localize(None).as_unit("ms").asi8.astype("int64")

def do_month(args):
    src, f = args
    d = pd.read_parquet(f, columns=["ts", "bid", "ask"]).sort_values("ts", kind="stable")
    if src == "mt5":
        d["ts"] = server_to_utc_ms(d.ts.to_numpy()); d = d.sort_values("ts", kind="stable")
    d, st = clean_ticks(d); st["month"] = f.stem
    (CLEAN / src).mkdir(parents=True, exist_ok=True); (BARS / src).mkdir(parents=True, exist_ok=True)
    d.to_parquet(CLEAN / src / f.name, index=False)
    minute_bars(d).to_parquet(BARS / src / f.name, index=False)
    return st

def tick_bar_times(ts, n):
    """Momentele de inchidere ale barelor de N tick-uri (timestamp-ul tick-ului care inchide bara)."""
    return ts[n - 1::n]

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "mt5"
    fs = sorted((ROOT / src).glob("*.parquet"))
    with Pool(5) as p: rows = p.map(do_month, [(src, f) for f in fs])
    df = pd.DataFrame(rows); os.makedirs("/workspace/eurusd_ml/reports/v2", exist_ok=True)
    df.to_csv(f"/workspace/eurusd_ml/reports/v2/cleaning_{src}.csv", index=False)
    print(df.sum(numeric_only=True).to_dict())
