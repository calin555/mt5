"""Descărcare tick-uri Dukascopy + agregare în bare.

Format Dukascopy: https://datafeed.dukascopy.com/datafeed/EURUSD/AAAA/LL/ZZ/HHh_ticks.bi5
  - LL = luna indexată de la 0 (ianuarie = 00) !
  - fișier comprimat LZMA; fiecare înregistrare are 20 octeți big-endian:
    uint32 ms_de_la_inceputul_orei, uint32 ask, uint32 bid, float32 ask_vol, float32 bid_vol
  - prețul real = valoare / 1e5 pentru EURUSD; orele sunt în UTC.
Un fișier gol (0 octeți) = oră fără tick-uri (weekend / sărbători).

Alternative documentate în README: `duka` (pip), `dukascopy-node` (npx).
"""
from __future__ import annotations

import lzma
import struct
import time
import random
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"
REC = np.dtype([("ms", ">u4"), ("ask", ">u4"), ("bid", ">u4"), ("askv", ">f4"), ("bidv", ">f4")])


def hour_path(raw_dir: str, sym: str, t: datetime) -> Path:
    return Path(raw_dir) / sym / f"{t:%Y}" / f"{t:%m}" / f"{t:%d}" / f"{t:%H}h.bi5"


def fetch_hour(sym: str, t: datetime, raw_dir: str, retries: int = 12) -> str:
    """Descarcă o oră (cu cache + reîncercări cu backoff). Returnează starea."""
    p = hour_path(raw_dir, sym, t)
    if p.exists():
        return "cached"
    # weekend complet: FX e închis sâmbătă și duminică până ~21-22 UTC
    url = URL.format(sym=sym, y=t.year, m=t.month - 1, d=t.day, h=t.hour)
    delay = 1.0
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(p)  # scriere atomică => resume sigur
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"")
                return "empty"
        except Exception:
            pass
        time.sleep(delay + random.random())
        delay = min(delay * 1.7, 30)
    return "fail"


def hours_between(start: str, end: str):
    t = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) + timedelta(days=1)
    now = datetime.now(timezone.utc) - timedelta(hours=2)
    while t < min(e, now):
        # sărim sâmbăta (piața închisă toată ziua)
        if t.weekday() != 5:
            yield t
        t += timedelta(hours=1)


def download(cfg: dict) -> dict:
    d = cfg["data"]
    # cele mai noi ore întâi: dacă serverul e lent, avem măcar istoricul recent
    hours = list(hours_between(d["start"], d["end"]))[::-1]
    stats: dict[str, int] = {}
    with ThreadPoolExecutor(d["workers"]) as ex:
        futs = [ex.submit(fetch_hour, cfg["symbol"], t, d["raw_dir"]) for t in hours]
        for i, f in enumerate(as_completed(futs), 1):
            s = f.result()
            stats[s] = stats.get(s, 0) + 1
            if i % 2000 == 0:
                print(f"[download] {i}/{len(hours)} {stats}", flush=True)
    print(f"[download] gata: {stats}")
    return stats


def decode_hour(path: Path, t: datetime, point: float) -> pd.DataFrame:
    """Decodează un fișier .bi5 într-un DataFrame cu tick-uri (timestamp UTC)."""
    raw = path.read_bytes()
    if not raw:
        return pd.DataFrame()
    arr = np.frombuffer(lzma.decompress(raw), dtype=REC)
    ts = pd.Timestamp(t).tz_convert(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t)
    return pd.DataFrame({
        "time": ts + pd.to_timedelta(arr["ms"].astype(np.int64), unit="ms"),
        "ask": arr["ask"].astype(np.float64) * point,
        "bid": arr["bid"].astype(np.float64) * point,
        "askv": arr["askv"].astype(np.float64),
        "bidv": arr["bidv"].astype(np.float64),
    })


def ticks_to_bars(ticks: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Agregă tick-uri în bare: OHLC bid și ask, spread mediu/max, nr. tick-uri, volum."""
    g = ticks.set_index("time").resample(rule, label="left", closed="left")
    out = pd.concat({
        "bid_open": g["bid"].first(), "bid_high": g["bid"].max(),
        "bid_low": g["bid"].min(), "bid_close": g["bid"].last(),
        "ask_open": g["ask"].first(), "ask_high": g["ask"].max(),
        "ask_low": g["ask"].min(), "ask_close": g["ask"].last(),
        "ticks": g["bid"].count(),
        "volume": (g["askv"].sum() + g["bidv"].sum()),
    }, axis=1)
    sp = (ticks["ask"] - ticks["bid"]).to_frame("sp").set_index(ticks["time"]).resample(rule, label="left", closed="left")["sp"]
    out["spread_mean"] = sp.mean()
    out["spread_max"] = sp.max()
    return out[out["ticks"] > 0]


def build_bars(cfg: dict, rules=("1min", "15min", "1h")) -> None:
    """Construiește barele lună cu lună (memorie mică), apoi le concatenează."""
    d = cfg["data"]
    out_dir = Path(d["bars_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: dict[str, list] = {r: [] for r in rules}
    hours = list(hours_between(d["start"], d["end"]))
    by_month: dict[str, list] = {}
    for t in hours:
        by_month.setdefault(f"{t:%Y-%m}", []).append(t)
    missing = 0
    for mon, hs in sorted(by_month.items()):
        frames = []
        for t in hs:
            p = hour_path(d["raw_dir"], cfg["symbol"], t)
            if not p.exists():
                missing += 1
                continue
            try:
                frames.append(decode_hour(p, t.replace(tzinfo=None), d["point"]))
            except Exception:
                missing += 1
        frames = [f for f in frames if len(f)]
        if not frames:
            continue
        ticks = pd.concat(frames, ignore_index=True)
        for r in rules:
            parts[r].append(ticks_to_bars(ticks, r))
        print(f"[build] {mon}: {len(ticks):,} tick-uri", flush=True)
    for r in rules:
        df = pd.concat(parts[r]).sort_index()
        df.to_parquet(out_dir / f"{cfg['symbol']}_{r}.parquet")
        print(f"[build] {r}: {len(df):,} bare {df.index[0]} -> {df.index[-1]}")
    print(f"[build] ore lipsă/nedescărcate: {missing}")


def load_bars(cfg: dict, tf: str | None = None) -> pd.DataFrame:
    tf = tf or cfg["timeframe"]
    rule = {"1h": "1h", "15m": "15min", "1m": "1min"}.get(tf, tf)
    return pd.read_parquet(Path(cfg["data"]["bars_dir"]) / f"{cfg['symbol']}_{rule}.parquet")


def load_mt5_csv(path: str) -> pd.DataFrame:
    """Barele exportate din MT5 demo (bid OHLC, tick_volume, spread în puncte)."""
    d = pd.read_csv(path)
    d["time"] = pd.to_datetime(d["time"], format="%Y.%m.%d %H:%M:%S")
    d = d.set_index("time").sort_index()
    return pd.DataFrame({
        "open": d["open"], "high": d["high"], "low": d["low"], "close": d["close"],
        "volume": d["tick_volume"].astype(float),
        "spread": d["spread"].astype(float) * 1e-5,
    })


def load_research_bars(cfg: dict) -> pd.DataFrame:
    """Bare unificate: open/high/low/close (bid), volume, spread (preț).
    Spread-ul folosit în costuri = max(spread măsurat, min_spread_pips)."""
    if cfg["source"] == "mt5":
        df = load_mt5_csv(cfg["mt5_csv"])
    else:
        b = load_bars(cfg)
        df = pd.DataFrame({"open": b.bid_open, "high": b.bid_high, "low": b.bid_low,
                           "close": b.bid_close, "volume": b.ticks.astype(float),
                           "spread": b.spread_mean})
    df = df[df.index >= cfg["research_start"]]
    floor = cfg["costs"]["min_spread_pips"] * cfg["costs"]["pip"]
    df["cost_spread"] = df["spread"].clip(lower=floor)
    return df
