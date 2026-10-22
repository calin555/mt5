"""Downloader Dukascopy politicos, reluabil, cel mai nou intai.
- un singur fir (concurenta 1), pauza intre cereri, backoff exponential pe 429/503/erori;
- fisierele .bi5 se salveaza pe disc; o ora deja descarcata (sau marcata goala) se sare;
- log de progres cu throughput si ETA.
Rulare: nohup python -m eurusd_ml.v2.duka_download --start 2015-01-01 --end 2026-10-22 &
"""
import argparse, time, os, random, urllib.request, urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

URL = "https://datafeed.dukascopy.com/datafeed/{s}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"  # luna e 0-indexata

def path_for(root, s, t):
    return Path(root) / s / f"{t:%Y}" / f"{t:%m}" / f"{t:%d}" / f"{t:%H}h.bi5"

def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research-downloader (low rate)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD"); ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None); ap.add_argument("--root", default="/workspace/ticks/duka_raw")
    ap.add_argument("--pause", type=float, default=0.6)
    a = ap.parse_args()
    start = datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc) if a.end else datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2))
    hours = []
    t = end
    while t >= start:
        wd = t.weekday()
        # weekend FX: sambata intreaga, vineri dupa 22 UTC, duminica inainte de 21 UTC -> fara date
        if not (wd == 5 or (wd == 4 and t.hour >= 22) or (wd == 6 and t.hour < 21)):
            hours.append(t)
        t -= timedelta(hours=1)
    total = len(hours); done = 0; fetched = 0; t0 = time.time(); backoff = 0
    print(f"{datetime.now():%F %T} START {total} ore {hours[-1]:%F} .. {hours[0]:%F %H}h", flush=True)
    for t in hours:
        p = path_for(a.root, a.symbol, t); done += 1
        if p.exists() or p.with_suffix(".empty").exists():
            continue
        url = URL.format(s=a.symbol, y=t.year, m=t.month - 1, d=t.day, h=t.hour)
        tries = 0
        while True:
            try:
                data = fetch(url); backoff = max(0, backoff // 2)
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp"); tmp.write_bytes(data); os.replace(tmp, p) if data else (p.with_suffix(".empty").touch(), tmp.unlink())
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    p.parent.mkdir(parents=True, exist_ok=True); p.with_suffix(".empty").touch(); break
                err = f"HTTP {e.code}"
            except Exception as e:
                err = repr(e)[:120]
            tries += 1; backoff = min(600, max(5, backoff * 2 or 5))
            print(f"{datetime.now():%F %T} {t:%F %H}h {err} try {tries} sleep {backoff}s", flush=True)
            time.sleep(backoff + random.random())
        fetched += 1
        time.sleep(a.pause + random.random() * 0.4)
        if fetched % 200 == 0:
            el = time.time() - t0; rate = fetched / el
            print(f"{datetime.now():%F %T} progres {done}/{total} ({100*done/total:.1f}%) curent {t:%F %H}h  {rate*3600:.0f} ore/h  ETA {(total-done)/rate/3600:.1f} h", flush=True)
    print(f"{datetime.now():%F %T} DONE {done}/{total}", flush=True)

if __name__ == "__main__":
    main()
