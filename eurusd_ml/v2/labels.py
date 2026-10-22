"""Etichetare triple-barrier pe drumul REAL de tick-uri, separat pentru long si short.

Pentru un eveniment la momentul t (ms UTC):
  - intrarea = primul tick cu ts >= t + latenta;
  - long intra la ASK si iese la BID; short intra la BID si iese la ASK;
  - bariere: TP = SL = k * sigma (sigma = volatilitatea asteptata pe orizont, calculata DOAR din trecut, in pret);
  - prima atingere se cauta tick cu tick; bariera de timp = ultimul tick inainte de t_intrare + H;
  - spread efectiv = max(spread real, floor): pe MetaQuotes-Demo spread-ul e adesea 0, nerealist, deci impunem un minim;
  - costuri: comision dus-intors + slippage pe fiecare parte, scazute din rezultat (in pips).
Rezultatul: pnl net in pips si momentul iesirii, pentru long si pentru short (separat!).
"""
import numpy as np
from numba import njit

PIP = 1e-4

@njit(cache=True)
def _sim(ts, bid, ask, ev_t, ev_sig, k, H, lat, floor, cost):
    n = ev_t.shape[0]
    out = np.full((n, 6), np.nan)  # pnl_long, exit_long, pnl_short, exit_short, entry_ts, reason_long*10+reason_short
    m = ts.shape[0]
    for i in range(n):
        t0 = ev_t[i] + lat
        j = np.searchsorted(ts, t0)
        if j >= m:
            continue
        te = ts[j]
        if te - t0 > 300000:  # fara cotatii 5 minute dupa semnal -> piata inchisa, sarim
            continue
        mid0 = 0.5 * (bid[j] + ask[j]); h0 = 0.5 * max(ask[j] - bid[j], floor)
        al = mid0 + h0; bs = mid0 - h0
        w = k * ev_sig[i]
        tpl = al + w; sll = al - w; tps = bs - w; sls = bs + w
        dl = 0; ds = 0; rl = 0; rs = 0
        tend = te + H
        jj = j + 1
        while jj < m and ts[jj] <= tend and (dl == 0 or ds == 0):
            mid = 0.5 * (bid[jj] + ask[jj]); h = 0.5 * max(ask[jj] - bid[jj], floor)
            b = mid - h; a = mid + h
            if dl == 0:
                if b <= sll:   # stop executat la pretul pietei (bid curent, poate fi mai rau decat bariera); TP = limita la bariera
                    out[i, 0] = b - al; out[i, 1] = ts[jj]; dl = 1; rl = 2
                elif b >= tpl:
                    out[i, 0] = tpl - al; out[i, 1] = ts[jj]; dl = 1; rl = 1
            if ds == 0:
                if a >= sls:
                    out[i, 2] = bs - a; out[i, 3] = ts[jj]; ds = 1; rs = 2
                elif a <= tps:
                    out[i, 2] = bs - tps; out[i, 3] = ts[jj]; ds = 1; rs = 1
            jj += 1
        last = jj - 1
        midl = 0.5 * (bid[last] + ask[last]); hl = 0.5 * max(ask[last] - bid[last], floor)
        if dl == 0:
            out[i, 0] = (midl - hl) - al; out[i, 1] = ts[last]; rl = 3
        if ds == 0:
            out[i, 2] = bs - (midl + hl); out[i, 3] = ts[last]; rs = 3
        out[i, 0] = out[i, 0] / 1e-4 - cost; out[i, 2] = out[i, 2] / 1e-4 - cost
        out[i, 4] = te; out[i, 5] = rl * 10 + rs
    return out

def simulate(ts, bid, ask, ev_t, ev_sig, k, H_ms, lat_ms=250, floor_pips=0.3, cost_pips=0.8):
    """cost_pips = comision dus-intors (0.6) + slippage 0.1 pe fiecare parte (0.2). Valori implicite documentate in README."""
    return _sim(ts.astype(np.int64), bid.astype(np.float64), ask.astype(np.float64), ev_t.astype(np.int64),
                ev_sig.astype(np.float64), float(k), np.int64(H_ms), np.int64(lat_ms), floor_pips * PIP, float(cost_pips))
