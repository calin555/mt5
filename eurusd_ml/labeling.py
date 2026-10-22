"""Triple-barrier labeling (López de Prado) + semnal primar pentru meta-labeling.

Pentru un eveniment la bara t (cunoscut la închiderea lui t):
  - intrare la deschiderea barei t+1
  - bariera sus/jos = intrare ± k * ATR(t)   (ATR calculat doar până la t)
  - bariera verticală = max_hold bare
  - dacă ambele bariere sunt atinse în aceeași bară => presupunem STOP (conservator)
Rezultatul: ieșire, preț ieșire, randament brut (în direcția side), cost, t1 (bara ieșirii,
folosită la purging în validare).
"""
import numpy as np
import pandas as pd
from .features import atr


def primary_signal(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Semnal primar simplu: +1 când EMA rapidă > EMA lentă, altfel -1."""
    c = df["close"]
    return np.sign(c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()).replace(0, np.nan)


def crossover_events(side: pd.Series) -> pd.Index:
    """Evenimente = barele unde semnalul primar își schimbă direcția."""
    ch = side.ne(side.shift(1)) & side.shift(1).notna() & side.notna()
    return side.index[ch]


def triple_barrier(df: pd.DataFrame, events: pd.Index, side: pd.Series, atr_window: int,
                   pt: float, sl: float, max_hold: int, cost_cfg: dict) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    sp = df["cost_spread"].to_numpy()
    a = atr(df, atr_window).to_numpy()
    pos = df.index.get_indexer(events)
    pip = cost_cfg["pip"]
    fixed = 2 * cost_cfg["slippage_pips"] * pip + cost_cfg["commission_per_lot_rt"] / 10.0 * pip  # 7$/lot ≈ 0.7 pip
    rows = []
    n = len(df)
    for t, ev in zip(pos, events):
        s = side.loc[ev]
        if t + 1 >= n or np.isnan(a[t]) or np.isnan(s):
            continue
        e = o[t + 1]
        up, dn = e + pt * a[t] if s > 0 else e + sl * a[t], e - sl * a[t] if s > 0 else e - pt * a[t]
        end = min(t + max_hold, n - 1)
        exit_px, j, kind = c[end], end, "vertical"
        for k in range(t + 1, end + 1):
            hit_up, hit_dn = h[k] >= up, l[k] <= dn
            if hit_up and hit_dn:          # ambiguu în bară: alegem stop-ul
                exit_px, j, kind = (dn, k, "sl") if s > 0 else (up, k, "sl")
                break
            if hit_up:
                exit_px, j, kind = up, k, ("pt" if s > 0 else "sl")
                break
            if hit_dn:
                exit_px, j, kind = dn, k, ("sl" if s > 0 else "pt")
                break
        gross = s * (exit_px - e)
        cost = 0.5 * sp[t + 1] + 0.5 * sp[j] + fixed      # spread la intrare + ieșire, slippage, comision
        rows.append((ev, df.index[t + 1], df.index[j], s, e, exit_px, kind, gross, cost))
    out = pd.DataFrame(rows, columns=["event", "entry_time", "t1", "side", "entry", "exit",
                                      "barrier", "gross", "cost"]).set_index("event")
    out["net"] = out["gross"] - out["cost"]
    out["net_pips"] = out["net"] / pip
    out["label"] = (out["net"] > 0).astype(int)       # 1 = trade-ul a câștigat DUPĂ costuri
    return out
