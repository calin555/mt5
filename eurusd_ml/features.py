"""Feature engineering STRICT CAUZAL.

Regula: feature-ul de la bara t folosește doar informație disponibilă la închiderea
barei t (rolling înapoi, fără shift negativ). Intrarea în poziție se face la
deschiderea barei t+1 (vezi labeling.py). Testul tests/test_features.py verifică
asta: feature-urile calculate pe date trunchiate == cele calculate pe tot setul.
"""
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def build_features(df: pd.DataFrame, windows=(4, 12, 24, 72, 168)) -> pd.DataFrame:
    c = df["close"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    a24 = atr(df, 24)
    for w in windows:
        f[f"ret_{w}"] = np.log(c / c.shift(w))                       # randament pe w bare
        f[f"vol_{w}"] = lr.rolling(w, min_periods=w).std()           # volatilitate realizată
        f[f"z_{w}"] = (c - c.rolling(w, min_periods=w).mean()) / (c.rolling(w, min_periods=w).std() + 1e-12)
        f[f"ema_dist_{w}"] = (c - c.ewm(span=w, adjust=False, min_periods=w).mean()) / a24  # momentum normalizat
    f["vol_ratio"] = f["vol_12"] / f["vol_168"]                       # regim de volatilitate
    f["atr_rel"] = a24 / c
    f["rsi_14"] = (rsi(c, 14) - 50) / 50
    f["rsi_2"] = (rsi(c, 2) - 50) / 50
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_n"] = macd / a24
    f["macd_hist_n"] = (macd - macd.ewm(span=9, adjust=False).mean()) / a24
    hi, lo = df["high"].rolling(72).max(), df["low"].rolling(72).min()
    f["donchian_pos"] = (c - lo) / (hi - lo + 1e-12)                  # poziția în canalul de 72 bare
    f["range_n"] = (df["high"] - df["low"]) / a24
    f["body_n"] = (c - df["open"]) / a24
    v = np.log1p(df["volume"])
    f["tickvol_z"] = (v - v.rolling(168, min_periods=168).mean()) / (v.rolling(168, min_periods=168).std() + 1e-12)
    f["spread_n"] = df["cost_spread"] / a24                           # spread raportat la volatilitate
    h = df.index.hour + df.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    f["dow"] = df.index.dayofweek / 4.0
    return f.replace([np.inf, -np.inf], np.nan)
