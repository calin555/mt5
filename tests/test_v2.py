"""Teste v2: etichete pe tick-uri corecte (bid/ask, prima atingere, SL inainte de TP), lipsa look-ahead in caracteristici,
purjare/embargo, unicitate."""
import numpy as np, pandas as pd
from eurusd_ml.v2.labels import simulate
from eurusd_ml.v2.dataset import features
from eurusd_ml.v2.research import purge, uniqueness, EMBARGO
from eurusd_ml.v2.clean_bars import minute_bars, clean_ticks

P = 1e-4
def ticks(mids, sp=0.0001, t0=1_000_000, dt=1000):
    m = np.array(mids, float); ts = t0 + np.arange(len(m)) * dt
    return ts, m - sp / 2, m + sp / 2

def test_long_enters_ask_exits_bid_tp():
    ts, b, a = ticks([1.1000, 1.1005, 1.1030, 1.1000])
    o = simulate(ts, b, a, np.array([1_000_000]), np.array([0.0020]), 1.0, 10**7, lat_ms=0, floor_pips=0, cost_pips=0)
    # intrare la ask 1.10005, TP = 1.10205; primul bid >= TP e la tick 2 (bid 1.10295) -> limit la TP => +20 pips
    assert abs(o[0, 0] - 20.0) < 1e-6 and o[0, 1] == ts[2]

def test_short_symmetric_and_separate():
    ts, b, a = ticks([1.1000, 1.0995, 1.0970, 1.1000])
    o = simulate(ts, b, a, np.array([1_000_000]), np.array([0.0020]), 1.0, 10**7, lat_ms=0, floor_pips=0, cost_pips=0)
    assert abs(o[0, 2] - 20.0) < 1e-6          # short castiga 20
    assert o[0, 0] < 0                          # long pierde (SL atins la tick 2, executat la bid)
    assert abs(o[0, 0] - ((1.0970 - 0.00005) - 1.10005) / P) < 1e-6

def test_time_barrier_and_costs_and_floor():
    ts, b, a = ticks([1.1, 1.1001, 1.1002, 1.1003], sp=0.0)
    o = simulate(ts, b, a, np.array([1_000_000]), np.array([0.01]), 1.0, 2500, lat_ms=0, floor_pips=0.4, cost_pips=0.8)
    # spread real 0 -> floor 0.4 pip; iesire la timp la tick 2 (ts<=t+2500): long = 2 pips - 0.4 - 0.8
    assert abs(o[0, 0] - (2.0 - 0.4 - 0.8)) < 1e-6 and o[0, 1] == ts[2]

def test_latency_uses_later_tick():
    ts, b, a = ticks([1.1, 1.2, 1.2, 1.2])
    o = simulate(ts, b, a, np.array([1_000_000]), np.array([0.5]), 1.0, 1500, lat_ms=500, floor_pips=0, cost_pips=0)
    assert o[0, 4] == ts[1]                     # nu poate intra la pretul de la momentul semnalului

def _bars(n=3000, seed=0):
    rng = np.random.default_rng(seed); ts = np.arange(n * 10) * 6000 + 10**12
    mid = 1.1 + np.cumsum(rng.normal(0, 1e-5, len(ts)))
    d = pd.DataFrame({"ts": ts, "bid": mid - 5e-6, "ask": mid + 5e-6})
    return d, minute_bars(d)

def test_features_causal():
    d, b = _bars()
    f1 = features(b)
    b2 = b.copy(); b2.loc[2000:, ["o", "h", "l", "c"]] *= 1.05; b2.loc[2000:, "n"] *= 3   # modificam viitorul
    f2 = features(b2)
    cols = [c for c in f1.columns if c not in ("ts",)]
    pd.testing.assert_frame_equal(f1.loc[:1999, cols], f2.loc[:1999, cols])

def test_minute_bar_label_is_end():
    d, b = _bars(10)
    assert (b.ts_end % 60000 == 0).all() and b.ts_end.iloc[0] > d.ts.iloc[0]

def test_purge_and_embargo():
    ts = np.arange(10) * 100; ex = ts + 150
    tr = purge(np.arange(10), 500, 600, ts, ex)
    assert 4 not in tr and 5 not in tr and 9 not in tr  # 4 se suprapune (iese la 550), 5-9 in test/embargo
    assert set(tr) == {0, 1, 2, 3}   # 3 iese la 450 < 500: pastrat

def test_uniqueness():
    e = np.array([0, 0, 600000]); x = np.array([60000 * 5 - 1, 60000 * 5 - 1, 60000 * 15])
    u = uniqueness(e, x)
    assert abs(u[0] - 0.5) < 1e-9 and abs(u[2] - 1.0) < 1e-9

def test_cleaning_removes_spike_and_zero():
    mids = np.full(300, 1.1); mids[150] = 1.2
    d = pd.DataFrame({"ts": np.arange(300) * 1000, "bid": mids - 5e-6, "ask": mids + 5e-6})
    d.loc[10, ["bid", "ask"]] = 0
    c, st = clean_ticks(d)
    assert st["bad_price"] == 1 and st["spikes"] == 1 and len(c) == 298

def test_server_time_to_utc():
    from eurusd_ml.v2.clean_bars import server_to_utc_ms
    t = lambda s: pd.Timestamp(s).value // 10**6
    assert server_to_utc_ms(np.array([t("2024-01-15 10:00")]))[0] == t("2024-01-15 08:00")   # iarna UTC+2
    assert server_to_utc_ms(np.array([t("2024-07-15 10:00")]))[0] == t("2024-07-15 07:00")   # vara UTC+3
