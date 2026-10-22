"""Teste: decodare bi5, agregare, cauzalitatea feature-urilor, triple-barrier, purging."""
import lzma
from datetime import datetime
import numpy as np
import pandas as pd
from eurusd_ml.data import REC, decode_hour, ticks_to_bars
from eurusd_ml.features import build_features
from eurusd_ml.labeling import triple_barrier
from eurusd_ml.validation import purged_kfold

COST = {"pip": 1e-4, "slippage_pips": 0.0, "commission_per_lot_rt": 0.0}


def synth(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    c = 1.1 + np.cumsum(rng.normal(0, 5e-4, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 2e-4, "low": np.minimum(o, c) - 2e-4,
                         "close": c, "volume": rng.integers(100, 1000, n).astype(float),
                         "spread": 1e-5, "cost_spread": 1e-4}, index=idx)


def test_decode_bi5(tmp_path):
    rec = np.array([(1000, 112345, 112340, 1.5, 2.0), (2500, 112350, 112344, 1.0, 1.0)], dtype=REC)
    p = tmp_path / "x.bi5"; p.write_bytes(lzma.compress(rec.tobytes(), format=lzma.FORMAT_ALONE))
    d = decode_hour(p, datetime(2024, 1, 3, 10), 1e-5)
    assert d["time"].iloc[0] == pd.Timestamp("2024-01-03 10:00:01")
    assert abs(d["ask"].iloc[0] - 1.12345) < 1e-12 and abs(d["bid"].iloc[1] - 1.12344) < 1e-12
    b = ticks_to_bars(d, "1h")
    assert b["ticks"].iloc[0] == 2 and abs(b["spread_max"].iloc[0] - 6e-5) < 1e-12


def test_features_no_lookahead():
    df = synth()
    full = build_features(df)
    cut = 1000
    part = build_features(df.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, atol=1e-9)


def test_future_shock_does_not_change_past():
    df = synth(); f1 = build_features(df)
    df2 = df.copy(); df2.iloc[1200:, df2.columns.get_loc("close")] *= 1.05
    f2 = build_features(df2)
    pd.testing.assert_frame_equal(f1.iloc[:1200], f2.iloc[:1200])


def test_triple_barrier_entry_after_event_and_t1():
    df = synth()
    ev = df.index[[300, 600]]
    side = pd.Series(1.0, index=df.index)
    tb = triple_barrier(df, ev, side, 24, 1.0, 1.0, 10, COST)
    assert (tb["entry_time"] > tb.index).all()
    assert (tb["t1"] >= tb["entry_time"]).all()
    assert ((tb["t1"] - tb["entry_time"]) <= pd.Timedelta(hours=10)).all()
    assert np.allclose(tb["entry"], df["open"].shift(-1).loc[ev])


def test_purged_kfold_no_overlap():
    df = synth()
    tb = triple_barrier(df, df.index[100:1400:5], pd.Series(1.0, index=df.index), 24, 1, 1, 20, COST)
    for tr, te in purged_kfold(tb, 5, pd.Timedelta(hours=24)):
        t0, t1 = tb["entry_time"].iloc[te].min(), tb["t1"].iloc[te].max()
        trn = tb.iloc[tr]
        assert not ((trn["t1"] >= t0) & (trn["entry_time"] <= t1)).any()
