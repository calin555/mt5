"""Validare fără scurgeri de informație.

PurgedKFold: folduri contigue în timp; din train scoatem (purge) orice eveniment al cărui
interval [intrare, t1] se suprapune cu intervalul de test, plus un embargo după test.
walk_forward_splits: fereastră de antrenare rulantă (N ani) -> test pe următoarele M luni.
"""
import numpy as np
import pandas as pd


def purged_kfold(events: pd.DataFrame, n_splits: int, embargo: pd.Timedelta):
    idx = np.arange(len(events))
    start, t1 = events["entry_time"].to_numpy(), events["t1"].to_numpy()
    for test in np.array_split(idx, n_splits):
        t0, tE = start[test[0]], t1[test].max() + embargo
        train = idx[(t1 < t0) | (start > tE)]
        yield train, test


def walk_forward_splits(events: pd.DataFrame, train_years: int, test_months: int, embargo: pd.Timedelta):
    first, last = events.index.min(), events.index.max()
    test_start = first + pd.DateOffset(years=train_years)
    while test_start < last:
        test_end = test_start + pd.DateOffset(months=test_months)
        tr_from = test_start - pd.DateOffset(years=train_years)
        test = np.where((events.index >= test_start) & (events.index < test_end))[0]
        # purge: trade-urile din train trebuie să se fi închis înainte de test (minus embargo)
        train = np.where((events.index >= tr_from) & (events["t1"] < test_start - embargo))[0]
        if len(test) and len(train) > 100:
            yield test_start, train, test
        test_start = test_end
