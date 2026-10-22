"""Orchestrare: date -> features -> etichete -> CV purjat -> walk-forward -> (o dată) holdout.

Două probleme ML (configurații limitate, toate înregistrate în reports/registry.csv):
  META: meta-labeling peste semnalul primar EMA(24/96); modelul decide DOAR dacă luăm trade-ul.
  DIR : predicția direcției pe evenimente la fiecare 6 bare (triple-barrier pe long);
        long dacă p > 0.55, short dacă p < 0.45, altfel stăm deoparte.
Holdout-ul (ultimele 12 luni) NU e atins de `train`; se evaluează o singură dată cu `holdout`.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from .config import ROOT
from .data import load_research_bars
from .features import build_features
from .labeling import primary_signal, crossover_events, triple_barrier
from .models import make_model
from .validation import purged_kfold, walk_forward_splits
from .backtest import non_overlapping, metrics, random_baseline, plot_equity

REP = ROOT / "reports"


def prepare(cfg):
    df = load_research_bars(cfg)
    F = build_features(df, cfg["features"]["windows"])
    L, P = cfg["labeling"], cfg["primary"]
    side = primary_signal(df, P["fast"], P["slow"])
    tb = lambda ev, s: triple_barrier(df, ev, s, L["atr_window"], L["pt_mult"], L["sl_mult"], L["max_hold"], cfg["costs"])
    meta = tb(crossover_events(side), side)
    meta_X = F.loc[meta.index].assign(side=meta["side"])
    step_ev = df.index[200::cfg["direction_step"]]
    longs = tb(step_ev, pd.Series(1.0, index=df.index))
    shorts = tb(step_ev, pd.Series(-1.0, index=df.index))
    common = longs.index.intersection(shorts.index)
    longs, shorts = longs.loc[common], shorts.loc[common]
    dir_X = F.loc[common]
    holdout_start = df.index.max() - pd.DateOffset(months=cfg["validation"]["holdout_months"])
    return dict(df=df, F=F, meta=meta, meta_X=meta_X, longs=longs, shorts=shorts, dir_X=dir_X,
                holdout_start=holdout_start)


def split_research(ev: pd.DataFrame, hs):
    """Research = evenimente închise complet înainte de începutul holdout-ului."""
    return ev["t1"] < hs, ev.index >= hs


def dir_trades(p: pd.Series, longs, shorts, band):
    lo = longs.loc[p.index[p > 0.5 + band]]
    sh = shorts.loc[p.index[p < 0.5 - band]]
    return pd.concat([lo, sh]).sort_index()


def fit_predict(name, cfg, Xtr, ytr, Xte):
    m = make_model(name, cfg["seed"])
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run_research(cfg):
    REP.mkdir(exist_ok=True)
    d = prepare(cfg)
    hs = d["holdout_start"]
    V = cfg["validation"]
    emb = pd.Timedelta(hours=V["embargo_bars"])
    reg, wf_store = [], {}
    tasks = {
        "META": (d["meta"], d["meta_X"], d["meta"]["label"]),
        "DIR": (d["longs"], d["dir_X"], d["longs"]["label"]),
    }
    print(f"[info] bare {d['df'].index[0]} -> {d['df'].index[-1]}, holdout de la {hs.date()}")
    print(f"[info] evenimente META={len(d['meta'])}, DIR={len(d['longs'])}")
    for task, (ev, X, y) in tasks.items():
        r_mask, _ = split_research(ev, hs)
        evr, Xr, yr = ev[r_mask], X[r_mask], y[r_mask]
        for name in cfg["models"]:
            # 1) CV purjat cu embargo (calitatea predicției)
            aucs, lls = [], []
            for tr, te in purged_kfold(evr, V["n_splits"], emb):
                p = fit_predict(name, cfg, Xr.iloc[tr], yr.iloc[tr], Xr.iloc[te])
                aucs.append(roc_auc_score(yr.iloc[te], p)); lls.append(log_loss(yr.iloc[te], p, labels=[0, 1]))
            # 2) walk-forward: predicții OOS lipite cap la cap
            oos = []
            for t0, tr, te in walk_forward_splits(evr, V["wf_train_years"], V["wf_test_months"], emb):
                oos.append(pd.Series(fit_predict(name, cfg, Xr.iloc[tr], yr.iloc[tr], Xr.iloc[te]), index=Xr.index[te]))
            p = pd.concat(oos)
            wf_store[(task, name)] = p
            if task == "META":
                cand = d["meta"].loc[p.index]
                tr_ = cand[p >= cfg["meta_threshold"]]
            else:
                cand = pd.concat([d["longs"].loc[p.index], d["shorts"].loc[p.index]])
                tr_ = dir_trades(p, d["longs"], d["shorts"], cfg["direction_band"])
            tr_ = non_overlapping(tr_)
            bars = d["df"].index[(d["df"].index >= p.index.min()) & (d["df"].index < hs)]
            m = metrics(tr_, bars)
            m.update(random_baseline(non_overlapping(cand) if task == "META" else cand, m["trades"], cfg["n_random"], cfg["seed"]))
            row = dict(stage="walkforward", task=task, model=name, cv_auc=round(np.mean(aucs), 4),
                       cv_auc_std=round(np.std(aucs), 4), cv_logloss=round(np.mean(lls), 4),
                       base_rate=round(float(yr.mean()), 3), oos_from=str(p.index.min().date()), **m)
            print(row, flush=True)
            reg.append(row)
    # baseline-uri fără ML pe aceeași perioadă WF
    p0 = wf_store[("META", cfg["models"][0])]
    bars = d["df"].index[(d["df"].index >= p0.index.min()) & (d["df"].index < hs)]
    prim = non_overlapping(d["meta"].loc[p0.index])
    reg.append(dict(stage="walkforward", task="META", model="primary_no_ml", **metrics(prim, bars)))
    pd_ = wf_store[("DIR", cfg["models"][0])]
    alll = non_overlapping(d["longs"].loc[pd_.index])
    reg.append(dict(stage="walkforward", task="DIR", model="always_long", **metrics(alll, bars)))
    out = pd.DataFrame(reg)
    out.to_csv(REP / "registry_walkforward.csv", index=False)
    # grafice equity OOS
    curves = {"Primar EMA (fără ML)": prim}
    for name in cfg["models"]:
        p = wf_store[("META", name)]
        curves[f"META {name}"] = non_overlapping(d["meta"].loc[p.index][p >= cfg["meta_threshold"]])
    plot_equity(curves, str(REP / "equity_wf_meta.png"), "Walk-forward OOS - meta-labeling (net, după costuri)")
    curves = {"Mereu long (fără ML)": alll}
    for name in cfg["models"]:
        p = wf_store[("DIR", name)]
        curves[f"DIR {name}"] = non_overlapping(dir_trades(p, d["longs"], d["shorts"], cfg["direction_band"]))
    plot_equity(curves, str(REP / "equity_wf_dir.png"), "Walk-forward OOS - predicție direcție (net, după costuri)")
    # selecție PRE-DECLARATĂ pentru holdout: modelul cu cel mai mare Sharpe WF per task
    best = {t: out[(out.task == t) & out.model.isin(cfg["models"])].sort_values("sharpe").iloc[-1]["model"] for t in ("META", "DIR")}
    json.dump(best, open(REP / "selected_models.json", "w"), indent=1)
    print("[select]", best)
    return out


def run_holdout(cfg, force: bool = False):
    flag = REP / "holdout_DONE.json"
    if flag.exists() and not force:
        print("Holdout-ul a fost deja evaluat (o singură dată). Rezultate:", flag.read_text())
        return
    best = json.load(open(REP / "selected_models.json"))
    d = prepare(cfg)
    hs = d["holdout_start"]
    V = cfg["validation"]
    emb = pd.Timedelta(hours=V["embargo_bars"])
    bars = d["df"].index[d["df"].index >= hs]
    res, curves_m, curves_d = [], {}, {}
    # META
    ev, X = d["meta"], d["meta_X"]
    r, h = split_research(ev, hs)
    tr = r & (ev["t1"] < hs - emb)
    p = pd.Series(fit_predict(best["META"], cfg, X[tr], ev["label"][tr], X[h]), index=X.index[h])
    prim = non_overlapping(ev[h])
    sel = non_overlapping(ev[h][p >= cfg["meta_threshold"]])
    m = metrics(sel, bars); m.update(random_baseline(prim, m["trades"], cfg["n_random"], cfg["seed"]))
    res += [dict(stage="holdout", task="META", model=best["META"], auc=round(roc_auc_score(ev["label"][h], p), 4), **m),
            dict(stage="holdout", task="META", model="primary_no_ml", **metrics(prim, bars))]
    curves_m = {"Primar EMA (fără ML)": prim, f"META {best['META']}": sel}
    # DIR
    L, S, X = d["longs"], d["shorts"], d["dir_X"]
    r, h = split_research(L, hs)
    tr = r & (L["t1"] < hs - emb)
    p = pd.Series(fit_predict(best["DIR"], cfg, X[tr], L["label"][tr], X[h]), index=X.index[h])
    sel = non_overlapping(dir_trades(p, L, S, cfg["direction_band"]))
    m = metrics(sel, bars); m.update(random_baseline(pd.concat([L[h], S[h]]), m["trades"], cfg["n_random"], cfg["seed"]))
    al = non_overlapping(L[h])
    res += [dict(stage="holdout", task="DIR", model=best["DIR"], auc=round(roc_auc_score(L["label"][h], p), 4), **m),
            dict(stage="holdout", task="DIR", model="always_long", **metrics(al, bars))]
    curves_d = {"Mereu long (fără ML)": al, f"DIR {best['DIR']}": sel}
    plot_equity({**curves_m, **curves_d}, str(REP / "equity_holdout.png"),
                f"HOLDOUT {hs.date()} -> {bars.max().date()} (evaluat o singură dată)")
    out = pd.DataFrame(res)
    out.to_csv(REP / "registry_holdout.csv", index=False)
    print(out.to_string())
    flag.write_text(json.dumps({"holdout_start": str(hs), "selected": best}))
