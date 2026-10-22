# EURUSD ML v2 – cercetare pe tick-uri

## Date (reale, numărate)
- **MT5 MetaQuotes-Demo 5057749905** (doar citire, `copy_ticks_range` COPY_TICKS_ALL pe săptămâni, script `eurusd_ml/v2/mt5_tick_dump_windows.py` rulat cu Python Windows sub Wine): **414.755.037 tick-uri brute**, 19.12.2011 – 22.10.2026. După curățare: **396.753.604** (eliminate: 256.064 prețuri invalide, 17.194.872 duplicate, 550.497 spike-uri). Raport lunar: `reports/v2/quality_mt5.csv`, `reports/v2/cleaning_mt5.csv`.
- **Ora serverului MT5 = ora New York + 7h** (UTC+2/+3). Convertit în UTC; verificat față de Dukascopy: diferența mediană a mid-ului 0,05 pips după conversie (15 pips fără). Fără această corecție toate caracteristicile orare ar fi greșite.
- Spread-ul MetaQuotes-Demo e adesea **0** (2021–2023 mediana 0,0) → nerealist; în backtest impunem spread minim 0,3 pips (cât arată Dukascopy) + comision 0,6 + slippage 0,2 pips dus-întors.
- Înainte de 2018 fluxul e eterogen (2014–2016: milioane de duplicate/spike-uri) → cercetarea folosește 2018-01 … 2026-10.
- **Dukascopy**: downloader politicos `eurusd_ml/v2/duka_download.py` (1 cerere odată, pauză, backoff până la 10 min pe 429/503). Serverul răspunde aproape numai cu 503/timeout din rețeaua noastră.

## Metodă
- Bare de 1 min din tick-uri (OHLC mid, nr. tick-uri, tick-uri sus/jos, varianță realizată, spread, OFI proxy); evenimente la 15 min (timp) sau la fiecare N tick-uri (bare de tick-uri). Bare de volum/dolar imposibile: MT5 nu dă volum real pe FX.
- Etichete triple-barrier **pe drumul real de tick-uri**, long la ASK/ieșire la BID, short invers, **separat** long și short, stop executat la piață, latență 250 ms.
- Sarcini: meta-labeling peste EMA și peste revenire la medie; `dir` cu două modele separate (repară defectul vechi).
- Ponderi de unicitate, purged K-fold + embargo 1 zi, walk-forward anual 2020–2025-10, registru 36 configurații, PBO (CSCV), Deflated Sharpe, holdout 22.10.2025–22.10.2026 evaluat O SINGURĂ DATĂ (`reports/v2/HOLDOUT_USED.json`).

## Rulare
```
python -m eurusd_ml.v2.ticks_store / clean_bars mt5 / dataset time15 1.0 240 / research / analyze wf / analyze holdout <id>
pytest tests/test_v2.py
```

## Rezultat
Toate cele 36 de configurații pierd în walk-forward (cea mai bună Sharpe −0,50). Holdout pentru cea aleasă (cfg27): −948 pips, Sharpe −2,56, mai slab decât selecția aleatoare. ML-ul nu aduce avantaj tranzacționabil. Detalii: `reports/v2/wf_summary.csv`, `wf_stats.json`, grafice `wf_equity_all.png`, `holdout_equity.png`.
