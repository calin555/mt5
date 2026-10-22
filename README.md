# eurusd_ml: cercetare ML pe EURUSD (H1), de la date la backtest

Proiect **doar de cercetare și paper trading**. Nu trimite niciun ordin.

## Concluzie (rezultate reale, 22 oct 2026)
**Machine learning-ul nu aduce niciun avantaj pe EURUSD H1 cu aceste date și features.**
- AUC în validarea purjată: 0,49–0,52, adică la nivelul întâmplării.
- Toate cele 8 configurații pierd după costuri în walk-forward (2015 – oct 2025).
- Pe holdout (oct 2025 – oct 2026, evaluat o singură dată) modelele ales pierd și ele: AUC 0,42 / 0,49.
- Filtrul ML pierde mai puțin decât semnalul primar, dar doar pentru că face mai puține tranzacții.
  Pierderea pe tranzacție nu e mai mică decât la o selecție aleatoare din aceleași semnale.

Tabelele complete sunt în `reports/registry_walkforward.csv` și `reports/registry_holdout.csv`.
Graficele sunt în `reports/equity_wf_meta.png`, `reports/equity_wf_dir.png` și `reports/equity_holdout.png`.

## Date
- **Dukascopy (sursa gratuită de tick-uri):** downloader-ul propriu e în `eurusd_ml/data.py`.
  - Descarcă în paralel, cu reîncercări și backoff, cache local și reluare.
  - Agregă în bare de 1m, 15m și 1h, cu OHLC bid/ask, spread mediu/maxim, număr de tick-uri și volum.
  - URL: `https://datafeed.dukascopy.com/datafeed/EURUSD/AAAA/LL/ZZ/HHh_ticks.bi5`. Luna e indexată de la 0.
  - Fișierul e comprimat LZMA, cu înregistrări big-endian `ms, ask, bid, askvol, bidvol`. Prețul = valoare / 1e5.
  - **Limitare reală:** pe 22 oct 2026 serverul a răspuns aproape numai cu 429/503 de pe acest computer.
    Am obținut circa 3 fișiere pe minut, deci 2018–2026 (~75.000 de ore) ar fi durat săptămâni.
    Am descărcat doar ~1 zi (21 oct 2026, 56.020 tick-uri). Din ea reiese un spread real de 0,30 pips median și 0,5 pips la p95.
  - Alternative: `pip install duka` (`duka EURUSD -s 2018-01-01 -e 2026-10-21`) sau
    `npx dukascopy-node -i eurusd -from 2018-01-01 -to 2026-10-21 -t tick -f csv`.
    Lovesc același server, deci au aceeași limitare.
- **Sursa folosită pentru research:** barele H1 EURUSD din contul **demo** MetaQuotes,
  exportate read-only din terminalul portabil `C:\MT5Demo`. Folosesc perioada 2012-01-02 → 2026-10-22, adică ~91.000 de bare.
  Câmpul `spread` din MT5 e adesea 0 (e minimul barei), așa că în costuri folosesc `max(spread, 1,0 pip)`.
  E o alegere conservatoare față de cele 0,3 pips măsurate la Dukascopy.

## Structură și rulare
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m eurusd_ml.cli download   # tick-uri Dukascopy -> data/raw (cache, reluare)
python -m eurusd_ml.cli build      # tick-uri -> data/bars/*.parquet (1m, 15m, 1h)
python -m eurusd_ml.cli train      # features, etichete, CV purjat, walk-forward, registru, grafice
python -m eurusd_ml.cli holdout    # holdout-ul, O SINGURĂ DATĂ (blocat după prima rulare)
python -m eurusd_ml.cli paper      # stub paper trading pe demo, ordinele sunt dezactivate
pytest tests
```
Setările sunt în `configs/default.yaml`. Cu `source: dukascopy` se folosesc barele construite din tick-uri.

| Fișier | Ce face |
|---|---|
| `data.py` | Descărcarea și decodarea .bi5, agregarea în bare, încărcarea exportului MT5. |
| `features.py` | Randamente (4–168 bare), volatilitate realizată, z-score, distanțe EMA/ATR, RSI, MACD normalizat, poziția în canalul Donchian, volum de tick-uri, spread/ATR, ora și ziua. Toate sunt **cauzale**: valoarea de la bara t folosește doar date până la închiderea lui t. |
| `labeling.py` | Triple-barrier: barierele sunt ±1,5×ATR(24), cu o barieră verticală la 24 de bare. Intrarea e la deschiderea barei următoare. Dacă ambele bariere sunt atinse în aceeași bară, se consideră stop. Eticheta = câștig după costuri. Tot aici e semnalul primar EMA 24/96 pentru meta-labeling. |
| `models.py` | Regresie logistică (baseline), Random Forest, LightGBM, XGBoost, toate puternic regularizate. Fără deep learning. |
| `validation.py` | K-fold purjat cu embargo de 24 de bare și walk-forward (3 ani de antrenare → 6 luni de test). |
| `backtest.py` | Costuri: spread la intrare și ieșire, slippage 0,2 pips pe fiecare parte, comision 7 USD/lot. Câte o poziție odată. Metrici: Sharpe, Sortino, DD maxim, win rate, PF, expunere. Baseline aleator din 1000 de extrageri. |
| `pipeline.py` | Orchestrarea, registrul de configurații și selecția **pre-declarată** pentru holdout (Sharpe WF maxim). |

**Cele două probleme ML** (4 modele × 2 = 8 configurații, toate înregistrate, fără tuning suplimentar):
1. **META:** modelul decide dacă se ia semnalul primar EMA(24/96) la crossover.
2. **DIR:** predicția direcției la fiecare 6 bare. Long dacă p > 0,55, short dacă p < 0,45, altfel fără poziție.

## Rezultate cheie (USD pe 1 lot, după costuri)
| Etapă | Strategie | Tranz. | Pips net | PF | Sharpe |
|---|---|---|---|---|---|
| WF 2015–2025 | Primar EMA fără ML | 814 | −979 | 0,90 | −0,42 |
| WF | META RF (cel mai bun) | 529 | −505 | 0,92 | −0,27 |
| WF | DIR RF (cel mai bun) | 1981 | −3449 | 0,85 | −0,97 |
| Holdout | Primar EMA | 67 | −160 | 0,76 | −0,99 |
| Holdout | META RF | 36 | −90 | 0,77 | −0,72 |
| Holdout | DIR RF | 45 | −100 | 0,72 | −0,95 |
