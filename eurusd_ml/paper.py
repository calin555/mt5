"""Paper trading (STUB) pe contul DEMO MetaQuotes 5057749905, read-only.

- Citește ultimele bare H1 din terminalul portabil C:\\MT5Demo (sub Wine, cu Python-ul
  Windows care are pachetul MetaTrader5), calculează semnalul și îl scrie în log.
- ORDINELE SUNT DEZACTIVATE: acest modul nu apelează niciodată order_send.
- Refuză orice alt cont decât demo-ul MetaQuotes.
Rulare (din Wine): python -m eurusd_ml.cli paper
"""
ALLOWED_LOGIN = 5057749905
ORDERS_ENABLED = False  # nu schimba: proiectul e doar research


def run_paper(cfg):
    try:
        import MetaTrader5 as mt5  # disponibil doar în Python-ul Windows (Wine)
    except ImportError:
        print("MetaTrader5 nu e instalat aici; rulează din Python-ul din Wine. Nimic de făcut.")
        return
    if not mt5.initialize(path=r"C:\MT5Demo\terminal64.exe", portable=True):
        print("init eșuat", mt5.last_error()); return
    acc = mt5.account_info()
    if acc is None or acc.login != ALLOWED_LOGIN or "MetaQuotes" not in acc.server:
        print("Cont nepermis, opresc."); mt5.shutdown(); return
    import pandas as pd
    r = pd.DataFrame(mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, 0, 500))
    mt5.shutdown()
    c = r["close"]
    sig = 1 if c.ewm(span=cfg["primary"]["fast"]).mean().iloc[-2] > c.ewm(span=cfg["primary"]["slow"]).mean().iloc[-2] else -1
    print(f"[paper] semnal primar pe ultima bară închisă: {sig:+d} (ordine dezactivate: {not ORDERS_ENABLED})")
