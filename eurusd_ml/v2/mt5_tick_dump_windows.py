# Citire tick-uri EURUSD din contul demo MetaQuotes 5057749905. Doar citire, fara login, fara ordine.
import MetaTrader5 as mt5, numpy as np, datetime as dt, os, sys, time
P=r"C:\MT5Demo\terminal64.exe"; OUT=r"C:\xmdemo\ticks"
if not mt5.initialize(path=P, portable=True, timeout=60000): print("INIT FAIL", mt5.last_error()); sys.exit(2)
a=mt5.account_info()
if a.login!=5057749905 or "MetaQuotes" not in a.server: print("WRONG ACCOUNT"); mt5.shutdown(); sys.exit(4)
mt5.symbol_select("EURUSD",True)
U=dt.timezone.utc
end=dt.datetime.now(U); wk=dt.timedelta(days=7); empty=0
t=dt.datetime(end.year,end.month,end.day,tzinfo=U)+dt.timedelta(days=1)
while empty<12:
    s=t-wk; fn=os.path.join(OUT,f"EURUSD_{s:%Y%m%d}.npy")
    if os.path.exists(fn): t=s; empty=0; continue
    r=None
    for k in range(4):
        r=mt5.copy_ticks_range("EURUSD",s,t,mt5.COPY_TICKS_ALL)
        if r is not None and len(r)>0: break
        time.sleep(3)
    n=0 if r is None else len(r)
    print(f"{s:%F} {n} err={mt5.last_error()}",flush=True)
    if n: np.save(fn,r); empty=0
    else: empty+=1
    t=s
mt5.shutdown(); print("DONE")
