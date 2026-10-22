"""Punct de intrare: python -m eurusd_ml.cli {download,build,train,backtest,all}"""
import argparse
from .config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "build", "train", "holdout", "paper"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.cmd == "download":
        from .data import download; download(cfg)
    elif a.cmd == "build":
        from .data import build_bars; build_bars(cfg)
    elif a.cmd == "train":
        from .pipeline import run_research; run_research(cfg)
    elif a.cmd == "holdout":
        from .pipeline import run_holdout; run_holdout(cfg)
    elif a.cmd == "paper":
        from .paper import run_paper; run_paper(cfg)


if __name__ == "__main__":
    main()
