"""Încărcarea configurației YAML."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(p) as f:
        cfg = yaml.safe_load(f)
    for k in ("raw_dir", "bars_dir"):
        cfg["data"][k] = str(ROOT / cfg["data"][k])
    return cfg
