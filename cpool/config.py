"""Config, paths, manifests."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path=None) -> dict:
    import os
    path = path or os.environ.get("CS_CONFIG")
    p = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_path"] = str(p)
    return cfg


def chash(cfg) -> str:
    c = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def path(rel) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def stage(cfg, name) -> Path:
    import os
    if name == "cache" and os.environ.get("CS_CACHE"):
        d = Path(os.environ["CS_CACHE"])      # e.g. Colab local disk (SQLite dislikes Drive)
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = path(cfg["out_dir"]) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(obj, p):
    import numpy as np

    def d(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=d), encoding="utf-8")


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def manifest(cfg, name, extra=None, fname="manifest.json"):
    vers = {}
    for p in ["numpy", "pandas", "numba", "kiwipiepy", "g2pk2", "python-mecab-ko", "ortools",
              "statsmodels", "kneed", "torch", "openai-whisper", "transformers", "edge-tts", "gTTS", "funasr", "modelscope"]:
        try:
            vers[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    save({"stage": name, "config_hash": chash(cfg), "seed": cfg["seed"],
          "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "python": sys.version.split()[0],
          "platform": platform.platform(), "packages": vers, **(extra or {})},
         stage(cfg, name) / fname)
