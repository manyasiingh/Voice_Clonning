"""Reproducibility helpers: seeding, library versions, git hash, run metadata, config files, JSON I/O."""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed):
    """Seed Python, NumPy and torch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """DataLoader worker seeding. torch gives each worker initial_seed = base_seed + worker_id, where
    base_seed comes from the DataLoader's generator; we propagate it to `random` and NumPy."""
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "unknown")
    except Exception:  # not installed / broken import
        return None


def library_versions():
    v = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for mod in ("torchaudio", "speechbrain", "numpy", "soundfile", "matplotlib", "yaml"):
        v[mod] = _version(mod)
    return v


def git_info():
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_json(path, obj):
    """Atomic JSON write (write to a temp file, then rename) so an interrupted run never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=2))
    os.replace(tmp, path)


def load_json(path):
    return json.loads(Path(path).read_text())


def write_run_meta(out_dir, config, name="run_meta.json"):
    """Print library versions and save config + versions + git hash next to the results."""
    meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv, "config": config,
            "versions": library_versions(), "git": git_info()}
    print("versions:", json.dumps(meta["versions"]))
    print("git:", json.dumps(meta["git"]))
    save_json(Path(out_dir) / name, meta)
    return meta


def parse_with_config(parser, argv=None):
    """Parse args where `--config file.yaml` provides defaults and explicit CLI flags override them."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    if known.config:
        import yaml

        cfg = yaml.safe_load(Path(known.config).read_text()) or {}
        dests = {a.dest for a in parser._actions}
        unknown = set(cfg) - dests
        if unknown:
            raise SystemExit(f"unknown keys in {known.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)
    return parser.parse_args(argv)


if __name__ == "__main__":  # python -m spkclone.utils [out_dir]  -> print + save environment info
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/env")
    v = {"versions": library_versions(), "git": git_info()}
    print(json.dumps(v, indent=2))
    save_json(out / "versions.json", v)
