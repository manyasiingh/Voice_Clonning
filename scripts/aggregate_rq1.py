#!/usr/bin/env python3
"""Build the RQ1 tables from results/rq1/ -> results/rq1/table.{csv,md} (and table_final_epoch.* if present).

- Main table: <run>/test_metrics.json = the DEV-SELECTED checkpoint. The `selected_epoch` column says which epoch
  was chosen (0 = the initial pretrained model: training never beat it on dev, so the row equals "pretrained").
- Final-epoch table: <run>/final_epoch/test_metrics.json = weights after the last epoch, not selected on dev.

Runs that differ only in seed (suffix _s<seed>) are grouped: mean +- std over seeds (std '-' for a single run),
and per run the speaker-bootstrap 95% CI of EER. Only files that exist are used; nothing is filled in.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def selected_epoch(run_dir, metrics):
    """From test_metrics.json if recorded, else re-derived from train_history.json with the same rule as train.py
    (lowest dev EER; epoch 0 is a candidate only when fine-tuning). None for runs without training (pretrained)."""
    if "selected_epoch" in metrics:
        return metrics["selected_epoch"]
    hist_file = run_dir / "train_history.json"
    if not hist_file.exists():
        return None
    hist = json.loads(hist_file.read_text())["history"]
    scratch = json.loads((run_dir / "run_meta.json").read_text())["config"].get("scratch", False)
    cand = [h for h in hist if h["epoch"] > 0] if scratch else hist
    return min(cand, key=lambda h: h["eer"])["epoch"]  # min() keeps the first of equal values, like train.py


def dev_curve(run_dir):
    f = run_dir / "train_history.json"
    if not f.exists():
        return None, None
    h = json.loads(f.read_text())["history"]
    return h[0]["eer"], h[-1]["eer"]


def build(runs, metrics_name):
    groups = defaultdict(list)
    for run_dir in runs:
        f = run_dir / metrics_name
        if f.exists():
            groups[re.sub(r"_s\d+$", "", run_dir.name)].append((run_dir, json.loads(f.read_text())))
    rows = []
    for name, items in sorted(groups.items()):
        eer = np.array([m["eer"] for _, m in items]) * 100
        dcf = np.array([m["min_dcf"] for _, m in items])
        sd = lambda v, d: f"{v.std(ddof=1):.{d}f}" if len(v) > 1 else "-"
        cis = [m.get("ci_speakers", {}).get("eer") for _, m in items]
        sel = [selected_epoch(r, m) for r, m in items]
        dev0 = [dev_curve(r) for r, _ in items]
        rows.append({
            "model": name, "n_seeds": len(items),
            "EER_mean_%": f"{eer.mean():.2f}", "EER_std_%": sd(eer, 2),
            "minDCF_mean": f"{dcf.mean():.3f}", "minDCF_std": sd(dcf, 3),
            "EER_speaker_CI95_%_per_run": "; ".join(f"{c['lo'] * 100:.2f}-{c['hi'] * 100:.2f}" for c in cis if c),
            "selected_epoch_per_run": ",".join("-" if s is None else str(s) for s in sel),
            "dev_EER_%_start->end_per_run": "; ".join("-" if a is None else f"{a * 100:.2f}->{b * 100:.2f}"
                                                      for a, b in dev0),
            "n_test_speakers": items[0][1]["n_speakers"], "n_trials": items[0][1]["n_trials"],
            "runs": ",".join(r.name for r, _ in items)})
    return rows


def write(rows, root, stem, note):
    with open(root / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    md = [note, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (root / f"{stem}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md) + "\n")


def main(root="results/rq1"):
    root = Path(root)
    runs = sorted(p for p in root.iterdir() if p.is_dir())
    main_rows = build(runs, "test_metrics.json")
    if not main_rows:
        sys.exit(f"no results found under {root}")
    write(main_rows, root, "table", "Dev-selected checkpoints (selected_epoch 0 = initial model kept).")
    final_rows = build(runs, "final_epoch/test_metrics.json")
    if final_rows:
        write(final_rows, root, "table_final_epoch", "Weights after the last epoch (NOT selected on dev).")


if __name__ == "__main__":
    main(*sys.argv[1:])
