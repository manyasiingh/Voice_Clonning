#!/usr/bin/env python3
"""Build the RQ1 table from results/rq1/<model>_s<seed>/test_metrics.json -> results/rq1/table.{csv,md}.

Runs that differ only in seed are grouped; we report mean +- std over seeds (std = 0 / '-' for a single run)
and, per run, the speaker-bootstrap 95% CI of EER. Only files that exist are used; nothing is filled in.
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(root="results/rq1"):
    root = Path(root)
    groups = defaultdict(list)
    for f in sorted(root.glob("*/test_metrics.json")):
        name = re.sub(r"_s\d+$", "", f.parent.name)
        groups[name].append((f.parent.name, json.loads(f.read_text())))
    if not groups:
        sys.exit(f"no results found under {root}")
    rows = []
    for name, runs in sorted(groups.items()):
        eer = np.array([r["eer"] for _, r in runs]) * 100
        dcf = np.array([r["min_dcf"] for _, r in runs])
        cis = [r.get("ci_speakers", {}).get("eer") for _, r in runs]
        ci_txt = "; ".join(f"{c['lo'] * 100:.2f}-{c['hi'] * 100:.2f}" for c in cis if c)
        sd = lambda v, d: f"{v.std(ddof=1):.{d}f}" if len(v) > 1 else "-"
        rows.append({"model": name, "n_seeds": len(runs), "EER_mean_%": f"{eer.mean():.2f}", "EER_std_%": sd(eer, 2),
                     "minDCF_mean": f"{dcf.mean():.3f}", "minDCF_std": sd(dcf, 3),
                     "EER_speaker_CI95_%_per_run": ci_txt, "n_test_speakers": runs[0][1]["n_speakers"],
                     "n_trials": runs[0][1]["n_trials"], "runs": ",".join(r for r, _ in runs)})
    with open(root / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (root / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:])
