#!/usr/bin/env python3
"""Thin wrapper that keeps the original CLI working:

  python scripts/speaker_verification_contrastive.py train   --out runs/spk [...same flags as before...]
  python scripts/speaker_verification_contrastive.py verify  --ckpt runs/spk/best.pt --a x.wav --b y.wav
  python scripts/speaker_verification_contrastive.py explain --ckpt pretrained --a x.wav --b y.wav --png why.png
  python scripts/speaker_verification_contrastive.py evaluate --ckpt pretrained --out results/rq1/pretrained   (new)

Change vs. the original: `train` now refuses to run when train/dev/test share speakers. The old quick test
(`--train_subset dev-clean`) needs `--allow_overlap`, because it was evaluating on its own training speakers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spkclone import explain, train, verify  # noqa: E402

CMDS = {"train": train.main, "verify": verify.verify_cli, "explain": explain.main, "evaluate": verify.evaluate_cli}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        sys.exit(__doc__)
    CMDS[sys.argv[1]](sys.argv[2:])
