"""Writes notebooks/kaggle_run.ipynb (run once with: python make_notebook.py <repo_dir>)."""
import json
import sys
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md("""
# Voice clones vs. speaker verification — Kaggle runner

Reproduces every result in `results/` from a clean Kaggle session.

**Session settings (right panel):** Accelerator = **GPU** (note which one you got: it is recorded in
`results/env/versions.json`), Internet = **On** (needs a phone-verified account).

Every long step is **resumable**: if the session dies, re-run the notebook from the top. Finished runs are
skipped (they already have `test_metrics.json`) and training continues from `last.pt`.
At the end, download `/kaggle/working/results.zip` and commit its contents to the repo.
""")

code("""
# ---- edit this one line: your GitHub repository ----
REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/speechbrain-voice-clone-vs-verification.git"

# LibriSpeech goes outside the saved working dir so it does not count against the 20 GB output.
# Check free space in the next cells; if /tmp is too small use "/kaggle/working/data" and delete it afterwards.
DATA_ROOT = "/tmp/data"
SEEDS = [0, 1, 2]
""")

code("""
import os
os.chdir("/kaggle/working")
if not os.path.exists("repo"):
    !git clone -q {REPO_URL} repo
else:
    !git -C repo pull -q
%cd /kaggle/working/repo
!git log -1 --oneline
""")

code("""
# Kaggle already has torch/torchaudio with CUDA; only add what is missing.
!pip install -q -r requirements.txt
!pip install -q -e . --no-deps
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!df -h /tmp /kaggle/working
!python -m spkclone.utils results/env
!pip freeze > results/env/pip_freeze.txt
""")

md("## 1. Data (dev-clean + test-clean, ~0.7 GB) and tests")

code("""
!python -c "from spkclone.data import ensure_librispeech as e; [e(s, '{DATA_ROOT}') for s in ('dev-clean', 'test-clean')]"
""")

code("""
# Unit tests + the encoder-equivalence test (SpkEncoder vs SpeechBrain EncoderClassifier) on real dev-clean speech.
os.environ["SPKCLONE_TEST_WAVS"] = f"{DATA_ROOT}/LibriSpeech/dev-clean"
!python -m pytest -s -q 2>&1 | tee results/env/pytest.txt
!cat results/env/encoder_equivalence.json
""")

md("""
## 2. Smoke tests (minutes) — original CLIs still work

Numbers from these cells are **meaningless** (training speakers overlap with evaluation speakers); they only prove
the pipeline runs end to end. Outputs go to `/tmp/smoke*`, not to `results/`.
""")

code("""
!python scripts/speaker_verification_contrastive.py train --config configs/smoke.yaml --root {DATA_ROOT} --out /tmp/smoke_spk
""")

code("""
# pick a 3-15 s dev-clean utterance as a reference voice (consented, public dataset speaker)
import glob, soundfile as sf
ref = next(f for f in sorted(glob.glob(f"{DATA_ROOT}/LibriSpeech/dev-clean/*/*/*.flac")) if 3 <= sf.info(f).duration <= 15)
print(ref, sf.info(ref).duration)
!python scripts/speechbrain_tts1.py single --text "Doctor Smith bought 3 apples." --plot --out /tmp/smoke_tts
!python scripts/speechbrain_tts1.py random --text "It was a sunny day." --plot --out /tmp/smoke_tts
!python scripts/speechbrain_tts1.py clone --ref {ref} --text "This is a short test sentence." --verify --plot --out /tmp/smoke_tts
!python scripts/speaker_verification_contrastive.py verify --ckpt pretrained --a {ref} --b /tmp/smoke_tts/clone_000.wav
!python scripts/speaker_verification_contrastive.py explain --ckpt pretrained --a {ref} --b /tmp/smoke_tts/clone_000.wav --png /tmp/smoke_tts/why.png
""")

md("""
## 3. RQ1 — verifier quality

Same fixed trials (eval_seed 0) for every model. Thresholds are calibrated on dev-clean, reported on test-clean.
""")

code("""
# pretrained VoxCeleb ECAPA-TDNN, no fine-tuning
!python -m spkclone.verify evaluate --ckpt pretrained --root {DATA_ROOT} --out results/rq1/pretrained
""")

code("""
# LibriSpeech train-clean-100 (~6.3 GB download, archive deleted after extraction)
!python -c "from spkclone.data import ensure_librispeech as e; e('train-clean-100', '{DATA_ROOT}')"
!df -h /tmp
""")

code("""
# SupCon fine-tuning, 3 seeds (resumable; re-run this cell after a timeout)
for seed in SEEDS:
    !python -m spkclone.train --config configs/rq1_finetune.yaml --root {DATA_ROOT} --seed {seed} --out runs/rq1/finetune_s{seed} --results results/rq1/finetune_s{seed}
""")

code("""
# small ECAPA from scratch, 3 seeds
for seed in SEEDS:
    !python -m spkclone.train --config configs/rq1_scratch.yaml --root {DATA_ROOT} --seed {seed} --out runs/rq1/scratch_s{seed} --results results/rq1/scratch_s{seed}
""")

code("""
# Ablation (fine-tuning, seed 0 only unless time allows): temperature 0.05 / 0.2 (0.1 is the main run),
# and K=2 with P=32 so the batch still has 64 crops (changing K alone would also halve the batch).
for name, extra in [("ablation_T0.05", "--temp 0.05"), ("ablation_T0.2", "--temp 0.2"), ("ablation_K2", "--K 2 --P 32")]:
    !python -m spkclone.train --config configs/rq1_finetune.yaml --root {DATA_ROOT} --seed 0 {extra} --out runs/rq1/{name}_s0 --results results/rq1/{name}_s0
""")

md("""
### RQ1 follow-up: final-epoch (not dev-selected) fine-tuned weights

In the first run no fine-tuning epoch beat the pretrained model on dev, so model selection kept the pretrained
weights and the "finetune" rows equal "pretrained". To see what training itself did (and to have a fine-tuned
verifier for RQ3), we also evaluate the weights after the last epoch. They are labelled as not selected on dev.
If `runs/` from that session still exists we only evaluate `last.pt` (minutes); otherwise the seed is re-trained
(~18 min each on a T4); train.py now writes `final_epoch/` itself.
""")

code("""
import os
FT_RUNS = [f"finetune_s{s}" for s in SEEDS] + ["ablation_T0.05_s0", "ablation_T0.2_s0", "ablation_K2_s0"]
EXTRA = {"ablation_T0.05_s0": "--temp 0.05", "ablation_T0.2_s0": "--temp 0.2", "ablation_K2_s0": "--K 2 --P 32"}
for run in FT_RUNS:
    if os.path.exists(f"results/rq1/{run}/final_epoch/test_metrics.json"):
        continue
    if os.path.exists(f"runs/rq1/{run}/last.pt"):
        !python -m spkclone.verify evaluate --ckpt runs/rq1/{run}/last.pt --root {DATA_ROOT} --out results/rq1/{run}/final_epoch
    else:
        seed = run.rsplit("_s", 1)[1]
        !python -m spkclone.train --config configs/rq1_finetune.yaml --root {DATA_ROOT} --seed {seed} {EXTRA.get(run, "")} --out runs/rq1/{run} --results results/rq1/{run} --force
""")

code("""
!python scripts/aggregate_rq1.py results/rq1
""")

md("""
## 4. RQ2–RQ4 — cloning fidelity, spoof vulnerability, explanation

TODO: `spkclone/eval_clone.py` is the next step of the project and is not written yet. Cells will be added here.
""")

code("""
!cd /kaggle/working/repo && zip -qr /kaggle/working/results.zip results && ls -lh /kaggle/working/results.zip
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1]) / "notebooks" / "kaggle_run.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(cells), "cells")
