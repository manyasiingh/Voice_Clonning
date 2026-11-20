"""SpkEncoder must reproduce speechbrain/spkrec-ecapa-voxceleb (EncoderClassifier.encode_batch).

Needs SpeechBrain + internet (downloads the pretrained model), so it is marked `slow`; it runs in the Kaggle
notebook. Real speech is used if SPKCLONE_TEST_WAVS points to a folder of audio files (the notebook sets it to
LibriSpeech dev-clean); synthetic harmonic signals are always included. The max abs difference is printed and
saved to results/env/encoder_equivalence.json.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("speechbrain")
pytestmark = pytest.mark.slow

ATOL = 1e-4


def synthetic_wavs(sr=16000):
    rng = np.random.default_rng(0)
    out = []
    for secs, f0 in [(1.5, 110.0), (3.0, 180.0), (6.0, 240.0)]:
        t = np.arange(int(secs * sr)) / sr
        x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 15)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        x = 0.1 * x / np.abs(x).max() + 0.005 * rng.standard_normal(len(t))
        out.append(("synthetic_%.1fs" % secs, torch.tensor(x, dtype=torch.float32)))
    return out


def real_wavs(n=4):
    root = os.environ.get("SPKCLONE_TEST_WAVS")
    if not root:
        return []
    from spkclone.data import load_wav

    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in {".wav", ".flac"})[:n]
    return [(p.name, load_wav(str(p), max_sec=10)) for p in files]


def test_spkencoder_matches_encoder_classifier():
    from spkclone.encoder import SpkEncoder, load_speechbrain_classifier
    from spkclone.utils import REPO_ROOT, save_json

    clf = load_speechbrain_classifier(device="cpu")
    enc = SpkEncoder("large")
    enc.emb.load_state_dict(clf.mods.embedding_model.state_dict())
    enc.eval()
    rows = []
    with torch.no_grad():
        for name, w in synthetic_wavs() + real_wavs():
            ref = clf.encode_batch(w[None]).squeeze()
            ours = enc(w[None]).squeeze()
            rows.append({"wav": name, "max_abs_diff": float((ref - ours).abs().max()),
                         "cosine": float(torch.nn.functional.cosine_similarity(ref, ours, dim=0))})
    worst = max(r["max_abs_diff"] for r in rows)
    print(f"\nencoder equivalence: max abs diff = {worst:.3e} over {len(rows)} wavs (tolerance {ATOL})")
    for r in rows:
        print("  ", r)
    save_json(REPO_ROOT / "results" / "env" / "encoder_equivalence.json",
              {"max_abs_diff": worst, "atol": ATOL, "passed": worst < ATOL, "rows": rows})
    assert worst < ATOL
