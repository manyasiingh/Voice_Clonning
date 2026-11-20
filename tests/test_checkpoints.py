"""Checkpoint formats written by train.py load back with the right architecture."""
import random

import numpy as np
import pytest
import torch

pytest.importorskip("speechbrain")

from spkclone.encoder import ARCH, SpkEncoder, infer_arch, load_encoder  # noqa: E402


@pytest.mark.parametrize("arch", list(ARCH))
def test_infer_arch(arch):
    assert infer_arch(SpkEncoder(arch).emb.state_dict()) == arch


def test_load_resume_checkpoint_without_arch(tmp_path):
    """last.pt files written before 'arch' was stored (first RQ1 runs) must still load."""
    enc = SpkEncoder("small")
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": None}
    torch.save({"emb": enc.emb.state_dict(), "opt": {}, "scaler": {}, "rng": rng,
                "state": {"epoch": 10, "step": 2000}}, tmp_path / "last.pt")
    loaded, meta = load_encoder(tmp_path / "last.pt", "cpu")
    assert meta["arch"] == "small" and meta["epoch"] == 10
    x = torch.randn(1, 16000)
    assert torch.allclose(loaded(x), enc.eval()(x), atol=1e-6)


def test_load_best_checkpoint(tmp_path):
    enc = SpkEncoder("small")
    torch.save({"emb": enc.emb.state_dict(), "arch": "small", "epoch": 3, "eer": 0.1, "thr": 0.5,
                "min_dcf": 0.2}, tmp_path / "best.pt")
    _, meta = load_encoder(tmp_path / "best.pt", "cpu")
    assert meta["epoch"] == 3 and meta["thr"] == 0.5
