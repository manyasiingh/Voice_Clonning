import io
import tarfile

import numpy as np
import pytest
import soundfile as sf

from spkclone.data import (DistinctSpeakerBatchSampler, PKDataset, assert_disjoint, build_index, load_crop,
                           safe_extract, session_of)


# ----- sampler: P distinct speakers per batch (fixes the i % n_speakers repetition) ----- #
def test_batches_have_distinct_speakers():
    s = DistinctSpeakerBatchSampler(n_speakers=20, P=16, n_batches=200, seed=0)
    batches = list(s)
    assert len(batches) == 200
    assert all(len(set(b)) == 16 for b in batches)
    assert all(0 <= j < 20 for b in batches for j in b)


def test_sampler_is_deterministic_and_epoch_dependent():
    a, b = DistinctSpeakerBatchSampler(50, 8, 10, seed=3), DistinctSpeakerBatchSampler(50, 8, 10, seed=3)
    assert list(a) == list(b)
    e0 = list(a)
    a.set_epoch(1)
    assert list(a) != e0


def test_sampler_rejects_too_large_P():
    with pytest.raises(ValueError):
        DistinctSpeakerBatchSampler(5, 8, 1)


# ----- index / splits / audio ----- #
def make_corpus(root, speakers=("s1", "s2"), utts=3, secs=1.0, sr=16000):
    rng = np.random.default_rng(0)
    for s in speakers:
        d = root / s / "chap1"
        d.mkdir(parents=True)
        for u in range(utts):
            sf.write(d / f"{s}-{u}.flac", (0.1 * rng.standard_normal(int(secs * sr))).astype("float32"), sr)
    return root


def test_build_index_and_session(tmp_path):
    idx = build_index(make_corpus(tmp_path))
    assert sorted(idx) == ["s1", "s2"] and len(idx["s1"]) == 3
    assert session_of(idx["s1"][0]) == "chap1"


def test_assert_disjoint(tmp_path):
    with pytest.raises(ValueError):
        assert_disjoint({"train": {"a": [], "b": []}, "test": {"b": []}})
    assert assert_disjoint({"train": {"a": []}, "test": {"b": []}}) == {}
    assert assert_disjoint({"train": {"a": []}, "dev": {"a": []}}, allow_overlap=True) == {"train&dev": 1}


def test_load_crop_lengths(tmp_path):
    idx = build_index(make_corpus(tmp_path, secs=1.0))
    f = idx["s1"][0]
    assert len(load_crop(f, 0.5, random_crop=True)) == 8000
    assert len(load_crop(f, 2.5, random_crop=False)) == 40000  # wrap-padded


def test_pk_dataset_item_shape(tmp_path):
    ds = PKDataset(build_index(make_corpus(tmp_path)), K=2, crop=0.5)
    x, j = ds[1]
    assert x.shape == (2, 8000) and j == 1 and len(ds) == 2


# ----- safe tar extraction ----- #
def _tar_with(tmp_path, name):
    p = tmp_path / "a.tar.gz"
    with tarfile.open(p, "w:gz") as t:
        data = b"hello"
        info = tarfile.TarInfo(name)
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    return p


def test_safe_extract_ok(tmp_path):
    out = tmp_path / "out"
    safe_extract(_tar_with(tmp_path, "LibriSpeech/x.txt"), out)
    assert (out / "LibriSpeech" / "x.txt").read_bytes() == b"hello"


def test_safe_extract_blocks_path_traversal(tmp_path):
    out = tmp_path / "out"
    with pytest.raises((tarfile.TarError, ValueError)):
        safe_extract(_tar_with(tmp_path, "../evil.txt"), out)
    assert not (tmp_path / "evil.txt").exists()
