import math

import numpy as np
import pytest

from spkclone.metrics import (acceptance_rate, bootstrap, det_curve, eer, eer_thr_mindcf, error_rates_at, min_dcf,
                              speaker_resampler, summarize_verification, threshold_at_far, trial_resampler)


def test_perfectly_separable_has_zero_eer():
    e, thr = eer(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]))
    assert e == 0.0 and thr == pytest.approx(0.8)


def test_known_eer_by_hand():
    # sorted: .9T .7N .6T .5N .4T .1N -> at thr .6: FNR = 1/3 (missed .4), FPR = 1/3 (accepted .7)
    s = np.array([0.9, 0.6, 0.4, 0.7, 0.5, 0.1])
    lab = np.array([1, 1, 1, 0, 0, 0])
    e, thr = eer(s, lab)
    assert e == pytest.approx(1 / 3) and thr == pytest.approx(0.6)


def test_det_includes_reject_everyone_point():
    thr, fnr, fpr = det_curve(np.array([0.3, 0.2]), np.array([1, 0]))
    assert thr[0] == np.inf and fnr[0] == 1.0 and fpr[0] == 0.0


def test_min_dcf_never_exceeds_one_reject_all():
    # one low target, ten high non-targets: every accepting threshold costs > 1, so the best is "reject all"
    s = np.r_[0.1, np.linspace(0.2, 1.0, 10)]
    lab = np.r_[1, np.zeros(10)]
    assert min_dcf(s, lab, 0.01) == pytest.approx(1.0)
    assert eer_thr_mindcf(s, lab)[2] == pytest.approx(1.0)


def test_ties_are_merged():
    thr, fnr, fpr = det_curve(np.array([0.5, 0.5]), np.array([1, 0]))
    assert len(thr) == 2
    assert eer(np.array([0.5, 0.5]), np.array([1, 0]))[0] == pytest.approx(0.5)


def test_gaussian_eer_matches_theory():
    rng = np.random.default_rng(0)
    d, n = 2.0, 20000
    s = np.r_[rng.normal(d, 1, n), rng.normal(0, 1, n)]
    lab = np.r_[np.ones(n), np.zeros(n)]
    theory = 0.5 * (1 + math.erf((-d / 2) / math.sqrt(2)))  # Phi(-d/2) = 0.1587
    assert eer(s, lab)[0] == pytest.approx(theory, abs=0.01)


def test_integer_weights_equal_duplication():
    rng = np.random.default_rng(1)
    s, lab = rng.normal(size=50), rng.random(50) < 0.4
    w = rng.integers(0, 3, 50)
    dup_s, dup_l = np.repeat(s, w), np.repeat(lab, w)
    assert eer(s, lab, w)[0] == pytest.approx(eer(dup_s, dup_l)[0])
    assert min_dcf(s, lab, 0.05, w) == pytest.approx(min_dcf(dup_s, dup_l, 0.05))


def test_threshold_at_far_respects_far():
    rng = np.random.default_rng(2)
    s = np.r_[rng.normal(2, 1, 500), rng.normal(0, 1, 500)]
    lab = np.r_[np.ones(500), np.zeros(500)]
    thr = threshold_at_far(s, lab, 0.01)
    r = error_rates_at(s, lab, thr)
    assert r["far"] <= 0.01
    # and one step lower would exceed it
    lower = s[~lab.astype(bool)][s[~lab.astype(bool)] < thr].max()
    assert error_rates_at(s, lab, lower)["far"] > 0.01


def test_acceptance_rate():
    assert acceptance_rate(np.array([0.1, 0.5, 0.9, 0.3]), 0.4) == pytest.approx(0.5)


def test_needs_both_classes():
    with pytest.raises(ValueError):
        det_curve(np.array([0.1, 0.2]), np.array([1, 1]))


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(3)
    s = np.r_[rng.normal(2, 1, 300), rng.normal(0, 1, 300)]
    lab = np.r_[np.ones(300), np.zeros(300)]
    point = eer(s, lab)[0]
    ci = bootstrap(lambda w: {"eer": eer(s, lab, w)[0]}, trial_resampler(len(s)), n_boot=200, seed=0)
    assert ci["eer"]["lo"] <= point <= ci["eer"]["hi"]
    assert ci["n_failed"] == 0


def test_speaker_resampler_weights():
    spk_a, spk_b = np.array([0, 0, 1]), np.array([0, 1, 1])
    w = speaker_resampler(spk_a, spk_b)(np.random.default_rng(0))
    assert w.shape == (3,) and (w >= 0).all()


def test_summarize_verification_runs():
    rng = np.random.default_rng(4)
    spk = rng.integers(0, 10, 200)
    spk_b = rng.integers(0, 10, 200)
    lab = spk == spk_b
    s = rng.normal(size=200) + 2 * lab
    res = summarize_verification(s, lab, spk, spk_b, n_boot=50)
    assert 0 <= res["eer"] <= 1 and "ci_speakers" in res and res["n_speakers"] == 10
