"""Verification metrics: DET curve, EER, minDCF, operating thresholds, attack acceptance, bootstrap CIs.

Convention everywhere: a trial is ACCEPTED (same speaker) when score >= threshold.
All functions accept optional per-trial `weights`; bootstrap resampling is implemented as integer weights
(how many times each trial / speaker was drawn), so one code path serves trial- and speaker-level bootstrap.
"""
import numpy as np


def det_curve(scores, labels, weights=None):
    """Return (thresholds, fnr, fpr) for every achievable operating point.

    The first point is "reject everyone" (threshold=+inf, fnr=1, fpr=0). Tied scores are merged, because
    a threshold cannot separate two equal scores.
    """
    s = np.asarray(scores, dtype=np.float64)
    lab = np.asarray(labels).astype(bool)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=np.float64)
    if not (s.shape == lab.shape == w.shape):
        raise ValueError("scores, labels and weights must have the same shape")
    order = np.argsort(-s, kind="mergesort")
    s, lab, w = s[order], lab[order], w[order]
    wt, wn = np.where(lab, w, 0.0), np.where(lab, 0.0, w)
    P, N = wt.sum(), wn.sum()
    if P <= 0 or N <= 0:
        raise ValueError("need at least one target and one non-target trial (with weight > 0)")
    last = np.r_[s[1:] != s[:-1], True]  # last position of each run of tied scores
    fnr = 1.0 - np.cumsum(wt)[last] / P
    fpr = np.cumsum(wn)[last] / N
    return np.r_[np.inf, s[last]], np.r_[1.0, fnr], np.r_[0.0, fpr]


def _eer_from_det(thr, fnr, fpr):
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2), float(thr[i])


def _dcf_from_det(fnr, fpr, p_target):
    # C_miss = C_fa = 1, normalised by the cost of the best trivial system (accept-all or reject-all).
    return float(((fnr * p_target + fpr * (1 - p_target)) / min(p_target, 1 - p_target)).min())


def eer(scores, labels, weights=None):
    """(EER, threshold). EER is taken at the DET point where |FNR - FPR| is smallest (no interpolation)."""
    return _eer_from_det(*det_curve(scores, labels, weights))


def min_dcf(scores, labels, p_target=0.01, weights=None):
    _, fnr, fpr = det_curve(scores, labels, weights)
    return _dcf_from_det(fnr, fpr, p_target)


def verification_stats(scores, labels, p_target=0.01, weights=None):
    thr, fnr, fpr = det_curve(scores, labels, weights)
    e, t = _eer_from_det(thr, fnr, fpr)
    return {"eer": e, "eer_threshold": t, "min_dcf": _dcf_from_det(fnr, fpr, p_target)}


def eer_thr_mindcf(scores, labels, p_tar=0.01):
    """Backward-compatible wrapper of the original script's function: (eer, threshold, minDCF)."""
    st = verification_stats(scores, labels, p_tar)
    return st["eer"], st["eer_threshold"], st["min_dcf"]


def threshold_at_far(scores, labels, far):
    """Lowest threshold whose false-accept rate is <= `far` (e.g. far=0.01 for the FAR=1% operating point)."""
    thr, _, fpr = det_curve(scores, labels)
    return float(thr[np.nonzero(fpr <= far)[0][-1]])  # fpr is non-decreasing, index 0 (fpr=0) always qualifies


def error_rates_at(scores, labels, thr):
    """Bona fide false-accept and false-reject rates at a fixed threshold."""
    s, lab = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    return {"threshold": float(thr), "far": float((s[~lab] >= thr).mean()), "frr": float((s[lab] < thr).mean()),
            "n_target": int(lab.sum()), "n_nontarget": int((~lab).sum())}


def acceptance_rate(scores, thr):
    """Fraction of trials accepted at `thr` (for spoof/attack trials this is the attack acceptance rate)."""
    s = np.asarray(scores, dtype=np.float64)
    return float((s >= thr).mean())


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def trial_resampler(n):
    """Resample trials with replacement -> integer weight per trial."""
    return lambda rng: np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)


def speaker_resampler(spk_a, spk_b):
    """Resample SPEAKERS with replacement. A trial between speakers a != b gets weight c[a]*c[b] (all pairings of
    the drawn copies); a same-speaker trial gets weight c[a]. spk_a/spk_b are integer speaker codes per trial."""
    spk_a, spk_b = np.asarray(spk_a), np.asarray(spk_b)
    n_spk = int(max(spk_a.max(), spk_b.max())) + 1

    def sample(rng):
        c = np.bincount(rng.integers(0, n_spk, n_spk), minlength=n_spk).astype(np.float64)
        return np.where(spk_a == spk_b, c[spk_a], c[spk_a] * c[spk_b])

    return sample


def bootstrap(stat_fn, sample_weights, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap. stat_fn(weights) -> dict of floats. Resamples where a statistic is undefined
    (stat_fn raises ValueError, e.g. no target trials drawn) are skipped and counted in n_failed."""
    rng = np.random.default_rng(seed)
    vals, failed = [], 0
    for _ in range(n_boot):
        try:
            vals.append(stat_fn(sample_weights(rng)))
        except ValueError:
            failed += 1
    out = {"n_boot": n_boot, "n_failed": failed, "level": 1 - alpha}
    if not vals:
        return out
    for k in vals[0]:
        v = np.array([d[k] for d in vals])
        out[k] = {"lo": float(np.quantile(v, alpha / 2)), "hi": float(np.quantile(v, 1 - alpha / 2)),
                  "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def summarize_verification(scores, labels, spk_a, spk_b, p_target=0.01, n_boot=1000, seed=0):
    """EER / minDCF with 95% bootstrap CIs over trials and over speakers."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)

    def stat(w):
        st = verification_stats(scores, labels, p_target, w)
        return {"eer": st["eer"], "min_dcf": st["min_dcf"]}

    res = verification_stats(scores, labels, p_target)
    res.update(p_target=p_target, n_trials=int(len(scores)), n_target=int(labels.sum()),
               n_nontarget=int((~labels).sum()), n_speakers=int(len(np.union1d(spk_a, spk_b))))
    if n_boot:
        res["ci_trials"] = bootstrap(stat, trial_resampler(len(scores)), n_boot, seed)
        res["ci_speakers"] = bootstrap(stat, speaker_resampler(spk_a, spk_b), n_boot, seed + 1)
    return res
