"""Embedding sets, trial scoring, dev-calibrated evaluation, two-file verification, optional VoxCeleb1-O trial lists.

Trial protocol (default, LibriSpeech): sample `eval_utts` utterances per held-out speaker with a FIXED seed
(the same trials for every model), crop each to `eval_crop` seconds (centred), and score ALL pairs.
Limitation: many pairs come from the same recording session (chapter), which is easier than VoxCeleb-style
cross-session trials, so these EERs are NOT comparable with published VoxCeleb numbers.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .data import build_index, load_crop, load_wav, resolve_split, session_of
from .encoder import load_encoder
from .metrics import error_rates_at, summarize_verification, threshold_at_far, verification_stats
from .utils import get_device, parse_with_config, save_json, write_run_meta


def sample_eval_utts(index, n_per_spk, seed=0):
    rng = random.Random(seed)
    items = []
    for spk, files in sorted(index.items()):
        items += [(spk, f) for f in sorted(rng.sample(files, min(n_per_spk, len(files))))]
    return items


@torch.no_grad()
def embed_files(enc, files, crop, device, bs=32):
    """L2-normalised embeddings (N, D). crop=None embeds full files one at a time (no padding issues)."""
    enc.eval()
    out = []
    if crop is None:
        for f in files:
            out.append(enc.embed(load_wav(f).to(device)).cpu())
        return torch.cat(out)
    for i in range(0, len(files), bs):
        wav = torch.stack([load_crop(f, crop, False) for f in files[i:i + bs]]).to(device)
        out.append(enc.embed(wav).cpu())
    return torch.cat(out)


def embed_set(enc, index, n_per_spk, crop, device, seed=0, bs=32):
    items = sample_eval_utts(index, n_per_spk, seed)
    emb = embed_files(enc, [f for _, f in items], crop, device, bs)
    return emb, [s for s, _ in items], [f for _, f in items]


def all_pair_trials(emb, spk, files=None):
    """Score every unordered pair. Returns dict of numpy arrays: scores, labels, spk_a, spk_b (int codes),
    and same_session (if files are given)."""
    sim = (emb @ emb.T).numpy()
    ia, ib = np.triu_indices(len(emb), 1)
    codes = {s: i for i, s in enumerate(sorted(set(spk)))}
    y = np.array([codes[s] for s in spk])
    t = {"scores": sim[ia, ib], "labels": y[ia] == y[ib], "spk_a": y[ia], "spk_b": y[ib]}
    if files is not None:
        sess = np.array([f"{s}/{session_of(f)}" for s, f in zip(spk, files)])
        t["same_session"] = sess[ia] == sess[ib]
    return t


def calibrate(dev_trials, far=0.01):
    """Thresholds from BONA FIDE dev trials only: the EER point and the FAR=1% point."""
    st = verification_stats(dev_trials["scores"], dev_trials["labels"])
    return {"eer_dev": st["eer"], "thr_eer": st["eer_threshold"], "far_target": far,
            "thr_far": threshold_at_far(dev_trials["scores"], dev_trials["labels"], far)}


def pca_plot(emb, spk, path, max_spk=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = emb.numpy() - emb.numpy().mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    xy = x @ vt[:2].T
    plt.figure(figsize=(6, 5))
    for s in sorted(set(spk))[:max_spk]:
        m = np.array([t == s for t in spk])
        plt.scatter(xy[m, 0], xy[m, 1], s=14, label=s)
    plt.title("Embeddings of held-out speakers (PCA)")
    plt.legend(fontsize=5, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_encoder(enc, idx_dv, idx_te, device, out_dir, eval_utts=15, eval_crop=6.0, eval_seed=0, n_boot=1000,
                     extra=None):
    """Calibrate on dev, report on test. Writes test_metrics.json, test_trials.npz, test_embeddings_pca.png.
    `extra` (dict) is stored in test_metrics.json too, e.g. which checkpoint / epoch was evaluated."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    e, s, f = embed_set(enc, idx_dv, eval_utts, eval_crop, device, eval_seed)
    cal = calibrate(all_pair_trials(e, s, f))
    e, s, f = embed_set(enc, idx_te, eval_utts, eval_crop, device, eval_seed)
    t = all_pair_trials(e, s, f)
    res = summarize_verification(t["scores"], t["labels"], t["spk_a"], t["spk_b"], n_boot=n_boot, seed=eval_seed)
    res["calibration_dev"] = cal
    res["at_dev_eer_threshold"] = error_rates_at(t["scores"], t["labels"], cal["thr_eer"])
    res["at_dev_far1_threshold"] = error_rates_at(t["scores"], t["labels"], cal["thr_far"])
    cross = ~t["same_session"]  # harder subset: pairs from different recording sessions (chapters)
    try:
        res["cross_session_only"] = verification_stats(t["scores"][cross], t["labels"][cross])
        res["cross_session_only"]["n_target"] = int(t["labels"][cross].sum())
    except ValueError:  # no cross-session target pairs in this sample
        res["cross_session_only"] = None
    res["protocol"] = dict(eval_utts=eval_utts, eval_crop=eval_crop, eval_seed=eval_seed,
                           note="all pairs of sampled utterances; many same-session pairs; not VoxCeleb-comparable")
    res.update(extra or {})
    save_json(out_dir / "test_metrics.json", res)
    np.savez_compressed(out_dir / "test_trials.npz", **t)
    pca_plot(e, s, out_dir / "test_embeddings_pca.png")
    ci = res.get("ci_speakers", {}).get("eer", {})
    print(f"TEST EER {res['eer'] * 100:.2f}% (speaker-bootstrap 95% CI {ci.get('lo', float('nan')) * 100:.2f}-"
          f"{ci.get('hi', float('nan')) * 100:.2f}%)  minDCF {res['min_dcf']:.3f}  "
          f"[{res['n_speakers']} speakers, {res['n_trials']} trials]")
    return res


# --------------------------------------------------------------------------- #
# VoxCeleb1-O style trial list: lines "label enrol_path test_path" relative to wav_root (user-provided data only)
# --------------------------------------------------------------------------- #
def load_trial_list(path):
    trials = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            lab, a, b = line.split()
            trials.append((int(lab), a, b))
    return trials


def evaluate_trial_list(enc, trials, wav_root, device, max_sec=None):
    files = sorted({p for _, a, b in trials for p in (a, b)})
    emb = {}
    for f in files:
        emb[f] = enc.embed(load_wav(str(Path(wav_root) / f), max_sec).to(device)).cpu()[0]
    scores = np.array([float(emb[a] @ emb[b]) for _, a, b in trials])
    labels = np.array([lab == 1 for lab, _, _ in trials])
    return verification_stats(scores, labels), scores, labels


# --------------------------------------------------------------------------- #
# CLIs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_cli(argv=None):
    p = argparse.ArgumentParser(description="Same / different speaker for two wav files.")
    p.add_argument("--ckpt", required=True, help="path to best.pt or 'pretrained'")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--max_sec", type=float, default=8.0)
    a = p.parse_args(argv)
    device = get_device()
    enc, meta = load_encoder(a.ckpt, device)
    thr = a.thr if a.thr is not None else meta.get("thr", 0.25)
    if a.thr is None and "thr" not in meta:
        print("NOTE: threshold 0.25 is uncalibrated; pass --thr from a dev calibration (test_metrics.json)")
    e = [enc.embed(load_wav(x, a.max_sec).to(device)) for x in (a.a, a.b)]
    s = (e[0] * e[1]).sum().item()
    print(f"cosine score = {s:.4f} | threshold = {thr:.4f} -> {'SAME speaker' if s >= thr else 'DIFFERENT speakers'}")
    return s


def evaluate_cli(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a checkpoint (or the pretrained model) on dev+test.")
    p.add_argument("--config")
    p.add_argument("--ckpt", default="pretrained")
    p.add_argument("--out", required=True)
    p.add_argument("--root", default="data")
    p.add_argument("--dev_subset", default="dev-clean")
    p.add_argument("--test_subset", default="test-clean")
    p.add_argument("--dev_dir")
    p.add_argument("--test_dir")
    p.add_argument("--eval_utts", type=int, default=15)
    p.add_argument("--eval_crop", type=float, default=6.0)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--trials", help="optional VoxCeleb1-O style trial list (needs --wav_root)")
    p.add_argument("--wav_root")
    p.add_argument("--force", action="store_true", help="re-run even if results exist")
    a = parse_with_config(p, argv)
    out = Path(a.out)
    if (out / "test_metrics.json").exists() and not a.force:
        print(f"{out}/test_metrics.json exists - skipping (use --force)")
        return
    write_run_meta(out, vars(a))
    device = get_device()
    enc, meta = load_encoder(a.ckpt, device)
    if a.trials:
        st, scores, labels = evaluate_trial_list(enc, load_trial_list(a.trials), a.wav_root, device)
        save_json(out / "trial_list_metrics.json", {**st, "trials": a.trials, "n_trials": len(scores)})
        print("trial list:", st)
    idx_dv = build_index(resolve_split(a.dev_dir, a.dev_subset, a.root))
    idx_te = build_index(resolve_split(a.test_dir, a.test_subset, a.root))
    evaluate_encoder(enc, idx_dv, idx_te, device, out, a.eval_utts, a.eval_crop, a.eval_seed, a.n_boot,
                     extra={"evaluated_checkpoint": {k: meta[k] for k in ("ckpt", "arch", "epoch") if k in meta}})


if __name__ == "__main__":
    import sys

    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    {"verify": verify_cli, "evaluate": evaluate_cli}.get(cmd, lambda _: sys.exit(
        "usage: python -m spkclone.verify {verify|evaluate} ..."))(rest)
