"""Why did the verifier score two utterances the way it did?

For one trial pair: gradient x input attribution on the normalised log-mel, time-window occlusion,
mel-band occlusion, and a deletion (faithfulness) test. `explain_many` aggregates band occlusion and the
deletion-test pass rate over many pairs (RQ4; exploratory).

Occlusion sets cells to 0. Because the model input is mean-normalised per utterance, 0 equals the per-bin
utterance mean, i.e. we replace the occluded region with "average" content for that utterance.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_wav
from .encoder import load_encoder, mean_norm
from .utils import get_device, save_json

N_MELS, BAND = 80, 8


def mel_centers_hz(n=N_MELS, fmax=8000):
    m = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda x: 700 * (10 ** (x / 2595) - 1)
    return inv(np.linspace(m(0), m(fmax), n + 2))[1:-1]


def band_labels():
    hz = mel_centers_hz()
    return [f"{hz[g]:.0f}-{hz[min(g + BAND - 1, N_MELS - 1)]:.0f}" for g in range(0, N_MELS, BAND)]


def explain_pair(enc, wav_a, wav_b, win=0.5, frac=0.2, n_random=8, seed=0):
    """wav_a/wav_b: 1-D 16 kHz tensors on the encoder's device. Returns a dict of numpy arrays / floats."""
    with torch.no_grad():
        xa = mean_norm(enc.fbank(wav_a[None].float()))  # (1, T, 80) = model input
        xb = mean_norm(enc.fbank(wav_b[None].float()))
    T_a, T_b = xa.shape[1], xb.shape[1]

    def cos(x, y):
        return F.cosine_similarity(enc.emb(x).squeeze(1), enc.emb(y).squeeze(1), dim=-1)[0]

    with torch.no_grad():
        base = cos(xa, xb).item()

    # 1) gradient x input of the cosine score w.r.t. both inputs (signed; + pushes toward SAME)
    ga, gb = xa.clone().requires_grad_(True), xb.clone().requires_grad_(True)
    cos(ga, gb).backward()
    att_a, att_b = (ga.grad * xa)[0].detach().cpu().numpy(), (gb.grad * xb)[0].detach().cpu().numpy()

    # 2) time-window occlusion (window -> utterance mean) -> score drop; 100 frames per second
    W, S = max(1, int(win * 100)), max(1, int(win * 50))

    @torch.no_grad()
    def occ_time(which):
        x = xa if which == "a" else xb
        pos, drop = [], []
        for t in range(0, max(1, x.shape[1] - W + 1), S):
            x2 = x.clone()
            x2[:, t:t + W] = 0
            s2 = cos(x2, xb).item() if which == "a" else cos(xa, x2).item()
            pos.append(t / 100)
            drop.append(base - s2)
        return np.array(pos), np.array(drop)

    pa, da = occ_time("a")
    pb, db = occ_time("b")

    # 3) mel-band occlusion on both utterances at once
    band_drop = []
    with torch.no_grad():
        for g in range(0, N_MELS, BAND):
            x1, x2 = xa.clone(), xb.clone()
            x1[:, :, g:g + BAND] = 0
            x2[:, :, g:g + BAND] = 0
            band_drop.append(base - cos(x1, x2).item())

    # 4) deletion test: removing the frames attribution calls "supporting" should lower the score more than
    #    removing random frames, which should lower it more than removing "opposing" frames.
    fa_imp, fb_imp = att_a.sum(1), att_b.sum(1)

    @torch.no_grad()
    def delete(ia, ib):
        x1, x2 = xa.clone(), xb.clone()
        x1[:, ia], x2[:, ib] = 0, 0
        return cos(x1, x2).item()

    ka, kb = max(1, int(frac * T_a)), max(1, int(frac * T_b))
    rng = np.random.default_rng(seed)
    rand = float(np.mean([delete(rng.choice(T_a, ka, replace=False), rng.choice(T_b, kb, replace=False))
                          for _ in range(n_random)]))
    deletion = {"base": base, "frac": frac,
                "remove_top_supporting": delete(np.argsort(-fa_imp)[:ka], np.argsort(-fb_imp)[:kb]),
                "remove_random": rand,
                "remove_top_opposing": delete(np.argsort(fa_imp)[:ka], np.argsort(fb_imp)[:kb])}
    deletion["faithful"] = bool(deletion["remove_top_supporting"] < rand < deletion["remove_top_opposing"])
    return {"score": base, "xa": xa[0].cpu().numpy(), "xb": xb[0].cpu().numpy(), "att_a": att_a, "att_b": att_b,
            "occ_a": (pa, da), "occ_b": (pb, db), "band_drop": np.array(band_drop), "deletion": deletion,
            "win": win}


def summary_text(r, thr):
    (pa, da), (pb, db), win = r["occ_a"], r["occ_b"], r["win"]
    lbl, d = band_labels(), r["deletion"]
    bi = int(np.argmax(r["band_drop"]))
    base = r["score"]
    return (
        f"Decision: {'SAME' if base >= thr else 'DIFFERENT'} speaker | score {base:.3f} vs threshold {thr:.3f} "
        f"(margin {base - thr:+.3f})\n"
        f"Most supportive window: A {pa[np.argmax(da)]:.1f}-{pa[np.argmax(da)] + win:.1f}s (drop {da.max():+.3f}), "
        f"B {pb[np.argmax(db)]:.1f}-{pb[np.argmax(db)] + win:.1f}s (drop {db.max():+.3f})\n"
        f"Most opposing window:  A {pa[np.argmin(da)]:.1f}-{pa[np.argmin(da)] + win:.1f}s (drop {da.min():+.3f}), "
        f"B {pb[np.argmin(db)]:.1f}-{pb[np.argmin(db)] + win:.1f}s (drop {db.min():+.3f})\n"
        f"Most influential band: {lbl[bi]} Hz (score drop {r['band_drop'][bi]:+.3f} when masked)\n"
        f"Deletion test (remove {d['frac']:.0%} frames): supporting->{d['remove_top_supporting']:.3f}, "
        f"random->{d['remove_random']:.3f}, opposing->{d['remove_top_opposing']:.3f} "
        f"[attribution faithful on this pair: {'yes' if d['faithful'] else 'NO - treat saliency maps with caution'}]"
    )


def plot_explanation(r, thr, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hz, text, win = mel_centers_hz(), summary_text(r, thr), r["win"]
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.8, 0.9])
    yt = [0, 20, 40, 60, 79]
    for c, (x, att, name) in enumerate([(r["xa"], r["att_a"], "A"), (r["xb"], r["att_b"], "B")]):
        T = x.shape[0]
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(x.T, origin="lower", aspect="auto", extent=[0, T / 100, 0, 80], cmap="magma")
        ax.set_title(f"{name}: normalised log-mel (model input)")
        ax.set_yticks(yt), ax.set_yticklabels([f"{hz[i]:.0f}" for i in yt]), ax.set_ylabel("Hz")
        ax = fig.add_subplot(gs[1, c])
        v = np.percentile(np.abs(att), 99) or 1.0
        ax.imshow(att.T, origin="lower", aspect="auto", extent=[0, T / 100, 0, 80], cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{name}: gradient x input  (red = pushes toward SAME, blue = toward DIFFERENT)")
        ax.set_yticks(yt), ax.set_yticklabels([f"{hz[i]:.0f}" for i in yt]), ax.set_ylabel("Hz")
    for c, ((p, d), name) in enumerate([(r["occ_a"], "A"), (r["occ_b"], "B")]):
        ax = fig.add_subplot(gs[2, c])
        ax.bar(p + win / 2, d, width=win * 0.5, color=np.where(d >= 0, "tab:red", "tab:blue"))
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{name}: score drop when this {win}s window is masked (red = was supporting)")
        ax.set_xlabel("time (s)")
    ax = fig.add_subplot(gs[3, 0])
    bd = r["band_drop"]
    ax.bar(band_labels(), bd, color=np.where(bd >= 0, "tab:red", "tab:blue"))
    ax.set_title("Score drop when a mel band is masked (both utterances)"), ax.set_xlabel("Hz")
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax = fig.add_subplot(gs[3, 1])
    ax.axis("off")
    ax.text(0, 1, text, va="top", fontsize=8, family="monospace", wrap=True)
    plt.tight_layout()
    plt.savefig(png, dpi=130)
    plt.close()


def result_json(r, thr):
    (pa, da), (pb, db) = r["occ_a"], r["occ_b"]
    return {"score": r["score"], "threshold": thr, "decision": "same" if r["score"] >= thr else "different",
            "deletion_test": r["deletion"], "occlusion_A": {"t": pa, "drop": da}, "occlusion_B": {"t": pb, "drop": db},
            "band_drop": dict(zip(band_labels(), r["band_drop"].tolist()))}


def explain_many(enc, pairs, device, max_sec=8.0, win=0.5):
    """pairs: list of (wav_path_a, wav_path_b). Aggregate band occlusion + deletion-test pass rate."""
    rows = []
    for i, (fa, fb) in enumerate(pairs):
        r = explain_pair(enc, load_wav(fa, max_sec).to(device), load_wav(fb, max_sec).to(device), win, seed=i)
        rows.append({"a": fa, "b": fb, "score": r["score"], "band_drop": r["band_drop"].tolist(),
                     "faithful": r["deletion"]["faithful"], "deletion": r["deletion"]})
    bd = np.array([x["band_drop"] for x in rows])
    passed = np.array([x["faithful"] for x in rows])
    return {"n_pairs": len(rows), "deletion_pass_rate": float(passed.mean()) if rows else None,
            "band_labels": band_labels(), "band_drop_mean": bd.mean(0).tolist() if rows else [],
            "band_drop_std": bd.std(0).tolist() if rows else [], "pairs": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description="Explain one verification decision.")
    p.add_argument("--ckpt", required=True, help="path to best.pt or 'pretrained'")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--thr", type=float, default=None)
    p.add_argument("--max_sec", type=float, default=8.0)
    p.add_argument("--png", default="explanation.png")
    p.add_argument("--win", type=float, default=0.5, help="occlusion window (s)")
    a = p.parse_args(argv)
    device = get_device()
    enc, meta = load_encoder(a.ckpt, device)
    thr = a.thr if a.thr is not None else meta.get("thr", 0.25)
    r = explain_pair(enc, load_wav(a.a, a.max_sec).to(device), load_wav(a.b, a.max_sec).to(device), a.win)
    print(summary_text(r, thr))
    plot_explanation(r, thr, a.png)
    save_json(Path(a.png).with_suffix(".json"), result_json(r, thr))
    print(f"saved {a.png}")


if __name__ == "__main__":
    main()
