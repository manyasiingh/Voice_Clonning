"""Fine-tune (default) or train from scratch an ECAPA-TDNN with a SupCon loss on P speakers x K utterances.

Resumable: `<out>/last.pt` is written after every epoch (model, optimiser, scaler, RNG states, history);
re-running the same command continues from it, and a finished run (test_metrics.json present) is skipped.
"""
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import DistinctSpeakerBatchSampler, PKDataset, assert_disjoint, build_index, resolve_split
from .encoder import SpkEncoder, load_encoder
from .losses import supcon_loss
from .utils import get_device, parse_with_config, save_json, seed_everything, worker_init_fn, write_run_meta
from .verify import all_pair_trials, embed_set, evaluate_encoder
from .metrics import verification_stats


def build_parser():
    t = argparse.ArgumentParser(description=__doc__)
    t.add_argument("--config", help="YAML file with defaults for any of these flags")
    t.add_argument("--out", default="runs/spk", help="checkpoints (large files)")
    t.add_argument("--results", default=None, help="small JSON/PNG outputs (default: --out)")
    t.add_argument("--root", default="data", help="where LibriSpeech is downloaded")
    t.add_argument("--train_subset", default="train-clean-100")
    t.add_argument("--dev_subset", default="dev-clean")
    t.add_argument("--test_subset", default="test-clean")
    t.add_argument("--train_dir")
    t.add_argument("--dev_dir")
    t.add_argument("--test_dir")
    t.add_argument("--allow_overlap", action="store_true", help="allow shared speakers across splits (smoke tests)")
    t.add_argument("--scratch", action="store_true", help="random init (small ECAPA) instead of fine-tuning")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--steps", type=int, default=200, help="steps per epoch")
    t.add_argument("--P", type=int, default=16, help="distinct speakers per batch")
    t.add_argument("--K", type=int, default=4, help="utterances per speaker per batch")
    t.add_argument("--crop", type=float, default=3.0)
    t.add_argument("--temp", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--eval_utts", type=int, default=15, help="utterances per held-out speaker for trials")
    t.add_argument("--eval_crop", type=float, default=6.0)
    t.add_argument("--eval_seed", type=int, default=0, help="trial sampling seed: keep fixed across models")
    t.add_argument("--n_boot", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--force", action="store_true", help="re-run even if test_metrics.json exists")
    return t


def augment(wav):
    """Light GPU augmentation: random gain (+-6 dB) on all, additive white noise (5-20 dB SNR) on ~half."""
    d = wav.device
    wav = wav * (10 ** (torch.empty(len(wav), 1, device=d).uniform_(-6, 6) / 20))
    noisy = torch.rand(len(wav), 1, device=d) < 0.5
    snr = torch.empty(len(wav), 1, device=d).uniform_(5, 20)
    pw = wav.pow(2).mean(1, keepdim=True)
    noise = torch.randn_like(wav) * (pw / 10 ** (snr / 10)).sqrt()
    return torch.where(noisy, wav + noise, wav)


def lr_at(step, lr, warmup, total):
    """Linear warm-up, then cosine decay to 2% of the peak learning rate."""
    if step < warmup:
        return lr * (step + 1) / warmup
    return lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def train(a):
    out, res_dir = Path(a.out), Path(a.results or a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (res_dir / "test_metrics.json").exists() and not a.force:
        print(f"{res_dir}/test_metrics.json exists - skipping (use --force)")
        return
    seed_everything(a.seed)
    device = get_device()
    amp = device == "cuda"
    write_run_meta(res_dir, vars(a))
    idx_tr = build_index(resolve_split(a.train_dir, a.train_subset, a.root))
    idx_dv = build_index(resolve_split(a.dev_dir, a.dev_subset, a.root))
    idx_te = build_index(resolve_split(a.test_dir, a.test_subset, a.root))
    assert_disjoint({"train": idx_tr, "dev": idx_dv, "test": idx_te}, a.allow_overlap)

    enc = SpkEncoder("small" if a.scratch else "large")
    if not a.scratch:
        enc.load_pretrained()
    enc.to(device)
    lr = a.lr or (1e-3 if a.scratch else 5e-5)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total = a.epochs * a.steps

    def dev_eval():
        e, s, _ = embed_set(enc, idx_dv, a.eval_utts, a.eval_crop, device, a.eval_seed)
        t = all_pair_trials(e, s)
        st = verification_stats(t["scores"], t["labels"])
        return {"eer": st["eer"], "thr": st["eer_threshold"], "min_dcf": st["min_dcf"]}

    def save_weights(m, epoch, name):
        torch.save({"emb": enc.emb.state_dict(), "arch": enc.arch, "epoch": int(epoch),
                    **{k: float(v) for k, v in m.items()}}, out / name)

    last = out / "last.pt"
    if last.exists():  # resume
        ck = torch.load(last, map_location="cpu", weights_only=False)  # own file; contains RNG states
        enc.emb.load_state_dict(ck["emb"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        _set_rng_state(ck["rng"])
        st = ck["state"]
        st.setdefault("best_epoch", None)  # resume files written before this field existed
        print(f"resumed from {last} at epoch {st['epoch']} (step {st['step']})")
    else:
        m = dev_eval()
        print(f"[ep 0] dev EER {m['eer'] * 100:.2f}%  minDCF {m['min_dcf']:.3f}  thr {m['thr']:.3f}  (initial model)")
        st = {"epoch": 0, "step": 0, "best_eer": math.inf, "best_epoch": None,
              "history": [{"epoch": 0, **m}], "losses": []}
        if not a.scratch:  # the pretrained model is a valid candidate for "best"
            st["best_eer"], st["best_epoch"] = m["eer"], 0
            save_weights(m, 0, "best.pt")

    ds = PKDataset(idx_tr, a.K, a.crop)
    sampler = DistinctSpeakerBatchSampler(len(ds), a.P, a.steps, seed=a.seed)
    for ep in range(st["epoch"] + 1, a.epochs + 1):
        sampler.set_epoch(ep)
        loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=a.workers,
                                             worker_init_fn=worker_init_fn,
                                             generator=torch.Generator().manual_seed(a.seed * 1000 + ep))
        enc.train()
        ep_losses = []
        for wav, spk in loader:
            P, K, T = wav.shape
            wav = augment(wav.view(P * K, T).to(device))
            labels = spk.repeat_interleave(K).to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(st["step"], lr, a.warmup, total)
            loss = supcon_loss(enc(wav, amp=amp), labels, a.temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(enc.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            ep_losses.append(loss.item())
            st["step"] += 1
        m = dev_eval()
        st["losses"] += ep_losses
        st["history"].append({"epoch": ep, "loss": float(np.mean(ep_losses)), **m})
        print(f"[ep {ep}] loss {np.mean(ep_losses):.3f} | dev EER {m['eer'] * 100:.2f}%  "
              f"minDCF {m['min_dcf']:.3f}  thr {m['thr']:.3f}")
        if m["eer"] < st["best_eer"]:
            st["best_eer"], st["best_epoch"] = m["eer"], ep
            save_weights(m, ep, "best.pt")
        st["epoch"] = ep
        torch.save({"emb": enc.emb.state_dict(), "arch": enc.arch, "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "rng": _rng_state(), "state": st}, last)
        save_json(res_dir / "train_history.json", {"history": st["history"], "losses": st["losses"],
                                                    "best_epoch": st["best_epoch"]})

    plot_losses(st, res_dir / "loss_curve.png")
    sel = {"selected_epoch": st["best_epoch"], "selection": "lowest dev EER (epoch 0 = initial model)"}
    # Main result: the dev-selected checkpoint. If no epoch beat the initial model, this IS the initial model.
    enc_best, _ = load_encoder(out / "best.pt", device)
    evaluate_encoder(enc_best, idx_dv, idx_te, device, res_dir, a.eval_utts, a.eval_crop, a.eval_seed, a.n_boot,
                     extra={**sel, "evaluated_checkpoint": "best.pt"})
    # Secondary: the weights after the last epoch, NOT selected on dev - shows what training itself did.
    evaluate_encoder(enc.eval(), idx_dv, idx_te, device, res_dir / "final_epoch", a.eval_utts, a.eval_crop,
                     a.eval_seed, a.n_boot, extra={**sel, "evaluated_checkpoint": f"final epoch {st['epoch']}"})


def plot_losses(st, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(st["losses"], lw=0.6)
    ax[0].set_xlabel("step"), ax[0].set_ylabel("SupCon loss"), ax[0].set_title("training loss")
    h = st["history"]
    ax[1].plot([x["epoch"] for x in h], [x["eer"] * 100 for x in h], marker="o")
    ax[1].set_xlabel("epoch"), ax[1].set_ylabel("dev EER (%)"), ax[1].set_title("dev EER (model selection)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(argv=None):
    train(parse_with_config(build_parser(), argv))


if __name__ == "__main__":
    main()
