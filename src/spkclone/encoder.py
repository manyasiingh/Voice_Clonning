"""ECAPA-TDNN speaker encoder, shared by training, verification, explanation and TTS evaluation.

SpkEncoder = Fbank(80) -> sentence mean normalisation -> ECAPA_TDNN. This reproduces the inference path of
`speechbrain/spkrec-ecapa-voxceleb` (EncoderClassifier.encode_batch with normalize=False), whose hyperparams.yaml
uses `Fbank(n_mels=80)` with default settings and `InputNormalization(norm_type=sentence, std_norm=False)`.
tests/test_encoder_equivalence.py checks this numerically and reports the max abs difference.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PRETRAINED_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
PRETRAINED_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"

ARCH = {  # "large" matches speechbrain/spkrec-ecapa-voxceleb; "small" is faster for scratch training
    "large": dict(channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                  dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192),
    "small": dict(channels=[512, 512, 512, 512, 1536], kernel_sizes=[5, 3, 3, 3, 1],
                  dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192),
}


def encoder_classifier_class():
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:  # SpeechBrain < 1.0
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier


def load_speechbrain_classifier(savedir=PRETRAINED_SAVEDIR, device="cpu"):
    """The official SpeechBrain inference wrapper (used as the reference in the equivalence test)."""
    return encoder_classifier_class().from_hparams(source=PRETRAINED_SOURCE, savedir=savedir,
                                                   run_opts={"device": device})


def mean_norm(x):
    """Sentence-level mean normalisation over time (axis 1), no variance normalisation.
    After this, a value of 0 in a (frame, mel-bin) cell equals that bin's utterance mean."""
    return x - x.mean(1, keepdim=True)


class SpkEncoder(nn.Module):
    def __init__(self, arch="large", n_mels=80):
        super().__init__()
        from speechbrain.lobes.features import Fbank
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

        self.arch = arch
        self.fbank = Fbank(n_mels=n_mels)
        self.emb = ECAPA_TDNN(n_mels, **ARCH[arch])

    def load_pretrained(self, savedir=PRETRAINED_SAVEDIR):
        clf = load_speechbrain_classifier(savedir, "cpu")
        self.emb.load_state_dict(clf.mods.embedding_model.state_dict())  # strict: fails loudly on mismatch
        return self

    def features(self, wav):
        """(B, samples) 16 kHz waveform -> (B, frames, 80) normalised log-mel = what ECAPA sees."""
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return mean_norm(self.fbank(wav.float()))

    def forward(self, wav, amp=False):
        x = self.features(wav)
        with torch.autocast(device_type=wav.device.type, dtype=torch.float16,
                            enabled=amp and wav.device.type == "cuda"):
            e = self.emb(x)
        return e.squeeze(1).float()

    @torch.no_grad()
    def embed(self, wav):
        """L2-normalised embeddings, so a dot product is the cosine score."""
        if wav.dim() == 1:
            wav = wav[None]
        return F.normalize(self(wav), dim=-1)


def infer_arch(state_dict):
    """'large' or 'small' from the width of the first TDNN block (1024 vs 512 channels)."""
    width = state_dict["blocks.0.conv.conv.weight"].shape[0]
    for name, cfg in ARCH.items():
        if cfg["channels"][0] == width:
            return name
    raise ValueError(f"unknown ECAPA width {width}")


def load_encoder(ckpt, device, savedir=PRETRAINED_SAVEDIR):
    """ckpt = 'pretrained', a best.pt / final.pt written by spkclone.train, or a last.pt resume checkpoint
    (= weights after the final epoch). Returns (encoder in eval mode, meta)."""
    if str(ckpt) == "pretrained":
        return SpkEncoder("large").load_pretrained(savedir).to(device).eval(), {"ckpt": "pretrained"}
    # best.pt / final.pt hold only tensors + plain scalars/strings, so the safe weights_only loader works.
    # last.pt is our own resume file and also holds RNG states (NumPy arrays), which needs weights_only=False.
    resume_file = Path(ckpt).name == "last.pt"
    d = torch.load(ckpt, map_location="cpu", weights_only=not resume_file)
    arch = d.get("arch") or infer_arch(d["emb"])
    enc = SpkEncoder(arch)
    enc.emb.load_state_dict(d["emb"])
    if resume_file:
        meta = {"epoch": d["state"]["epoch"], "step": d["state"]["step"]}
    else:
        meta = {k: v for k, v in d.items() if k != "emb"}
    meta.update(ckpt=str(ckpt), arch=arch)
    return enc.to(device).eval(), meta
