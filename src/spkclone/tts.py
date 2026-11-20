"""Speech generation with SpeechBrain (Tacotron2 + HiFi-GAN).

Modes
  clone   zero-shot voice cloning: speak TEXT in the voice of a short reference wav
          (MSTacotron2 trained on LibriTTS; vocoder HiFi-GAN 22.05 kHz). Note: MSTacotron2 embeds the reference
          with its own encoder `speechbrain/spkrec-ecapa-voxceleb-mel-spec` (per its hyperparams.yaml), which is a
          different checkpoint from the verifier `speechbrain/spkrec-ecapa-voxceleb` but the same model family.
  random  same model, randomly sampled voice
  single  fixed single-speaker voice (Tacotron2 + HiFi-GAN trained on LJSpeech)

Text is normalised with spkclone.textnorm BEFORE sentence splitting (so "Dr. Smith" is not split at "Dr.").
English only. Only clone voices you have permission to use.
"""
import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .data import load_wav
from .textnorm import normalize
from .utils import get_device, seed_everything

TTS_SR = 22050
MODELS = {
    "multi": ("speechbrain/tts-mstacotron2-libritts", "speechbrain/tts-hifigan-libritts-22050Hz"),
    "single": ("speechbrain/tts-tacotron2-ljspeech", "speechbrain/tts-hifigan-ljspeech"),
}


def get_classes():
    try:
        from speechbrain.inference.TTS import MSTacotron2, Tacotron2
        from speechbrain.inference.vocoders import HIFIGAN
    except ImportError:  # older SpeechBrain
        from speechbrain.pretrained import HIFIGAN, MSTacotron2, Tacotron2
    return MSTacotron2, Tacotron2, HIFIGAN


def split_text(text, max_chars=180):
    """Tacotron2 attention degrades on long inputs, so synthesise sentence-sized chunks."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?;:])\s+", text.strip()) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + len(p) + 1 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def prepare_reference(path, tmpdir):
    """Reference audio -> 16 kHz mono wav file (what MSTacotron2.clone_voice expects) + the tensor."""
    w = load_wav(path, sr=16000)
    out = str(Path(tmpdir) / "ref16k.wav")
    sf.write(out, w.numpy(), 16000)
    return out, w


class TTSRunner:
    """Loads the TTS + vocoder once; `synth` returns (wav float32 numpy @22.05 kHz, mel, alignment)."""

    def __init__(self, mode, device, savedir="pretrained_models"):
        MSTacotron2, Tacotron2, HIFIGAN = get_classes()
        self.mode, opts = mode, {"device": device}
        tts_src, voc_src = MODELS["single" if mode == "single" else "multi"]
        cls = Tacotron2 if mode == "single" else MSTacotron2
        self.tts = cls.from_hparams(source=tts_src, savedir=f"{savedir}/{tts_src.split('/')[1]}", run_opts=opts)
        self.voc = HIFIGAN.from_hparams(source=voc_src, savedir=f"{savedir}/{voc_src.split('/')[1]}", run_opts=opts)

    @torch.no_grad()
    def synth(self, text, ref_path=None):
        if self.mode == "clone":
            mel, _, align = self.tts.clone_voice(text, ref_path)
        elif self.mode == "random":
            mel, _, align = self.tts.generate_random_voice(text)
        else:
            mel, _, align = self.tts.encode_text(text)
        wav = self.voc.decode_batch(mel).squeeze().detach().cpu().numpy().astype(np.float32)
        return wav, mel, align


def plot_synthesis(mel, align, path, title):
    """SpeechBrain Tacotron2 returns mel as (batch, n_mels, frames) and alignments as
    (batch, decoder_steps, encoder_steps) with one decoder step per mel frame (n_frames_per_step = 1).
    We check that instead of guessing the orientation, and fail loudly if the installed version differs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel = mel.detach().cpu().float().numpy()
    al = align.detach().cpu().float().numpy()
    mel = mel[0] if mel.ndim == 3 else mel
    al = al[0] if al.ndim == 3 else al
    n_mels, n_frames = mel.shape
    dec_steps, enc_steps = al.shape
    if dec_steps != n_frames:
        raise ValueError(f"alignment shape {al.shape} does not match mel shape {mel.shape} "
                         "(expected (decoder_steps == mel frames, encoder_steps)); check the SpeechBrain version")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
    ax[0].set_title("generated mel-spectrogram")
    ax[0].set_xlabel("frame"), ax[0].set_ylabel("mel bin")
    ax[1].imshow(al.T, origin="lower", aspect="auto", cmap="viridis", extent=[0, dec_steps, 0, enc_steps])
    ax[1].set_title("attention alignment (want a clean diagonal)")
    ax[1].set_xlabel("decoder step (= mel frame)"), ax[1].set_ylabel("encoder step (text token)")
    fig.suptitle(title[:100], fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def speaker_similarity(enc, ref16k, wav22k, device):
    """ECAPA cosine between reference (16 kHz tensor) and generated (22.05 kHz numpy) audio, encoder loaded once."""
    import torchaudio

    gen16k = torchaudio.functional.resample(torch.from_numpy(wav22k), TTS_SR, 16000)
    e1, e2 = enc.embed(ref16k.to(device)), enc.embed(gen16k.to(device))
    return float((e1 * e2).sum())


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["clone", "random", "single"])
    p.add_argument("--text", nargs="*", default=[], help="one or more sentences/paragraphs")
    p.add_argument("--text_file", help="text file, one paragraph per line")
    p.add_argument("--ref", help="reference wav for `clone`")
    p.add_argument("--out", default="tts_out")
    p.add_argument("--verify", action="store_true", help="speaker similarity: reference vs generated (clone mode)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--pause", type=float, default=0.3, help="silence between chunks in all.wav (s)")
    p.add_argument("--max_chars", type=int, default=180)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_textnorm", action="store_true", help="disable number/abbreviation normalisation")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    texts = list(a.text)
    if a.text_file:
        texts += [l.strip() for l in Path(a.text_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not texts:
        raise SystemExit("Provide --text or --text_file")
    if a.mode == "clone" and not a.ref:
        raise SystemExit("`clone` needs --ref reference.wav")

    device = get_device()
    seed_everything(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = TTSRunner(a.mode, device)
    tmp = tempfile.mkdtemp()
    ref_path, ref16k = prepare_reference(a.ref, tmp) if a.mode == "clone" else (None, None)
    enc = None
    if a.verify and a.mode == "clone":
        from .encoder import load_encoder

        enc, _ = load_encoder("pretrained", device)  # loaded ONCE, same code path as the verifier

    pause = np.zeros(int(a.pause * TTS_SR), dtype=np.float32)
    all_wavs, n = [], 0
    for text in texts:
        if not a.no_textnorm:
            text = normalize(text)
        for chunk in split_text(text, a.max_chars):
            wav, mel, align = runner.synth(chunk, ref_path)
            sf.write(out / f"{a.mode}_{n:03d}.wav", wav, TTS_SR)
            msg = f"[{n:03d}] {len(wav) / TTS_SR:.1f}s  {chunk[:70]}"
            if enc is not None:
                msg += f"  | speaker cos-sim to reference: {speaker_similarity(enc, ref16k, wav, device):.3f}"
            print(msg)
            if a.plot:
                plot_synthesis(mel, align, out / f"{a.mode}_{n:03d}.png", chunk)
            all_wavs += [wav, pause]
            n += 1
    sf.write(out / f"{a.mode}_all.wav", np.concatenate(all_wavs[:-1]), TTS_SR)
    print(f"\nSaved {n} clips + {a.mode}_all.wav in {out}/  (sample rate {TTS_SR} Hz)")


if __name__ == "__main__":
    main()
