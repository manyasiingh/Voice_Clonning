"""Compatibility shims so the pretrained SpeechBrain TTS models load with current library versions.

Each shim only ADDS a missing old name pointing to its renamed / equivalent replacement, and does nothing on the
older versions that still have the name. Found in the first Kaggle smoke tests; see docs/decisions.md.
"""


def patch_mel_spectrogram_name():
    """speechbrain/tts-mstacotron2-libritts hyperparams.yaml refers to `Tacotron2.mel_spectogram` (typo).
    SpeechBrain renamed it to `mel_spectrogram` (PR #3063; v1.1.x) with the same arguments."""
    import speechbrain.lobes.models.Tacotron2 as t2

    if not hasattr(t2, "mel_spectogram") and hasattr(t2, "mel_spectrogram"):
        t2.mel_spectogram = t2.mel_spectrogram


def patch_tokenizer_encode_plus():
    """MSTacotron2's grapheme-to-phoneme step embeds words with a BERT tokenizer via `tokenizer.encode_plus(text,
    return_tensors="pt")`. transformers 5 removed `encode_plus`; for a single text it is equivalent to calling the
    tokenizer directly, `tokenizer(text, return_tensors="pt")`, which returns the same BatchEncoding."""
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:  # transformers not installed: nothing to patch
        return
    if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
        def encode_plus(self, text, text_pair=None, **kwargs):
            return self(text, text_pair=text_pair, **kwargs)

        PreTrainedTokenizerBase.encode_plus = encode_plus


def apply_tts_patches():
    patch_mel_spectrogram_name()
    patch_tokenizer_encode_plus()
