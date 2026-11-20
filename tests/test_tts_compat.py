"""Compatibility shims needed to load speechbrain/tts-mstacotron2-libritts with current library versions."""
import pytest


def test_mel_spectogram_name_resolves_after_patch():
    pytest.importorskip("speechbrain")
    from spkclone.compat import patch_mel_spectrogram_name

    patch_mel_spectrogram_name()
    import speechbrain.lobes.models.Tacotron2 as t2

    assert callable(t2.mel_spectogram)


def test_tokenizer_has_encode_plus_after_patch():
    pytest.importorskip("transformers")
    from spkclone.compat import patch_tokenizer_encode_plus

    patch_tokenizer_encode_plus()
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    assert callable(PreTrainedTokenizerBase.encode_plus)
