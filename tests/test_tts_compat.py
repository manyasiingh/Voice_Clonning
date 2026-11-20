"""The MSTacotron2 hyperparams.yaml needs `speechbrain.lobes.models.Tacotron2.mel_spectogram` to resolve."""
import pytest

pytest.importorskip("speechbrain")


def test_mel_spectogram_name_resolves_after_patch():
    from spkclone.tts import patch_mel_spectrogram_name

    patch_mel_spectrogram_name()
    import speechbrain.lobes.models.Tacotron2 as t2

    assert callable(t2.mel_spectogram)
