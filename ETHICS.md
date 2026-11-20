# Ethics

**Purpose.** This is a defensive evaluation: it measures how vulnerable a speaker-verification system is to
synthetic speech, so that such systems can be deployed (or not) with an honest picture of the risk.

**Data.** Only public datasets whose speakers recorded for public release: LibriSpeech (OpenSLR SLR12; the page
states "License: CC BY 4.0") and the LibriTTS-trained SpeechBrain models. TODO: confirm the LibriTTS licence text on
its own page before the report is written. No voices of real, named people are cloned, and the repo contains no demo
that impersonates a named individual.

**What is (not) shipped.** The repo is an evaluation pipeline (generate clones of dataset speakers, score them,
report acceptance rates). It is not a one-click "clone anyone's voice" tool, and no cloned audio is committed.
The `clone` CLI that existed before this project is kept for compatibility; only use it on voices you have
permission to use.

**Limits of the conclusions.** One TTS system (MSTacotron2, labelled work-in-progress by its authors), one verifier
family (ECAPA-TDNN), English read speech. A low attack acceptance rate here does not mean a deployed system is safe
against better TTS systems; a high one does not generalise to every verifier.
