# Can zero-shot TTS voice clones fool an ECAPA-TDNN speaker verifier?

*Evaluating cloning fidelity, contrastive fine-tuning, and explainability with SpeechBrain.*

> **Status:** tests pass on Kaggle (61/61; encoder reproduces SpeechBrain's pretrained model exactly).
> RQ1 has been run; RQ2-RQ4 are not run yet. Every number below comes from a file in `results/` produced by
> `notebooks/kaggle_run.ipynb`.

**Finding so far (RQ1):** on held-out LibriSpeech speakers the off-the-shelf VoxCeleb ECAPA-TDNN reaches 1.71% EER,
and supervised-contrastive fine-tuning on LibriSpeech train-clean-100 did **not** improve it: in all three seeds the
dev EER never dropped below its starting value, so model selection kept the pretrained weights.

## Summary

Speaker verification (SV) is used for voice authentication. Zero-shot text-to-speech can imitate a voice from a few
seconds of audio. This project uses SpeechBrain end to end to (a) build and calibrate an ECAPA-TDNN verifier,
(b) generate cloned speech of held-out LibriSpeech speakers with MSTacotron2 + HiFi-GAN, (c) measure how close the
clones are to the real speaker and how often the verifier accepts them, and (d) look at what the verifier is
listening to. It is a **defensive evaluation on public, consented data** (see [ETHICS.md](ETHICS.md)).

## Research questions

| | Question | Status |
|---|---|---|
| RQ1 | How well does the pretrained VoxCeleb ECAPA-TDNN separate held-out LibriSpeech speakers, and does SupCon fine-tuning change EER / minDCF? (pretrained vs fine-tuned vs scratch-small, 3 seeds, bootstrap CIs) | run; final-epoch evaluation of fine-tuned weights TODO |
| RQ2 | How similar are clones to the real target, compared with real-same, real-different and random-voice pairs? | TODO: `eval_clone.py` |
| RQ3 | At thresholds calibrated on bona fide dev trials only, what fraction of clone trials is accepted? | TODO: `eval_clone.py` |
| RQ4 | (Exploratory) Do occlusion / attribution patterns differ between bona fide and clone trials? Is the attribution faithful? | explain code ready, TODO: aggregation run |

## Results

### RQ1 — verifier quality

LibriSpeech test-clean: 40 speakers, 15 utterances each (6 s centred crops), all 179,700 pairs (4,200 target).
The same trials are used for every model. Thresholds are calibrated on dev-clean (40 other speakers).
Source: [`results/rq1/table.md`](results/rq1/table.md) and `results/rq1/<run>/test_metrics.json`.

| Model (checkpoint chosen by lowest dev EER) | Seeds | Test EER % | Speaker-bootstrap 95% CI of EER % | minDCF (p=0.01) | Selected epoch |
|---|---|---|---|---|---|
| Pretrained ECAPA-TDNN (VoxCeleb), no fine-tuning | – | 1.71 | 0.81–2.83 | 0.122 | – |
| + SupCon fine-tuning (T=0.1, P=16, K=4, 10 × 200 steps) | 3 | 1.71 ± 0.00 | 0.81–2.83 (each seed) | 0.122 | 0, 0, 0 |
| Small ECAPA-TDNN from scratch (SupCon) | 3 | 5.06 ± 0.20 | 3.17–7.10, 3.40–7.57, 3.14–7.31 | 0.443 ± 0.042 | 10, 10, 10 |

± is the standard deviation over seeds.
- **Fine-tuning row:** "Selected epoch 0" means no fine-tuning epoch beat the initial model on dev, so this row *is* the
  pretrained model. It is not an independent result.
- **Fine-tuning on dev:** EER started at 1.74%, and its lowest value after any epoch was 1.76% / 1.78% / 1.86%
  (seeds 0/1/2). After 10 epochs it was 1.88% / 1.91% / 1.88%.
- **Ablations (seed 0 only):** after 10 epochs, dev EER was 3.64% at temperature 0.05, 2.79% at temperature 0.2, and 1.90% at K=2 with P=32.
  None was selected over the initial model either. Source: `results/rq1/*/train_history.json`.
- **Scratch:** the small model's dev EER was still falling at epoch 10 (7.90–8.17%), so it is under-trained, not converged.
- **Not yet measured:** the test EER of the fine-tuned weights after the last epoch (not dev-selected). That is
  the next notebook cell (`results/rq1/<run>/final_epoch/`).

Operating points of the pretrained model on test-clean, at thresholds set on dev-clean
(`at_dev_eer_threshold`, `at_dev_far1_threshold` in `results/rq1/pretrained/test_metrics.json`):

| Threshold (from dev) | Value | Test false-accept rate | Test false-reject rate |
|---|---|---|---|
| dev EER point | 0.313 | 1.27% | 2.05% |
| dev FAR = 1% point | 0.337 | 0.74% | 2.60% |

Trial-level bootstrap CIs are much narrower than speaker-level ones: pretrained EER 1.51–2.02% (trials) vs 0.81–2.83%
(speakers). The speaker-level interval is the honest one, since it reflects that there are only 40 test speakers.

## How to reproduce

1. Push this repo to GitHub.
2. On Kaggle: New Notebook -> File -> Import notebook -> `notebooks/kaggle_run.ipynb`; set Accelerator = GPU,
   Internet = On; edit `REPO_URL` in the first code cell.
3. Run all. Long cells are resumable; re-run after a timeout. Download `/kaggle/working/results.zip` at the end.

The same code runs on any machine with a GPU:

```bash
pip install -r requirements.txt && pip install -e . --no-deps
python -m pytest -s
python -m spkclone.verify evaluate --ckpt pretrained --out results/rq1/pretrained
python -m spkclone.train --config configs/rq1_finetune.yaml --seed 0 --out runs/rq1/finetune_s0 --results results/rq1/finetune_s0
```

The original commands still work through thin wrappers: `scripts/speaker_verification_contrastive.py`
(`train` / `verify` / `explain`, plus a new `evaluate`) and `scripts/speechbrain_tts1.py` (`clone` / `random` / `single`).

Runtime and GPU (first run, `results/env/versions.json` + run start times in `results/rq1/*/run_meta.json`):
Tesla T4, PyTorch 2.11, SpeechBrain 1.1.2. About 18 min per fine-tuning run and about 8 min per scratch run
(10 epochs × 200 steps each); all RQ1 training (9 runs) took about 2.3 GPU hours.

## Limitations (known before running anything)

- **Trial protocol.** All pairs of a fixed sample of utterances per held-out speaker; many pairs come from the same
  recording session, which is expected to be easier than VoxCeleb-style cross-session trials. On this sample the
  cross-session-only subset was not harder for the pretrained model (EER 1.62%, 1,665 target trials, vs 1.71% for
  all pairs), but it is still read speech in clean conditions. These EERs must **not** be compared with published
  VoxCeleb numbers.
- **One fine-tuning recipe.** Fine-tuning used a single learning rate (5e-5) and 10 × 200 steps. A null result for
  this recipe does not show that no fine-tuning recipe helps. VoxCeleb1-O is supported via
  `--trials` only if you obtain the data yourself (it needs registration).
- **"Unseen" speakers not confirmed.** The MSTacotron2 model card says only that it was trained on LibriTTS, not which
  subsets, so we cannot yet confirm that LibriSpeech test-clean speakers were unseen by the TTS model.
- **Shared encoder family.** MSTacotron2 conditions on `speechbrain/spkrec-ecapa-voxceleb-mel-spec`; the verifier is
  `speechbrain/spkrec-ecapa-voxceleb`. They are different checkpoints of the same model family, which may make clones
  look more similar to this verifier than to an unrelated one.
- One TTS system, one verifier family, English read speech only. The TTS model is labelled work-in-progress by its authors.
- 40 test speakers: confidence intervals are bootstrapped over speakers and will be wide.

## What I would do next

- Write `eval_clone.py` (RQ2 + RQ3) and the RQ4 aggregation, then run them.
- RQ1: try a gentler fine-tuning recipe (e.g. lower learning rate, frozen early layers) to test whether the null
  result is about the recipe or about the domain (the pretrained model is already strong on clean read speech).
- A second, independent verifier (e.g. a non-ECAPA model) to test the shared-encoder-family concern.
- Intelligibility of clones via ASR WER (model to be looked up, not guessed).
- Phase 2: Hindi / Hinglish extension - test whether the verifier transfers to Hindi and code-switched speech,
  and how much speaker identity survives in cross-lingual clones from Hindi-capable TTS systems.

## Repo map

```
src/spkclone/   encoder, data, losses, metrics, train, verify, explain, tts, textnorm, compat, utils
scripts/        original-CLI wrappers, aggregate_rq1.py, make_notebook.py
configs/        YAML per experiment (smoke, rq1_finetune, rq1_scratch)
tests/          pytest: metrics, loss, sampler/data, textnorm, checkpoints, encoder equivalence (slow)
notebooks/      kaggle_run.ipynb - reproduces every result
results/        small json/csv/png/npz written by the scripts (committed)
```
