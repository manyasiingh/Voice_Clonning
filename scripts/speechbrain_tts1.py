#!/usr/bin/env python3
"""Thin wrapper that keeps the original TTS CLI working (same modes and flags, plus --no_textnorm):

  python scripts/speechbrain_tts1.py clone  --ref my_voice.wav --text "Hello, this is a cloned voice." --verify --plot
  python scripts/speechbrain_tts1.py random --text "Mary had a little lamb." "It was a sunny day."
  python scripts/speechbrain_tts1.py single --text_file sentences.txt --out out_single
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spkclone.tts import main  # noqa: E402

if __name__ == "__main__":
    main()
