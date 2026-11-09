#!/usr/bin/env python3
"""Слова озвучки S3 с таймкодами (Whisper) → s3/work/asr/<файл>.json. Нужно для субтитров точно в голос.
  pip install faster-whisper
  python3 tools/s3blender/transcribe_s3.py s3/voice/s3e1_*.mp3
"""
import json, subprocess, sys, os
import numpy as np
from pathlib import Path
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "s3" / "work" / "asr"
out.mkdir(parents=True, exist_ok=True)
m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=os.cpu_count())
for f in sys.argv[1:]:
    pcm = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", f, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                                       capture_output=True).stdout, np.float32)
    segs, _ = m.transcribe(pcm, language="ru", word_timestamps=True, beam_size=5)
    words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)} for s in segs for w in s.words]
    (out / (Path(f).stem + ".json")).write_text(json.dumps(words, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"✓ {Path(f).stem}: {' '.join(w['w'] for w in words)}", flush=True)
