#!/usr/bin/env python3
"""Распознаёт голосовые файлы с таймкодами слов -> s2/work/asr/<имя>.json"""
import json, subprocess, sys
import numpy as np
from pathlib import Path
from faster_whisper import WhisperModel
m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device="cpu", compute_type="int8", cpu_threads=4)
out = Path("work/asr"); out.mkdir(parents=True, exist_ok=True)
for f in sys.argv[1:]:
    pcm = np.frombuffer(subprocess.run(["ffmpeg","-v","error","-i",f,"-ac","1","-ar","16000","-f","f32le","-"],capture_output=True).stdout, np.float32)
    segs, info = m.transcribe(pcm, language="ru", word_timestamps=True, vad_filter=False, beam_size=5)
    words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3), "p": round(w.probability, 2)}
             for s in segs for w in s.words]
    (out / (Path(f).stem + ".json")).write_text(json.dumps(words, ensure_ascii=False, indent=0))
    print(f"✓ {Path(f).stem}: {' '.join(w['w'] for w in words)}", flush=True)
