#!/usr/bin/env bash
# Скачивает последние N шортсов канала в оригинале (английский звук + английские субтитры)
# Использование: tools/shorts_analysis/download.sh [канал] [N]
set -euo pipefail
CHANNEL="${1:-https://www.youtube.com/@ricksclips3000/shorts}"
N="${2:-20}"
OUT="$(cd "$(dirname "$0")/../.." && pwd)/shorts"
mkdir -p "$OUT"

yt-dlp \
  --playlist-end "$N" \
  -f "bv*[height<=1920]+ba/b" --merge-output-format mp4 \
  --write-info-json --write-thumbnail --convert-thumbnails jpg \
  --write-subs --write-auto-subs --sub-langs "en.*,en" --sub-format vtt \
  -o "$OUT/%(playlist_index)02d_%(id)s.%(ext)s" \
  "$CHANNEL"
