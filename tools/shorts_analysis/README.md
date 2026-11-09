# Покадровый разбор шортсов

1. `tools/shorts_analysis/download.sh https://www.youtube.com/@ricksclips3000/shorts 20` — последние 20 шортсов в оригинале (англ. звук + англ. субтитры) в `shorts/`.
2. `python3 -I tools/shorts_analysis/analyze.py shorts/` — разбор КАЖДОГО кадра на 30 fps.

Нужно: `pip install yt-dlp opencv-python-headless numpy`, ffmpeg. Сети нужен доступ к youtube.com и *.googlevideo.com.

Результат в `shorts/analysis/`: `<ролик>.json` (планы, камера, свет/тени, цвет, звук, текст, покадровые ряды), `<ролик>_sheet.jpg` (лист планов), `summary.md` (сводка по всем роликам).

Проверено на синтетических клипах: панорама (0.54 против истинных 0.555 ширины кадра), наезд (0.464 против ln 1.59 = 0.464), тряска, склейки внутри одной сцены.

Дополнительно:
- `python3 -I tools/shorts_analysis/transitions.py <папка>` — все смены картинки, включая наплывы и пролёты камеры (полоса субтитров и HUD исключены).
- `python3 -I tools/shorts_analysis/compare.py shorts/ref shorts/our > shorts/compare.md` — медиана референсов против наших.
- Анализатор работает параллельно на всех ядрах.
