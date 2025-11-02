# src/app.py

import re
import time
from pathlib import Path

import cv2

from .capture import open_video, read_frame, is_usable
from .detect_text import get_full_ocr
from .layout import summarize
from .memory import Timeline
from .tts import speak
from .ui import draw_items


def _norm(s: str) -> str:
    """Normalize text for de-duplication."""
    return re.sub(r"\s+", " ", s).strip().lower()


def run(
    video_path: str,
    visualize: bool = True,
    print_text: bool = True,         # print recognized lines to stdout
    log_to_file: bool = False,       # also append to logs/ocr_<ts>.txt
    auto_read_short: bool = False,   # optional short TTS summary per frame
    conf_threshold: float = 0.60,    # min confidence to print/speak
    flush_secs: float = 10.0,        # forget printed cache every N seconds
):
    """
    Main loop:
      - Reads frames from video_path
      - Runs OCR per frame (get_full_ocr)
      - Prints newly seen high-confidence lines
      - Optional visualization with boxes and labels

    Keys:
      q / ESC : quit
    """
    cap = open_video(video_path)
    mem = Timeline()

    printed_cache = set()
    last_flush = time.time()

    log_file = None
    if log_to_file:
        Path("logs").mkdir(exist_ok=True)
        log_file = Path("logs") / f"ocr_{int(time.time())}.txt"

    while True:
        frame = read_frame(cap)
        if frame is None:
            break  # EOF

        if not is_usable(frame):
            if visualize:
                cv2.imshow("view", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
            continue

        # Full OCR for this frame
        items = get_full_ocr(frame)
        mem.add(items)

        # Periodically clear printed cache to allow re-printing after a while
        now = time.time()
        if now - last_flush >= flush_secs:
            printed_cache.clear()
            last_flush = now

        # Print new, high-confidence lines
        if print_text and items:
            for it in items:
                conf = float(it.get("conf", 0.0))
                txt = it.get("text", "").strip()
                if conf < conf_threshold or not txt:
                    continue
                key = _norm(txt)
                if not key or key in printed_cache:
                    continue
                printed_cache.add(key)
                line = f"[{conf:.2f}] {txt}"
                print(line)
                if log_file:
                    with log_file.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")
                        
        # Visualization
        if visualize:
            vis = draw_items(frame, items, show_conf=False, wrap=True, max_chars=32)
            cv2.imshow("view", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break