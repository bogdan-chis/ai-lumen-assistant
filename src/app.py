import re
import time
from queue import Queue
import cv2

from .capture import open_video, read_frame, is_usable
from .detect_text import get_full_ocr
from .memory import Timeline
from .tts import speak_ro
from .llm import answer
from .ui import draw_items


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def run(
    video_path: str,
    visualize: bool = True,
    conf_threshold: float = 0.60,
    flush_secs: float = 10.0,
    queue_limit: int = 5,             # trigger LLM call when queue reaches this size
):
    cap = open_video(video_path)
    mem = Timeline()
    printed_cache = set()
    last_flush = time.time()
    text_queue = Queue()

    while True:
        frame = read_frame(cap)
        if frame is None:
            break

        if not is_usable(frame):
            if visualize:
                cv2.imshow("view", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
            continue

        items = get_full_ocr(frame)
        mem.add(items)

        now = time.time()
        if now - last_flush >= flush_secs:
            printed_cache.clear()
            last_flush = now

        for it in items:
            conf = float(it.get("conf", 0.0))
            txt = it.get("text", "").strip()
            if conf < conf_threshold or not txt:
                continue

            key = _norm(txt)
            if not key or key in printed_cache:
                continue

            printed_cache.add(key)
            text_queue.put(txt)

            # when enough text is accumulated, summarize it
            if text_queue.qsize() >= queue_limit:
                snippets = []
                while not text_queue.empty():
                    snippets.append(text_queue.get())

                try:
                    # LLM summarization
                    summary = answer("Ce se află în fața mea?", snippets)
                    print(f"LLM output: {summary}")
                    speak_ro(summary)
                except Exception as e:
                    print(f"Error during LLM or TTS: {e}")

        if visualize:
            vis = draw_items(frame, items, show_conf=False, wrap=True, max_chars=32)
            cv2.imshow("view", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
