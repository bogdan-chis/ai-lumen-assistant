from .capture import open_video, read_frame, is_usable
from .detect_text import detect_boxes
from .ocr import ocr_crops
from .layout import summarize
from .memory import Timeline
from .rag import retrieve
from .llm import answer
from .tts import speak
import cv2
from .detect_text import get_full_ocr
from .ui import draw_items

def run(video_path: str, auto_read_short=True, visualize=True):
    cap = open_video(video_path)
    mem = Timeline()

    while True:
        frame = read_frame(cap)
        if frame is None:
            break  # EOF

        if not is_usable(frame):
            if visualize:
                cv2.imshow("Blind assistant", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break
            continue    

        items = get_full_ocr(frame)
        mem.add(items)
        
        if auto_read_short and items:
            s = summarize(items, max_len=120)
            if s:
                speak(s)

        vis = draw_items(frame, items, show_conf=False, wrap=True, max_chars=32)
        cv2.imshow("view", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        if key == ord('a'):
            resp = answer("expiry date", retrieve("expiry date", mem.get_corpus()))
            speak(resp)