from .capture import open_stream, read_frame, is_usable
from .detect_text import detect_boxes
from .ocr import ocr_crops
from .layout import summarize
from .memory import Timeline
from .rag import retrieve
from .llm import answer
from .tts import speak
from .ui import draw_boxes
import cv2

def run(video=0, auto_read_short=True):
    cap = open_stream(video)
    mem = Timeline()
    while True:
        frame = read_frame(cap)
        if frame is None: break
        if not is_usable(frame): 
            cv2.imshow("view", frame)
            if cv2.waitKey(1) == 27: break
            continue

        boxes = detect_boxes(frame)
        items = ocr_crops(frame, boxes)
        mem.add(items)

        if auto_read_short and items:
            s = summarize(items, max_len=120)
            if s: speak(s)

        vis = draw_boxes(frame, items)
        cv2.imshow("view", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: break
        if key == ord('a'):  # ask last question over memory
            resp = answer("What is the expiry date?", retrieve("expiry date", mem.get_corpus()))
            speak(resp)
