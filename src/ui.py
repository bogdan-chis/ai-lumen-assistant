import cv2
import math
from typing import List, Dict, Tuple

def _fit_font(img_w: int, img_h: int) -> Tuple[int, float, int]:
    # scale text size with image size
    base = min(img_w, img_h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, base / 2000.0)
    thickness = max(1, int(base / 800))
    return font, scale, thickness

def _put_label_with_bg(img, text, org, font, scale, thickness, txt_color=(0,0,0), bg_color=(0,255,0)):
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    # background rectangle
    cv2.rectangle(img, (x, y - th - baseline), (x + tw, y + baseline), bg_color, -1)
    # text
    cv2.putText(img, text, (x, y), font, scale, txt_color, thickness, cv2.LINE_AA)

def draw_items(img_bgr, items: List[Dict], show_conf: bool = True, wrap: bool = True, max_chars: int = 40):
    """
    Draws boxes and text on a copy of the frame.
    items: list of {"box": (x1,y1,x2,y2), "text": str, "conf": float}
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    font, scale, th = _fit_font(w, h)

    for it in items:
        x1, y1, x2, y2 = it["box"]
        txt = it.get("text", "").strip()
        conf = it.get("conf", 0.0)
        label = txt if not show_conf else f"{txt}  ({conf:.2f})"

        # draw box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), max(1, th))

        # optional wrapping to avoid off-screen labels
        display_lines = [label]
        if wrap and len(label) > max_chars:
            display_lines = []
            s = label
            while len(s) > max_chars:
                display_lines.append(s[:max_chars])
                s = s[max_chars:]
            if s:
                display_lines.append(s)

        # draw stacked labels at the top-left of the box
        y_text = max(0, y1 - 4)
        for i, line in enumerate(reversed(display_lines)):  # draw last line closest to box
            # compute size to offset each line
            (tw, th_text), base = cv2.getTextSize(line, font, scale, th)
            y_text -= (th_text + base + 2)
            if y_text < 0:
                y_text = y1 + (i+1)*(th_text + base + 4)  # fallback: draw inside box
            _put_label_with_bg(out, line, (x1, y_text), font, scale, th, txt_color=(0,0,0), bg_color=(0,255,0))

    return out