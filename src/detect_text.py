import cv2
import numpy as np
import torch
import easyocr

# Use GPU if available; otherwise CPU.
_GPU = torch.cuda.is_available()
_reader = easyocr.Reader(['ro'], gpu=_GPU)

def _to_xyxy(bbox):
    # bbox: 4 points [[x,y], ...] -> (x1,y1,x2,y2)
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    return (min(xs), min(ys), max(xs), max(ys))

def detect_boxes(img, min_conf=0.5):
    """
    Returns list of (x1, y1, x2, y2) boxes for regions with text.
    """
    # EasyOCR expects RGB
    img_rgb = img[:, :, ::-1] if img.shape[-1] == 3 else img
    results = _reader.readtext(
        img_rgb,
        paragraph=False,
        text_threshold=0.4,     # tweakable
        low_text=0.3,           # tweakable
        link_threshold=0.3,     # tweakable
        slope_ths=0.2,          # improves on tilted text
        detail=1                # return [bbox, text, conf]
    )
    boxes = []
    for bbox, text, conf in results:
        if not text or float(conf) < min_conf:
            continue
        boxes.append(_to_xyxy(bbox))
    return boxes

def get_full_ocr(img, min_conf=0.5):
    """
    Returns list of dicts: {"box": (x1,y1,x2,y2), "text": str, "conf": float}
    """
    img_rgb = img[:, :, ::-1] if img.shape[-1] == 3 else img
    results = _reader.readtext(
        img_rgb,
        paragraph=False,
        text_threshold=0.4,
        low_text=0.3,
        link_threshold=0.3,
        slope_ths=0.2,
        detail=1
    )
    items = []
    for bbox, text, conf in results:
        conf = float(conf)
        if not text or conf < min_conf:
            continue
        items.append({
            "box": _to_xyxy(bbox),
            "text": text.strip(),
            "conf": conf
        })
    return items
