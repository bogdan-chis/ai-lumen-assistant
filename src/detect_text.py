from rapidocr_onnxruntime import RapidOCR

# Force CUDA first, then CPU fallback
_rapid = RapidOCR(providers=["CUDAExecutionProvider", "CPUExecutionProvider"], text_score=0.5)

def detect_boxes(img):
    result, _ = _rapid(img)
    boxes = []
    for (box, txt, score) in result or []:
        xs = [int(p[0]) for p in box]; ys = [int(p[1]) for p in box]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes

def get_full_ocr(img):
    result, _ = _rapid(img)
    items = []
    for (box, txt, score) in result or []:
        xs = [int(p[0]) for p in box]; ys = [int(p[1]) for p in box]
        items.append({"box": (min(xs), min(ys), max(xs), max(ys)),
                      "text": txt, "conf": float(score)})
    return items
