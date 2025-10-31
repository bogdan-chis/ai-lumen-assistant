from paddleocr import PaddleOCR

_ocr_det = PaddleOCR(use_angle_cls=True, det=True, rec=False, lang='en')

def detect_boxes(img):
    result = _ocr_det.ocr(img, det=True, rec=False)
    # Flatten and return boxes as x1,y1,x2,y2...
    boxes = []
    for line in result:
        for (poly, _) in line:
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return boxes
