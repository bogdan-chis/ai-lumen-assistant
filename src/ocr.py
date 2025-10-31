from paddleocr import PaddleOCR

_rec = PaddleOCR(use_angle_cls=True, det=False, rec=True, lang='en')

def ocr_crops(img, boxes):
    texts = []
    for (x1,y1,x2,y2) in boxes:
        crop = img[y1:y2, x1:x2]
        res = _rec.ocr(crop, det=False, rec=True)
        if res and res[0]:
            txt = " ".join([r[0][0] for r in res[0]])
            conf = float(res[0][0][1])
            texts.append({"box": (x1,y1,x2,y2), "text": txt, "conf": conf})
    return texts
