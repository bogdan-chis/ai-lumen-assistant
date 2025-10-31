import cv2

def draw_boxes(img, items):
    out = img.copy()
    for it in items:
        x1,y1,x2,y2 = it["box"]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 1)
    return out