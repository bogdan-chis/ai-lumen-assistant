import cv2

def open_stream(src=0):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 840)
    return cap

def read_frame(cap):
    ok, frame = cap.read()
    if not ok: return None
    return frame

def is_usable(frame, blur_thresh=80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var >= blur_thresh
