import cv2
import os

def open_video(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not hasattr(cv2, "VideoCapture"):
        raise RuntimeError("OpenCV build is broken: cv2.VideoCapture missing")
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap

def read_frame(cap):
    ok, frame = cap.read()
    return frame if ok else None

def is_usable(frame, blur_thresh: float = 80.0):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var >= blur_thresh