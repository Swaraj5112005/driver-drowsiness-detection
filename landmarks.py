"""MediaPipe FaceMesh helpers shared by data preparation and the live demo."""
import cv2
import numpy as np
import mediapipe as mp
import config

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_BOX = [61, 291, 0, 17, 13, 14, 78, 308]


class FaceLandmarker:
    def __init__(self, static=False):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def _crop(gray, pts, scale, size=config.IMG_SIZE):
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * scale
    h, w = gray.shape
    xa, xb = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
    ya, yb = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
    if xb - xa < 4 or yb - ya < 4:
        return None
    return cv2.resize(gray[ya:yb, xa:xb], (size, size), interpolation=cv2.INTER_AREA)


def eye_crops(gray, pts):
    crops = [_crop(gray, pts[LEFT_EYE], 2.2), _crop(gray, pts[RIGHT_EYE], 2.2)]
    return [c for c in crops if c is not None]


def mouth_crop(gray, pts):
    return _crop(gray, pts[MOUTH_BOX], 1.6)


def ear(pts, idx):
    p = pts[idx]
    v = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    return v / (2.0 * np.linalg.norm(p[0] - p[3]) + 1e-6)


def mar(pts):
    return np.linalg.norm(pts[13] - pts[14]) / (np.linalg.norm(pts[78] - pts[308]) + 1e-6)
