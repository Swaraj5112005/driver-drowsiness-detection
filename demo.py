"""Step 4: live prototype (webcam or video file).
  python demo.py --heuristic                              # no trained models needed (EAR/MAR rules) - test your pipeline first
  python demo.py --eye-model outputs/eye_improved.keras --yawn-model outputs/yawn_improved.keras
  python demo.py --source my_test_video.mp4 --eye-model ... --yawn-model ...
Press q to quit. Alerts are logged to outputs/alerts_log.csv.
"""
import argparse
import csv
from pathlib import Path
import time
import cv2
import numpy as np
from landmarks import FaceLandmarker, eye_crops, mouth_crop, ear, mar, LEFT_EYE, RIGHT_EYE
from temporal import DrowsinessMonitor

COL = {"OK": (0, 200, 0), "CAUTION": (0, 200, 255), "ALERT": (0, 0, 255)}


def beep():
    try:
        import winsound
        winsound.Beep(1200, 300)
    except ImportError:
        print("\a", end="", flush=True)


def prob(model, imgs):
    x = np.stack(imgs).astype("float32")[..., None]
    return float(model(x, training=False).numpy().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam or path to a video")
    ap.add_argument("--eye-model"); ap.add_argument("--yawn-model")
    ap.add_argument("--heuristic", action="store_true", help="use EAR/MAR rules instead of CNNs")
    ap.add_argument("--eye-thr", type=float, default=0.5)
    ap.add_argument("--yawn-thr", type=float, default=0.5)
    a = ap.parse_args()

    eye_m = yawn_m = None
    if not a.heuristic:
        if not (a.eye_model and a.yawn_model):
            raise SystemExit("Give --eye-model and --yawn-model, or use --heuristic")
        from tensorflow import keras
        eye_m, yawn_m = keras.models.load_model(a.eye_model), keras.models.load_model(a.yawn_model)

    is_file = not a.source.isdigit()
    cap = cv2.VideoCapture(a.source if is_file else int(a.source))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {a.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    face, mon = FaceLandmarker(), DrowsinessMonitor()
    last_beep, last_state, no_face, idx = 0.0, "OK", 0, 0
    Path("outputs").mkdir(parents=True, exist_ok=True)
    log = open("outputs/alerts_log.csv", "a", newline=""); w = csv.writer(log)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps if is_file else time.time()
        idx += 1
        pts = face(frame)
        info, msg = None, ""
        if pts is None:
            no_face += 1
            msg = "NO FACE DETECTED" if no_face > fps * 2 else ""
        else:
            no_face = 0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if a.heuristic:
                closed = (ear(pts, LEFT_EYE) + ear(pts, RIGHT_EYE)) / 2 < 0.20
                yawning = mar(pts) > 0.55
                pe = py = None
            else:
                ec, mc = eye_crops(gray, pts), mouth_crop(gray, pts)
                pe = prob(eye_m, ec) if ec else 0.0
                py = prob(yawn_m, [mc]) if mc is not None else 0.0
                closed, yawning = pe >= a.eye_thr, py >= a.yawn_thr
            info = mon.update(t, closed, yawning)
            msg = f"eyes {'CLOSED' if closed else 'open'} | yawn {'YES' if yawning else 'no'}"
            if info["state"] == "ALERT" and last_state != "ALERT":
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), round(t, 2), "|".join(info["reasons"]),
                            round(info["perclos"], 2), info["yawns"]]); log.flush()
            last_state = info["state"]
            if info["state"] == "ALERT" and time.time() - last_beep > 2:
                beep(); last_beep = time.time()

        st = info["state"] if info else "OK"
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), COL[st], 8 if st == "ALERT" else 2)
        cv2.putText(frame, st, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, COL[st], 3)
        if info:
            cv2.putText(frame, f"PERCLOS {info['perclos']:.2f}  closure {info['closure_s']:.1f}s  yawns/min {info['yawns']}",
                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if info["reasons"]:
                cv2.putText(frame, ", ".join(info["reasons"]), (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, msg, (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Driver Drowsiness Monitor (prototype)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows(); log.close()


if __name__ == "__main__":
    main()
