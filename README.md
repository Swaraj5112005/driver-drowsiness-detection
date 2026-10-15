# Driver Drowsiness Detection - Eye State + Yawning (CNN)

## Setup
```
# Windows
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe test_temporal.py         # should print 4 PASS lines

# Linux/macOS
python3.12 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python test_temporal.py
```
The MediaPipe pin uses the legacy `mp.solutions.face_mesh` API. Python 3.14
cannot install a compatible release for that pin; create the environment with
Python 3.10-3.12 instead.
On Windows, install the [Microsoft Visual C++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
if TensorFlow or MediaPipe fails to load with a missing DLL error.
Training is much faster on Google Colab (free GPU). Upload the project folder + datasets there.
Native Windows TensorFlow has no GPU support - use Colab or WSL2 for GPU.

## Datasets
* MRL Eye: https://mrl.cs.vsb.cz/eyedataset.html  (unzip anywhere)
* YawDD: IEEE DataPort, doi 10.21227/e1qm-hb90 (free account) - unzip so .avi files are under one folder

## Pipeline
```
.\venv\Scripts\python.exe prepare_mrl.py   --root "C:\path\to\MRL"
.\venv\Scripts\python.exe prepare_yawdd.py --root "C:\path\to\YawDD"
foreach ($arch in @("simple", "improved", "mobilenet")) {
  .\venv\Scripts\python.exe train.py --manifest data/mrl_manifest.csv  --task eye  --arch $arch
  .\venv\Scripts\python.exe train.py --manifest data/yawn_manifest.csv --task yawn --arch $arch
  .\venv\Scripts\python.exe evaluate.py --manifest data/mrl_manifest.csv  --model "outputs/eye_$arch.keras"
  .\venv\Scripts\python.exe evaluate.py --manifest data/yawn_manifest.csv --model "outputs/yawn_$arch.keras"
}
.\venv\Scripts\python.exe demo.py --heuristic
.\venv\Scripts\python.exe demo.py --eye-model outputs/eye_improved.keras --yawn-model outputs/yawn_improved.keras
```
Outputs (curves, confusion matrices, report JSONs, summary.csv, worst-error lists) go to `outputs/`.

## Important design notes (mention these in your report)
* Splits are by SUBJECT, so no person is in both train and test (prevents inflated accuracy).
* Positive class = 1 = eye CLOSED / YAWNING (the drowsy indicators); recall on class 1 matters most.
* YawDD frame labels are WEAK labels derived from mouth-aspect-ratio + video name. Spot-check ~100 crops.
* MRL images are close-up infrared eye crops; webcam crops come from MediaPipe landmarks, so there is a
  domain gap. Report it honestly and test on your own webcam clips (different lighting, glasses).
* Thresholds (0.5, 1.5 s closure, PERCLOS 0.40) are tunable in `temporal.py` / `demo.py`.

## Suggested timeline (maps to your 6 phases)
Wk1 setup + datasets + manifests | Wk2 baseline CNN | Wk3 improved + MobileNetV2 | Wk4 tuning + demo |
Wk5 evaluation + error analysis + own test videos | Wk6 report + PPT

## Feature ideas
1 Per-driver calibration (first 10 s sets EAR/MAR baseline)   2 Grad-CAM explainability
3 Head-nod / head-pose (solvePnP on FaceMesh)                 4 Distraction (looking away) detection
5 TFLite INT8 + Raspberry Pi/Android latency table            6 Learned temporal model (LSTM / 1D-CNN) vs rule
7 Low-light robustness (CLAHE + dark augmentation, IR test)   8 Streamlit dashboard + trip fatigue report
9 Escalating alerts (beep -> voice -> Telegram/SMS message)   10 Blink-rate / microsleep statistics
