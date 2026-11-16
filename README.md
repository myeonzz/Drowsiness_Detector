# Aizu - Driver Drowsiness Monitoring (Intelligent Systems simulation)

A Python simulation of the Aizu drowsiness monitor. No hardware: it runs on a
Windows laptop with a webcam or video files.

Two small CNNs (96x96 grayscale input) do the classification:

| Model | Classes | Input |
|---|---|---|
| eye model | Open / Closed | square crop around each eye |
| face model | no_yawn / yawn | face crop |

MediaPipe Face Landmarker finds the face and eyes. The same crop code
(`common.py`) is used for training and for live prediction.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every path, model setting and alert threshold |
| `common.py` | Face landmarks, crops, preprocessing, augmentation, split, TFLite runner |
| `train.py` | Trains both models, saves `.keras`, float TFLite and INT8 TFLite |
| `evaluate.py` | Test-set metrics and charts for float vs INT8 |
| `alerts.py` | PERCLOS, 4-level alert logic, sounds, event clip recorder |
| `demo.py` | Real-time demo (webcam or video file) |
| `scenarios.py` | Runs the test videos in `./scenarios`, writes `results/scenarios.csv` |
| `test_alerts.py` | Checks the alert rules with made-up signals |

## Setup (Windows, PowerShell)

Needs Python 3.13 (64-bit). From the project folder:

```powershell
cd C:\Users\Myeonzz\Documents\IS_Finals
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands below call `.\.venv\Scripts\python.exe` directly, so you do not
need to activate the environment (activation is often blocked by PowerShell's
execution policy).

## Run

```powershell
# 1. Train both models (about 15 minutes on CPU). Add "eye" or "face" to train one.
.\.venv\Scripts\python.exe train.py

# 2. Evaluate -> charts and summary.csv in .\results
.\.venv\Scripts\python.exe evaluate.py

# 3. Check the alert rules (no camera needed)
.\.venv\Scripts\python.exe test_alerts.py

# 4. Live demo with the webcam (Q or Esc to quit)
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe demo.py --source scenarios\yawning.mp4
.\.venv\Scripts\python.exe demo.py --mute --no-record

# 5. Scenario tests -> .\results\scenarios.csv
.\.venv\Scripts\python.exe scenarios.py
.\.venv\Scripts\python.exe scenarios.py --show
```

To retrain from scratch after changing crop settings, delete the `cache`
folder first.

## Outputs

| Path | Content |
|---|---|
| `models\eye_model.keras`, `models\face_model.keras` | Float Keras models |
| `models\*_float.tflite`, `models\*_int8.tflite` | TFLite float32 and INT8 models |
| `results\summary.csv` | Accuracy, precision, recall, F1, AUC, size, inference time |
| `results\<model>_confusion_matrix.png` | Confusion matrices (float and INT8) |
| `results\<model>_roc.png` | ROC curves |
| `results\<model>_training_curves.png` | Training / validation accuracy and loss |
| `results\face_dropped.json` | yawn / no_yawn images dropped because no face was found |
| `results\scenarios.csv` | Expected vs highest level per scenario video |
| `events\*.avi` + `*.json` | 10 s clip (5 s before + 5 s after) and log for each Level 2/3 event |

## Alert levels

| Level | Trigger | Action |
|---|---|---|
| 0 NORMAL (green) | - | none |
| 1 CAUTION (yellow) | a yawn of 3 s or more, or 3 yawns in 10 min, or PERCLOS >= 15% | one chime, "Signs of fatigue. Consider a break." |
| 2 WARNING (orange) | PERCLOS >= 25%, or eyes closed >= 1 s twice within 60 s | 3 beeps every 10 s, "Drowsiness detected. Pull over soon." |
| 3 CRITICAL (red) | eyes closed continuously >= 2 s | continuous alarm until eyes are open 2 s, "WAKE UP" |

Rules implemented in `alerts.py`:

- Two Level-3 events within 5 min hold Level 2 as the minimum for 10 min ("Stop driving").
- The level drops one step only after 60 s with no trigger at that level.
- A frame counts as closed only if P(Closed) >= 0.7. One open frame does not reset a closure.
- If the eyes are not visible for 3 s (dark sunglasses, no face), eye status
  becomes "unknown", the screen shows "Eyes not visible" and only the
  yawn-based Level 1 triggers are used.
- PERCLOS needs 10 s of eye data before it can trigger, so one blink in the
  first second does not raise an alert.
- The same trigger cannot log, chime or record again within 60 s (a PERCLOS
  hovering around a threshold would otherwise repeat). Each microsleep always counts.

All thresholds are in `config.py`.

## Scenario videos

Put the test clips in `.\scenarios`. The expected level is taken from a
keyword in the file name (`SCENARIO_EXPECTED` in `config.py`):

| File name | Expected highest level |
|---|---|
| `alert.mp4` | 0 |
| `normal_blinking.mp4` | 0 |
| `slow_eye_closure.mp4` | 3 (eyes closed for 2 s or more) |
| `yawning.mp4` | 1 (hold one yawn for 3 s or more) |
| `sunglasses.mp4` | 0 (eye status should become "unknown") |
| `low_light.mp4` | 0 |

Change the numbers in `SCENARIO_EXPECTED` if a clip is acted differently, for
example a low-light clip with a microsleep should expect 3. Video files are
processed on the video's own clock, so results do not depend on laptop speed.

## Changes from the Kaggle reference notebook

- Two binary models instead of one 4-class model.
- MediaPipe Face Landmarker replaces the Haar cascade. The installed OpenCV 5
  wheel no longer ships the Haar XML files, and using one detector for both
  training crops and live crops keeps them consistent.
- 96x96 grayscale input instead of 145x145 RGB.
- Current Keras API (`argmax` on the softmax output, no `predict_classes`).
- The face crop is applied at prediction time too.
- Stratified 70/15/15 split with a fixed seed; the notebook validated on its test set.
- Augmentation: flip, +/-15 degree rotation, brightness, contrast, gamma,
  slight blur, noise and downscale-then-upscale (low-resolution camera).

## Known limits

- The yawn / no_yawn photos are frames from a few drivers' videos, and the
  file names do not identify the driver. A random split therefore puts the
  same people in train and test, so the face model's test scores are
  optimistic. The scenario videos of your own faces are the honest check.
- The sunglasses check compares eye-region brightness with the face. It
  catches dark lenses; mirrored or lightly tinted lenses can pass as eyes.
- Sounds use `winsound`, so they only play on Windows.
