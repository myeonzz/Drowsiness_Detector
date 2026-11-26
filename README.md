# Aizu - Driver Drowsiness Monitoring (Intelligent Systems simulation)

A Python simulation of the Aizu drowsiness monitor. No hardware: it runs on a
Windows laptop with a webcam or video files.

Two small CNNs (96x96 grayscale input) do the classification:

| Model | Classes | Input |
|---|---|---|
| eye model | Open / Closed | square crop around each eye |
| mouth model | no_yawn / yawn | square crop around the mouth |

OpenCV YuNet detects the face in the live demo and locates face-relative eye
and mouth regions. The mouth classifier is trained from the mouth-crop images
in `train\no_yawn` and `train\yawn`.

Yawns are classified by the mouth CNN. See "Yawn detection" below.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every path, model setting and alert threshold |
| `common.py` | YuNet detection and crops, preprocessing, augmentation, split, TFLite runner |
| `train.py` | Trains selected models, saves `.keras`, float TFLite and INT8 TFLite |
| `evaluate.py` | Test-set metrics and charts for float vs INT8 |
| `alerts.py` | Offense-count alert logic, sounds, alert log, event clip recorder |
| `demo.py` | Real-time demo (webcam or video file) |
| `scenarios.py` | Runs the test videos in `./scenarios`, writes `results/scenarios.csv` |
| `test_alerts.py` | Checks the alert rules with made-up signals |
| `yawn_test.py` | Guided 2.5-minute webcam test. Eye check: open, droopy, closed. Yawn check: talking, neutral, 3 yawns |

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
# 1. Train both models. Add "eye" or "mouth" to train one.
.\.venv\Scripts\python.exe train.py
# Retrain the mouth CNN from scratch using only train\no_yawn and train\yawn:
.\.venv\Scripts\python.exe train.py mouth --fresh

# 2. Evaluate -> charts and summary.csv in .\results
.\.venv\Scripts\python.exe evaluate.py

# 3. Check the alert rules (no camera needed)
.\.venv\Scripts\python.exe test_alerts.py

# 4. Live demo with the webcam
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe demo.py --source scenarios\yawning.mp4
.\.venv\Scripts\python.exe demo.py --mute --wide --no-record

# 5. Scenario tests -> .\results\scenarios.csv
.\.venv\Scripts\python.exe scenarios.py
.\.venv\Scripts\python.exe scenarios.py --show

# 6. Guided eye and yawn test on the webcam (about 2.5 minutes)
#    -> .\results\eye_test.csv and yawn_test.csv
.\.venv\Scripts\python.exe yawn_test.py
.\.venv\Scripts\python.exe yawn_test.py --part eyes
.\.venv\Scripts\python.exe yawn_test.py --part yawn
```

`--fresh` rebuilds the selected model's cached crops before training. The
mouth model reads only `train\no_yawn` and `train\yawn`; it starts with new
weights on each training run. YuNet is used by the live demo to find the face
and derive its mouth crop, not to detect faces in these mouth-close-up images.

## Demo keys

| Key | Action |
|---|---|
| `W` | 160-degree mode on/off ("160° MODE" is shown when on) |
| `M` | Mute on/off ("MUTED" is shown when on) |
| `Q` or `Esc` | Quit |

`--wide` and `--mute` start the demo with those modes already on.

## Demo window

- **Camera view (left):** the tracked face, eye and mouth boxes. Eye labels show the CNN's open/closed result and confidence; the mouth label shows the CNN's yawn probability. The border flashes red during the alarm.
- **Details panel (right):** the alert level, eye and mouth CNN results, PERCLOS P80, offense count and history, time until the next offense expires, FPS, 160-degree mode and sound state.

## Camera settings

| Setting | Value |
|---|---|
| Webcam capture | Native resolution: the largest of 1920x1080 / 1280x720 the camera accepts, else its default |
| Processing resolution | `PROCESS_RES = "svga"`: every frame is resized to 800x600 (4:3) before any detection. This is Aizu's processing resolution on the ESP32-S3 |
| Optional preset | `PROCESS_RES = "720p"`: 1280x720, the device's ceiling |
| Model input | Eye and mouth crops are resized to 96x96 grayscale |
| 160-degree mode | Frame is downscaled to 400x300 and scaled back to 800x600 before detection, to mimic the low pixel density of the 160-degree lens |
| Frame rate | Processing is capped at 10 fps; the actual rate is shown on screen |
| Display window | 4:3 camera view with a details panel on its right. Freely resizable, kept at that shape. Window size does not affect detection |
| Event clips | Saved at 640x480 MJPEG: the camera view with its labeled boxes, without the panel |

A 16:9 camera image is centre-cropped to 4:3 (not squashed) before the resize.
Video files go through the same resize, so scenario results match the live demo.

## Eye closure detection

YuNet locates the eye regions, then the eye CNN decides whether they are open
or closed:

| Signal | How it works |
|---|---|
| Eye CNN | Each eye crop is classified; both eyes must have `P(Closed) >= CLOSED_CONF_MIN` (0.70 by default) |
| Smoothing | An exponential moving average (`EYE_SMOOTH_ALPHA` in `config.py`) reduces frame-to-frame score changes |
| PERCLOS (P80) | Share of frames in the last 60 s classified as closed. Shown on screen; it does not trigger alerts |

The offense rules are unchanged: closed frames for 2 s is an offense, and 3 s
is the Level 3 override. Per-frame eye CNN scores, decisions and PERCLOS are
logged to `events\frames_<start time>.csv`.

## Yawn detection

| Step | What happens |
|---|---|
| Face and mouth region | YuNet detects the face; `common.py` derives a mouth crop from the face box |
| Mouth CNN | Classifies the 96x96 grayscale crop as `no_yawn` or `yawn` |
| Threshold | A yawn frame is counted when P(yawn) reaches `YAWN_CNN_MIN` in `config.py` |
| Yawn offense | Yawn frames continuously for 4 s (one other frame in between is allowed) |

The details panel shows the CNN's P(yawn) and the configured threshold. The
per-frame log records the mouth CNN score and yawn decision.

## Outputs

| Path | Content |
|---|---|
| `models\eye_model.keras`, `models\mouth_model.keras` | Float Keras models |
| `models\*_float.tflite`, `models\*_int8.tflite` | TFLite float32 and INT8 models |
| `results\summary.csv` | Accuracy, precision, recall, F1, AUC, size, inference time |
| `results\<model>_confusion_matrix.png` | Confusion matrices (float and INT8) |
| `results\<model>_roc.png` | ROC curves |
| `results\<model>_training_curves.png` | Training / validation accuracy and loss |
| `results\mouth_dropped.json` | Per-class image counts dropped while building the mouth cache |
| `results\yawn_methods.csv` | Held-out test metrics for the mouth CNN |
| `results\eye_test.csv`, `eye_test_closure.png` | Guided eye check: % of frames each rule calls closed in the open, droopy and closed parts; closure plot |
| `results\yawn_test.csv`, `yawn_test_mouth_cnn.png` | Guided yawn check: mouth CNN scores by test phase |
| `results\guided_test_frames.csv` | Every frame's signals from the guided test |
| `results\scenarios.csv` | Expected vs highest level per scenario video |
| `events\*.avi` + `*.json` | 10 s clip (5 s before + 5 s after) and log for each event that results in Level 2 or 3 |
| `events\alerts_<start time>.json` | One log per demo session listing every offense and override: time, level, trigger, reason and offense count |
| `events\frames_<start time>.csv` | One log per demo session with eye and mouth CNN scores, decisions and PERCLOS |

## Alert levels (offense count)

An **offense** is one of:

- eyes closed continuously for 2 s or more (a frame is closed per
  `EYE_CLOSED_RULE`, see "Eye closure detection"; one open frame does not
  reset the closure), or
- a yawn lasting 4 s or more.

Each event is one offense; a long event still counts once. Offenses are
counted over a rolling 10-minute window, and the count sets the level:

| Level | Offenses in the last 10 min | Sound | Voice / text |
|---|---|---|---|
| 0 NORMAL (green) | 0 | none | none |
| 1 CAUTION (yellow) | 1 | one soft two-note chime, once per offense | "Signs of fatigue. Consider a break." |
| 2 WARNING (orange) | 2 | 3 short sharp beeps, repeated every 10 s while at Level 2 | "Drowsiness detected. Pull over soon." |
| 3 CRITICAL (red) | 3 or more, **or** eyes closed continuously for 3 s (override, whatever the count) | continuous loud two-tone alarm; stops only after the eyes are open for 2 s straight | "WAKE UP" (spoken every 3 s during the alarm) |

Rules implemented in `alerts.py`:

- The level drops one step each time the oldest offense leaves the 10-minute window.
- The 3 s override does not add a second offense for the same closure. It
  holds Level 3 until that closure leaves the window; the level then follows
  the offense count again.
- The alarm starts on every event that results in Level 3. The banner stays
  red after the alarm stops, until the level drops.
- If the eyes are not visible for 3 s (dark sunglasses, no face), eye status
  becomes "unknown", the screen shows "Eyes not visible" and only yawns are
  counted. An alarm that starts while the eyes are hidden stops after 5 s,
  because there are no eyes to wait for.
- The screen shows the offense count and a window timer (time until the
  oldest offense leaves, which is when the level next drops). PERCLOS (P80,
  60 s) is shown for information; it does not trigger anything.
- Every offense and override is written to `events\alerts_<start time>.json`
  with time, level, trigger, reason and offense count. Events that result in
  Level 2 or 3 also save a 10 s clip.

The tones are generated with numpy and played with `winsound`, so no audio
files are needed. The text is spoken with `pyttsx3` (Windows voice); if that is
missing or fails, the text is shown on screen only. `VOICE_ENABLED = False`
in `config.py` turns the voice off.

All thresholds are in `config.py`.

## Scenario videos

Put the test clips in `.\scenarios`. The expected level is taken from a
keyword in the file name (`SCENARIO_EXPECTED` in `config.py`):

| File name | Expected highest level |
|---|---|
| `alert.mp4` | 0 |
| `normal_blinking.mp4` | 0 |
| `slow_eye_closure.mp4` | 3 (eyes closed for 3 s or more: override) |
| `yawning.mp4` | 1 (one yawn held for 4 s or more = 1 offense) |
| `sunglasses.mp4` | 0 (eye status should become "unknown") |
| `low_light.mp4` | 0 |

Change the numbers in `SCENARIO_EXPECTED` if a clip is acted differently, for
example a low-light clip with a microsleep should expect 3. Video files are
processed on the video's own clock, so results do not depend on laptop speed.

## Changes from the Kaggle reference notebook

- Two binary models instead of one 4-class model.
- OpenCV YuNet locates faces and eyes in live frames; CNNs classify eye
  closure and yawning.
- 96x96 grayscale input instead of 145x145 RGB.
- Current Keras API (`argmax` on the softmax output, no `predict_classes`).
- The yawn model sees a mouth crop instead of the whole face, with the same
  crop at prediction time. A whole-face yawn model gave false positives on
  the live webcam.
- Stratified 70/15/15 split with a fixed seed; the notebook validated on its test set.
- Augmentation: flip, +/-15 degree rotation, brightness, contrast, gamma,
  slight blur, noise and downscale-then-upscale (low-resolution camera).

## Known limits

- The yawn / no_yawn photos are frames from a few drivers' videos, and the
  file names do not identify the driver. A random split therefore puts the
  same people in train and test, so the mouth model's test scores are
  optimistic. The scenario videos of your own faces are the honest check.
- The sunglasses check compares eye-region brightness with the face. It
  catches dark lenses; mirrored or lightly tinted lenses can pass as eyes.
- Sounds use `winsound`, so they only play on Windows.
- The alarm is played in 0.5 s pieces, so it pulses slightly and can take up
  to about half a second to stop.
