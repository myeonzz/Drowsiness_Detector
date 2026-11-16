# Aizu - Driver Drowsiness Monitoring (Intelligent Systems simulation)

A Python simulation of the Aizu drowsiness monitor. No hardware: it runs on a
Windows laptop with a webcam or video files.

Two small CNNs (96x96 grayscale input) do the classification:

| Model | Classes | Input |
|---|---|---|
| eye model | Open / Closed | square crop around each eye |
| mouth model | no_yawn / yawn | square crop around the mouth |

MediaPipe Face Landmarker finds the face, eyes and lips. The same crop code
(`common.py`) is used for training and for live prediction.

Yawns are detected from the lip landmarks first (Mouth Aspect Ratio); the
mouth model is a second check. See "Yawn detection" below.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every path, model setting and alert threshold |
| `common.py` | Face landmarks, crops, preprocessing, augmentation, split, TFLite runner |
| `train.py` | Trains both models, saves `.keras`, float TFLite and INT8 TFLite |
| `evaluate.py` | Test-set metrics and charts for float vs INT8 |
| `alerts.py` | Offense-count alert logic, sounds, alert log, event clip recorder |
| `demo.py` | Real-time demo (webcam or video file) |
| `scenarios.py` | Runs the test videos in `./scenarios`, writes `results/scenarios.csv` |
| `test_alerts.py` | Checks the alert rules with made-up signals |
| `yawn_test.py` | Guided 2-minute webcam test: talking, neutral, 3 yawns; reports false and missed yawns |

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
# 1. Train both models (about 15 minutes on CPU). Add "eye" or "mouth" to train one.
.\.venv\Scripts\python.exe train.py

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

# 6. Guided yawn test on the webcam (about 2 minutes) -> .\results\yawn_test.csv
.\.venv\Scripts\python.exe yawn_test.py
```

To retrain from scratch after changing crop settings, delete the `cache`
folder first.

## Demo keys

| Key | Action |
|---|---|
| `W` | 160-degree mode on/off ("160° MODE" is shown when on) |
| `M` | Mute on/off ("MUTED" is shown when on) |
| `C` | Recalibrate the yawn threshold (5 s, mouth closed) |
| `Q` or `Esc` | Quit |

`--wide` and `--mute` start the demo with those modes already on.

## Demo window

- **Camera view (left):** only the tracked boxes, each with a name tag: `face` (magenta), each eye as `open 0.98` (green) or `closed 0.93` (red), and the mouth as `mouth 0.02` (green) or `yawn 0.71` (red). The number is the model confidence for eyes and the MAR for the mouth. Calibration and test prompts also appear here, and the border flashes red during the alarm.
- **Details panel (right):** the alert level card with its message, then Driver (eyes, mouth, MAR against its threshold, PERCLOS), Offenses (count, kind of each, last one, time to the next drop), System (fps, 160-degree mode, sound) and the keys.

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

## Yawn detection

| Step | What happens |
|---|---|
| MAR | Mouth Aspect Ratio = vertical inner-lip opening / mouth width, from the lip landmarks. About 0.01 with the mouth closed; the dataset's yawn photos have a median of 0.71 |
| Calibration | For the first 5 s the demo shows "CALIBRATING" (keep the mouth closed, normal face) and takes the median MAR as the driver's baseline. Press `C` to redo it |
| Threshold | baseline + `MAR_YAWN_OFFSET` (0.35), never below 0.30. If the mouth was open during calibration (baseline above 0.20) a default baseline of 0.02 is used instead |
| Manual threshold | Set `MAR_THRESHOLD_MANUAL` in `config.py` (for example `0.40`) to skip calibration |
| Mouth CNN | When MAR is above the threshold, the mouth-crop CNN must also say "yawn" (`YAWN_USE_CNN = True`). Set it to `False` to use MAR alone |
| Yawn offense | Yawn frames continuously for 4 s (one other frame in between is allowed) |

The details panel shows the live value as `MAR 0.12 (yawn >= 0.36)`,
plus the CNN score while the mouth is open. No yawns are counted during
calibration. Every frame's MAR is logged to `events\mar_<start time>.csv`.

On the dataset's mouth test photos (`results\yawn_methods.csv`), MAR alone and
"MAR and CNN agree" give the same result: no false yawns and 7 of 107 yawn
photos missed, all with the mouth only partly open (MAR 0.21-0.36). So the CNN neither helps nor
hurts there. On the live webcam, run `yawn_test.py`: its
`frames_vetoed_by_cnn` column shows whether the CNN is blocking real yawns.
If it is, turn it off.

## Outputs

| Path | Content |
|---|---|
| `models\eye_model.keras`, `models\mouth_model.keras` | Float Keras models |
| `models\*_float.tflite`, `models\*_int8.tflite` | TFLite float32 and INT8 models |
| `results\summary.csv` | Accuracy, precision, recall, F1, AUC, size, inference time |
| `results\<model>_confusion_matrix.png` | Confusion matrices (float and INT8) |
| `results\<model>_roc.png` | ROC curves |
| `results\<model>_training_curves.png` | Training / validation accuracy and loss |
| `results\mouth_dropped.json` | yawn / no_yawn images dropped because no face was found |
| `results\yawn_methods.csv` | Yawn decision per test photo: MAR alone, mouth CNN alone, both agreeing |
| `results\yawn_test.csv`, `yawn_test_mar.csv`, `yawn_test_mar.png` | Guided yawn test: false / missed yawns per phase, MAR per frame, MAR plot |
| `results\scenarios.csv` | Expected vs highest level per scenario video |
| `events\*.avi` + `*.json` | 10 s clip (5 s before + 5 s after) and log for each event that results in Level 2 or 3 |
| `events\alerts_<start time>.json` | One log per demo session listing every offense and override: time, level, trigger, reason and offense count |
| `events\mar_<start time>.csv` | One log per demo session with every frame's MAR, threshold, CNN score and yawn decision |

## Alert levels (offense count)

An **offense** is one of:

- eyes closed continuously for 2 s or more (P(Closed) >= 0.7; one open frame
  does not reset the closure), or
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
  oldest offense leaves, which is when the level next drops). PERCLOS (60 s)
  is still shown for information; it no longer triggers anything.
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
- MediaPipe Face Landmarker replaces the Haar cascade. The installed OpenCV 5
  wheel no longer ships the Haar XML files, and using one detector for both
  training crops and live crops keeps them consistent.
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
