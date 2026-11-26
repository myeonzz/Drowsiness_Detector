# Aizu System Testing Guide

This guide defines how to collect evidence for the model, alert logic, live
pipeline, robustness, HCI, usability, real-time performance and IoT requirements.
Use the actual result from each command or test. Do not copy example numbers into
the report.

## 1. Before testing

Run commands from the repository folder:

```powershell
cd "C:\Users\lyzette dominguez\Documents\GitHub\Drowsiness_Detector"
.\.venv\Scripts\python.exe check_camera.py
.\.venv\Scripts\python.exe test_alerts.py
```

Record the camera index selected by `check_camera.py`. The alert-rule test must
print `14 tests passed`.

Create these folders if needed:

```powershell
New-Item -ItemType Directory -Force results, events, scenarios
```

Use one condition label for each live test and repeat the test more than once.
Examples:

- `no glasses, normal light`
- `glasses, normal light`
- `no glasses, glare`
- `glasses, low light`
- `sunglasses, normal light`
- `head turn, normal light`

## 2. Train and evaluate the models

Clear the cache before retraining:

```powershell
Remove-Item -Recurse -Force cache
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
```

Use these forms to test one model at a time:

```powershell
.\.venv\Scripts\python.exe train.py eye
.\.venv\Scripts\python.exe train.py mouth
.\.venv\Scripts\python.exe evaluate.py eye
.\.venv\Scripts\python.exe evaluate.py mouth
```

The output and generated files are:

- `results/summary.csv`
- `results/eye_confusion_matrix.png`
- `results/mouth_confusion_matrix.png`
- `results/eye_roc.png`
- `results/mouth_roc.png`
- `results/eye_training_curves.png`
- `results/mouth_training_curves.png`
- `results/yawn_methods.csv`

For each model, record:

- Test-image count
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion-matrix counts: TN, FP, FN, TP
- Keras size and inference time
- Float TFLite size and inference time
- INT8 TFLite size and inference time

Targets:

| Model | Positive class | Target |
|---|---|---|
| Eye | Closed | F1 >= 0.85 |
| Mouth | Yawn | F1 >= 0.80 |
| Both | INT8 accuracy | Drop <= 2 percentage points from float |

Calculate:

```text
INT8 accuracy drop = float TFLite accuracy - INT8 TFLite accuracy
Size reduction = (1 - INT8 size / float Keras size) x 100
```

Do not use a 100% or 1.000 result unless it appears in `summary.csv`.

## 3. Test the alert rules

Run:

```powershell
.\.venv\Scripts\python.exe test_alerts.py
```

The automated checks cover:

- Normal blinking
- Short closures
- Low-confidence closures
- 2-second eye offense
- 4-second yawn offense
- Level 1, 2 and 3 transitions
- 3-second microsleep override
- Alarm stop timing
- 10-minute offense window
- One-frame closure tolerance
- Sunglasses/yawns-only behavior

Record the passing count and save the terminal output in the test evidence folder.

## 4. Test scenario videos

Place named videos in `scenarios/`:

```text
alert.mp4
normal_blinking.mp4
slow_eye_closure.mp4
yawning.mp4
sunglasses.mp4
low_light.mp4
```

Run:

```powershell
.\.venv\Scripts\python.exe scenarios.py
.\.venv\Scripts\python.exe scenarios.py --show
```

Use `--record` only if Level 2/3 clips are required:

```powershell
.\.venv\Scripts\python.exe scenarios.py --record
```

The expected levels are:

| Scenario | Expected highest level |
|---|---:|
| Alert/normal | 0 |
| Normal blinking | 0 |
| Slow eye closure | 3 |
| Yawning | 1 |
| Sunglasses | 0 |
| Low light | 0 |

Record in `results/scenarios.csv`:

- Expected level
- Highest level reached
- Time to highest level
- Trigger types
- Correct/incorrect result

For event scoring, count one continuous closure or yawn as one event. Calculate:

```text
Event precision = detected true events / all detected events
Event recall = detected true events / all expected events
Event F1 = 2 x precision x recall / (precision + recall)
Miss rate = missed events / expected events
False alarms per hour = false alarms / normal-test-minutes x 60
```

Targets:

- Event recall: at least 90%
- False alarms: at most 1 per hour
- Time to alert after a 2-second eye closure: at most 1 second
- Expected scenario level matched

## 5. Run the guided eye test

Run one condition at a time:

```powershell
.\.venv\Scripts\python.exe eye_test.py --label "no glasses, normal light"
.\.venv\Scripts\python.exe eye_test.py --label "glasses, normal light"
.\.venv\Scripts\python.exe eye_test.py --label "no glasses, glare"
.\.venv\Scripts\python.exe eye_test.py --label "glasses and glare"
```

The sequence is:

| Phase | Duration | Required action |
|---|---:|---|
| Ready | 6 s | Eyes open, neutral mouth |
| Eyes open | 20 s | Look at the camera and blink normally |
| Droopy | 10 s | Hold a half-closed eye position |
| Eyes closed | 5 s | Keep both eyes shut |

Ignore the first 1.5 seconds of every measured phase. Record:

- Frames counted
- Frames with no eyes found
- Closed-frame percentage
- Mean P(Closed)
- Saved plot

Outputs:

- `results/eye_test_live.csv`
- `results/eye_test_<condition>.png`

Repeat the test at least three times for the normal condition and for each
major condition.

## 6. Collect more live yawn and no-yawn data

The existing public folders already contain 2,528 yawn images and 2,591
no-yawn images. However, live webcam images come from a different visual domain.
Collect 300 or more images of each state using the live crop method:

```powershell
.\.venv\Scripts\python.exe collect_mouth_dataset.py --label no_yawn --count 300
.\.venv\Scripts\python.exe collect_mouth_dataset.py --label yawn --count 300
```

The collector:

- Detects the face with YuNet
- Extracts the same mouth crop used by the live demo
- Converts it to 96 x 96 grayscale
- Saves it under `train\no_yawn` or `train\yawn`
- Captures one crop every 200 ms

Collection procedure:

1. Sit 40–60 cm from the webcam.
2. Keep a neutral expression for the no-yawn capture.
3. Keep a visible yawn mouth opening for the yawn capture.
4. Vary face angle, lighting and distance across the session.
5. Do not include frames during the 3-second countdown.

After collecting data:

```powershell
Remove-Item cache/mouth.npz -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe train.py mouth
.\.venv\Scripts\python.exe evaluate.py mouth
```

Keep the local data balanced. For example, add approximately the same number
of no-yawn and yawn images.

## 7. Run the guided yawn test

Run:

```powershell
.\.venv\Scripts\python.exe yawn_test.py --part yawn
```

If eye and yawn testing are combined, run:

```powershell
.\.venv\Scripts\python.exe yawn_test.py
```

The sequence is:

| Phase | Duration | Required action |
|---|---:|---|
| Talk | 30 s | Talk continuously but do not yawn |
| Neutral | 30 s | Keep the mouth closed |
| Yawn 1 | 15 s | Yawn widely and hold it for at least 4 s |
| Yawn 2 | 15 s | Yawn widely and hold it for at least 4 s |
| Yawn 3 | 15 s | Yawn widely and hold it for at least 4 s |

The yawn prompt can be started by saying a phrase, counting aloud or reading
text. Mark the actual yawn start times in the run notes.

Record:

- False yawns during talk
- False yawns during neutral state
- Missed yawns
- Detected yawns
- Mouth-CNN P(Yawn) peak
- Longest continuous yawn decision
- Number of frames above the 0.50 threshold
- Start and end time of each detected offense
- Time from first sustained yawn decision to offense

Outputs:

- `results/yawn_test.csv`
- `results/guided_test_frames.csv`
- `results/yawn_test_mouth_cnn.png`

Repeat the complete yawn test at least three times. Report mean event precision,
recall and F1, rather than a single run.

## 8. Run robustness tests

Use the same recorded procedure for every condition:

1. Run the eye test.
2. Run the yawn test.
3. Repeat each test at least three times.
4. Record the same normal condition as the baseline.

Recommended conditions:

| Condition | Eye test | Yawn test |
|---|---|---|
| Normal, no glasses | Required | Required |
| Glasses | Required | Optional |
| Sunglasses | Required | Required |
| Low light | Required | Required |
| Glare | Required | Optional |
| Head turn | Required | Required |
| Different participants | Required | Required |

For each condition, calculate:

```text
Delta accuracy = condition accuracy - normal accuracy
Delta recall = condition recall - normal recall
False alarm rate = false alarms / measured normal duration in hours
```

Do not add a condition to the report until all repetitions have a saved CSV,
plot and run label.

## 9. Measure real-time performance

Run a stable, uninterrupted demo:

```powershell
.\.venv\Scripts\python.exe demo.py --mute --no-record
```

Record for at least 5 minutes:

- Mean FPS
- FPS standard deviation
- Minimum FPS
- Mean eye-model inference time
- Mean mouth-model inference time
- End-to-end frame processing time
- CPU percentage
- RAM usage
- Resolution
- Processing mode: normal or 160-degree

End-to-end latency should be measured by timestamping frame receipt and the
completed detection alert. Do not use per-model inference time as the frame
throughput or full-pipeline latency.

Measure model size from `summary.csv` or with:

```powershell
Get-ChildItem models/*.tflite, models/*.keras |
  Select-Object Name, @{Name="Size_KB";Expression={[math]::Round($_.Length/1KB,2)}}
```

## 10. Check the 160-degree mode

Run:

```powershell
.\.venv\Scripts\python.exe demo.py --mute --wide --no-record
```

Record one normal-light run and one low-light run. Compare:

- Detection recall
- False alarm rate
- Mean FPS
- Eye and mouth confidence
- Number of frames with no face or no eyes

The wide mode is only the simulation of the 160-degree camera. It does not
replace testing the board camera.

## 11. Alert, sound and recording check

Test without the microphone or with microphone access disabled.

Verify:

- 2-second eye closure creates one caution
- 3-second eye closure creates the critical override
- 4-second yawn creates one caution
- Repeated frames do not create repeated offenses for the same event
- 10-minute window removes old offenses
- Level 2 and Level 3 clips contain 5 seconds before and 5 seconds after
- A no-eyes/missing-face event can timeout the alarm
- Speech alerts stop when the alarm condition stops
- Audio clips are not recorded

Save the alert log and generated clips under `events/`.

## 12. HCI evaluation

Evaluate the prototype against the interview and survey findings. Create a table
with:

| Finding | Prototype response | Met? | Evidence |

Mark every finding as `Yes`, `Partly` or `No`. Use direct quotes or a short
observation as evidence.

The test should cover:

- Recording consent and privacy
- Adjustable alert settings
- Night/dark displays
- Offline trip and clip storage
- Rest reminders
- Fleet Level 3 notification
- Call-driver action
- False-alarm reporting
- Manager permissions and privacy

Do not report all findings as met unless each one has evidence.

## 13. Online usability test

Recruit:

- 12 drivers for the 68-point SUS target
- 2 fleet managers

Record participant ID, user type, task result and open comments.

For SUS, use this formula:

```text
SUS = (sum of odd-item responses - 5
       + 25 - sum of even-item responses) x 2.5
```

Where odd items use `response - 1` and even items use `5 - response`.

Report:

- Driver mean SUS
- Driver minimum and maximum
- Each fleet manager SUS
- Task completion: Yes
- Task completion: Yes, with difficulty
- Task completion: No
- Main open-answer problems
- Changes made after feedback

Tasks for drivers:

- Set up the device
- Find the alert-volume setting
- Find the last Level 2 alert
- Play an alert clip
- Delete an alert clip
- Find a nearby rest stop

Tasks for fleet managers:

- Respond to a Level 3 alert
- Open an incident record
- Find the driver with the most offenses
- Mark an event as a false alarm
- Find this week's report

Use participant results rather than the test administrator's opinion.
The online test measures self-reported task completion, not hands-on driving
time or error rate.

## 14. IoT requirements traceability

For each requirement, record:

| Requirement | Implemented part | Evidence | Status |

Use `Met`, `Partly met` or `Not met`.

Key requirements:

- Detect eye closure and yawning
- Work in day and night conditions
- Alert offline within approximately 1 second
- Store only event clips
- Use 96x96 grayscale model input
- Run as an affordable car add-on
- Do not record audio

Do not mark a requirement as met only because it is implemented in software.
Link it to a test result or measured artifact.

## 15. Limitations and conclusion

State the actual limitations supported by the tests:

- Models use public datasets, not a local Filipino-driver dataset
- The eye/mouth split may not separate every person
- Infrared and visible-light camera behavior may differ
- Sunglasses, masks and extreme head turns may hide facial features
- Yawns shorter than 4 seconds are intentionally not offenses
- Online usability measures reported task completion, not live driving behavior
- Real board FPS and resource use must be measured on the board

The conclusion should answer each project question using measured results:

1. Can the models classify Closed and Yawn?
2. Do the models meet their F1 targets?
3. Does INT8 preserve accuracy and reduce size?
4. Does the pipeline detect events and alert at the required speed?
5. Does the HCI design meet the findings?
6. Is the prototype easy to use?
7. Does the IoT design fit within flash, PSRAM and storage limits?
