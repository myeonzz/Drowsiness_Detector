# Automated Test Results

Date: 2026-11-21

## 1. Held-out model test

The evaluated rows are stored in `results/summary.csv`.

| Model | Positive class | Test images | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Eye INT8 | Closed | 920 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| Mouth INT8 | Yawn | 979 | 0.9663 | 0.9746 | 0.9563 | 0.9654 | 0.9929 | 12 | 21 |

The expanded mouth dataset used 6,522 images:
3,316 no-yawn images and 3,206 yawn images. This combines `no_yawn` with
`no_yawn2` and `yawn` with `yawn2`. With a fixed seed of 42, the stratified
70/15/15 split produced 4,565 training images, 978 validation images and
979 held-out test images. The test set contains 498 no-yawn and 481 yawn
images. Yawn is the positive class.

Training ran for all 100 epochs. The lowest validation loss occurred at epoch
92: training accuracy 94.370%, validation accuracy 96.728% and validation loss
0.083239.

Eye target: Closed F1 >= 0.85. **Met (1.000).**

Yawn target: Yawn F1 >= 0.80. **Met (0.9654).**

INT8 accuracy drop:

- Eye: 0.000 percentage points
- Mouth: 0.210 percentage points **improvement** (Float: 96.42%; INT8: 96.63%)
- Target: at most 2 percentage points. **Met for both models.**

Model size comparison:

| Model | Float Keras size | Float TFLite size | INT8 TFLite size |
|---|---:|---:|---:|
| Eye | 1231.14 KB | 385.68 KB | 107.68 KB |
| Mouth | 1231.14 KB | 385.68 KB | 107.68 KB |

INT8 storage reduction from Float Keras: **91.3%**.

Generated figures:

- `results/eye_confusion_matrix.png`
- `results/mouth_confusion_matrix.png`
- `results/eye_roc.png`
- `results/mouth_roc.png`
- `results/eye_training_curves.png`
- `results/mouth_training_curves.png`
- `results/mouth_training_curves.png`

## 2. Alert-rule test

Command:

```powershell
.\.venv\Scripts\python.exe test_alerts.py
```

Result: **14 of 14 tests passed**.

Covered:

- Normal blinking
- Short eye closures
- Short yawns
- 2-second eye offense
- 4-second yawn offense
- Level 1, 2 and 3 transitions
- 3-second microsleep override
- Alarm release after 2 seconds of open eyes
- 10-minute rolling offense window
- Open-frame tolerance
- Yawn-only behavior when eyes are unknown

The two stale alarm expectations were corrected to match the existing
`OVERRIDE_ONLY_WHILE_CLOSED=True` setting.

## 3. Guided live yawn test

Command:

```powershell
.\.venv\Scripts\python.exe yawn_test.py --part yawn
```

Sequence: 30 s talking, 30 s neutral, then three 15-second yawn phases.

| Phase | Duration | Expected yawns | Detected yawns | False yawns | Missed yawns |
|---|---:|---:|---:|---:|---:|
| Talk | 30 s | 0 | 0 | 0 | 0 |
| Neutral | 30 s | 0 | 0 | 0 | 0 |
| Yawn 1 | 15 s | 1 | 0 | 0 | 1 |
| Yawn 2 | 15 s | 1 | 1 | 0 | 0 |
| Yawn 3 | 15 s | 1 | 1 | 0 | 0 |

Live event metrics:

- True detections: 2
- False detections: 0
- Missed detections: 1
- Event precision: 1.000
- Event recall: 0.667
- Event F1: 0.800
- Miss rate: 33.3%
- False alarm rate during 60 s of normal talk/neutral: 0 per hour

The target event recall of at least 90% was **not met in this run**.

The live yawn result should be interpreted as one recorded participant run:

- Talk: 30 seconds with no yawn expected
- Neutral: 30 seconds with no yawn expected
- Yawn 1: 15 seconds, one yawn expected
- Yawn 2: 15 seconds, one yawn expected
- Yawn 3: 15 seconds, one yawn expected

The first yawn phase produced no offense, while the second and third phases
produced one detected yawn each. No false yawn was detected during talking or
while the mouth was neutral. The resulting live score was 2 of 3 expected
yawns detected, or 66.7% event recall.

The current live webcam result was lower than the held-out offline result.
The likely reason is a data-domain mismatch: the public yawn images are close-up
mouth photographs, while the live system sees 800 x 600 webcam frames followed
by a YuNet mouth crop and 96 x 96 grayscale conversion.

Generated:

- `results/yawn_test.csv`
- `results/guided_test_frames.csv`
- `results/yawn_test_mouth_cnn.png`

## 4. Real-time performance

Command was a 60-second no-display webcam run on camera 0.

| Measure | Result |
|---|---:|
| Duration | 60.12 s |
| Frames | 452 |
| Processing FPS | 7.5 |
| Native camera resolution | 1280 x 720 |
| Processing resolution | 800 x 600 |
| Mean CPU | 39.19% |
| Mean RAM | 191.62 MB |

The 10-fps processing limit was not reached in this run. The measured full-pipeline
FPS was 7.5 FPS.

## 5. Tests not completed automatically

Scenario videos:

- No test scenario videos exist in `scenarios/` except `.gitkeep`.
- Run `python scenarios.py` after adding named MP4/AVI/MOV/MKV/WebM files.

Human eye robustness:

- The 41-second webcam eye procedure is implemented in `eye_test.py`.
- It requires a person and should be repeated for glasses, glare, low light,
  sunglasses and head turns.

HCI evaluation:

- The interview and survey findings require human participants and responses.
- This cannot be completed by an automated model or pipeline test.

Online usability:

- 12 drivers and 2 fleet managers are required.
- SUS responses and task-completion answers must come from participants.

No fabricated HCI or SUS values have been inserted.
