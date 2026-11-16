"""Aizu - driver drowsiness monitoring simulation.

Every path, model setting and alert threshold lives here so it can be tuned
in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------- paths ---
DATA_DIR = ROOT / "train"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
EVENTS_DIR = ROOT / "events"
SCENARIOS_DIR = ROOT / "scenarios"
CACHE_DIR = ROOT / "cache"

# MediaPipe Face Landmarker (the current replacement for "Face Mesh").
# train.py / demo.py download it automatically if it is missing.
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ------------------------------------------------------------------ data ---
SEED = 42
IMG_SIZE = 96                      # 96x96 grayscale input for both models
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15               # -> 70 / 15 / 15 split

# name -> (negative class folder, positive class folder). Index 1 is the
# "drowsy" class, so model output[1] = P(Closed) or P(yawn).
MODELS = {
    "eye": ("Open", "Closed"),
    "face": ("no_yawn", "yawn"),
}

# Crop geometry (shared by training and the live demo).
FACE_MARGIN = 0.10                 # extra border around the landmark box
EYE_CROP_SCALE = 1.7               # square side = scale * eye-corner distance
EYE_CROP_SHIFT_UP = 0.12           # shift crop up (fraction of side) to include the brow

# -------------------------------------------------------------- training ---
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 15
WARMUP_EPOCHS = 20                 # early stopping is off for the first epochs
AUG_ROTATION_DEG = 15              # +/- degrees
AUG_LOWRES_RANGE = (28, 64)        # downscale to this size, then back to 96
REPRESENTATIVE_SAMPLES = 300       # for INT8 calibration

# ------------------------------------------------------------- real time ---
TARGET_FPS = 10
FRAME_SIZE = (640, 480)            # all processing/recording at this size
MODEL_VARIANT = "int8"             # "int8" or "float" TFLite model in the demo

# -------------------------------------------------- per-frame decisions ---
CLOSED_CONF_MIN = 0.70             # count "closed" only if P(Closed) >= this
YAWN_CONF_MIN = 0.70               # count "yawn" only if P(yawn) >= this
OPEN_FRAME_TOLERANCE = 1           # open frames allowed inside a closure
YAWN_FRAME_TOLERANCE = 1           # non-yawn frames allowed inside a yawn

# Sunglasses / occlusion check: eyes are "not visible" when the eye region is
# much darker than the rest of the face (median gray ratio below this).
# 0.45 flagged dark sunglasses and no bare eyes on the dataset's car photos.
EYE_DARK_RATIO = 0.45
EYE_DARK_BOX_SCALE = 0.9           # checked box side = scale * eye-corner distance
EYES_UNKNOWN_AFTER_S = 3.0         # no visible eyes for this long -> "unknown"

# ---------------------------------------------------------------- alerts ---
PERCLOS_WINDOW_S = 60.0            # rolling PERCLOS window
PERCLOS_MIN_DATA_S = 10.0          # need this much eye data before PERCLOS can trigger
PERCLOS_CAUTION = 0.15             # Level 1
PERCLOS_WARNING = 0.25             # Level 2

YAWN_LONG_S = 3.0                  # Level 1: one yawn lasting this long
YAWN_COUNT_MIN_S = 1.0             # a yawn must last this long to be counted
YAWN_COUNT = 3                     # Level 1: this many yawns ...
YAWN_COUNT_WINDOW_S = 600.0        # ... within 10 minutes

LONG_CLOSURE_S = 1.0               # Level 2: closures of this length ...
LONG_CLOSURE_COUNT = 2             # ... at least twice ...
LONG_CLOSURE_WINDOW_S = 60.0       # ... within 60 s

MICROSLEEP_S = 2.0                 # Level 3: eyes closed continuously this long
ALARM_OPEN_RELEASE_S = 2.0         # alarm stops after eyes open this long

ESCALATION_L3_COUNT = 2            # this many Level-3 events ...
ESCALATION_WINDOW_S = 300.0        # ... within 5 min ...
ESCALATION_HOLD_S = 600.0          # ... hold Level 2 minimum for 10 min
DEESCALATE_AFTER_S = 60.0          # drop one level after this long with no trigger
EVENT_COOLDOWN_S = 60.0            # same trigger cannot log/chime/record again sooner

LEVEL_NAMES = {0: "NORMAL", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}
LEVEL_COLORS_BGR = {0: (60, 170, 60), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 230)}
LEVEL_TEXT = {
    0: "",
    1: "Signs of fatigue. Consider a break.",
    2: "Drowsiness detected. Pull over soon.",
    3: "WAKE UP",
}
STOP_DRIVING_TEXT = "Stop driving"
EYES_NOT_VISIBLE_TEXT = "Eyes not visible"

# ---------------------------------------------------------------- sounds ---
CHIME_HZ, CHIME_MS = 880, 350              # Level 1: one soft chime
BEEP_HZ, BEEP_MS, BEEP_REPEAT_S = 1500, 150, 10.0   # Level 2: 3 beeps / 10 s
ALARM_HZ, ALARM_MS = 2800, 500             # Level 3: continuous alarm

# ------------------------------------------------------- event recording ---
RECORD_LEVELS = (2, 3)             # only these levels are recorded
CLIP_PRE_S = 5.0
CLIP_POST_S = 5.0
RECORD_OVERLAY = True              # save the annotated frame (False = raw frame)

# ------------------------------------------------------------- scenarios ---
# Expected highest alert level, matched by keyword in the video file name.
# The first matching keyword wins, so keep the more specific ones first.
SCENARIO_EXPECTED = {
    "normal_blink": 0,
    "blink": 0,
    "slow_eye": 3,
    "closure": 3,
    "microsleep": 3,
    "yawn": 1,
    "sunglasses": 0,
    "low_light": 0,
    "lowlight": 0,
    "alert": 0,
}
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
