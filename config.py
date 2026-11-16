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
# The mouth model sees only a mouth crop cut from the yawn / no_yawn photos.
MODELS = {
    "eye": ("Open", "Closed"),
    "mouth": ("no_yawn", "yawn"),
}

# Crop geometry (shared by training and the live demo).
FACE_MARGIN = 0.10                 # extra border around the landmark box
EYE_CROP_SCALE = 1.7               # square side = scale * eye-corner distance
EYE_CROP_SHIFT_UP = 0.12           # shift crop up (fraction of side) to include the brow
MOUTH_CROP_SCALE = 1.5             # square side = scale * mouth outline size

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
TARGET_FPS = 10                    # processing is capped at this rate
# Every frame is resized to this resolution before any detection.
#   "svga": 800x600 (4:3) - Aizu's processing resolution on the ESP32-S3
#   "720p": 1280x720      - optional; this is the device's ceiling
PROCESS_RES = "svga"
PROCESS_RES_PRESETS = {"svga": (800, 600), "720p": (1280, 720)}
FRAME_SIZE = PROCESS_RES_PRESETS[PROCESS_RES]
# The webcam is opened at its native resolution: the first of these sizes the
# camera accepts (otherwise its default), then resized to FRAME_SIZE.
CAMERA_NATIVE_TRY = ((1920, 1080), (1280, 720))
# "160-degree mode" (W key): shrink the frame by this factor and scale it back
# up before detection, to mimic the low pixel density of the 160-degree lens
# (800x600 -> 400x300 -> 800x600).
LENS160_SCALE = 0.5
PANEL_WIDTH = 340                  # details panel beside the camera view (pixels)
MODEL_VARIANT = "int8"             # "int8" or "float" TFLite model in the demo

# -------------------------------------------------- per-frame decisions ---
CLOSED_CONF_MIN = 0.70             # count "closed" only if P(Closed) >= this
YAWN_CONF_MIN = 0.70               # count "yawn" only if P(yawn) >= this
OPEN_FRAME_TOLERANCE = 1           # open frames allowed inside a closure
YAWN_FRAME_TOLERANCE = 1           # non-yawn frames allowed inside a yawn

# Startup calibration (C key repeats it): the driver keeps the eyes open and
# the mouth closed while the open-eye EAR and closed-mouth MAR are measured.
CALIBRATION_S = 5.0

# Eye closure. Two signals per frame:
#   cnn: the eye CNN says Closed with confidence >= CLOSED_CONF_MIN
#   ear: Eye Aspect Ratio closure % = 1 - EAR / EAR_open, at or above EAR_CLOSURE_MIN
#        (EAR_open is the driver's own open-eye EAR, so eye shape does not matter)
# EYE_CLOSED_RULE decides how they combine into "this frame is closed":
#   "cnn" | "ear" | "either" (cnn OR ear) | "both" (cnn AND ear)
EYE_CLOSED_RULE = "either"
EAR_CLOSURE_MIN = 0.80             # "80 % closed"; also the PERCLOS (P80) criterion
EAR_CLOSED_MANUAL = None           # e.g. 0.08: closed when EAR <= this, instead of the 80 % rule
EAR_OPEN_DEFAULT = 0.30            # used when calibration is rejected
EAR_OPEN_MIN = 0.15                # calibration is rejected if the eyes were closed / hidden

# Yawn detection. Primary signal: Mouth Aspect Ratio from the lip landmarks
# (MAR = inner-lip opening / mouth width). A yawn frame = MAR at or above the
# threshold. The threshold is set per driver by a short calibration at
# startup (mouth closed, normal face): threshold = baseline + MAR_YAWN_OFFSET.
MAR_THRESHOLD_MANUAL = None        # e.g. 0.40 to skip calibration and force a threshold
MAR_YAWN_OFFSET = 0.35             # threshold = calibrated baseline + this
MAR_THRESHOLD_MIN = 0.30           # the threshold never goes below this
MAR_BASELINE_DEFAULT = 0.02        # used when calibration is rejected
MAR_BASELINE_MAX = 0.20            # calibration is rejected if the mouth was open
# Secondary check: the mouth-crop CNN must also say "yawn" before a frame counts.
YAWN_USE_CNN = True
YAWN_CNN_MIN = 0.50                # P(yawn) from the mouth CNN needed to agree

# Sunglasses / occlusion check: eyes are "not visible" when the eye region is
# much darker than the rest of the face (median gray ratio below this).
# 0.45 flagged dark sunglasses and no bare eyes on the dataset's car photos.
EYE_DARK_RATIO = 0.45
EYE_DARK_BOX_SCALE = 0.9           # checked box side = scale * eye-corner distance
EYES_UNKNOWN_AFTER_S = 3.0         # no visible eyes for this long -> "unknown"

# ---------------------------------------------------------------- alerts ---
# Offense-count system. Level = number of offenses in the rolling window:
# 0 offenses = Level 0, 1 = Level 1, 2 = Level 2, 3 or more = Level 3.
OFFENSE_CLOSURE_S = 2.0            # offense: eyes closed continuously this long
OFFENSE_YAWN_S = 4.0               # offense: one yawn lasting this long
OFFENSE_WINDOW_S = 600.0           # offenses are counted over the last 10 minutes
MICROSLEEP_S = 3.0                 # override: eyes closed this long -> Level 3 at once
ALARM_OPEN_RELEASE_S = 2.0         # Level-3 alarm stops after eyes open this long
ALARM_NO_EYES_S = 5.0              # alarm length when the eyes are not visible (sunglasses)
PERCLOS_WINDOW_S = 60.0            # rolling PERCLOS (P80) window, shown on screen only

LEVEL_NAMES = {0: "NORMAL", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}
LEVEL_COLORS_BGR = {0: (60, 170, 60), 1: (0, 215, 255), 2: (0, 140, 255), 3: (0, 0, 230)}
LEVEL_TEXT = {
    0: "",
    1: "Signs of fatigue. Consider a break.",
    2: "Drowsiness detected. Pull over soon.",
    3: "WAKE UP",
}
EYES_NOT_VISIBLE_TEXT = "Eyes not visible"

# ---------------------------------------------------------------- sounds ---
# Generated tones, louder and harsher at each level (volume is 0-1).
CHIME_HZ, CHIME_MS, CHIME_VOLUME = (880, 1175), 280, 0.25    # Level 1: soft two-note chime
BEEP_HZ, BEEP_MS, BEEP_GAP_MS, BEEP_VOLUME = 1800, 130, 90, 0.6   # Level 2: 3 sharp beeps
BEEP_REPEAT_S = 10.0                                         # ... every 10 s
ALARM_HZ, ALARM_MS, ALARM_VOLUME = (2400, 3000), 250, 1.0    # Level 3: two-tone alarm
VOICE_ENABLED = True               # speak the alert text (needs pyttsx3)
VOICE_RATE = 175                   # words per minute
WAKE_UP_REPEAT_S = 3.0             # "WAKE UP" is spoken this often during the alarm

# ------------------------------------------------------- event recording ---
RECORD_LEVELS = (2, 3)             # only these levels are recorded
CLIP_PRE_S = 5.0
CLIP_POST_S = 5.0
CLIP_SIZE = (640, 480)             # saved clips are scaled to this size
RECORD_OVERLAY = True              # save the camera view with labeled boxes (False = raw frame)

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
