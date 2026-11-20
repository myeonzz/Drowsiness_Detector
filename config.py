"""Aizu - driver drowsiness monitoring simulation.

Every path, model setting and alert threshold lives here so it can be tuned
in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------- paths ---
DATA_DIR = Path(r"D:\Downloads\train")
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
EVENTS_DIR = ROOT / "events"
SCENARIOS_DIR = ROOT / "scenarios"
CACHE_DIR = ROOT / "cache"

# MediaPipe Face Landmarker (the current replacement for "Face Mesh").
# Now used ONLY by the mouth / yawn side (MAR + mouth crop).
# train.py / demo.py download it automatically if it is missing.
FACE_LANDMARKER_PATH = MODELS_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# YuNet face + eye-centre detector (OpenCV). Used for the EYE pipeline instead
# of MediaPipe. Downloaded automatically if missing.
YUNET_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_SCORE = 0.6                  # face detection confidence

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

# Eye dataset options (train.py).
EYE_DATA_FULLFACE = False          # False: images are eye close-ups (used as given)
                                   # True: full-face frames, cropped with YuNet
EYE_GROUP_SEP = "_"                # person id = file name up to the first separator;
                                   # the eye split keeps each person in one set

# Crop geometry (shared by training and the live demo).
FACE_MARGIN = 0.10                 # extra border around the landmark box
MOUTH_CROP_SCALE = 1.5             # square side = scale * mouth outline size
# Eye crops now come from YuNet eye centres: square side = scale * eye-to-eye
# distance. Tune by comparing saved live crops with the training images.
EYE_CROP_SCALE_YUNET = 0.70
EYE_CROP_SHIFT_UP = 0.12           # shift crop up (fraction of side) to include the brow
# Legacy (MediaPipe eye corners); no longer used by the eye pipeline.
EYE_CROP_SCALE = 1.7

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
# Eye P(closed) is smoothed with an exponential moving average before it is
# compared with CLOSED_CONF_MIN: smoothed = a * new + (1 - a) * previous.
# 1.0 = no smoothing; lower = smoother but slower to react.
EYE_SMOOTH_ALPHA = 0.4
OPEN_FRAME_TOLERANCE = 4           # open frames allowed inside a closure (was 1)
YAWN_FRAME_TOLERANCE = 1           # non-yawn frames allowed inside a yawn

# Startup calibration (C key repeats it): the driver keeps the mouth closed
# while the closed-mouth MAR is measured. There is no eye calibration any more.
CALIBRATION_S = 5.0

# Eye closure. The eye CNN is the only eye signal: a frame is "closed" when
# P(Closed) of both eyes is >= CLOSED_CONF_MIN. No MediaPipe / EAR.
EYE_CLOSED_RULE = "cnn"
EYES_CLOSED_WARN_S = 2.0           # red "EYES CLOSED" banner on the camera view after this many seconds
# True: the level card follows the eyes in real time (open eyes -> NORMAL at once;
#       closed 2 s -> CAUTION; closed 3 s -> CRITICAL "WAKE UP").
# False: the original offense-count level, which stays up until offenses leave
#        the 10-minute window.
LEVEL_REALTIME = False

# ---- Legacy EAR settings: NOT used by the new eye pipeline. Kept only so ----
# ---- older scripts (yawn_test.py, scenarios.py) that import them still run. --
EAR_CLOSURE_MIN = 0.80
DROOPY_CLOSURE_MIN = 0.25
EAR_CLOSED_MANUAL = None
EAR_OPEN_DEFAULT = 0.30
EAR_OPEN_MIN = 0.12

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
# If no trained mouth model is found, demo.py falls back to MAR only.
YAWN_USE_CNN = True
YAWN_CNN_MIN = 0.50                # P(yawn) from the mouth CNN needed to agree

# Sunglasses / occlusion check: eyes are "not visible" when the eye region is
# much darker than the rest of the face (median gray ratio below this).
# 0.45 flagged dark sunglasses and no bare eyes on the dataset's car photos.
EYE_DARK_RATIO = 0.45
EYE_DARK_BOX_SCALE_YUNET = 0.375   # checked box side = scale * eye-to-eye distance
EYE_DARK_BOX_SCALE = 0.9           # legacy (MediaPipe eye corners), unused
EYES_UNKNOWN_AFTER_S = 3.0         # no visible eyes for this long -> "unknown"

# ---------------------------------------------------------------- alerts ---
# Offense-count system. Level = number of offenses in the rolling window:
# 0 offenses = Level 0, 1 = Level 1, 2 = Level 2, 3 or more = Level 3.
OFFENSE_CLOSURE_S = 2.0            # offense: eyes closed continuously this long
OFFENSE_YAWN_S = 4.0               # offense: one yawn lasting this long
OFFENSE_WINDOW_S = 600.0           # offenses are counted over the last 10 minutes
MICROSLEEP_S = 3.0                 # override: eyes closed this long -> Level 3 at once
# True: a 3 s closure holds Level 3 only while the eyes stay closed; after they open
# the level falls back to the offense count. False: holds for the whole 10-minute window.
OVERRIDE_ONLY_WHILE_CLOSED = True
ALARM_OPEN_RELEASE_S = 2.0         # Level-3 alarm stops after eyes open this long
ALARM_NO_EYES_S = 5.0              # alarm length when the eyes are not visible (sunglasses)
PERCLOS_WINDOW_S = 60.0            # rolling PERCLOS window (CNN-closed frames), shown on screen only

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