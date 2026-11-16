"""Shared helpers: face landmarks, cropping, preprocessing, augmentation,
dataset split and TFLite inference. Used by train / evaluate / demo so the
exact same crop is applied at training and prediction time."""
import os
import urllib.request

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np

import config

# Landmark indices (MediaPipe face mesh topology): eye corners.
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)


# ----------------------------------------------------------- face finder ---
def ensure_landmarker_model():
    path = config.FACE_LANDMARKER_PATH
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe face landmarker -> {path}")
        urllib.request.urlretrieve(config.FACE_LANDMARKER_URL, path)
    return path


class FaceFinder:
    """MediaPipe Face Landmarker wrapper. Returns landmarks in pixels."""

    def __init__(self, video=False):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._video = video
        model_bytes = ensure_landmarker_model().read_bytes()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def find(self, bgr, t=None):
        """Return (N, 2) float array of landmark pixel coords, or None."""
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            ts = max(int((t or 0) * 1000), self._last_ts + 1)  # must increase
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        h, w = bgr.shape[:2]
        return np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float32)

    def close(self):
        self._landmarker.close()


# -------------------------------------------------------------- cropping ---
def _square_box(cx, cy, side, shape):
    """Clip a square box to the image. Returns (x0, y0, x1, y1) or None."""
    h, w = shape[:2]
    half = side / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def face_box(landmarks, shape):
    x0, y0 = landmarks.min(axis=0)
    x1, y1 = landmarks.max(axis=0)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * config.FACE_MARGIN)
    return _square_box((x0 + x1) / 2, (y0 + y1) / 2, side, shape)


def eye_boxes(landmarks, shape):
    """Square boxes around each eye (eye + brow, like the training close-ups)."""
    boxes = []
    for a, b in (RIGHT_EYE_CORNERS, LEFT_EYE_CORNERS):
        pa, pb = landmarks[a], landmarks[b]
        side = float(np.linalg.norm(pa - pb)) * config.EYE_CROP_SCALE
        cx, cy = (pa + pb) / 2
        boxes.append(_square_box(cx, cy - side * config.EYE_CROP_SHIFT_UP, side, shape))
    return boxes


def crop(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def preprocess(img):
    """Any BGR/gray crop -> 96x96 uint8 grayscale (the model input)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if img.shape[0] >= config.IMG_SIZE else cv2.INTER_LINEAR
    return cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=interp)


def eyes_look_occluded(gray_frame, landmarks, boxes):
    """Sunglasses check: both eye openings much darker than the face.
    Also True when an eye crop falls outside the frame."""
    fb = face_box(landmarks, gray_frame.shape)
    if fb is None or any(b is None for b in boxes):
        return True
    face_med = float(np.median(crop(gray_frame, fb))) + 1e-6
    ratios = []
    for a, b in (RIGHT_EYE_CORNERS, LEFT_EYE_CORNERS):
        pa, pb = landmarks[a], landmarks[b]
        side = float(np.linalg.norm(pa - pb)) * config.EYE_DARK_BOX_SCALE
        cx, cy = (pa + pb) / 2
        tight = _square_box(cx, cy, side, gray_frame.shape)   # eye opening only
        if tight is None:
            return True
        ratios.append(float(np.median(crop(gray_frame, tight))) / face_med)
    return max(ratios) < config.EYE_DARK_RATIO


# ---------------------------------------------------------- augmentation ---
def augment(img, rng):
    """Augment one 96x96 uint8 gray image: flip, +/-15 deg rotation, brightness /
    contrast / gamma, slight blur, noise and low-res (downscale then upscale)."""
    s = config.IMG_SIZE
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    angle = rng.uniform(-config.AUG_ROTATION_DEG, config.AUG_ROTATION_DEG)
    m = cv2.getRotationMatrix2D((s / 2, s / 2), angle, rng.uniform(0.9, 1.1))
    m[:, 2] += rng.uniform(-0.06, 0.06, size=2) * s
    img = cv2.warpAffine(img, m, (s, s), borderMode=cv2.BORDER_REPLICATE)

    x = img.astype(np.float32) / 255.0
    x = (x - 0.5) * rng.uniform(0.6, 1.4) + 0.5 + rng.uniform(-0.2, 0.2)  # contrast, brightness
    x = np.clip(x, 0, 1) ** rng.uniform(0.5, 2.0)                          # gamma
    if rng.random() < 0.3:
        x = x + rng.normal(0, rng.uniform(0.01, 0.03), x.shape)            # sensor noise
    img = (np.clip(x, 0, 1) * 255).astype(np.uint8)

    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.5:                                                 # low-res camera
        low = int(rng.integers(config.AUG_LOWRES_RANGE[0], config.AUG_LOWRES_RANGE[1] + 1))
        img = cv2.resize(cv2.resize(img, (low, low), interpolation=cv2.INTER_AREA),
                         (s, s), interpolation=cv2.INTER_LINEAR)
    return img


# --------------------------------------------------------------- dataset ---
def load_split(name):
    """Load cached 96x96 data for a model and return the fixed 70/15/15 split:
    dict with (X, y) tuples under 'train', 'val', 'test'."""
    from sklearn.model_selection import train_test_split

    data = np.load(config.CACHE_DIR / f"{name}.npz")
    X, y = data["X"], data["y"]
    hold = config.VAL_FRACTION + config.TEST_FRACTION
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=hold, stratify=y, random_state=config.SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=config.TEST_FRACTION / hold, stratify=y_tmp,
        random_state=config.SEED)
    return {"train": (X_tr, y_tr), "val": (X_val, y_val), "test": (X_te, y_te)}


def model_path(name, variant):
    """variant: 'keras', 'float' (TFLite float32) or 'int8' (TFLite INT8)."""
    suffix = {"keras": ".keras", "float": "_float.tflite", "int8": "_int8.tflite"}[variant]
    return config.MODELS_DIR / f"{name}_model{suffix}"


# ---------------------------------------------------------------- TFLite ---
class TFLiteModel:
    """Runs a float or INT8 TFLite model on 96x96 uint8 gray images and
    returns float probabilities [P(negative), P(positive)]."""

    def __init__(self, path):
        import tensorflow as tf

        import warnings

        with warnings.catch_warnings():          # hide the LiteRT migration notice
            warnings.simplefilter("ignore")
            self._interp = tf.lite.Interpreter(model_path=str(path), num_threads=2)
        self._interp.allocate_tensors()
        self._inp = self._interp.get_input_details()[0]
        self._out = self._interp.get_output_details()[0]

    def predict(self, img):
        x = img.reshape(1, config.IMG_SIZE, config.IMG_SIZE, 1).astype(self._inp["dtype"])
        self._interp.set_tensor(self._inp["index"], x)
        self._interp.invoke()
        y = self._interp.get_tensor(self._out["index"])[0]
        scale, zero = self._out["quantization"]
        if scale:                                    # de-quantize INT8 output
            y = (y.astype(np.float32) - zero) * scale
        return y.astype(np.float32)
