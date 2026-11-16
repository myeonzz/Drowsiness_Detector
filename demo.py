"""Real-time drowsiness demo on a webcam or a video file.

    python demo.py                       # webcam 0
    python demo.py --source 1            # another webcam
    python demo.py --source scenarios\\yawning.mp4
    python demo.py --mute --no-record

Keys:  W = 160-degree mode on/off   M = mute on/off
       C = recalibrate the yawn threshold   Q or Esc = quit
"""
import argparse
import csv
import time
from datetime import datetime

import cv2
import numpy as np

import common
import config as cfg
from alerts import AlertEngine, AlertLog, EventRecorder, SoundPlayer

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "Aizu - drowsiness monitor"


class Pipeline:
    """frame -> face landmarks -> eye crops -> eye model (eyes closed?)
                               -> lip landmarks -> MAR, optionally confirmed by
                                  the mouth-crop CNN (yawning?)."""

    def __init__(self, variant=None):
        variant = variant or cfg.MODEL_VARIANT
        self.finder = common.FaceFinder(video=True)
        self.eye_model = common.TFLiteModel(common.model_path("eye", variant))
        self.mouth_model = (common.TFLiteModel(common.model_path("mouth", variant))
                            if cfg.YAWN_USE_CNN else None)
        self.mar_threshold = cfg.MAR_THRESHOLD_MANUAL     # None until calibrated
        self.mar_baseline = None
        self.calibration_note = "manual" if self.mar_threshold is not None else ""
        self._calib_start, self._calib = None, []

    def recalibrate(self):
        """Start the 5 s MAR calibration again (C key)."""
        self.mar_threshold, self.mar_baseline = None, None
        self._calib_start, self._calib = None, []

    def _calibrate(self, mar, t):
        """Collect closed-mouth MAR for a few seconds, then set the threshold.
        Returns the seconds of calibration left."""
        if self._calib_start is None:
            self._calib_start = t
        self._calib.append(mar)
        left = cfg.MAR_CALIBRATION_S - (t - self._calib_start)
        if left > 0:
            return left
        baseline = float(np.median(self._calib))
        if baseline > cfg.MAR_BASELINE_MAX:   # mouth was open: do not trust it
            baseline, self.calibration_note = cfg.MAR_BASELINE_DEFAULT, "default (mouth was open)"
        else:
            self.calibration_note = "calibrated"
        self.mar_baseline = baseline
        self.mar_threshold = max(baseline + cfg.MAR_YAWN_OFFSET, cfg.MAR_THRESHOLD_MIN)
        return 0.0

    def process(self, frame, t):
        """Returns p_closed / p_yawn (None when not measurable), MAR and boxes.
        p_yawn is 1.0 / 0.0: the yawn decision for this frame."""
        out = {"face_box": None, "eye_boxes": [], "eye_probs": [], "mouth_box": None,
               "p_closed": None,
               "p_yawn": None, "mar": None, "mar_threshold": self.mar_threshold,
               "p_mouth_cnn": None, "calibrating": None, "face_found": False,
               "occluded": False}
        landmarks = self.finder.find(frame, t)
        if landmarks is None:
            return out
        out["face_found"] = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out["face_box"] = common.face_box(landmarks, frame.shape)

        # ---- yawn: MAR first, mouth CNN must agree if it is enabled ----
        mar = out["mar"] = common.mouth_aspect_ratio(landmarks)
        out["mouth_box"] = common.mouth_box(landmarks, frame.shape)
        if self.mar_threshold is None:
            out["calibrating"] = self._calibrate(mar, t)
            out["mar_threshold"] = self.mar_threshold
            out["p_yawn"] = 0.0               # no yawns are counted while calibrating
        else:
            yawning = mar >= self.mar_threshold
            if yawning and self.mouth_model is not None and out["mouth_box"] is not None:
                mouth = common.preprocess(common.crop(gray, out["mouth_box"]))   # 96x96 gray
                out["p_mouth_cnn"] = float(self.mouth_model.predict(mouth)[1])
                yawning = out["p_mouth_cnn"] >= cfg.YAWN_CNN_MIN
            out["p_yawn"] = 1.0 if yawning else 0.0

        boxes = common.eye_boxes(landmarks, frame.shape)
        out["eye_boxes"] = [b for b in boxes if b is not None]
        out["occluded"] = common.eyes_look_occluded(gray, landmarks, boxes)
        if not out["occluded"]:
            probs = [self.eye_model.predict(common.preprocess(common.crop(gray, b)))[1]
                     for b in boxes]
            out["eye_probs"] = [float(p) for p in probs]
            out["p_closed"] = float(np.min(probs))       # both eyes must look closed
        else:
            out["eye_probs"] = [None] * len(out["eye_boxes"])
        return out

    def close(self):
        self.finder.close()


# ----------------------------------------------------------------- frames ---
def fit_frame(frame, size):
    """Centre-crop to the target aspect ratio, then resize (no distortion)."""
    w, h = size
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    if fw * h > fh * w:                      # source is wider: trim the sides
        nw = int(round(fh * w / h))
        x = (fw - nw) // 2
        frame = frame[:, x:x + nw]
    elif fw * h < fh * w:                    # source is taller: trim top/bottom
        nh = int(round(fw * h / w))
        y = (fh - nh) // 2
        frame = frame[y:y + nh]
    interp = cv2.INTER_AREA if frame.shape[1] > w else cv2.INTER_LINEAR
    return cv2.resize(frame, (w, h), interpolation=interp)


def lens160(frame):
    """160-degree mode: throw away detail (e.g. 800x600 -> 400x300 -> 800x600)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (int(w * cfg.LENS160_SCALE), int(h * cfg.LENS160_SCALE)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def to_clip(frame):
    """Scale a frame into the 640x480 clip size (black bars if not 4:3)."""
    w, h = cfg.CLIP_SIZE
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    if (nw, nh) == (w, h):
        return resized
    canvas = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def open_camera(index):
    """Open a webcam at its native resolution (largest size it accepts)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return cap, None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # full fps at high res
    for w, h in cfg.CAMERA_NATIVE_TRY:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if got == (w, h):
            break
    ok, frame = cap.read()
    return cap, ((frame.shape[1], frame.shape[0]) if ok else None)


# ------------------------------------------------------------------- GUI ---
# The window is the camera view (boxes with labels only) plus a details panel
# on its right, outside the picture.
PANEL_BG = (32, 28, 26)
PANEL_LINE = (70, 64, 60)
TEXT_MAIN = (240, 240, 240)
TEXT_DIM = (160, 160, 160)
BOX_FACE = (255, 0, 255)                       # magenta
BOX_OK = (0, 220, 0)                           # green: eye open / mouth normal
BOX_BAD = (0, 0, 255)                          # red: eye closed / yawning
BOX_UNKNOWN = (170, 170, 170)                  # gray: not classified


def _put(img, text, org, scale=0.55, color=TEXT_MAIN, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _outlined(img, text, org, scale, color, thick=2):
    """Text readable on top of the video: black copies shifted by 1 px behind
    it. (A thicker stroke changes letter spacing in OpenCV 5.)"""
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _labeled_box(img, box, label, color, scale=0.5):
    """Rectangle with a filled name tag on top, like an object detector."""
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    (tw, th), base = cv2.getTextSize(label, FONT, scale, 1)
    tag_h = th + base + 6
    ty = y0 - tag_h if y0 - tag_h >= 0 else y1          # under the box at the top edge
    tx = max(0, min(x0 - 1, img.shape[1] - tw - 8))
    cv2.rectangle(img, (tx, ty), (tx + tw + 8, ty + tag_h), color, -1)
    dark = sum(color) > 380                              # dark text on light tags
    cv2.putText(img, label, (tx + 4, ty + th + 3), FONT, scale,
                (0, 0, 0) if dark else (255, 255, 255), 1, cv2.LINE_AA)


def draw_camera(frame, det, state, note=""):
    """Camera view: only the tracked boxes with their labels, plus the
    messages that need the driver's attention (calibration, test prompts)."""
    img = frame.copy()
    h, w = img.shape[:2]

    if det["face_box"]:
        _labeled_box(img, det["face_box"], "face", BOX_FACE, 0.55)
    for box, p in zip(det["eye_boxes"], det["eye_probs"]):
        if p is None:
            _labeled_box(img, box, "eye ?", BOX_UNKNOWN)
        elif p >= cfg.CLOSED_CONF_MIN:
            _labeled_box(img, box, f"closed {p:.2f}", BOX_BAD)
        else:
            _labeled_box(img, box, f"open {1 - p:.2f}", BOX_OK)
    if det["mouth_box"]:
        if det["mar_threshold"] is None:
            _labeled_box(img, det["mouth_box"], f"mouth {det['mar']:.2f}", BOX_UNKNOWN)
        elif det["p_yawn"]:
            _labeled_box(img, det["mouth_box"], f"yawn {det['mar']:.2f}", BOX_BAD)
        else:
            _labeled_box(img, det["mouth_box"], f"mouth {det['mar']:.2f}", BOX_OK)

    if note:
        _outlined(img, note, (14, h - 56), 0.9, (0, 255, 255))
    if det["calibrating"]:
        _outlined(img, f"CALIBRATING {det['calibrating']:.0f} s - mouth closed, normal face",
                  (14, h - 20), 0.75, (0, 255, 255))
    if state["alarm"] and int(state["t"] * 4) % 2 == 0:       # flashing border
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
    return img


def _wrap(text, scale, thick, max_w):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and cv2.getTextSize(trial, FONT, scale, thick)[0][0] > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


def draw_panel(height, det, state, fps, wide=False, muted=False):
    """Details panel shown beside the camera view."""
    pw = cfg.PANEL_WIDTH
    img = np.full((height, pw, 3), PANEL_BG, np.uint8)
    pad, y = 16, 0

    # ---- alert level card ----
    color = cfg.LEVEL_COLORS_BGR[state["level"]]
    cv2.rectangle(img, (0, 0), (pw, 112), color, -1)
    _put(img, f"LEVEL {state['level']}", (pad, 34), 0.7, (0, 0, 0), 2)
    _put(img, cfg.LEVEL_NAMES[state["level"]], (pad, 68), 0.95, (0, 0, 0), 2)
    for i, line in enumerate(_wrap(state["text"], 0.5, 1, pw - 2 * pad)[:2]):
        _put(img, line, (pad, 90 + 18 * i), 0.5, (0, 0, 0), 1)
    y = 112

    def section(title):
        nonlocal y
        y += 26
        _put(img, title, (pad, y), 0.5, TEXT_DIM, 1)
        cv2.line(img, (pad, y + 8), (pw - pad, y + 8), PANEL_LINE, 1)
        y += 8

    def row(label, value, value_color=TEXT_MAIN):
        nonlocal y
        y += 22
        _put(img, label, (pad, y), 0.52, TEXT_DIM)
        _put(img, value, (pad + 118, y), 0.52, value_color)

    def line(text, color=TEXT_MAIN):
        nonlocal y
        y += 20
        _put(img, text, (pad + 8, y), 0.5, color)

    red, green, yellow = (90, 90, 255), (110, 220, 110), (0, 215, 255)

    # ---- driver ----
    section("DRIVER")
    if det["p_closed"] is not None:
        p = det["p_closed"]
        closed = p >= cfg.CLOSED_CONF_MIN
        row("Eyes", f"{'Closed' if closed else 'Open'}  {max(p, 1 - p):.2f}",
            red if closed else green)
    elif state["eyes_unknown"]:
        row("Eyes", cfg.EYES_NOT_VISIBLE_TEXT, yellow)
    else:
        row("Eyes", "--")
    if det["mar"] is None:
        row("Mouth", "Face not found", yellow)
    elif det["mar_threshold"] is None:
        row("Mouth", "Calibrating...", yellow)
        row("MAR", f"{det['mar']:.2f}")
    else:
        row("Mouth", "YAWN" if det["p_yawn"] else "Normal", red if det["p_yawn"] else green)
        row("MAR", f"{det['mar']:.2f}  (yawn >= {det['mar_threshold']:.2f})")
        if det["p_mouth_cnn"] is not None:
            row("Mouth CNN", f"{det['p_mouth_cnn']:.2f}")
    row("PERCLOS 60 s", f"{state['perclos'] * 100:.1f}%")

    # ---- offenses ----
    section(f"OFFENSES  (last {cfg.OFFENSE_WINDOW_S / 60:g} min)")
    row("Count", str(state["offense_count"]))
    kinds = state["offense_kinds"]
    for i, kind in enumerate(kinds[-3:], start=max(len(kinds) - 3, 0) + 1):
        line(f"{i}. {kind}")                     # what kind each offense was
    if state["override"]:                        # why it is Level 3 below 3 offenses
        line(f"+ Override: eyes closed {cfg.MICROSLEEP_S:g} s", red)
    if state["last_alert"]:
        label, ago = state["last_alert"]
        row("Last", f"{label}, {int(ago) // 60}:{int(ago) % 60:02d} ago")
    left = state["window_remaining_s"]
    row("Next drop in", "--" if left is None else f"{int(left) // 60}:{int(left) % 60:02d}")

    # ---- system ----
    section("SYSTEM")
    row("FPS", f"{fps:.1f}")
    row("160 deg mode", "ON" if wide else "off",
        yellow if wide else TEXT_DIM)
    row("Sound", "MUTED" if muted else "on", yellow if muted else TEXT_DIM)

    # ---- keys ----
    _put(img, "W 160 mode   M mute", (pad, height - 34), 0.48, TEXT_DIM)
    _put(img, "C calibrate   Q quit", (pad, height - 14), 0.48, TEXT_DIM)
    cv2.line(img, (0, 0), (0, height), PANEL_LINE, 1)
    return img


def _keep_aspect(view_w, view_h):
    """The window may be resized freely; snap its height back to the view's
    aspect ratio (4:3 camera + panel). Display size never affects detection."""
    try:
        _, _, w, h = cv2.getWindowImageRect(WINDOW)
    except cv2.error:
        return
    want = int(round(w * view_h / view_w))
    if w > 0 and abs(h - want) > 2:
        cv2.resizeWindow(WINDOW, w, want)


# ------------------------------------------------------------------- main ---
class MarLog:
    """Per-frame MAR log: events/mar_<start>.csv."""

    def __init__(self, out_dir=None):
        out_dir = out_dir or cfg.EVENTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"mar_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self._file = open(self.path, "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["time_s", "mar", "threshold", "mouth_cnn", "yawn_frame",
                            "calibrating"])

    def add(self, t, det):
        fmt = lambda v: "" if v is None else f"{v:.4f}"
        self._csv.writerow([f"{t:.2f}", fmt(det["mar"]), fmt(det["mar_threshold"]),
                            fmt(det["p_mouth_cnn"]),
                            "" if det["p_yawn"] is None else int(det["p_yawn"]),
                            int(bool(det["calibrating"]))])

    def close(self):
        self._file.close()


def run(source=0, display=True, sound=True, muted=False, record=True, events_dir=None,
        variant=None, verbose=True, wide=False, max_seconds=None, on_frame=None):
    """Run the full pipeline on a webcam index or video path. Returns a summary
    (highest level, time it was reached, events). Video files use the video's
    own clock, so results do not depend on how fast this laptop is.
    on_frame(t, det, state) is called for every processed frame; the text it
    returns (if any) is shown on screen."""
    live = isinstance(source, int)
    if live:
        cap, native = open_camera(source)
        step, src_fps = 1, None
    else:
        cap, native = cv2.VideoCapture(str(source)), None
    if not cap.isOpened() or (live and native is None):
        cap.release()
        raise RuntimeError(f"Cannot open video source: {source}")
    if live:
        if verbose:
            print(f"Camera {source} opened at {native[0]}x{native[1]}; "
                  f"processing at {cfg.FRAME_SIZE[0]}x{cfg.FRAME_SIZE[1]}")
    else:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps != src_fps or src_fps > 240:
            src_fps = 30.0
        step = max(1, round(src_fps / cfg.TARGET_FPS))     # sample at ~10 fps

    pipeline = Pipeline(variant)
    engine = AlertEngine()
    player = SoundPlayer(enabled=sound, muted=muted)
    if verbose and sound:
        print("Voice alerts: " + ("on (pyttsx3)" if player.voice_available
                                  else "not available, text only"))
    recorder = EventRecorder(events_dir, source=str(source)) if record else None
    alert_log = AlertLog(events_dir, source=str(source)) if record else None
    mar_log = MarLog(events_dir) if record else None
    announced = False
    if display:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, cfg.FRAME_SIZE[0] + cfg.PANEL_WIDTH, cfg.FRAME_SIZE[1])

    period = 1.0 / cfg.TARGET_FPS                          # 10 fps cap
    start = last_wall = time.monotonic()
    fps, frame_idx, processed, t, all_events = 0.0, -1, 0, 0.0, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % step:
                continue
            t = time.monotonic() - start if live else frame_idx / src_fps
            if max_seconds is not None and t > max_seconds:
                break
            frame = fit_frame(frame, cfg.FRAME_SIZE)       # before any detection
            if wide:
                frame = lens160(frame)
            det = pipeline.process(frame, t)
            state = engine.update(t, det["p_closed"], det["p_yawn"])
            processed += 1

            now = time.monotonic()
            if processed > 1:
                inst = 1.0 / max(now - last_wall, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            last_wall = now

            if mar_log:
                mar_log.add(t, det)
            if verbose and not announced and pipeline.mar_threshold is not None:
                announced = True
                base = ("" if pipeline.mar_baseline is None
                        else f", baseline {pipeline.mar_baseline:.3f}")
                print(f"Yawn threshold: MAR >= {pipeline.mar_threshold:.3f} "
                      f"({pipeline.calibration_note}{base})")
            note = on_frame(t, det, state) if on_frame else ""
            camera = draw_camera(frame, det, state, note or "")      # boxes + labels
            if display:                                              # camera | details
                shown = np.hstack([camera, draw_panel(camera.shape[0], det, state, fps,
                                                      wide, player.muted)])
            player.update(state)
            if recorder:
                recorder.add_frame(t, to_clip(camera if cfg.RECORD_OVERLAY else frame))
            for ev in state["events"]:
                all_events.append(ev)
                if alert_log:
                    alert_log.add(ev)
                if recorder:
                    recorder.add_event(ev)
                if verbose:
                    print(f"[{ev['time']:7.1f}s] Level {ev['level']} "
                          f"{cfg.LEVEL_NAMES[ev['level']]}: {ev['reason']} "
                          f"(PERCLOS {ev['perclos'] * 100:.1f}%)")

            if display:
                cv2.imshow(WINDOW, shown)
                _keep_aspect(shown.shape[1], shown.shape[0])
                # hold 10 fps: webcam -> fixed period, video -> follow video clock
                target = (last_wall + period) if live else (start + t + period)
                wait_ms = max(1, int((target - time.monotonic()) * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("w"), ord("W")):
                    wide = not wide
                if key in (ord("m"), ord("M")):
                    player.muted = not player.muted
                if key in (ord("c"), ord("C")) and cfg.MAR_THRESHOLD_MANUAL is None:
                    pipeline.recalibrate()
                    announced = False
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break                                  # window closed with X
            elif live:
                time.sleep(max(0.0, last_wall + period - time.monotonic()))
    finally:
        cap.release()
        player.close()
        if recorder:
            recorder.close()
        if mar_log:
            mar_log.close()
        pipeline.close()
        if display:
            cv2.destroyAllWindows()

    return {
        "duration_s": round(t, 2), "frames": processed, "fps": round(fps, 1),
        "native_resolution": native,
        "mar_threshold": pipeline.mar_threshold,
        "mar_baseline": pipeline.mar_baseline,
        "calibration": pipeline.calibration_note,
        "mar_log": str(mar_log.path) if mar_log else None,
        "max_level": engine.max_level,
        "time_to_max_s": round(engine.max_level_time, 2) if engine.max_level else None,
        "events": all_events,
        "clips": [str(p) for p in recorder.saved] if recorder else [],
        "alert_log": str(alert_log.path) if alert_log and alert_log.alerts else None,
        "sounds": [name for _, name in player.played],
        "spoken": list(player.voice.spoken) if player.voice else [],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aizu real-time drowsiness demo")
    ap.add_argument("--source", default="0", help="webcam index or video file (default 0)")
    ap.add_argument("--mute", action="store_true", help="start muted (M toggles)")
    ap.add_argument("--wide", action="store_true", help="start in 160-degree mode (W toggles)")
    ap.add_argument("--no-record", action="store_true", help="no event clips or alert log")
    ap.add_argument("--no-display", action="store_true", help="run without a window")
    ap.add_argument("--variant", choices=("int8", "float"), default=cfg.MODEL_VARIANT)
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    summary = run(src, display=not args.no_display, muted=args.mute, wide=args.wide,
                  record=not args.no_record, variant=args.variant)
    print(f"\nProcessed {summary['frames']} frames over {summary['duration_s']} s. "
          f"Highest level: {summary['max_level']} ({cfg.LEVEL_NAMES[summary['max_level']]}). "
          f"Events: {len(summary['events'])}. Clips saved: {len(summary['clips'])}.")
    if summary["alert_log"]:
        print(f"Alert log: {summary['alert_log']}")
