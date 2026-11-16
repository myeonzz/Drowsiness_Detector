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
        out = {"face_box": None, "eye_boxes": [], "mouth_box": None, "p_closed": None,
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
            out["p_closed"] = float(np.min(probs))       # both eyes must look closed
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


# ---------------------------------------------------------------- overlay ---
def _text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    # Black outline from shifted copies at the same thickness. (A thicker
    # stroke changes the letter spacing in OpenCV 5, which shows as ghosting.)
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def _text_right(img, text, y, scale=0.7, color=(255, 255, 255), thick=2):
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thick)
    x = img.shape[1] - tw - 12
    _text(img, text, (x, y), scale, color, thick)
    return x, tw


def draw_overlay(frame, det, state, fps, wide=False, muted=False, note=""):
    img = frame.copy()
    h, w = img.shape[:2]
    color = cfg.LEVEL_COLORS_BGR[state["level"]]

    if det["face_box"]:
        x0, y0, x1, y1 = det["face_box"]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    for x0, y0, x1, y1 in det["eye_boxes"] + ([det["mouth_box"]] if det["mouth_box"] else []):
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)

    # colored banner with the current level and its message
    cv2.rectangle(img, (0, 0), (w, 66), color, -1)
    cv2.putText(img, f"LEVEL {state['level']}  {cfg.LEVEL_NAMES[state['level']]}", (12, 28),
                FONT, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, state["text"], (12, 56), FONT, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    if det["p_closed"] is not None:
        p = det["p_closed"]
        eye_line = f"Eyes: {'Closed' if p >= 0.5 else 'Open'} {max(p, 1 - p):.2f}"
    elif state["eyes_unknown"]:
        eye_line = f"Eyes: unknown - {cfg.EYES_NOT_VISIBLE_TEXT}"
    else:
        eye_line = "Eyes: --"
    if det["mar"] is None:
        face_line = "Face: not found"
    elif det["mar_threshold"] is None:
        face_line = f"Mouth: MAR {det['mar']:.2f} (calibrating)"
    else:
        face_line = (f"Mouth: MAR {det['mar']:.2f} / {det['mar_threshold']:.2f}  "
                     f"{'YAWN' if det['p_yawn'] else 'no yawn'}")
        if det["p_mouth_cnn"] is not None:
            face_line += f"  CNN {det['p_mouth_cnn']:.2f}"
    perclos = f"PERCLOS (60 s): {state['perclos'] * 100:.1f}%"
    window_min = cfg.OFFENSE_WINDOW_S / 60
    offenses = f"Offenses ({window_min:g} min): {state['offense_count']}"
    left = state["window_remaining_s"]
    if left is None:
        timer = "Window timer: --"
    else:                                    # time until the oldest offense leaves
        timer = f"Window timer: {int(left) // 60}:{int(left) % 60:02d} to next drop"
    lines = [eye_line, face_line, perclos, offenses]
    kinds = state["offense_kinds"]
    for i, kind in enumerate(kinds[-4:], start=max(len(kinds) - 4, 0) + 1):
        lines.append(f"   {i}. {kind}")          # what kind each offense was
    if state["override"]:                        # why it is Level 3 below 3 offenses
        lines.append(f"   + Override: eyes closed {cfg.MICROSLEEP_S:g} s")
    if state["last_alert"]:
        label, ago = state["last_alert"]
        lines.append(f"Last: {label}, {int(ago) // 60}:{int(ago) % 60:02d} ago")
    lines += [timer, f"FPS: {fps:.1f}"]
    for i, line in enumerate(lines):
        _text(img, line, (12, 92 + 24 * i))

    y = 96
    if wide:                                 # "160° MODE" (font has no degree sign)
        x, _ = _text_right(img, "160  MODE", y, color=(255, 255, 0))
        (tw, th), _ = cv2.getTextSize("160", FONT, 0.7, 2)
        cv2.circle(img, (x + tw + 7, y - th + 3), 4, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(img, (x + tw + 7, y - th + 3), 4, (255, 255, 0), 2, cv2.LINE_AA)
        y += 30
    if muted:
        _text_right(img, "MUTED", y, color=(200, 200, 200))

    if det["calibrating"]:
        _text(img, f"CALIBRATING {det['calibrating']:.0f} s - mouth closed, normal face",
              (12, h - 48), 0.8, (0, 255, 255), 2)
    if note:
        _text(img, note, (12, h - 84), 1.0, (0, 255, 255), 2)
    _text(img, "W: 160 mode   M: mute   C: calibrate   Q: quit", (w - 398, h - 12), 0.5)
    if state["eyes_unknown"]:
        _text(img, cfg.EYES_NOT_VISIBLE_TEXT, (12, h - 16), 0.9, (0, 215, 255), 2)
    if state["alarm"] and int(state["t"] * 4) % 2 == 0:       # flashing border
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
    return img


def _keep_aspect(frame_w, frame_h):
    """The window may be resized freely; snap its height back to the frame's
    aspect ratio (4:3 for SVGA). Display size never affects detection."""
    try:
        _, _, w, h = cv2.getWindowImageRect(WINDOW)
    except cv2.error:
        return
    want = int(round(w * frame_h / frame_w))
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
        cv2.resizeWindow(WINDOW, *cfg.FRAME_SIZE)

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
            shown = draw_overlay(frame, det, state, fps, wide, player.muted, note or "")
            player.update(state)
            if recorder:
                recorder.add_frame(t, to_clip(shown if cfg.RECORD_OVERLAY else frame))
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
                _keep_aspect(*cfg.FRAME_SIZE)
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
