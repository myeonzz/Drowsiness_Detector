"""Real-time drowsiness demo on a webcam or a video file.

    python demo.py                       # webcam 0
    python demo.py --source 1            # another webcam
    python demo.py --source scenarios\\yawning.mp4
    python demo.py --mute --no-record

Press Q or Esc to quit.
"""
import argparse
import time

import cv2
import numpy as np

import common
import config as cfg
from alerts import AlertEngine, EventRecorder, SoundPlayer

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Pipeline:
    """frame -> face landmarks -> eye crops -> eye model, face crop -> face model."""

    def __init__(self, variant=None):
        variant = variant or cfg.MODEL_VARIANT
        self.finder = common.FaceFinder(video=True)
        self.eye_model = common.TFLiteModel(common.model_path("eye", variant))
        self.face_model = common.TFLiteModel(common.model_path("face", variant))

    def process(self, frame, t):
        """Returns p_closed / p_yawn (None when not measurable) and the boxes."""
        out = {"face_box": None, "eye_boxes": [], "p_closed": None, "p_yawn": None,
               "face_found": False, "occluded": False}
        landmarks = self.finder.find(frame, t)
        if landmarks is None:
            return out
        out["face_found"] = True
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        out["face_box"] = common.face_box(landmarks, frame.shape)
        if out["face_box"] is not None:      # same face crop as in training
            face = common.preprocess(common.crop(gray, out["face_box"]))
            out["p_yawn"] = float(self.face_model.predict(face)[1])

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


def letterbox(frame):
    """Resize to 640x480 keeping the aspect ratio (black bars if needed)."""
    w, h = cfg.FRAME_SIZE
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    scale = min(w / fw, h / fh)
    nw, nh = int(round(fw * scale)), int(round(fh * scale))
    canvas = np.zeros((h, w, 3), np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas


def _text(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame, det, state, fps):
    img = frame.copy()
    w = img.shape[1]
    color = cfg.LEVEL_COLORS_BGR[state["level"]]

    if det["face_box"]:
        x0, y0, x1, y1 = det["face_box"]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
    for x0, y0, x1, y1 in det["eye_boxes"]:
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)

    # colored banner with the current level and its message
    cv2.rectangle(img, (0, 0), (w, 58), color, -1)
    cv2.putText(img, f"LEVEL {state['level']}  {cfg.LEVEL_NAMES[state['level']]}", (10, 24),
                FONT, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, state["text"], (10, 49), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if det["p_closed"] is not None:
        p = det["p_closed"]
        eye_line = f"Eyes: {'Closed' if p >= 0.5 else 'Open'} {max(p, 1 - p):.2f}"
    elif state["eyes_unknown"]:
        eye_line = f"Eyes: unknown - {cfg.EYES_NOT_VISIBLE_TEXT}"
    else:
        eye_line = "Eyes: --"
    if det["p_yawn"] is not None:
        p = det["p_yawn"]
        face_line = f"Mouth: {'yawn' if p >= 0.5 else 'no_yawn'} {max(p, 1 - p):.2f}"
    else:
        face_line = "Face: not found"
    perclos = f"PERCLOS (60 s): {state['perclos'] * 100:.1f}%"
    if not state["perclos_ok"]:
        perclos += " (warming up)"
    lines = [eye_line, face_line, perclos, f"Yawns (10 min): {state['yawn_count']}",
             f"FPS: {fps:.1f}"]
    for i, line in enumerate(lines):
        _text(img, line, (10, 82 + 22 * i))

    if state["eyes_unknown"]:
        _text(img, cfg.EYES_NOT_VISIBLE_TEXT, (10, img.shape[0] - 16), 0.8, (0, 215, 255), 2)
    if state["alarm"] and int(state["t"] * 4) % 2 == 0:       # flashing border
        cv2.rectangle(img, (0, 0), (w - 1, img.shape[0] - 1), (0, 0, 255), 12)
    return img


def run(source=0, display=True, sound=True, record=True, events_dir=None,
        variant=None, verbose=True):
    """Run the full pipeline on a webcam index or video path. Returns a summary
    (highest level, time it was reached, events). Video files use the video's
    own clock, so results do not depend on how fast this laptop is."""
    live = isinstance(source, int)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if live else cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    if live:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_SIZE[1])
        step, src_fps = 1, None
    else:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps != src_fps or src_fps > 240:
            src_fps = 30.0
        step = max(1, round(src_fps / cfg.TARGET_FPS))     # sample at ~10 fps

    pipeline = Pipeline(variant)
    engine = AlertEngine()
    player = SoundPlayer(enabled=sound)
    recorder = EventRecorder(events_dir, source=str(source)) if record else None
    period = 1.0 / cfg.TARGET_FPS
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
            frame = letterbox(frame)
            det = pipeline.process(frame, t)
            state = engine.update(t, det["p_closed"], det["p_yawn"])
            processed += 1

            now = time.monotonic()
            if processed > 1:
                inst = 1.0 / max(now - last_wall, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
            last_wall = now

            shown = draw_overlay(frame, det, state, fps)
            player.update(state)
            if recorder:
                recorder.add_frame(t, shown if cfg.RECORD_OVERLAY else frame)
            for ev in state["events"]:
                all_events.append(ev)
                if recorder:
                    recorder.add_event(ev)
                if verbose:
                    print(f"[{ev['time']:7.1f}s] Level {ev['level']} "
                          f"{cfg.LEVEL_NAMES[ev['level']]}: {ev['trigger']} "
                          f"(PERCLOS {ev['perclos'] * 100:.1f}%)")

            if display:
                cv2.imshow("Aizu - drowsiness monitor (Q to quit)", shown)
                # hold ~10 fps: webcam -> fixed period, video -> follow video clock
                target = (last_wall + period) if live else (start + t + period)
                wait_ms = max(1, int((target - time.monotonic()) * 1000))
                if cv2.waitKey(wait_ms) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            elif live:
                time.sleep(max(0.0, last_wall + period - time.monotonic()))
    finally:
        cap.release()
        player.close()
        if recorder:
            recorder.close()
        pipeline.close()
        if display:
            cv2.destroyAllWindows()

    return {
        "duration_s": round(t, 2), "frames": processed,
        "max_level": engine.max_level,
        "time_to_max_s": round(engine.max_level_time, 2) if engine.max_level else None,
        "events": all_events,
        "clips": [str(p) for p in recorder.saved] if recorder else [],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aizu real-time drowsiness demo")
    ap.add_argument("--source", default="0", help="webcam index or video file (default 0)")
    ap.add_argument("--mute", action="store_true", help="no alert sounds")
    ap.add_argument("--no-record", action="store_true", help="do not save event clips")
    ap.add_argument("--no-display", action="store_true", help="run without a window")
    ap.add_argument("--variant", choices=("int8", "float"), default=cfg.MODEL_VARIANT)
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    summary = run(src, display=not args.no_display, sound=not args.mute,
                  record=not args.no_record, variant=args.variant)
    print(f"\nProcessed {summary['frames']} frames over {summary['duration_s']} s. "
          f"Highest level: {summary['max_level']} ({cfg.LEVEL_NAMES[summary['max_level']]}). "
          f"Events: {len(summary['events'])}. Clips saved: {len(summary['clips'])}.")
