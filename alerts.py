"""Alert logic: PERCLOS, closure / yawn tracking, the 4-level tier system,
alert sounds and event-triggered clip recording. No camera code here, so the
logic can be tested with made-up signals (see test_alerts.py)."""
import json
import threading
import time
from collections import deque
from datetime import datetime

import cv2

import config as cfg


class _Run:
    """Tracks one continuous stretch of 'active' frames (eyes closed, or
    yawning), tolerating a few inactive frames without resetting."""

    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.start = None        # time the run began
        self.last = None         # time of the last active frame
        self.misses = 0

    @property
    def duration(self):
        return 0.0 if self.start is None else self.last - self.start

    def update(self, t, active):
        """Returns the duration of a run that just ended, else None."""
        if active:
            if self.start is None:
                self.start = t
            self.last, self.misses = t, 0
            return None
        if self.start is None:
            return None
        self.misses += 1
        if self.misses > self.tolerance:
            return self.reset()
        return None

    def reset(self):
        ended = self.duration if self.start is not None else None
        self.start = self.last = None
        self.misses = 0
        return ended


class AlertEngine:
    """Feed one sample per processed frame with update(); read the state dict."""

    def __init__(self):
        self.level = 0
        self.level_entered = 0.0
        self.last_trigger = {1: None, 2: None, 3: None}
        self.eye_samples = deque()                 # (t, closed) for PERCLOS
        self.closure = _Run(cfg.OPEN_FRAME_TOLERANCE)
        self.closure_flags = set()                 # thresholds passed by this closure
        self.long_closures = deque()               # times of >= 1 s closures
        self.l3_events = deque()                   # times of microsleeps
        self.yawn = _Run(cfg.YAWN_FRAME_TOLERANCE)
        self.yawn_times = deque()                  # end times of counted yawns
        self.alarm = False                         # Level-3 alarm sounding
        self.open_since = None
        self.unseen_since = None                   # eyes not visible since
        self.eyes_unknown = False
        self.hold_until = None                     # "Stop driving" Level-2 floor
        self.prev_triggers = set()
        self.last_event = {}                       # trigger name -> last event time
        self.max_level = 0
        self.max_level_time = 0.0

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _prune(dq, t, window, key=lambda v: v):
        while dq and t - key(dq[0]) > window:
            dq.popleft()

    def perclos(self):
        """(value 0-1, enough_data). Share of closed frames in the last 60 s."""
        if not self.eye_samples:
            return 0.0, False
        closed = sum(1 for _, c in self.eye_samples if c)
        span = self.eye_samples[-1][0] - self.eye_samples[0][0]
        return closed / len(self.eye_samples), span >= cfg.PERCLOS_MIN_DATA_S

    # ------------------------------------------------------------- update --
    def update(self, t, p_closed, p_yawn):
        """t: seconds. p_closed: P(eyes closed) or None if eyes not visible.
        p_yawn: P(yawn) or None if no face. Returns the current state dict."""
        # ---- eyes ----
        if p_closed is None:
            self.open_since = None
            if self.unseen_since is None:
                self.unseen_since = t
            if t - self.unseen_since >= cfg.EYES_UNKNOWN_AFTER_S and not self.eyes_unknown:
                self.eyes_unknown = True
                self.closure.reset()
                self.closure_flags.clear()
            eye_status = "unknown" if self.eyes_unknown else "?"
        else:
            self.unseen_since, self.eyes_unknown = None, False
            closed = p_closed >= cfg.CLOSED_CONF_MIN
            eye_status = "closed" if closed else "open"
            self.eye_samples.append((t, closed))
            if self.closure.update(t, closed) is not None:
                self.closure_flags.clear()
            if closed:
                self.open_since = None
            elif self.open_since is None:
                self.open_since = t
            if self.closure.duration >= cfg.LONG_CLOSURE_S and "long" not in self.closure_flags:
                self.closure_flags.add("long")
                self.long_closures.append(t)
            if self.closure.duration >= cfg.MICROSLEEP_S and "micro" not in self.closure_flags:
                self.closure_flags.add("micro")
                self.l3_events.append(t)
                self.alarm = True
                if len(self.l3_events) >= cfg.ESCALATION_L3_COUNT:
                    self.hold_until = t + cfg.ESCALATION_HOLD_S
            if (self.alarm and self.open_since is not None
                    and t - self.open_since >= cfg.ALARM_OPEN_RELEASE_S):
                self.alarm = False

        self._prune(self.eye_samples, t, cfg.PERCLOS_WINDOW_S, key=lambda v: v[0])
        self._prune(self.long_closures, t, cfg.LONG_CLOSURE_WINDOW_S)
        self._prune(self.l3_events, t, cfg.ESCALATION_WINDOW_S)

        # ---- yawns ----
        yawning = p_yawn is not None and p_yawn >= cfg.YAWN_CONF_MIN
        ended = self.yawn.update(t, yawning)
        if ended is not None and ended >= cfg.YAWN_COUNT_MIN_S:
            self.yawn_times.append(t)
        self._prune(self.yawn_times, t, cfg.YAWN_COUNT_WINDOW_S)

        # ---- triggers: name -> level ----
        perclos, perclos_ok = self.perclos()
        triggers = {}
        if self.yawn.duration >= cfg.YAWN_LONG_S:
            triggers["long_yawn"] = 1
        if len(self.yawn_times) >= cfg.YAWN_COUNT:
            triggers["frequent_yawns"] = 1
        if not self.eyes_unknown:              # eye-based triggers need visible eyes
            if perclos_ok and perclos >= cfg.PERCLOS_CAUTION:
                triggers["perclos_caution"] = 1
            if perclos_ok and perclos >= cfg.PERCLOS_WARNING:
                triggers["perclos_warning"] = 2
            if len(self.long_closures) >= cfg.LONG_CLOSURE_COUNT:
                triggers["repeated_closures"] = 2
        if self.alarm:
            triggers["microsleep"] = 3

        # An event = a trigger switching on. A trigger hovering around its
        # threshold must not fire again and again, hence the cooldown
        # (every microsleep is its own event, so it has none).
        events = []
        for name, lvl in triggers.items():
            if name in self.prev_triggers:
                continue
            last = self.last_event.get(name)
            if name != "microsleep" and last is not None and t - last < cfg.EVENT_COOLDOWN_S:
                continue
            self.last_event[name] = t
            events.append({"time": round(t, 2), "level": lvl, "trigger": name,
                           "perclos": round(perclos, 4)})
        self.prev_triggers = set(triggers)

        # ---- level: escalate at once, de-escalate one step per quiet 60 s ----
        top = max(triggers.values(), default=0)
        for lvl in (1, 2, 3):
            if top >= lvl:
                self.last_trigger[lvl] = t
        stop_driving = self.hold_until is not None and t < self.hold_until
        floor = 2 if stop_driving else 0
        if top > self.level:
            self.level, self.level_entered = top, t
        elif self.level > floor:
            quiet_from = max(self.level_entered, self.last_trigger[self.level] or 0.0)
            if t - quiet_from >= cfg.DEESCALATE_AFTER_S:
                self.level, self.level_entered = self.level - 1, t
        if self.level < floor:
            self.level, self.level_entered = floor, t
        if self.level > self.max_level:
            self.max_level, self.max_level_time = self.level, t

        text = cfg.LEVEL_TEXT[self.level]
        if stop_driving and self.level == 2:
            text = cfg.STOP_DRIVING_TEXT
        return {
            "t": t, "level": self.level, "text": text, "alarm": self.alarm,
            "perclos": perclos, "perclos_ok": perclos_ok, "eye_status": eye_status,
            "eyes_unknown": self.eyes_unknown, "yawning": yawning,
            "yawn_count": len(self.yawn_times), "stop_driving": stop_driving,
            "triggers": triggers, "events": events,
        }


# -------------------------------------------------------------------- sound
class SoundPlayer:
    """Plays alert sounds on a background thread (Windows winsound)."""

    def __init__(self, enabled=True):
        try:
            import winsound
            self._ws = winsound
        except ImportError:                      # not on Windows -> silent
            self._ws, enabled = None, False
        self.enabled = enabled
        self._level, self._alarm, self._chime = 0, False, False
        self._last_beeps = None
        self._stop = False
        if enabled:
            threading.Thread(target=self._loop, daemon=True).start()

    def update(self, state):
        self._alarm = state["alarm"]
        if state["level"] != self._level:
            self._last_beeps = None              # beep right away on entering Level 2
        self._level = state["level"]
        if state["level"] == 1 and any(e["level"] == 1 for e in state["events"]):
            self._chime = True

    def _loop(self):
        while not self._stop:
            if self._alarm:                      # Level 3: continuous alarm
                self._ws.Beep(cfg.ALARM_HZ, cfg.ALARM_MS)
            elif self._level == 2 and (self._last_beeps is None or
                                       time.monotonic() - self._last_beeps >= cfg.BEEP_REPEAT_S):
                self._last_beeps = time.monotonic()
                for _ in range(3):               # Level 2: 3 short beeps every 10 s
                    self._ws.Beep(cfg.BEEP_HZ, cfg.BEEP_MS)
                    time.sleep(0.1)
            elif self._chime:                    # Level 1: one soft chime
                self._chime = False
                self._ws.Beep(cfg.CHIME_HZ, cfg.CHIME_MS)
            else:
                time.sleep(0.03)

    def close(self):
        self._stop = True


# ---------------------------------------------------------------- recording
class EventRecorder:
    """Keeps the last 5 s of frames in memory. On a Level 2/3 event it saves
    5 s before + 5 s after as an MJPEG .avi plus a JSON log. Nothing is
    written to disk unless an event happens."""

    def __init__(self, out_dir=None, source=""):
        self.out_dir = out_dir or cfg.EVENTS_DIR
        self.source = source
        self.buffer = deque()                    # (t, frame)
        self.pending = []                        # clips still collecting frames
        self.saved = []

    def add_frame(self, t, frame):
        self.buffer.append((t, frame))
        while self.buffer and t - self.buffer[0][0] > cfg.CLIP_PRE_S:
            self.buffer.popleft()
        for clip in self.pending:
            clip["frames"].append(frame)
        for clip in [c for c in self.pending if t >= c["end"]]:
            self._write(clip)

    def add_event(self, event):
        """Call after add_frame() for the frame where the event happened."""
        if event["level"] not in cfg.RECORD_LEVELS:
            return
        self.pending.append({
            "event": event, "end": event["time"] + cfg.CLIP_POST_S,
            "frames": [f for _, f in self.buffer], "wall": datetime.now(),
        })

    def _write(self, clip):
        self.pending.remove(clip)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ev = clip["event"]
        stem = f"{clip['wall']:%Y%m%d_%H%M%S}_L{ev['level']}_{ev['trigger']}"
        path = self.out_dir / f"{stem}.avi"
        n = 1
        while path.exists():                     # never overwrite an earlier clip
            n += 1
            path = self.out_dir / f"{stem}_{n}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"),
                                 cfg.TARGET_FPS, cfg.FRAME_SIZE)
        for frame in clip["frames"]:
            writer.write(frame)
        writer.release()
        log = {
            "time": clip["wall"].isoformat(timespec="seconds"),
            "session_time_s": ev["time"],
            "level": ev["level"],
            "level_name": cfg.LEVEL_NAMES[ev["level"]],
            "trigger": ev["trigger"],
            "perclos": ev["perclos"],
            "source": self.source,
            "clip": path.name,
            "clip_frames": len(clip["frames"]),
        }
        path.with_suffix(".json").write_text(json.dumps(log, indent=2))
        self.saved.append(path)

    def close(self):
        """Save clips that were still waiting for their 5 s 'after' part."""
        for clip in list(self.pending):
            self._write(clip)
