"""Alert logic: closure / yawn tracking, the offense-count level system,
alert sounds, the alert log and event-triggered clip recording. No camera code here, so the
logic can be tested with made-up signals (see test_alerts.py)."""
import io
import json
import threading
import time
import wave
from collections import deque
from datetime import datetime

import cv2
import numpy as np

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
    """Offense-count alert logic. Feed one sample per processed frame with
    update(); read the returned state dict.

    Offense = eyes closed continuously >= 2 s, or a yawn lasting >= 4 s. One
    long event counts once. Level = offenses in the last 10 minutes
    (0, 1, 2, 3+). Eyes closed >= 3 s goes straight to Level 3."""

    def __init__(self):
        self.level = 0
        self.offenses = deque()                    # (t, kind) inside the window
        self.microsleep_t = None                   # time of the last >= 3 s closure
        self.eye_samples = deque()                 # (t, closed) for PERCLOS (display only)
        self.closure = _Run(cfg.OPEN_FRAME_TOLERANCE)
        self.closure_flags = set()                 # thresholds passed by this closure
        self.yawn = _Run(cfg.YAWN_FRAME_TOLERANCE)
        self.yawn_counted = False                  # this yawn is already an offense
        self.alarm = False                         # Level-3 alarm sounding
        self.alarm_timeout = None                  # only when the eyes are not visible
        self.open_since = None
        self.unseen_since = None                   # eyes not visible since
        self.eyes_unknown = False
        self.max_level = 0
        self.max_level_time = 0.0
        self.last_alert = None                     # (t, trigger) of the latest alert

    def perclos(self):
        """PERCLOS: share of closed frames in the last 60 s (shown on screen
        only). The demo feeds it the P80 flag: EAR closure of 80 % or more."""
        if not self.eye_samples:
            return 0.0
        return sum(1 for _, c in self.eye_samples if c) / len(self.eye_samples)

    def _level(self, t):
        if self.microsleep_t is not None and t - self.microsleep_t <= cfg.OFFENSE_WINDOW_S:
            return 3                               # override holds while in the window
        return min(len(self.offenses), 3)

    def _start_alarm(self, t, eyes_open_now):
        self.alarm = True
        # "open for 2 s straight" is counted from the moment the alarm starts
        self.open_since = t if eyes_open_now else None
        # with the eyes hidden (sunglasses) there is nothing to wait for
        self.alarm_timeout = t + cfg.ALARM_NO_EYES_S if self.eyes_unknown else None

    # ------------------------------------------------------------- update --
    def update(self, t, p_closed, p_yawn, perclos_closed=None):
        """t: seconds. p_closed: P(eyes closed) or None if eyes not visible.
        p_yawn: P(yawn) or None if no face. perclos_closed: whether this frame
        counts as closed for PERCLOS (default: the same as the closed decision).
        Returns the current state dict."""
        new = []                                   # (trigger, is_offense) this frame

        # ---- eyes ----
        eyes_open_now = False
        if p_closed is None:
            self.open_since = None
            if self.unseen_since is None:
                self.unseen_since = t
            if t - self.unseen_since >= cfg.EYES_UNKNOWN_AFTER_S and not self.eyes_unknown:
                self.eyes_unknown = True           # from here on: yawn offenses only
                self.closure.reset()
                self.closure_flags.clear()
            eye_status = "unknown" if self.eyes_unknown else "?"
        else:
            self.unseen_since, self.eyes_unknown = None, False
            closed = p_closed >= cfg.CLOSED_CONF_MIN
            eyes_open_now = not closed
            eye_status = "closed" if closed else "open"
            self.eye_samples.append((t, closed if perclos_closed is None else bool(perclos_closed)))
            if self.closure.update(t, closed) is not None:
                self.closure_flags.clear()         # closure ended
            if closed:
                self.open_since = None
            elif self.open_since is None:
                self.open_since = t
            if (self.closure.duration >= cfg.OFFENSE_CLOSURE_S
                    and "offense" not in self.closure_flags):
                self.closure_flags.add("offense")  # one long closure counts once
                new.append(("eye_closure", True))
            if self.closure.duration >= cfg.MICROSLEEP_S and "micro" not in self.closure_flags:
                self.closure_flags.add("micro")
                self.microsleep_t = t
                new.append(("microsleep", False))  # same closure: no extra offense
        while self.eye_samples and t - self.eye_samples[0][0] > cfg.PERCLOS_WINDOW_S:
            self.eye_samples.popleft()

        # ---- yawns ----
        yawning = p_yawn is not None and p_yawn >= cfg.YAWN_CONF_MIN
        if self.yawn.update(t, yawning) is not None:
            self.yawn_counted = False              # yawn ended
        if self.yawn.duration >= cfg.OFFENSE_YAWN_S and not self.yawn_counted:
            self.yawn_counted = True               # one long yawn counts once
            new.append(("long_yawn", True))

        # ---- offenses in the rolling window -> level ----
        while self.offenses and t - self.offenses[0][0] > cfg.OFFENSE_WINDOW_S:
            self.offenses.popleft()                # oldest offense leaves: level drops
        events = []
        perclos = self.perclos()
        for trigger, is_offense in new:
            if is_offense:
                self.offenses.append((t, trigger))
            level = self._level(t)
            events.append({
                "time": round(t, 2), "level": level, "trigger": trigger,
                "reason": trigger_reason(trigger), "offense_count": len(self.offenses),
                "perclos": round(perclos, 4),
            })
            self.last_alert = (t, trigger)
            if level == 3:
                self._start_alarm(t, eyes_open_now)
        self.level = self._level(t)
        if self.level > self.max_level:
            self.max_level, self.max_level_time = self.level, t

        # ---- Level-3 alarm: stops only after the eyes are open 2 s straight ----
        if self.alarm:
            opened = (self.open_since is not None
                      and t - self.open_since >= cfg.ALARM_OPEN_RELEASE_S)
            timed_out = self.alarm_timeout is not None and t >= self.alarm_timeout
            if opened or (timed_out and self.eyes_unknown):
                self.alarm = False

        oldest = self.offenses[0][0] if self.offenses else None
        return {
            "t": t, "level": self.level, "text": cfg.LEVEL_TEXT[self.level],
            "alarm": self.alarm, "perclos": perclos, "eye_status": eye_status,
            "eyes_unknown": self.eyes_unknown, "yawning": yawning,
            "offense_count": len(self.offenses),
            # kind of each offense in the window, oldest first
            "offense_kinds": [trigger_label(kind) for _, kind in self.offenses],
            # True while a >= 3 s closure is holding Level 3
            "override": (self.microsleep_t is not None
                         and t - self.microsleep_t <= cfg.OFFENSE_WINDOW_S),
            # latest offense / override: (label, seconds ago), or None
            "last_alert": None if self.last_alert is None else
                          (trigger_label(self.last_alert[1]), t - self.last_alert[0]),
            # seconds until the oldest offense leaves the window (None if no offenses)
            "window_remaining_s": None if oldest is None else cfg.OFFENSE_WINDOW_S - (t - oldest),
            "events": events,
        }


# -------------------------------------------------------------------- sound
SAMPLE_RATE = 22050


def _tone(freq, ms, volume, sharp=False, decay=False):
    """One generated tone as float samples in [-1, 1]."""
    t = np.arange(int(SAMPLE_RATE * ms / 1000)) / SAMPLE_RATE
    samples = np.sin(2 * np.pi * freq * t)
    if sharp:                                    # squarer wave = harsher sound
        samples = np.clip(samples * 4, -1, 1)
    env = np.exp(-4 * t / t[-1]) if decay else np.ones_like(t)
    edge = min(len(t) // 4, int(SAMPLE_RATE * 0.005))   # 5 ms fade, no clicks
    env[:edge] *= np.linspace(0, 1, edge)
    env[-edge:] *= np.linspace(1, 0, edge)
    return samples * env * volume


def _silence(ms):
    return np.zeros(int(SAMPLE_RATE * ms / 1000))


def _wav(samples):
    """Float samples -> in-memory WAV file (16-bit mono)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def build_sounds():
    """The three alert sounds, generated with numpy (no audio files)."""
    chime = np.concatenate([_tone(f, cfg.CHIME_MS, cfg.CHIME_VOLUME, decay=True)
                            for f in cfg.CHIME_HZ])
    beep = np.concatenate([_tone(cfg.BEEP_HZ, cfg.BEEP_MS, cfg.BEEP_VOLUME, sharp=True),
                           _silence(cfg.BEEP_GAP_MS)])
    alarm = np.concatenate([_tone(f, cfg.ALARM_MS, cfg.ALARM_VOLUME, sharp=True)
                            for f in cfg.ALARM_HZ])
    return {"chime": _wav(chime), "beeps": _wav(np.tile(beep, 3)), "alarm": _wav(alarm)}


class Voice:
    """Speaks alert texts with pyttsx3 on its own thread. If pyttsx3 is not
    installed or fails, it stays silent and the text is shown on screen only."""

    def __init__(self):
        self.available = False
        self.spoken = []                         # texts actually spoken (for testing)
        self._pending = None
        self._stop = False
        self._ready = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        self._ready.wait(timeout=10)

    def say(self, text):
        self._pending = text                     # newest message wins

    def _loop(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", cfg.VOICE_RATE)
            self.available = True
        except Exception:
            return
        finally:
            self._ready.set()
        while not self._stop:
            text, self._pending = self._pending, None
            if text is None:
                time.sleep(0.05)
                continue
            try:
                engine.say(text)
                engine.runAndWait()
                self.spoken.append(text)
            except Exception:
                self.available = False
                return

    def close(self):
        self._stop = True


class SoundPlayer:
    """Escalating alert sounds on a background thread.
    Level 1: one soft chime per offense. Level 2: 3 sharp beeps every 10 s.
    Level 3: continuous loud alarm while the engine's alarm flag is on (it
    clears only after the eyes have been open for 2 s). Each sound comes with
    the spoken alert text when a voice is available."""

    def __init__(self, enabled=True, muted=False, voice=True):
        try:
            import winsound
            self._ws = winsound
        except ImportError:                      # not on Windows -> silent
            self._ws, enabled = None, False
        self.enabled = enabled
        self.muted = muted
        self.voice = Voice() if (enabled and voice and cfg.VOICE_ENABLED) else None
        self.played = []                         # (wall time, sound name) for testing
        self._level, self._alarm, self._chime = 0, False, False
        self._last_beeps = self._last_wake = None
        self._stop = False
        if enabled:
            self._sounds = build_sounds()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    @property
    def voice_available(self):
        return bool(self.voice and self.voice.available)

    def update(self, state):
        self._alarm = state["alarm"]
        if state["level"] != self._level:
            self._last_beeps = None              # beep right away on entering Level 2
        self._level = state["level"]
        if state["level"] == 1 and any(e["level"] == 1 for e in state["events"]):
            self._chime = True                   # once per Level-1 trigger

    def _play(self, name):
        self.played.append((time.monotonic(), name))
        self._ws.PlaySound(self._sounds[name], self._ws.SND_MEMORY)   # blocks until done

    def _say(self, text):
        if self.voice_available:
            self.voice.say(text)

    def _loop(self):
        while not self._stop:
            now = time.monotonic()
            if self.muted:
                self._chime = False
                self._last_beeps = self._last_wake = None
                time.sleep(0.03)
            elif self._alarm:                    # Level 3: continuous alarm
                if self._last_wake is None or now - self._last_wake >= cfg.WAKE_UP_REPEAT_S:
                    self._last_wake = now
                    self._say(cfg.LEVEL_TEXT[3])
                self._play("alarm")
            elif self._level == 2 and (self._last_beeps is None or
                                       now - self._last_beeps >= cfg.BEEP_REPEAT_S):
                self._last_beeps = now           # Level 2: 3 beeps every 10 s
                self._play("beeps")
                self._say(cfg.LEVEL_TEXT[2])
            elif self._chime:                    # Level 1: one soft chime
                self._chime = False
                self._play("chime")
                self._say(cfg.LEVEL_TEXT[1])
            else:
                self._last_wake = None
                time.sleep(0.03)

    def close(self):
        self._stop = True
        if self.enabled:
            self._thread.join(timeout=2)
        if self.voice:
            self.voice.close()


# ---------------------------------------------------------------- alert log
def trigger_label(name):
    """Short on-screen name for the kind of offense."""
    return {
        "eye_closure": f"Eyes closed {cfg.OFFENSE_CLOSURE_S:g} s",
        "long_yawn": f"Yawn {cfg.OFFENSE_YAWN_S:g} s",
        "microsleep": f"Eyes closed {cfg.MICROSLEEP_S:g} s (override)",
    }[name]


def trigger_reason(name):
    """Plain-language reason for an alert, using the current thresholds."""
    return {
        "eye_closure": f"Offense: eyes closed continuously for {cfg.OFFENSE_CLOSURE_S:g} s",
        "long_yawn": f"Offense: yawn lasting {cfg.OFFENSE_YAWN_S:g} s",
        "microsleep": (f"Override: eyes closed continuously for {cfg.MICROSLEEP_S:g} s "
                       "(straight to Level 3)"),
    }[name]


class AlertLog:
    """Session log of every offense and override: events/alerts_<start>.json.
    Rewritten after each alert, so it is complete even if the demo crashes."""

    def __init__(self, out_dir=None, source=""):
        self.out_dir = out_dir or cfg.EVENTS_DIR
        self.path = self.out_dir / f"alerts_{datetime.now():%Y%m%d_%H%M%S}.json"
        self.source = source
        self.alerts = []

    def add(self, event):
        self.alerts.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "session_time_s": event["time"],
            "level": event["level"],
            "level_name": cfg.LEVEL_NAMES[event["level"]],
            "trigger": event["trigger"],
            "reason": event["reason"],
            "offense_count": event["offense_count"],
            "perclos": event["perclos"],
        })
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"source": self.source, "alerts": self.alerts}, indent=2))


# ---------------------------------------------------------------- recording
class EventRecorder:
    """Keeps the last 5 s of frames in memory. On a Level 2/3 event it saves
    5 s before + 5 s after as a 640x480 MJPEG .avi plus a JSON log. Nothing is
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
                                 cfg.TARGET_FPS, cfg.CLIP_SIZE)
        for frame in clip["frames"]:
            writer.write(frame)
        writer.release()
        log = {
            "time": clip["wall"].isoformat(timespec="seconds"),
            "session_time_s": ev["time"],
            "level": ev["level"],
            "level_name": cfg.LEVEL_NAMES[ev["level"]],
            "trigger": ev["trigger"],
            "reason": ev["reason"],
            "offense_count": ev["offense_count"],
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
