"""Checks the alert rules with made-up signals (no camera or models needed).

    python test_alerts.py
"""
import config as cfg
from alerts import AlertEngine

DT = 1.0 / cfg.TARGET_FPS
OPEN, CLOSED, YAWN, NO = 0.02, 0.98, 0.95, 0.05


class Sim:
    def __init__(self):
        self.engine, self.t, self.state, self.events = AlertEngine(), 0.0, None, []

    def run(self, seconds, eye=OPEN, yawn=NO):
        """Feed constant signals for some seconds; returns the last state."""
        for _ in range(round(seconds / DT)):
            self.state = self.engine.update(self.t, eye, yawn)
            self.events += self.state["events"]
            self.t += DT
        return self.state


def test_normal_blinking_stays_normal():
    s = Sim()
    for _ in range(30):                     # 0.3 s blink every 4 s, 2 min
        s.run(3.7)
        s.run(0.3, eye=CLOSED)
    assert s.engine.max_level == 0, s.engine.max_level


def test_low_confidence_closed_is_ignored():
    s = Sim()
    s.run(15)
    s.run(5, eye=0.65)                      # below the 0.7 confidence bar
    assert s.engine.max_level == 0


def test_long_yawn_is_caution():
    s = Sim()
    s.run(12)
    s.run(3.5, yawn=YAWN)
    assert s.state["level"] == 0            # under 4 s: nothing yet
    s.run(1.0, yawn=YAWN)
    assert s.state["level"] == 1 and "long_yawn" in s.state["triggers"]


def test_three_yawns_is_caution():
    s = Sim()
    for _ in range(3):
        s.run(20)
        s.run(1.5, yawn=YAWN)
    s.run(1)
    assert s.state["level"] == 1 and "frequent_yawns" in s.state["triggers"]


def test_perclos_levels():
    s = Sim()
    for _ in range(12):                     # closed 0.8 s of every 5 s = 16 %
        s.run(4.2)
        s.run(0.8, eye=CLOSED)
    assert s.state["level"] == 1, s.state
    for _ in range(22):                     # closed 0.9 s of every 3 s = 30 %
        s.run(2.1)
        s.run(0.9, eye=CLOSED)
    assert s.state["level"] == 2 and "perclos_warning" in s.state["triggers"]
    cautions = [e for e in s.events if e["trigger"] == "perclos_caution"]
    assert len(cautions) <= 2, cautions     # hovering at 15 % must not spam events


def test_two_long_closures_is_warning():
    s = Sim()
    s.run(20)
    s.run(1.3, eye=CLOSED)
    assert s.state["level"] == 0
    s.run(10)
    s.run(1.3, eye=CLOSED)
    assert s.state["level"] == 2 and "repeated_closures" in s.state["triggers"]


def test_one_open_frame_does_not_reset_closure():
    s = Sim()
    s.run(20)
    s.run(1.7, eye=CLOSED)
    s.run(DT, eye=OPEN)                     # single open frame inside the closure
    s.run(1.5, eye=CLOSED)
    assert s.state["level"] == 3
    s2 = Sim()
    s2.run(20)
    s2.run(1.7, eye=CLOSED)
    s2.run(2 * DT, eye=OPEN)                # two open frames: closure restarts
    s2.run(1.5, eye=CLOSED)
    assert s2.state["level"] < 3


def test_microsleep_alarm_and_deescalation():
    s = Sim()
    s.run(20)
    s.run(1.9, eye=CLOSED)
    assert s.state["level"] < 2             # under 2 s: no warning yet
    s.run(0.3, eye=CLOSED)
    assert s.state["level"] == 2 and "eyes_closed" in s.state["triggers"]
    assert not s.state["alarm"]             # 2 s closure: Level 2, no alarm
    s.run(0.9, eye=CLOSED)
    assert s.state["level"] == 3 and s.state["alarm"] and s.state["text"] == "WAKE UP"
    s.run(1.5)
    assert s.state["alarm"]                 # eyes open < 2 s: alarm still on
    s.run(1.0)
    assert not s.state["alarm"] and s.state["level"] == 3
    s.run(58)
    assert s.state["level"] == 3            # not yet 60 s quiet
    s.run(4)
    assert s.state["level"] == 2
    s.run(61)
    assert s.state["level"] == 1
    s.run(61)
    assert s.state["level"] == 0


def test_two_microsleeps_hold_level_2():
    s = Sim()
    s.run(20)
    s.run(3.5, eye=CLOSED)
    s.run(60)
    s.run(3.5, eye=CLOSED)
    s.run(300)                              # 5 quiet minutes later ...
    assert s.state["level"] == 2 and s.state["stop_driving"]
    assert s.state["text"] == cfg.STOP_DRIVING_TEXT
    s.run(cfg.ESCALATION_HOLD_S)            # hold over: free to drop again
    assert s.state["level"] < 2 and not s.state["stop_driving"]


def test_sunglasses_only_yawn_triggers():
    s = Sim()
    s.run(2)
    s.run(2.5, eye=None)
    assert not s.state["eyes_unknown"]
    s.run(1.0, eye=None)
    assert s.state["eyes_unknown"] and s.state["eye_status"] == "unknown"
    s.run(60, eye=None)
    assert s.engine.max_level == 0
    s.run(4.5, eye=None, yawn=YAWN)
    assert s.state["level"] == 1


def test_events_are_logged_once():
    s = Sim()
    s.run(20)
    s.run(4.0, eye=CLOSED)
    s.run(10)
    micro = [e for e in s.events if e["trigger"] == "microsleep"]
    assert len(micro) == 1 and micro[0]["level"] == 3


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"{len(tests)} tests passed")
