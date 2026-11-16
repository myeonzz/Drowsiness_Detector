"""Checks the offense-count alert rules with made-up signals (no camera or
models needed).

    python test_alerts.py
"""
import config as cfg
from alerts import AlertEngine

DT = 1.0 / cfg.TARGET_FPS
OPEN, CLOSED, YAWN, NO = 0.02, 0.98, 0.95, 0.05
WINDOW = cfg.OFFENSE_WINDOW_S


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
    assert s.engine.max_level == 0 and s.state["offense_count"] == 0


def test_short_closures_and_yawns_are_not_offenses():
    s = Sim()
    for _ in range(5):
        s.run(10)
        s.run(1.8, eye=CLOSED)              # under 2 s
        s.run(10)
        s.run(3.5, yawn=YAWN)               # under 4 s
    assert s.engine.max_level == 0 and not s.events


def test_low_confidence_closed_is_ignored():
    s = Sim()
    s.run(15)
    s.run(5, eye=0.65)                      # below the 0.7 confidence bar
    assert s.engine.max_level == 0


def test_two_second_closure_is_one_offense():
    s = Sim()
    s.run(20)
    s.run(1.9, eye=CLOSED)
    assert s.state["level"] == 0
    s.run(0.3, eye=CLOSED)                  # 2.2 s: offense, stays under 3 s
    assert s.state["level"] == 1 and s.state["offense_count"] == 1
    assert not s.state["alarm"]
    s.run(0.5, eye=CLOSED)                  # still the same closure
    assert s.state["offense_count"] == 1    # one long event counts once


def test_four_second_yawn_is_one_offense():
    s = Sim()
    s.run(12)
    s.run(3.9, yawn=YAWN)
    assert s.state["level"] == 0            # under 4 s: nothing yet
    s.run(0.3, yawn=YAWN)
    assert s.state["level"] == 1 and s.events[-1]["trigger"] == "long_yawn"
    s.run(6, yawn=YAWN)                     # a 10 s yawn is still one offense
    assert s.state["offense_count"] == 1


def test_levels_follow_offense_count():
    s = Sim()
    s.run(20)
    s.run(2.2, eye=CLOSED)
    assert s.state["level"] == 1
    s.run(30)
    s.run(4.2, yawn=YAWN)
    assert s.state["level"] == 2 and s.state["text"] == cfg.LEVEL_TEXT[2]
    s.run(30)
    s.run(2.2, eye=CLOSED)
    assert s.state["level"] == 3 and s.state["alarm"] and s.state["text"] == "WAKE UP"
    s.run(30)
    s.run(4.2, yawn=YAWN)                   # 4th offense: still Level 3
    assert s.state["level"] == 3 and s.state["offense_count"] == 4
    assert [e["level"] for e in s.events] == [1, 2, 3, 3]
    assert [e["offense_count"] for e in s.events] == [1, 2, 3, 4]


def test_microsleep_override_goes_straight_to_level_3():
    s = Sim()
    s.run(20)
    s.run(2.9, eye=CLOSED)
    assert s.state["level"] == 1            # offense at 2 s
    s.run(0.3, eye=CLOSED)                  # 3 s: override
    assert s.state["level"] == 3 and s.state["alarm"]
    assert s.state["offense_count"] == 1    # the closure still counts once
    assert [e["trigger"] for e in s.events] == ["eye_closure", "microsleep"]


def test_alarm_stops_only_after_eyes_open_two_seconds():
    s = Sim()
    s.run(20)
    s.run(3.5, eye=CLOSED)
    assert s.state["alarm"]
    s.run(1.5)
    assert s.state["alarm"]                 # open 1.5 s: alarm still on
    s.run(0.5, eye=CLOSED)                  # closed again: the 2 s restart
    s.run(1.5)
    assert s.state["alarm"]
    s.run(1.0)
    assert not s.state["alarm"] and s.state["level"] == 3


def test_third_offense_by_yawn_sounds_alarm_at_least_two_seconds():
    s = Sim()
    for _ in range(3):
        s.run(20)
        s.run(4.2, yawn=YAWN)               # eyes open the whole time
    assert s.state["level"] == 3 and s.state["alarm"]
    s.run(1.5)
    assert s.state["alarm"]
    s.run(1.0)
    assert not s.state["alarm"]


def test_deescalates_one_level_as_each_offense_leaves_the_window():
    s = Sim()
    s.run(10)
    s.run(2.2, eye=CLOSED)                  # offense 1 at ~12 s
    s.run(100)
    s.run(2.2, eye=CLOSED)                  # offense 2 at ~114 s
    s.run(100)
    s.run(2.2, eye=CLOSED)                  # offense 3 at ~216 s
    assert s.state["level"] == 3
    s.run(WINDOW - 210)                     # just before offense 1 expires
    assert s.state["level"] == 3
    s.run(10)
    assert s.state["level"] == 2            # offense 1 left
    s.run(100)
    assert s.state["level"] == 1            # offense 2 left
    s.run(100)
    assert s.state["level"] == 0 and s.state["window_remaining_s"] is None


def test_window_timer_counts_down_to_oldest_offense():
    s = Sim()
    s.run(10)
    s.run(2.2, eye=CLOSED)
    s.run(60)
    left = s.state["window_remaining_s"]
    assert WINDOW - 62 < left < WINDOW - 59, left


def test_microsleep_level_ends_when_it_leaves_the_window():
    s = Sim()
    s.run(10)
    s.run(3.5, eye=CLOSED)
    s.run(WINDOW - 10)
    assert s.state["level"] == 3
    s.run(15)
    assert s.state["level"] == 0


def test_one_open_frame_does_not_reset_closure():
    s = Sim()
    s.run(20)
    s.run(1.2, eye=CLOSED)
    s.run(DT, eye=OPEN)                     # single open frame inside the closure
    s.run(1.0, eye=CLOSED)
    assert s.state["level"] == 1
    s2 = Sim()
    s2.run(20)
    s2.run(1.2, eye=CLOSED)
    s2.run(2 * DT, eye=OPEN)                # two open frames: closure restarts
    s2.run(1.0, eye=CLOSED)
    assert s2.state["level"] == 0


def test_sunglasses_counts_yawns_only():
    s = Sim()
    s.run(2)
    s.run(2.5, eye=None)
    assert not s.state["eyes_unknown"]
    s.run(1.0, eye=None)
    assert s.state["eyes_unknown"] and s.state["eye_status"] == "unknown"
    s.run(60, eye=None)
    assert s.engine.max_level == 0
    for _ in range(3):
        s.run(10, eye=None)
        s.run(4.2, eye=None, yawn=YAWN)
    assert s.state["level"] == 3 and s.state["alarm"]
    assert all(e["trigger"] == "long_yawn" for e in s.events)
    s.run(cfg.ALARM_NO_EYES_S + 0.5, eye=None)
    assert not s.state["alarm"]             # no eyes to wait for: alarm times out


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"{len(tests)} tests passed")
