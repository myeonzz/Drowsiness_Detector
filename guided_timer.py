"""Beeping timer for the guided eye test: open 20 s, droopy 10 s, closed 12 s.

Run it in a SECOND PowerShell window while demo.py is running:
    python guided_timer.py

Start it right after the demo's calibration text disappears. It gives a short
get-ready countdown, then:
    1 beep   -> EYES OPEN      (20 s)
    2 beeps  -> DROOPY         (10 s, half closed)
    3 beeps  -> EYES CLOSED    (12 s)
    long beep -> END: open your eyes and press Q in the demo window

The beep is played by this script, not by demo.py, so it still works with
demo.py --mute. Keep this window out of the camera's view.
"""
import argparse
import sys
import time

try:
    import winsound

    def tone(freq, ms):
        winsound.Beep(freq, ms)
except ImportError:  # not Windows: fall back to the terminal bell
    def tone(freq, ms):
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(ms / 1000)


def beeps(n, freq=1000, ms=180, gap=0.15):
    for _ in range(n):
        tone(freq, ms)
        time.sleep(gap)


def countdown(seconds, label):
    """Print a live countdown for `seconds`, starting at the moment of the call."""
    end = time.monotonic() + seconds
    while True:
        left = end - time.monotonic()
        if left <= 0:
            break
        print(f"\r{label:<14}{left:5.1f} s left ", end="", flush=True)
        time.sleep(0.1)
    print(f"\r{label:<14}  done        ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ready", type=float, default=3.0, help="get-ready seconds")
    ap.add_argument("--open-s", type=float, default=20.0)
    ap.add_argument("--droopy-s", type=float, default=10.0)
    ap.add_argument("--closed-s", type=float, default=12.0)
    a = ap.parse_args()

    print("Start this right after the demo's calibration text disappears.")
    for i in range(int(a.ready), 0, -1):
        print(f"Starting in {i}...", end="\r", flush=True)
        tone(600, 80)
        time.sleep(max(0.0, 1.0 - 0.08))
    print(" " * 30, end="\r")

    t0 = time.monotonic()
    phases = [
        ("EYES OPEN", 1, a.open_s),
        ("DROOPY", 2, a.droopy_s),
        ("EYES CLOSED", 3, a.closed_s),
    ]
    for name, n, secs in phases:
        # count the phase from the first beep, so each phase is exactly its length
        start = time.monotonic()
        print(f"[{start - t0:5.1f} s] {name}")
        beeps(n)
        countdown(max(0.0, secs - (time.monotonic() - start)), name)

    print(f"[{time.monotonic() - t0:5.1f} s] END - open your eyes and press Q in the demo window, "
          "then run phase_table.py")
    tone(800, 900)


if __name__ == "__main__":
    main()
