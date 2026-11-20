"""Guided eye test: on-screen timer + beeps, results table made automatically.

    python guided_eye_test.py --label "no glasses, normal light"

Timeline (default):  6 s get ready (eyes open, mouth closed, mouth calibration)
                     20 s EYES OPEN      -> 1 beep at the start
                     10 s DROOPY         -> 2 beeps at the start
                     12 s EYES CLOSED    -> 3 beeps at the start
                     end                 -> 1 long low beep
Short low ticks sound in the last 3 s of every phase as a countdown.

Uses demo.Pipeline through demo.run(), so it must sit in the same folder as
demo.py. Only the eye CNN is evaluated. Alerts and sounds are off.

Output: printed table, results\\eye_test_cnn.csv (appended), 
        results\\eye_test_<label>_frames.csv and results\\eye_test_<label>.png
"""
import argparse
import csv
import re
import threading
import time
from pathlib import Path

import numpy as np

import config as cfg
import demo

try:
    import winsound
except ImportError:                      # not Windows: no beeps, text only
    winsound = None

RESULTS = Path(__file__).resolve().parent / "results"
EMA_W = 0.4                              # weight of the newest frame (smoothed column)


def beeps(pattern):
    """pattern = [(Hz, ms), ...]; played in a thread so the video never stalls."""
    if winsound is None:
        return

    def _play():
        for hz, ms in pattern:
            winsound.Beep(hz, ms)
            time.sleep(0.08)
    threading.Thread(target=_play, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description="Guided eye test with timer and beeps")
    ap.add_argument("--label", required=True, help='e.g. "glasses, normal light"')
    ap.add_argument("--source", default="0", help="webcam index (default 0)")
    ap.add_argument("--lead", type=float, default=6, help="get-ready seconds")
    ap.add_argument("--open", type=float, default=20, dest="t_open")
    ap.add_argument("--droopy", type=float, default=10)
    ap.add_argument("--closed", type=float, default=12)
    ap.add_argument("--skip", type=float, default=1.5, help="seconds ignored at the start of each phase")
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source

    phases = [  # name, expected, duration, on-screen text, start beeps
        ("lead-in", None, args.lead, "GET READY - eyes open, mouth closed", [(600, 150)]),
        ("eyes open", "open", args.t_open, "EYES OPEN", [(900, 250)]),
        ("droopy", "not closed", args.droopy, "DROOPY (half closed)", [(1100, 250)] * 2),
        ("eyes closed", "closed", args.closed, "EYES CLOSED", [(1400, 250)] * 3),
    ]
    starts, t0 = [], 0.0
    for p in phases:
        starts.append(t0)
        t0 += p[2]
    total = t0

    st = {"idx": -1, "tick": None, "ema": None}
    rows = []        # (t, phase_idx, found, p, p_closed, ema, closed_smooth)

    def locate(t):
        for i, p in enumerate(phases):
            if t < starts[i] + p[2]:
                return i, starts[i] + p[2] - t
        return None, 0.0

    def on_frame(t, det, state):
        idx, left = locate(t)
        if idx is None:
            return "DONE - press Q"
        if idx != st["idx"]:
            st["idx"], st["tick"] = idx, None
            beeps(phases[idx][4])
        tick = int(np.ceil(left))
        if tick in (3, 2, 1) and tick != st["tick"] and left > 0.05:
            st["tick"] = tick
            beeps([(500, 60)])
        if idx > 0:                                   # lead-in is not recorded
            p = det["p_eye_cnn"]
            found = det["p_closed"] is not None and p is not None
            ema = closed_s = None
            if found:
                st["ema"] = p if st["ema"] is None else EMA_W * p + (1 - EMA_W) * st["ema"]
                ema = st["ema"]
                closed_s = ema >= cfg.CLOSED_CONF_MIN
            rows.append((t, idx, found, p if found else None,
                         bool(det["p_closed"]) if found else None, ema, closed_s))
        return f"{phases[idx][3]}  {left:.0f} s"

    summary = demo.run(src, display=True, sound=False, record=False, verbose=True,
                       max_seconds=total + 0.3, on_frame=on_frame)
    if winsound:
        winsound.Beep(500, 700)

    if summary["duration_s"] < total - 1:
        print("\nStopped early, nothing saved. Run it again and let it finish.")
        return

    # ---- table ----
    slug = re.sub(r"[^a-z0-9]+", "_", args.label.lower()).strip("_")
    RESULTS.mkdir(exist_ok=True)
    table = []
    print(f"\nEYE TEST (CNN only): {args.label}")
    print(f"{'phase':<12}{'expected':<12}{'frames':>7}{'no_eyes':>9}{'closed_%':>10}"
          f"{'smoothed_%':>12}{'mean_P(closed)':>16}")
    for i in (1, 2, 3):
        name, expected = phases[i][0], phases[i][1]
        sel = [r for r in rows if r[1] == i and r[0] - starts[i] >= args.skip]
        n = len(sel)
        found = [r for r in sel if r[2]]
        raw = 100 * np.mean([r[4] for r in found]) if found else float("nan")
        sm_rows = [r for r in found if r[6] is not None]
        sm = 100 * np.mean([r[6] for r in sm_rows]) if sm_rows else float("nan")
        mp = float(np.mean([r[3] for r in found])) if found else float("nan")
        table.append((args.label, name, expected, n, n - len(found), raw, sm, mp))
        print(f"{name:<12}{expected:<12}{n:>7}{n - len(found):>9}{raw:>10.1f}{sm:>12.1f}{mp:>16.3f}")
    print("closed_% = share of frames with eyes found that the CNN called closed "
          "(open and droopy should be low, closed should be high).")
    print(f"smoothed_% = same after an exponential moving average (weight {EMA_W}). "
          f"First {args.skip} s of each phase ignored.")

    # ---- files ----
    csv_path = RESULTS / "eye_test_cnn.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "phase", "expected", "frames", "no_eyes",
                        "closed_pct", "closed_pct_smoothed", "mean_p_closed"])
        for r in table:
            w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.1f}", f"{r[6]:.1f}", f"{r[7]:.3f}"])
    frames_path = RESULTS / f"eye_test_{slug}_frames.csv"
    with open(frames_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "phase", "eyes_found", "p_closed", "closed", "p_smoothed", "closed_smoothed"])
        for t, i, fd, p, c, e, cs in rows:
            w.writerow([f"{t:.2f}", phases[i][0], int(fd), "" if p is None else f"{p:.4f}",
                        "" if c is None else int(c), "" if e is None else f"{e:.4f}",
                        "" if cs is None else int(cs)])
    print(f"Saved {csv_path}\nSaved {frames_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.6))
        shade = {1: "#cfe8cf", 2: "#ffe3b8", 3: "#f5c2c2"}
        for i in (1, 2, 3):
            ax.axvspan(starts[i], starts[i] + phases[i][2], color=shade[i], alpha=0.6, lw=0)
            ax.text(starts[i] + 0.3, 1.03, phases[i][0], fontsize=8)
        ts = [r[0] for r in rows if r[2]]
        ax.plot(ts, [r[3] for r in rows if r[2]], lw=1, label="P(closed)")
        ax.plot(ts, [r[5] for r in rows if r[2]], lw=1.5, label="smoothed")
        ax.axhline(cfg.CLOSED_CONF_MIN, color="k", ls="--", lw=0.8, label=f"threshold {cfg.CLOSED_CONF_MIN}")
        ax.set_ylim(-0.02, 1.1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("P(closed)")
        ax.set_title(f"Eye CNN P(closed) over time: {args.label}")
        ax.legend(loc="center left", fontsize=8)
        fig.tight_layout()
        png = RESULTS / f"eye_test_{slug}.png"
        fig.savefig(png, dpi=150)
        print(f"Saved {png}")
    except Exception as err:
        print(f"Plot skipped ({err})")


if __name__ == "__main__":
    main()
