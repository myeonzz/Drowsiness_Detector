"""Guided eye test for the CNN-only eye pipeline.

Run from the project folder, one condition at a time:

    .\\.venv\\Scripts\\python.exe eye_test.py --label "no glasses, normal light"
    .\\.venv\\Scripts\\python.exe eye_test.py --label "glasses, normal light"
    .\\.venv\\Scripts\\python.exe eye_test.py --label "no glasses, glare"
    .\\.venv\\Scripts\\python.exe eye_test.py --label "glasses and glare"

It opens the demo window and tells you what to do on screen (and with a short beep):

    6 s   get ready (eyes open, mouth neutral)
    20 s  EYES OPEN
    10 s  DROOPY   (half close your eyes and hold the same level)
    5 s   EYES CLOSED

It stops by itself after 41 s. The first 1.5 s of each phase are ignored. For each
phase it prints and saves: frames counted, frames with no eyes found, the share of
frames the CNN calls closed, and the mean P(closed). "No eyes" frames cannot raise
an alert, so that column matters for glasses and glare.

Saved:  results\\eye_test_live.csv (one row per phase, appended)
        results\\eye_test_<label>.png (P(closed) over time)
"""
import argparse
import csv
import re
from datetime import datetime

import numpy as np

import config as cfg
import demo

READY_S, SKIP_S = 6.0, 1.5
PHASES = [          # name, what the CNN should say, length in seconds
    ("eyes open", "open", 20.0),
    ("droopy", "not closed", 10.0),
    ("eyes closed", "closed", 5.0),
]
PROMPTS = [
    "EYES OPEN - look at the camera",
    "DROOPY - half close your eyes, hold it",
    "EYES CLOSED - keep them shut",
]
STARTS, t_acc = [], READY_S
for _name, _exp, _len in PHASES:
    STARTS.append(t_acc)
    t_acc += _len
TOTAL_S = t_acc                                   # 41 s


def phase_at(t):
    """-1 = get ready, 0..2 = the three phases, 3 = finished."""
    if t < READY_S:
        return -1
    for i, start in enumerate(STARTS):
        if t < start + PHASES[i][2]:
            return i
    return 3


def beep(enabled):
    if not enabled:
        return
    try:
        import winsound
        winsound.Beep(1000, 150)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="Guided eye test (CNN only)")
    ap.add_argument("--label", default="test", help='condition, e.g. "glasses, glare"')
    ap.add_argument("--source", type=int, default=0, help="webcam index")
    ap.add_argument("--no-beep", action="store_true", help="no beep at each phase change")
    ap.add_argument("--record", action="store_true", help="also save alert clips and logs")
    args = ap.parse_args()

    counted = [[] for _ in PHASES]                # (p_closed, p_cnn) per counted frame
    trace = []                                    # (t, p_cnn or nan, phase) for the plot
    last = {"phase": -2}

    def on_frame(t, det, state):
        idx = phase_at(t)
        if idx != last["phase"]:
            last["phase"] = idx
            if idx >= 0:
                beep(not args.no_beep)
        p_closed = det.get("p_closed")            # 1.0 / 0.0, or None if no eyes were found
        p_cnn = det.get("p_eye_cnn")
        trace.append((t, np.nan if p_cnn is None else float(p_cnn), idx))
        if 0 <= idx < len(PHASES) and t >= STARTS[idx] + SKIP_S:
            counted[idx].append((p_closed, p_cnn))
        if idx == -1:
            return f"GET READY ({READY_S - t:.0f} s): eyes open, mouth neutral"
        if idx < len(PHASES):
            left = STARTS[idx] + PHASES[idx][2] - t
            return f"{idx + 1}/3 {PROMPTS[idx]} ({left:.0f} s)"
        return "DONE"

    print(f"Eye test: {args.label}  ({TOTAL_S:.0f} s). The window will close by itself.")
    demo.run(args.source, display=True, sound=False, record=args.record,
             verbose=True, max_seconds=TOTAL_S + 0.3, on_frame=on_frame)

    # ---------------------------------------------------------------- results
    rows = []
    print(f"\nEYE TEST (CNN only): {args.label}")
    print(f"{'phase':<12} {'expected':<11} {'frames':>6} {'no_eyes':>8} "
          f"{'closed_%':>9} {'mean_P(closed)':>15}")
    for (name, expected, _), data in zip(PHASES, counted):
        frames = len(data)
        no_eyes = sum(1 for pc, _ in data if pc is None)
        seen = [pc for pc, _ in data if pc is not None]
        closed_pct = 100.0 * sum(1 for pc in seen if pc) / len(seen) if seen else float("nan")
        probs = [p for _, p in data if p is not None]
        mean_p = float(np.mean(probs)) if probs else float("nan")
        rows.append([args.label, name, expected, frames, no_eyes,
                     f"{closed_pct:.1f}", f"{mean_p:.3f}"])
        print(f"{name:<12} {expected:<11} {frames:>6} {no_eyes:>8} "
              f"{closed_pct:>9.1f} {mean_p:>15.3f}")
    print("closed_% = share of frames with eyes found that the CNN called closed "
          "(open and droopy should be low, closed should be high).")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.RESULTS_DIR / "eye_test_live.csv"
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label", "phase", "expected", "frames", "no_eyes_frames",
                        "closed_pct", "mean_p_closed", "saved_at"])
        stamp = datetime.now().isoformat(timespec="seconds")
        for r in rows:
            w.writerow(r + [stamp])
    print(f"Saved {csv_path}")

    try:                                          # P(closed) over time, for the paper
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = np.array([x[0] for x in trace])
        ps = np.array([x[1] for x in trace])
        fig, ax = plt.subplots(figsize=(8, 3.2))
        shades = ["#d9f0d3", "#fde0b8", "#f6c6c6"]
        for i, (name, _, length) in enumerate(PHASES):
            ax.axvspan(STARTS[i], STARTS[i] + length, color=shades[i], label=name)
        ax.plot(ts, ps, color="black", linewidth=1.2)
        ax.axhline(cfg.CLOSED_CONF_MIN, color="red", linestyle="--", linewidth=1,
                   label=f"closed threshold ({cfg.CLOSED_CONF_MIN:g})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("P(closed)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(args.label)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
        fig.tight_layout()
        slug = re.sub(r"[^A-Za-z0-9]+", "_", args.label).strip("_").lower() or "test"
        png = cfg.RESULTS_DIR / f"eye_test_{slug}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        print(f"Saved {png}")
    except Exception as err:
        print(f"(plot skipped: {err})")


if __name__ == "__main__":
    main()
