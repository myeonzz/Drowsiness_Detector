"""Per-phase results of a timed eye test, from the newest demo frame log.

1. Run:  .\\.venv\\Scripts\\python.exe demo.py --mute      (do NOT use --no-record)
2. After the CALIBRATING text disappears, start a stopwatch:
       eyes open 20 s  ->  droopy 10 s  ->  eyes closed 5 s, then press Q
3. Run:  .\\.venv\\Scripts\\python.exe phase_table.py --label "no glasses, normal light"

It reads the newest events\\frames_*.csv, takes the end of the calibration as time 0,
ignores the first 1.5 s of each phase, and prints for each phase: frames counted,
frames with no eyes found, the share of frames the CNN calls closed, and the mean
P(closed). It appends the rows to results\\eye_test_cnn.csv and saves a plot of
P(closed) over time to results\\eye_test_<label>.png.

Options: --file <path>   use a specific frame log
         --start <s>     use this time (in log seconds) as the start of the eyes-open phase
"""
import argparse
import csv
import re
from datetime import datetime

import numpy as np

import config as cfg

SKIP_S = 1.5
PHASES = [("eyes open", "open", 20.0), ("droopy", "not closed", 10.0), ("eyes closed", "closed", 5.0)]


def num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description="Per-phase CNN results from a demo frame log")
    ap.add_argument("--label", default="test")
    ap.add_argument("--file", default=None)
    ap.add_argument("--start", type=float, default=None)
    args = ap.parse_args()

    if args.file:
        path = args.file
    else:
        logs = sorted(cfg.EVENTS_DIR.glob("frames_*.csv"))
        if not logs:
            raise SystemExit(f"No frames_*.csv in {cfg.EVENTS_DIR}. Run demo.py without --no-record first.")
        path = logs[-1]
    print(f"Reading {path}")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("The frame log is empty.")

    t = np.array([num(r["time_s"]) for r in rows], dtype=float)
    calib = [r.get("calibrating", "0") == "1" for r in rows]
    if args.start is not None:
        t0 = args.start
    elif any(calib):
        last = max(i for i, c in enumerate(calib) if c)      # last calibrating frame
        t0 = t[min(last + 1, len(t) - 1)]
    else:
        t0 = t[0]
    print(f"Eyes-open phase starts at {t0:.1f} s of the log (the log is {t[-1]:.1f} s long).")

    closed_flag = [num(r.get("eye_closed")) for r in rows]   # 1 / 0 / None (no eyes)
    p_cnn = [num(r.get("eye_cnn")) for r in rows]

    out, start = [], t0
    print(f"\nEYE TEST (CNN only): {args.label}")
    print(f"{'phase':<12} {'expected':<11} {'frames':>6} {'no_eyes':>8} "
          f"{'closed_%':>9} {'mean_P(closed)':>15}")
    for name, expected, length in PHASES:
        idx = [i for i in range(len(t)) if start + SKIP_S <= t[i] < start + length]
        frames = len(idx)
        no_eyes = sum(1 for i in idx if closed_flag[i] is None)
        seen = [closed_flag[i] for i in idx if closed_flag[i] is not None]
        closed_pct = 100.0 * sum(1 for v in seen if v) / len(seen) if seen else float("nan")
        probs = [p_cnn[i] for i in idx if p_cnn[i] is not None]
        mean_p = float(np.mean(probs)) if probs else float("nan")
        out.append([args.label, name, expected, frames, no_eyes, f"{closed_pct:.1f}", f"{mean_p:.3f}"])
        print(f"{name:<12} {expected:<11} {frames:>6} {no_eyes:>8} {closed_pct:>9.1f} {mean_p:>15.3f}")
        start += length
    print("closed_% = share of frames with eyes found that the CNN called closed "
          "(open and droopy should be low, closed should be high).")
    if t[-1] < t0 + sum(p[2] for p in PHASES) - 1:
        print("NOTE: the log ends before the 35 s of phases were finished, so the later phases are incomplete.")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.RESULTS_DIR / "eye_test_cnn.csv"
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label", "phase", "expected", "frames", "no_eyes_frames",
                        "closed_pct", "mean_p_closed", "saved_at"])
        stamp = datetime.now().isoformat(timespec="seconds")
        for r in out:
            w.writerow(r + [stamp])
    print(f"Saved {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = t - t0
        ps = np.array([np.nan if p is None else p for p in p_cnn], dtype=float)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        shades, s = ["#d9f0d3", "#fde0b8", "#f6c6c6"], 0.0
        for i, (name, _, length) in enumerate(PHASES):
            ax.axvspan(s, s + length, color=shades[i], label=name)
            s += length
        ax.plot(ts, ps, color="black", linewidth=1.2)
        ax.axhline(cfg.CLOSED_CONF_MIN, color="red", linestyle="--", linewidth=1,
                   label=f"closed threshold ({cfg.CLOSED_CONF_MIN:g})")
        ax.set_xlim(-2, s + 2)
        ax.set_xlabel("Time since the eyes-open phase started (s)")
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
