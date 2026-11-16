"""Guided eye and yawn test on the webcam (about 2.5 minutes).

Follow the yellow prompts on screen. A beep marks every change of phase, so
you know when to open your eyes again.

    1. calibration   6 s   eyes open, mouth closed, normal face
    Eye check:
    2. eyes open    20 s   look at the camera normally (blinking is fine)
    3. droopy       10 s   eyes half-closed / droopy, but not shut
    4. eyes closed   5 s   eyes fully closed until the beep
    Yawn check:
    5. talk         30 s   talk normally (count aloud, read something)
    6. neutral      30 s   mouth closed, look at the camera
    7. yawn x3      15 s each: one real, wide yawn held for 4 s or more

Eye check report: how each rule (cnn / ear / either / both) classifies each
part. Yawn check report: false yawns (during talk / neutral) and missed yawns
(yawn slots with no detection).

    python yawn_test.py                  # everything
    python yawn_test.py --part eyes      # only the eye check
    python yawn_test.py --part yawn      # only the yawn check
    python yawn_test.py --source my_recording.mp4

Writes to .\\results: eye_test.csv, eye_test_closure.png, yawn_test.csv,
yawn_test_mar.png and guided_test_frames.csv (every frame's signals).
"""
import argparse
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as cfg
import demo

EYE_SETTLE_S = 1.5        # ignore the start of each eye phase (time to react)
RULES = ("cnn", "ear", "either", "both")

# (name, seconds, prompt, kind)
CALIBRATION = [("calibration", cfg.CALIBRATION_S + 1, "", "calibration")]
EYE_PHASES = [
    ("eyes open", 20, "EYES OPEN: look at the camera", "eye"),
    ("droopy", 10, "DROOPY: eyes half-closed, not shut", "eye"),
    ("eyes closed", 5, "CLOSE your eyes until the beep", "eye"),
]
YAWN_PHASES = [
    ("talk", 30, "TALK normally", "quiet"),
    ("neutral", 30, "NEUTRAL: mouth closed", "quiet"),
    ("yawn 1", 15, "YAWN 1 of 3: hold 4 s or more", "yawn"),
    ("yawn 2", 15, "YAWN 2 of 3: hold 4 s or more", "yawn"),
    ("yawn 3", 15, "YAWN 3 of 3: hold 4 s or more", "yawn"),
]
# What a correct system should say in each eye phase.
EYE_EXPECTED = {"eyes open": "open", "droopy": "open (not 80% closed)", "eyes closed": "closed"}


def longest_run(times, flags):
    """Longest stretch of True flags in seconds, allowing the same number of
    non-yawn frames inside a yawn as the alert logic does."""
    best, start, misses = 0.0, None, 0
    for t, flag in zip(times, flags):
        if flag:
            start = t if start is None else start
            misses = 0
            best = max(best, t - start)
        elif start is not None:
            misses += 1
            if misses > cfg.YAWN_FRAME_TOLERANCE:
                start = None
    return best


def beep():
    try:
        import winsound
        threading.Thread(target=winsound.Beep, args=(1000, 200), daemon=True).start()
    except ImportError:
        pass


def shade_phases(ax, phases, top):
    start = 0.0
    for i, (name, seconds, _, _) in enumerate(phases):
        ax.axvspan(start, start + seconds, color="gray", alpha=0.08 if i % 2 else 0.18)
        ax.text(start + seconds / 2, top, name, ha="center", va="top", fontsize=8)
        start += seconds


def eye_report(df, phases):
    """Per eye phase: % of frames each rule calls closed, and the medians."""
    rows, start = [], 0.0
    for name, seconds, _, kind in phases:
        end = start + seconds
        if kind == "eye":
            part = df[(df["time_s"] >= start + EYE_SETTLE_S) & (df["time_s"] < end)
                      & df["eye_cnn"].notna()]
            row = {"phase": name, "expected": EYE_EXPECTED[name], "frames": len(part),
                   "median_ear": round(part["ear"].median(), 3),
                   "median_closure_pct": round(part["closure"].median() * 100, 1),
                   "median_cnn_p_closed": round(part["eye_cnn"].median(), 3)}
            for rule in RULES:
                closed = [demo.eye_closed_by_rule(c, e, rule)
                          for c, e in zip(part["cnn_closed"], part["ear_closed"])]
                row[f"closed_pct_{rule}"] = round(100 * sum(closed) / max(len(closed), 1), 1)
            rows.append(row)
        start = end
    return pd.DataFrame(rows)


def yawn_report(df, phases, yawn_times):
    rows, start = [], 0.0
    for name, seconds, _, kind in phases:
        end = start + seconds
        if kind in ("quiet", "yawn"):
            part = df[(df["time_s"] >= start) & (df["time_s"] < end)]
            expected = 1 if kind == "yawn" else 0
            found = sum(1 for t in yawn_times if start <= t < end)
            above = part["mar"] >= part["mar_threshold"]
            rows.append({
                "phase": name, "start_s": start, "end_s": end,
                "expected_yawns": expected, "detected_yawns": found,
                "false_yawns": max(found - expected, 0),
                "missed_yawns": max(expected - found, 0),
                "peak_mar": round(part["mar"].max(), 3) if part["mar"].notna().any() else None,
                "longest_yawn_run_s": round(longest_run(
                    part["time_s"].tolist(), (part["yawn_frame"] == 1).tolist()), 1),
                "frames_above_threshold": int(above.sum()),
                "frames_vetoed_by_cnn": int((above & (part["yawn_frame"] == 0)
                                             & part["mouth_cnn"].notna()).sum()),
            })
        start = end
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Guided eye and yawn test")
    ap.add_argument("--source", default="0", help="webcam index or a video of the same routine")
    ap.add_argument("--part", choices=("all", "eyes", "yawn"), default="all")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    phases = (CALIBRATION + (EYE_PHASES if args.part in ("all", "eyes") else [])
              + (YAWN_PHASES if args.part in ("all", "yawn") else []))
    total = sum(p[1] for p in phases)

    def phase_at(t):
        end = 0.0
        for i, (_, seconds, _, _) in enumerate(phases):
            end += seconds
            if t < end:
                return i, end - t
        return None, 0.0

    frames, last_phase = [], [0]

    def on_frame(t, det, state):
        i, left = phase_at(t)
        if i != last_phase[0]:                 # phase changed: audible cue
            last_phase[0] = i
            if isinstance(source, int):
                beep()
        frames.append({
            "time_s": round(t, 2), "phase": phases[i][0] if i is not None else "",
            "ear": det["ear"], "ear_open": det["ear_open"], "closure": det["closure"],
            "eye_cnn": det["p_eye_cnn"], "cnn_closed": det["cnn_closed"],
            "ear_closed": det["ear_closed"], "eye_closed": det["p_closed"],
            "perclos_p80": state["perclos"],
            "mar": det["mar"], "mar_threshold": det["mar_threshold"],
            "mouth_cnn": det["p_mouth_cnn"], "yawn_frame": det["p_yawn"],
        })
        if i is None or not phases[i][2]:
            return ""
        return f"{phases[i][2]}  ({left:.0f} s)"

    summary = demo.run(source, display=not args.no_display, sound=False, record=False,
                       max_seconds=total, on_frame=on_frame)
    df = pd.DataFrame(frames)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(cfg.RESULTS_DIR / "guided_test_frames.csv", index=False)
    pd.set_option("display.width", 240)
    print(f"\nCalibration: {summary['calibration']}; open-eye EAR {summary['ear_open']}; "
          f"yawn threshold MAR >= {summary['mar_threshold'] or 0:.3f}")

    # ------------------------------------------------------------- eyes ---
    if args.part in ("all", "eyes"):
        eyes = eye_report(df, phases)
        eyes.to_csv(cfg.RESULTS_DIR / "eye_test.csv", index=False)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["time_s"], df["closure"] * 100, linewidth=1.2, label="EAR closure %")
        ax.plot(df["time_s"], df["eye_cnn"] * 100, linewidth=1.0, alpha=0.7,
                label="eye CNN P(closed) %")
        ax.axhline(cfg.EAR_CLOSURE_MIN * 100, color="red", linestyle="--",
                   label=f"closed at {cfg.EAR_CLOSURE_MIN:.0%} closure")
        ax.axhline(cfg.CLOSED_CONF_MIN * 100, color="orange", linestyle=":",
                   label=f"CNN closed at {cfg.CLOSED_CONF_MIN:.2f}")
        ax.set(xlabel="time (s)", ylabel="%", xlim=(0, total), ylim=(0, 112),
               title="Eye check: EAR closure and eye CNN over time")
        shade_phases(ax, phases, 110)
        ax.legend(loc="center left", fontsize=8)
        fig.tight_layout()
        fig.savefig(cfg.RESULTS_DIR / "eye_test_closure.png", dpi=150)
        plt.close(fig)
        print("\nEYE CHECK - % of frames each rule calls closed "
              f"(first {EYE_SETTLE_S} s of each phase ignored):")
        print(eyes.to_string(index=False))
        print(f"Rule in use: {cfg.EYE_CLOSED_RULE}")

    # ------------------------------------------------------------ yawns ---
    if args.part in ("all", "yawn"):
        yawn_times = [e["time"] for e in summary["events"] if e["trigger"] == "long_yawn"]
        yawns = yawn_report(df, phases, yawn_times)
        yawns.to_csv(cfg.RESULTS_DIR / "yawn_test.csv", index=False)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["time_s"], df["mar"], linewidth=1.2, label="MAR")
        if summary["mar_threshold"] is not None:
            ax.axhline(summary["mar_threshold"], color="red", linestyle="--",
                       label=f"threshold {summary['mar_threshold']:.2f}")
        for t in yawn_times:
            ax.axvline(t, color="green", linewidth=2, alpha=0.7)
        top = max(1.0, float(df["mar"].max() or 0) * 1.15)
        ax.set(xlabel="time (s)", ylabel="MAR", xlim=(0, total), ylim=(0, top),
               title="Yawn check: MAR over time (green line = yawn counted)")
        shade_phases(ax, phases, top * 0.98)
        ax.legend(loc="center left", fontsize=8)
        fig.tight_layout()
        fig.savefig(cfg.RESULTS_DIR / "yawn_test_mar.png", dpi=150)
        plt.close(fig)
        print("\nYAWN CHECK:")
        print(yawns.to_string(index=False))
        print(f"Mouth CNN check: {'on' if cfg.YAWN_USE_CNN else 'off'}")
        print(f"False yawns (talk + neutral): {int(yawns['false_yawns'].sum())}")
        print(f"Missed yawns: {int(yawns['missed_yawns'].sum())} of "
              f"{int(yawns['expected_yawns'].sum())}")

    if summary["duration_s"] < total - 2:
        print(f"\nNOTE: the run stopped at {summary['duration_s']} s of {total:.0f} s, "
              "so later phases are incomplete.")
    print(f"\nSaved results in {cfg.RESULTS_DIR}")
