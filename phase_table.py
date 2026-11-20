"""Per-phase results of the eye CNN from a demo frame log (events\\frames_*.csv).

Usage (run right after a guided demo run):
    python phase_table.py --label "no glasses, normal light"

How the phases are found
    The test ends when you press Q at the end of the closed phase, so the
    phases are counted back from the LAST row of the log:
        closed  = last 12 s
        droopy  = the 10 s before that
        open    = the 20 s before that
    The first 1.5 s of every phase is ignored. If you pressed Q late, pass
    --trim-end <seconds> to cut that extra time off the end.

What "called closed" means
    The raw per-frame CNN call: min(P(closed) of both eyes) >= CLOSED_CONF_MIN
    (read from config.py, 0.70 if unreadable). No smoothing and no
    closure tolerance are applied here.
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    import config as cfg
    THRESH = float(cfg.CLOSED_CONF_MIN)
except Exception:
    THRESH = 0.70

PHASES = ("eyes open", "droopy", "eyes closed")
EXPECTED = {"eyes open": "open", "droopy": "not closed", "eyes closed": "closed"}


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def latest_log(folder):
    files = sorted(Path(folder).glob("frames_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"No frames_*.csv found in {folder}. Run demo.py first.")
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help='condition, e.g. "glasses, glare"')
    ap.add_argument("--file", help="frame log to read (default: newest in events/)")
    ap.add_argument("--events-dir", default="events")
    ap.add_argument("--out", default="results/eye_test_cnn.csv")
    ap.add_argument("--open-s", type=float, default=20.0)
    ap.add_argument("--droopy-s", type=float, default=10.0)
    ap.add_argument("--closed-s", type=float, default=12.0)
    ap.add_argument("--skip-s", type=float, default=1.5, help="ignored at the start of each phase")
    ap.add_argument("--trim-end", type=float, default=0.0, help="seconds to cut off the end (late Q)")
    ap.add_argument("--threshold", type=float, default=THRESH)
    a = ap.parse_args()

    path = Path(a.file) if a.file else latest_log(a.events_dir)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} is empty.")
    if "time_s" not in rows[0] or "eye_cnn" not in rows[0]:
        sys.exit(f"Columns time_s / eye_cnn not found. Columns are: {list(rows[0].keys())}")

    t = [num(r["time_s"]) for r in rows]
    p = [num(r["eye_cnn"]) for r in rows]
    has_occ = "occluded" in rows[0]
    occ = [str(r.get("occluded", "")).strip().lower() in ("1", "true", "yes", "y") for r in rows]

    t_end = max(x for x in t if x is not None) - a.trim_end
    b_closed = t_end - a.closed_s
    b_droopy = b_closed - a.droopy_s
    b_open = b_droopy - a.open_s
    t_start = min(x for x in t if x is not None)
    windows = {
        "eyes open": (b_open + a.skip_s, b_droopy),
        "droopy": (b_droopy + a.skip_s, b_closed),
        "eyes closed": (b_closed + a.skip_s, t_end),
    }
    if b_open < t_start - 0.5:
        print(f"WARNING: the log is only {t_end - t_start:.1f} s long but the three phases need "
              f"{a.open_s + a.droopy_s + a.closed_s:.0f} s. The open phase is cut short.")

    out_rows = []
    print(f"\nEYE TEST (CNN only, raw per-frame call, no smoothing): {a.label}")
    print(f"log: {path}   threshold P(closed) >= {a.threshold:.2f}")
    head = f"{'phase':<12}{'expected':<12}{'frames':>7}{'no_eyes':>9}"
    if has_occ:
        head += f"{'occluded':>10}"
    print(head + f"{'closed_%':>10}{'mean_P(closed)':>16}")
    for name in PHASES:
        lo, hi = windows[name]
        idx = [i for i, x in enumerate(t) if x is not None and lo <= x < hi]
        frames = len(idx)
        found = [p[i] for i in idx if p[i] is not None]
        no_eyes = frames - len(found)
        n_occ = sum(1 for i in idx if occ[i] and p[i] is None) if has_occ else None
        called = sum(1 for v in found if v >= a.threshold)
        pct = 100.0 * called / len(found) if found else float("nan")
        mean_p = sum(found) / len(found) if found else float("nan")
        line = f"{name:<12}{EXPECTED[name]:<12}{frames:>7}{no_eyes:>9}"
        if has_occ:
            line += f"{n_occ:>10}"
        print(line + f"{pct:>10.1f}{mean_p:>16.3f}")
        out_rows.append([a.label, name, EXPECTED[name], frames, no_eyes,
                         "" if n_occ is None else n_occ,
                         f"{pct:.1f}", f"{mean_p:.3f}", a.threshold, path.name])
    print("closed_% = share of frames with eyes found that the CNN called closed "
          "(open and droopy should be low, closed should be high).")
    print("no_eyes = frames with no CNN result (face not found"
          + (" or eyes flagged occluded" if not has_occ else "") + ").")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "phase", "expected", "frames", "no_eyes", "occluded",
                        "closed_pct", "mean_p_closed", "threshold", "log_file"])
        w.writerows(out_rows)
    print(f"Saved {out}")

    # P(closed) over time, with the three phases shaded
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.2))
        xs = [(x - b_open) for x, v in zip(t, p) if x is not None and v is not None]
        ys = [v for x, v in zip(t, p) if x is not None and v is not None]
        ax.plot(xs, ys, lw=1, color="tab:blue")
        ax.axhline(a.threshold, color="gray", ls="--", lw=1)
        for name, color in zip(PHASES, ("#d9f2d9", "#fff0cc", "#f8d7d7")):
            lo, hi = windows[name]
            ax.axvspan(lo - a.skip_s - b_open, hi - b_open, color=color, zorder=0)
            ax.text((lo - a.skip_s + hi) / 2 - b_open, 1.02, name, ha="center", fontsize=8)
        ax.set_ylim(-0.02, 1.1)
        ax.set_xlabel("time since start of open phase (s)")
        ax.set_ylabel("P(closed)")
        ax.set_title(a.label, fontsize=10)
        fig.tight_layout()
        png = out.parent / ("eye_test_" + "".join(c if c.isalnum() else "_" for c in a.label).strip("_") + ".png")
        fig.savefig(png, dpi=150)
        print(f"Saved {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
