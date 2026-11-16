"""Run every test video in ./scenarios and compare the highest alert level
reached with the expected one. Writes ./results/scenarios.csv.

The expected level comes from keywords in the file name (see
SCENARIO_EXPECTED in config.py), e.g. alert.mp4, normal_blinking.mp4,
slow_eye_closure.mp4, yawning.mp4, sunglasses.mp4, low_light.mp4.

    python scenarios.py            # headless, fast
    python scenarios.py --show     # also show the video window
"""
import argparse

import pandas as pd

import config as cfg
import demo


def expected_level(filename):
    stem = filename.lower().replace("-", "_").replace(" ", "_")
    for keyword, level in cfg.SCENARIO_EXPECTED.items():
        if keyword in stem:
            return level
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="show the video while processing")
    ap.add_argument("--record", action="store_true", help="save Level 2/3 clips to ./events")
    args = ap.parse_args()

    cfg.SCENARIOS_DIR.mkdir(exist_ok=True)
    videos = sorted(p for p in cfg.SCENARIOS_DIR.iterdir()
                    if p.suffix.lower() in cfg.VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"No videos found in {cfg.SCENARIOS_DIR}. Add your test clips "
                         "(e.g. alert.mp4, yawning.mp4) and run again.")

    rows = []
    for video in videos:
        print(f"\n--- {video.name} ---")
        expected = expected_level(video.name)
        s = demo.run(video, display=args.show, sound=False, record=args.record)
        triggers = sorted({e["trigger"] for e in s["events"]})
        rows.append({
            "video": video.name,
            "duration_s": s["duration_s"],
            "expected_level": "?" if expected is None else expected,
            "highest_level": s["max_level"],
            "highest_level_name": cfg.LEVEL_NAMES[s["max_level"]],
            "time_to_highest_s": "" if s["time_to_max_s"] is None else s["time_to_max_s"],
            "triggers": ";".join(triggers),
            "correct": "?" if expected is None else ("yes" if s["max_level"] == expected else "no"),
        })

    df = pd.DataFrame(rows)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    out = cfg.RESULTS_DIR / "scenarios.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.to_string(index=False))
    known = df[df["correct"] != "?"]
    print(f"\n{(known['correct'] == 'yes').sum()} of {len(known)} scenarios correct. Saved {out}")
