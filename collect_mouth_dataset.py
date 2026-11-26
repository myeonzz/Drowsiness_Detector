"""Capture 96x96 mouth crops for retraining the yawn CNN."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

import common
import config
import demo


def main():
    ap = argparse.ArgumentParser(description="Collect webcam mouth-crop samples")
    ap.add_argument("--label", choices=("no_yawn", "yawn"), required=True)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--source", default="0")
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="countdown before capturing")
    args = ap.parse_args()

    if args.count < 1:
        raise SystemExit("--count must be at least 1")

    source = int(args.source) if args.source.isdigit() else args.source
    live = isinstance(source, int)
    cap, native = demo.open_camera(source) if live else (
        cv2.VideoCapture(str(source)), None)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera/video source: {source}")

    finder = common.EyeFinder()
    out_dir = config.DATA_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    start = time.monotonic()
    last_capture = -1.0
    window = "Collect mouth dataset"

    print(f"Collecting {args.count} {args.label} samples. Press Q or Esc to stop.")
    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                break
            frame = demo.fit_frame(frame, config.FRAME_SIZE)
            found = finder.find(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            box = None
            if found is not None:
                box = common.mouth_box_yunet(found, frame.shape)

            elapsed = time.monotonic() - start
            if elapsed < args.seconds:
                message = f"GET READY {args.label}: {args.seconds - elapsed:.1f}s"
            elif box is not None:
                message = f"Capturing {args.label}: {saved}/{args.count}"
            else:
                message = "Face not detected"

            display = frame.copy()
            if box is not None:
                x0, y0, x1, y1 = box
                cv2.rectangle(display, (x0, y0), (x1, y1), (0, 220, 0), 2)

            cv2.putText(display, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

            if elapsed >= args.seconds and box is not None:
                now = time.monotonic()
                if now - last_capture >= 0.2:
                    crop = common.preprocess(common.crop(gray, box))
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = out_dir / f"live_{stamp}.jpg"
                    cv2.imwrite(str(path), crop)
                    saved += 1
                    last_capture = now
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} images to {out_dir}")


if __name__ == "__main__":
    main()
