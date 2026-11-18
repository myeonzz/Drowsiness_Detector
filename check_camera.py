"""Quick diagnostic script to test connected webcams and find the active camera index."""
import cv2

def check_cameras(max_indices=4):
    print("Scanning available camera indices...")
    found_any = False

    for idx in range(max_indices):
        for api_name, api in [("DirectShow", cv2.CAP_DSHOW), ("Default", cv2.CAP_ANY)]:
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    mean_b = float(frame.mean())
                    max_b = int(frame.max())
                    h, w = frame.shape[:2]
                    status = "OK (Active)" if max_b > 10 else "BLACK / BLOCKED"
                    print(f"  Camera index {idx} [{api_name}]: {w}x{h}, Brightness={mean_b:.1f}/255, Max={max_b} -> {status}")
                    if max_b > 10:
                        found_any = True
                cap.release()

    if not found_any:
        print("\nAll camera feeds are black or inaccessible.")
        print("Check:")
        print("  1. Laptop physical privacy shutter / slider over the webcam lens.")
        print("  2. Function key (F8, F9, F10 or Fn+Camera).")
        print("  3. Windows Settings > Privacy & security > Camera > Allow desktop apps.")
    else:
        print("\nTo run the demo with an active camera, use:")
        print("  .\\.venv\\Scripts\\python.exe demo.py --source <index>")

if __name__ == "__main__":
    check_cameras()

