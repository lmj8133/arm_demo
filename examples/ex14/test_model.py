#!/usr/bin/env python3
"""Quick model diagnostic - capture one frame and show all detections."""

import sys
import cv2


def main():
    # 1. Check model file
    model_path = sys.argv[1] if len(sys.argv) > 1 else "plate_recog_best.pt"
    print(f"[1] Model: {model_path}")

    from pathlib import Path

    if not Path(model_path).exists():
        print("    ERROR: File not found!")
        return 1
    print(f"    OK: File exists ({Path(model_path).stat().st_size / 1024:.1f} KB)")

    # 2. Load model
    print("[2] Loading model...")
    from ultralytics import YOLO

    model = YOLO(model_path)
    print("    OK: Model loaded")
    print(f"    Classes: {model.names}")

    # 3. Capture frame
    print("[3] Capturing frame from camera 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("    ERROR: Cannot open camera")
        return 1

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("    ERROR: Cannot capture frame")
        return 1

    h, w = frame.shape[:2]
    print(f"    OK: Frame captured ({w}x{h})")

    # Save raw frame
    cv2.imwrite("debug_raw.jpg", frame)
    print("    Saved: debug_raw.jpg")

    # 4. Run inference with very low threshold
    print("[4] Running inference (conf=0.01)...")
    results = model(frame, conf=0.01, verbose=True)

    # 5. Show all detections
    print("[5] Results:")
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            print("    No detections!")
        else:
            print(f"    Found {len(boxes)} detection(s):")
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = result.names.get(cls_id, str(cls_id))
                print(f"      [{i}] {cls_name}: conf={conf:.3f}, bbox={xyxy}")

                # Draw on frame
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{cls_name} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

    # Save annotated frame
    cv2.imwrite("debug_annotated.jpg", frame)
    print("[6] Saved: debug_annotated.jpg")

    print("\nDone! Check debug_raw.jpg and debug_annotated.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
