#!/usr/bin/env python3
"""Test plate detection without arm control.

Captures frames from camera and runs YOLOv8 detection.
Saves detection results to images for verification.

Usage:
    uv run python examples/ex14/test_detector.py
    uv run python examples/ex14/test_detector.py --camera 0 --model plate_recog_best.pt
    uv run python examples/ex14/test_detector.py --save-dir ./detections
"""

import argparse
import sys
import os
import time

import cv2

from camera import CameraCapture, CameraCaptureError
from detector import PlateDetector, PlateDetectorError


def main():
    parser = argparse.ArgumentParser(description="Test plate detection")
    parser.add_argument(
        "--camera", default="0",
        help="Camera device index or path (default: 0)"
    )
    parser.add_argument(
        "--model", default="plate_recog_best.pt",
        help="Path to YOLOv8 model file (default: plate_recog_best.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Directory to save detection images (optional)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable GUI display, save images instead"
    )
    args = parser.parse_args()

    # Parse camera device
    try:
        camera_device = int(args.camera)
    except ValueError:
        camera_device = args.camera

    # Create save directory if specified
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[INFO] Saving images to: {args.save_dir}")

    print("=" * 50)
    print("Plate Detection Test")
    print("=" * 50)

    # Initialize camera
    print(f"[INFO] Opening camera: {camera_device}")
    try:
        camera = CameraCapture(device=camera_device)
        camera.open()
        print(f"[OK] Camera: {camera.frame_size}")
    except CameraCaptureError as e:
        print(f"[ERROR] {e}")
        return 1

    # Initialize detector
    print(f"[INFO] Loading model: {args.model}")
    try:
        detector = PlateDetector(args.model, conf_threshold=args.conf)
        print(f"[OK] Model loaded")
    except PlateDetectorError as e:
        print(f"[ERROR] {e}")
        camera.release()
        return 1

    print()
    print("[INFO] Running detection...")
    if not args.no_display:
        print("[INFO] Press 'q' to quit, 's' to save current frame")
        cv2.namedWindow("Detection Test", cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    save_count = 0

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("[WARNING] Failed to capture frame")
                continue

            frame_count += 1
            h, w = frame.shape[:2]

            # Run detection
            detections = detector.detect(frame)

            # Draw results
            display_frame = frame.copy()

            for det in detections:
                x1, y1, x2, y2 = det.as_xyxy()
                cx, cy = det.center
                norm_x, norm_y = det.normalized_center(w, h)

                # Draw bbox
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(
                    display_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

                # Draw center
                cv2.circle(display_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                # Print detection info
                print(
                    f"[DET] {det.class_name}: "
                    f"conf={det.confidence:.2f}, "
                    f"center=({cx:.0f}, {cy:.0f}), "
                    f"norm=({norm_x:.3f}, {norm_y:.3f}), "
                    f"area={det.area:.0f}px"
                )

            # Show detection count
            count_text = f"Detections: {len(detections)}"
            cv2.putText(
                display_frame, count_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

            # Display or save
            if not args.no_display:
                cv2.imshow("Detection Test", display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("[INFO] Quit")
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"detection_{save_count:04d}.jpg"
                    if args.save_dir:
                        save_path = os.path.join(args.save_dir, save_path)
                    cv2.imwrite(save_path, display_frame)
                    print(f"[SAVED] {save_path}")
                    save_count += 1
            else:
                # Headless mode: auto-save if detections found
                if detections and args.save_dir:
                    save_path = os.path.join(
                        args.save_dir, f"detection_{save_count:04d}.jpg"
                    )
                    cv2.imwrite(save_path, display_frame)
                    print(f"[SAVED] {save_path}")
                    save_count += 1

                # Run for limited frames in headless mode
                if frame_count >= 100:
                    print(f"[INFO] Captured {frame_count} frames")
                    break

                time.sleep(0.1)

    except KeyboardInterrupt:
        print()
        print("[INFO] Interrupted")

    # Cleanup
    if not args.no_display:
        cv2.destroyAllWindows()
    camera.release()

    print(f"[OK] Done. Processed {frame_count} frames, saved {save_count} images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
