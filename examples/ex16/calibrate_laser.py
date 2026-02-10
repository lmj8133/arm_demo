#!/usr/bin/env python3
"""Laser calibration — record + analyze → LaserProfile.

Supports two modes:
    1. Record from camera then auto-analyze (--camera)
    2. Analyze an existing video file     (--video)

Example:
    # Record from camera → calibrate (press 'r' to toggle recording)
    python3 examples/ex16/calibrate_laser.py --camera 0 --rotate 90

    # Analyze an existing video
    python3 examples/ex16/calibrate_laser.py --video recording.avi

    # Record with custom duration limit
    python3 examples/ex16/calibrate_laser.py --camera 0 --duration 10

    # Manual ROI (skip quad detection)
    python3 examples/ex16/calibrate_laser.py --video recording.avi --roi 170,180,380,390
"""

import argparse
import datetime
import os
import sys
import time

# Add paths for local imports
sys.path.insert(0, os.path.dirname(__file__))

from laser_tracker import (
    DEFAULT_PROFILE_PATH,
    ROTATE_FLAGS,
    LaserProfile,
    LaserTracker,
    parse_roi,
)


def print_progress(current: int, total: int) -> None:
    """Print progress bar."""
    pct = current * 100 // max(1, total)
    bar_len = 40
    filled = bar_len * current // max(1, total)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {pct}% ({current}/{total})", end="", flush=True)


def record_video(
    camera: str,
    output_path: str,
    rotate: int = 0,
    width: int = 640,
    height: int = 480,
    duration: float = 0,
) -> bool:
    """Record calibration video from camera.

    Shows live preview first. Press 'r' to start recording,
    press 'r' again to stop and proceed to analysis. Press 'q' to abort.

    Args:
        camera: Camera device index (e.g. "0") or path
        output_path: Output video file path
        rotate: Rotation in degrees (0, 90, 180, 270)
        width: Camera capture width
        height: Camera capture height
        duration: Max recording duration in seconds (0 = unlimited)

    Returns:
        True if recording succeeded (at least 1 frame written)
    """
    import cv2

    rotate_flag = ROTATE_FLAGS.get(rotate)

    try:
        dev = int(camera)
    except ValueError:
        dev = camera

    cap = cv2.VideoCapture(dev)
    if isinstance(dev, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {camera}")
        return False

    # Read first frame to get actual dimensions (after rotation)
    ret, first = cap.read()
    if not ret:
        print("[ERROR] Cannot read from camera")
        cap.release()
        return False
    if rotate_flag is not None:
        first = cv2.rotate(first, rotate_flag)

    h, w = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    win = "Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    print(f"[PREVIEW] Camera {w}x{h} @ {fps:.0f}fps")
    print("[PREVIEW] Press 'r' to start recording, 'q' to quit")

    recording = False
    writer = None
    frame_count = 0
    start_time = 0.0
    elapsed = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rotate_flag is not None:
            frame = cv2.rotate(frame, rotate_flag)

        # Write frame if recording
        if recording:
            writer.write(frame)
            frame_count += 1
            elapsed = time.monotonic() - start_time

            # Check duration limit
            if duration > 0 and elapsed >= duration:
                print(f"\n[RECORD] Duration limit reached ({duration:.0f}s)")
                break

        # Build display
        display = frame.copy()
        if recording:
            # Red REC indicator
            cv2.circle(display, (w - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(display, f"REC {elapsed:.1f}s  frames={frame_count}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display, "[r] stop recording", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        else:
            cv2.putText(display, "PREVIEW", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(display, "[r] start recording  [q] quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if not recording:
                # Start recording
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    print(f"[ERROR] Cannot create video writer: {output_path}")
                    break
                recording = True
                start_time = time.monotonic()
                # Write the current frame (the one visible when 'r' was pressed)
                writer.write(frame)
                frame_count = 1
                dur_text = f" (max {duration:.0f}s)" if duration > 0 else ""
                print(f"[RECORD] Started{dur_text} — press 'r' again to stop")
            else:
                # Stop recording
                break

        elif key == ord("q"):
            if recording:
                # Stop recording and still analyze
                break
            else:
                # Quit without recording
                cap.release()
                cv2.destroyWindow(win)
                print("[INFO] Aborted (no recording)")
                return False

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyWindow(win)

    if frame_count > 0:
        print(f"[RECORD] Saved {frame_count} frames ({elapsed:.1f}s) -> {output_path}")
        return True

    print("[WARNING] No frames recorded")
    return False


def run_preview(
    video_path: str,
    profile: LaserProfile,
    roi: tuple | None,
    rotate: int,
    interval: int = 30,
) -> None:
    """Show sampled detection results with the derived profile."""
    import cv2

    rotate_flag = ROTATE_FLAGS.get(rotate)

    tracker = LaserTracker.from_profile(profile, roi=roi)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video for preview: {video_path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = "Calibration Preview"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("[PREVIEW] Press any key for next frame, 'q' to close")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % interval != 0:
            continue

        if rotate_flag is not None:
            frame = cv2.rotate(frame, rotate_flag)

        target = tracker.detect(frame)
        display = frame.copy()

        # Draw ROI
        if tracker.roi:
            x1, y1, x2, y2 = tracker.roi
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 100, 0), 1)

        # Draw detection
        if target:
            cx, cy = int(target.cx), int(target.cy)
            cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            radius = max(12, int(target.area ** 0.5 * 3))
            cv2.circle(display, (cx, cy), radius, (0, 255, 0), 1)
            label = f"br={target.brightness:.0f} a={target.area:.0f}"
            cv2.putText(display, label, (cx + 15, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        info = f"Frame {frame_idx}/{total}"
        det_str = "DETECTED" if target else "NOT DETECTED"
        cv2.putText(display, f"{info} - {det_str}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(win, display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyWindow(win)


def analyze_and_save(
    video_path: str,
    output: str,
    manual_roi: tuple | None,
    no_quad: bool,
    skip_frames: int,
    rotate: int,
    no_hue: bool,
    preview: bool,
) -> int:
    """Analyze a video file and save the derived LaserProfile."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print()
    print("=" * 50)
    print("Laser Calibration — Analyze")
    print("=" * 50)
    print(f"  Video:      {video_path}")
    print(f"  Frames:     {total} ({fps:.0f} fps, {w}x{h})")
    print(f"  Rotate:     {rotate}")
    roi_label = (
        f"manual {manual_roi}" if manual_roi
        else ("auto-quad" if not no_quad else "full frame")
    )
    print(f"  ROI:        {roi_label}")
    print()

    print("[CALIBRATE] Analyzing video...")
    try:
        profile, stats = LaserTracker.calibrate_profile_from_video(
            video_path=video_path,
            roi=manual_roi,
            auto_quad=not no_quad,
            skip_frames=skip_frames,
            rotate=rotate,
            progress_callback=print_progress,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        return 1

    print()  # newline after progress bar

    if no_hue:
        profile.use_hue_filter = False
        print("[INFO] Hue filter force-disabled (--no-hue)")

    profile.save(output)

    # Print summary
    roi_info = stats.get("roi")
    roi_source = stats.get("roi_source", "unknown")
    roi_str = f"{roi_source.upper()} {roi_info}" if roi_info else roi_source.upper()

    detected = stats["frames_detected"]
    processed = stats["frames_processed"]
    rate = stats["detection_rate"]

    print()
    print("Laser Calibration Results")
    print(f"  Video:     {os.path.basename(video_path)} ({total} frames, {fps:.0f}fps)")
    print(f"  ROI:       {roi_str}")
    print(f"  Detected:  {detected}/{processed} ({rate:.1%})")
    print(f"  brightness_threshold:  {profile.brightness_threshold}")
    print(f"  max_saturation:        {profile.max_saturation:>3}")
    print(f"  dot_area:              [{profile.min_dot_area}, {profile.max_dot_area}]")
    print(f"  use_hue_filter:        {profile.use_hue_filter}")
    if profile.use_hue_filter:
        print(f"  hue_red_low_upper:     {profile.hue_red_low_upper:>3}")
        print(f"  hue_red_high_lower:    {profile.hue_red_high_lower}")
        print(f"  hue_min_saturation:    {profile.hue_min_saturation:>3}")
    print(f"  Saved to: {output}")

    if preview:
        print()
        run_preview(video_path, profile, roi_info, rotate)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Laser calibration — record from camera or analyze existing video",
    )

    # Source: --camera or --video
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--camera", metavar="DEV", default=None,
        help="Camera device index or path (e.g. 0, /dev/video0). "
             "Records a calibration video, then auto-analyzes it.",
    )
    source.add_argument(
        "--video", metavar="PATH", default=None,
        help="Existing video file to analyze",
    )

    # Output
    parser.add_argument(
        "-o", "--output", default=DEFAULT_PROFILE_PATH, metavar="PATH",
        help=f"Output profile JSON path (default: {DEFAULT_PROFILE_PATH})",
    )

    # Recording options
    parser.add_argument(
        "--duration", type=float, default=0, metavar="SEC",
        help="Max recording duration in seconds (0 = unlimited, press 'q' to stop)",
    )
    parser.add_argument(
        "--save-video", metavar="PATH", default=None,
        help="Path to save recorded video (default: auto-generated timestamp name)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Camera capture width (default: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Camera capture height (default: 480)",
    )

    # Analysis options
    parser.add_argument(
        "--roi", type=str, default=None,
        help="Manual ROI as x1,y1,x2,y2 (skips quad detection)",
    )
    parser.add_argument(
        "--no-quad", action="store_true",
        help="Disable automatic quad detection",
    )
    parser.add_argument(
        "--rotate", type=int, choices=[0, 90, 180, 270], default=0,
        help="Rotate frames CW by degrees (default: 0)",
    )
    parser.add_argument(
        "--skip-frames", type=int, default=0, metavar="N",
        help="Process every N-th frame (0 = all)",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show sampled detection results after analysis (requires display)",
    )
    parser.add_argument(
        "--no-hue", action="store_true",
        help="Force-disable hue filter in output profile",
    )
    args = parser.parse_args()

    # Parse manual ROI
    manual_roi = None
    if args.roi:
        try:
            manual_roi = parse_roi(args.roi)
        except ValueError as e:
            print(f"[ERROR] Invalid ROI: {e}")
            return 1

    # Determine video path
    if args.camera is not None:
        # --- Record mode ---
        if args.save_video:
            video_path = args.save_video
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"calibration_{ts}.avi"

        print("=" * 50)
        print("Laser Calibration — Record + Analyze")
        print("=" * 50)

        ok = record_video(
            camera=args.camera,
            output_path=video_path,
            rotate=args.rotate,
            width=args.width,
            height=args.height,
            duration=args.duration,
        )
        if not ok:
            return 1
    else:
        video_path = args.video

    # --- Analyze ---
    return analyze_and_save(
        video_path=video_path,
        output=args.output,
        manual_roi=manual_roi,
        no_quad=args.no_quad,
        skip_frames=args.skip_frames,
        rotate=args.rotate,
        no_hue=args.no_hue,
        preview=args.preview,
    )


if __name__ == "__main__":
    sys.exit(main())
