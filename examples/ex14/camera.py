"""USB camera capture module.

Provides CameraCapture class for capturing frames from USB cameras.

Example:
    with CameraCapture(device=0, width=640, height=480) as cam:
        ret, frame = cam.read()
        if ret:
            cv2.imshow("Frame", frame)
"""

from typing import Tuple, Union

import cv2
import numpy as np


class CameraCaptureError(Exception):
    """Exception raised for camera-related errors."""
    pass


class CameraCapture:
    """USB camera frame capture with OpenCV.

    Example:
        # Using context manager (recommended)
        with CameraCapture(device=0) as cam:
            ret, frame = cam.read()

        # Manual usage
        cam = CameraCapture(device="/dev/video0")
        cam.open()
        ret, frame = cam.read()
        cam.release()
    """

    def __init__(
        self,
        device: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """Initialize CameraCapture.

        Args:
            device: Camera device index (0) or path ("/dev/video0")
            width: Frame width in pixels (default: 640)
            height: Frame height in pixels (default: 480)
            fps: Target frame rate (default: 30)
        """
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: cv2.VideoCapture = None
        self._is_open = False

    def open(self) -> "CameraCapture":
        """Open camera device.

        Returns:
            self for method chaining

        Raises:
            CameraCaptureError: If camera cannot be opened
        """
        if self._is_open:
            return self

        self._cap = cv2.VideoCapture(self.device)

        if not self._cap.isOpened():
            raise CameraCaptureError(
                f"Failed to open camera device: {self.device}"
            )

        # Set resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Verify actual settings
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width != self.width or actual_height != self.height:
            # Update to actual values
            self.width = actual_width
            self.height = actual_height

        self._is_open = True
        return self

    def read(self) -> Tuple[bool, np.ndarray]:
        """Capture a single frame.

        Returns:
            Tuple of (success, frame). frame is BGR numpy array.

        Raises:
            CameraCaptureError: If camera is not open
        """
        if not self._is_open or self._cap is None:
            raise CameraCaptureError("Camera is not open. Call open() first.")

        return self._cap.read()

    def release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    @property
    def is_open(self) -> bool:
        """Check if camera is currently open."""
        return self._is_open

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get current frame size (width, height)."""
        return (self.width, self.height)

    def __enter__(self) -> "CameraCapture":
        """Enter context manager - open camera."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - release camera."""
        self.release()

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return (
            f"CameraCapture(device={self.device}, "
            f"size={self.width}x{self.height}, status={status})"
        )
