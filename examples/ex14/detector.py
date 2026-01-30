"""YOLOv8 plate detection module.

Provides PlateDetector class for detecting license plates using YOLOv8.

Example:
    detector = PlateDetector("plate_recog_best.pt", conf_threshold=0.5)
    detections = detector.detect(frame)
    for det in detections:
        print(f"Plate at {det.center} with confidence {det.confidence:.2f}")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    """Single detection result.

    Attributes:
        x1: Bounding box left coordinate (pixels)
        y1: Bounding box top coordinate (pixels)
        x2: Bounding box right coordinate (pixels)
        y2: Bounding box bottom coordinate (pixels)
        confidence: Detection confidence (0-1)
        class_id: Class index
        class_name: Class name string
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point (cx, cy) in pixels."""
        cx = (self.x1 + self.x2) / 2
        cy = (self.y1 + self.y2) / 2
        return (cx, cy)

    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Get bounding box area in pixels squared."""
        return self.width * self.height

    def normalized_center(self, img_width: int, img_height: int) -> Tuple[float, float]:
        """Get center normalized to 0-1 range.

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            (norm_x, norm_y) where both values are in [0, 1]
        """
        cx, cy = self.center
        return (cx / img_width, cy / img_height)

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        """Get bounding box as integer (x1, y1, x2, y2)."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def __repr__(self) -> str:
        cx, cy = self.center
        return (
            f"Detection({self.class_name}, "
            f"center=({cx:.0f}, {cy:.0f}), "
            f"conf={self.confidence:.2f})"
        )


class PlateDetectorError(Exception):
    """Exception raised for detector-related errors."""

    pass


class PlateDetector:
    """YOLOv8 plate detector.

    Example:
        detector = PlateDetector("plate_recog_best.pt")
        detections = detector.detect(frame)
        largest = detector.detect_largest(frame)
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        """Initialize PlateDetector.

        Args:
            model_path: Path to YOLOv8 model file (.pt)
            conf_threshold: Minimum confidence threshold (0-1, default: 0.5)
            device: Device to run inference on (None=auto, "cpu", "cuda:0")

        Raises:
            PlateDetectorError: If model cannot be loaded
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

        if not self.model_path.exists():
            raise PlateDetectorError(f"Model file not found: {self.model_path}")

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
            if device:
                self._model.to(device)
        except ImportError:
            raise PlateDetectorError(
                "ultralytics package not installed. Run: pip install ultralytics"
            )
        except Exception as e:
            raise PlateDetectorError(f"Failed to load model: {e}") from e

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a frame.

        Args:
            frame: BGR image as numpy array (H, W, C)

        Returns:
            List of Detection objects sorted by confidence (descending)
        """
        results = self._model(frame, verbose=False, conf=self.conf_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names.get(cls_id, str(cls_id))

                det = Detection(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )
                detections.append(det)

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_largest(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect and return the largest detection by area.

        Args:
            frame: BGR image as numpy array (H, W, C)

        Returns:
            Detection with largest bounding box area, or None if no detections
        """
        detections = self.detect(frame)
        if not detections:
            return None

        return max(detections, key=lambda d: d.area)

    def __repr__(self) -> str:
        return (
            f"PlateDetector(model={self.model_path.name}, conf={self.conf_threshold})"
        )
