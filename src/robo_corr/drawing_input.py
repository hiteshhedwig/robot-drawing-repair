"""Small OpenCV drawing tablet that records ordered, disconnected strokes."""

from dataclasses import dataclass, field

import cv2
import numpy as np

from robo_corr.error_detection import detect_missing_regions, dotted_error_overlay


WINDOW_NAME = "robo_corr manual input"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512
ERASER_RADIUS = 16


@dataclass
class StrokeRecorder:
    """Hold the desired drawing and the robot's observed/current drawing."""

    image: np.ndarray = field(
        default_factory=lambda: np.full(
            (IMAGE_HEIGHT, IMAGE_WIDTH, 3), 255, dtype=np.uint8
        )
    )
    strokes: list[list[tuple[int, int]]] = field(default_factory=list)
    current_image: np.ndarray = field(
        default_factory=lambda: np.full(
            (IMAGE_HEIGHT, IMAGE_WIDTH, 3), 255, dtype=np.uint8
        )
    )
    erase_points: list[tuple[int, int]] = field(default_factory=list)
    drawing: bool = False
    erasing: bool = False
    _version: int = 0
    _metrics_version: int = -1
    _missing_percent: float = 0.0
    _error_map: np.ndarray = field(
        default_factory=lambda: np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    )
    _error_overlay: np.ndarray = field(
        default_factory=lambda: np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    )
    disturbance_version: int = 0
    repair_preview: list[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        self.image.fill(255)
        self.current_image.fill(255)
        self.strokes.clear()
        self.erase_points.clear()
        self.drawing = False
        self.erasing = False
        self.repair_preview.clear()
        self.disturbance_version += 1
        self._version += 1

    def draw_robot_segment(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Update the simulated robot-output canvas from marker motion."""
        cv2.line(self.current_image, start, end, (25, 25, 25), 3, cv2.LINE_AA)
        self._version += 1

    def rebuild_current(
        self, pixel_segments: list[tuple[tuple[int, int], tuple[int, int]]]
    ) -> None:
        """Rebuild the simulated robot-output canvas after erasing."""
        self.current_image.fill(255)
        for start, end in pixel_segments:
            cv2.line(self.current_image, start, end, (25, 25, 25), 3, cv2.LINE_AA)
        self._version += 1

    @property
    def missing_percent(self) -> float:
        self._refresh_error_metrics()
        return self._missing_percent

    @property
    def error_map(self) -> np.ndarray:
        self._refresh_error_metrics()
        return self._error_map.copy()

    def _refresh_error_metrics(self) -> None:
        if self._metrics_version == self._version:
            return
        result = detect_missing_regions(
            self.image, self.current_image, reference_strokes=self.strokes
        )
        self._missing_percent = result.missing_percent
        self._error_map = result.error_map
        self._error_overlay = dotted_error_overlay(result.error_map)
        self._metrics_version = self._version

    def as_arrays(self) -> list[np.ndarray]:
        """Return non-empty strokes in the format used by the robot planner."""
        return [
            np.asarray(stroke, dtype=float)
            for stroke in self.strokes
            if len(stroke) > 1
        ]

    def display_image(self, status: str = "READY") -> np.ndarray:
        """Build the visible tablet image without changing recorded pixels."""
        self._refresh_error_metrics()
        left = self.image.copy()
        right = self.current_image.copy()
        right[self._error_overlay] = (0, 0, 255)
        for repair_stroke in self.repair_preview:
            integer_points = np.rint(repair_stroke).astype(np.int32)
            if len(integer_points) > 1:
                cv2.polylines(right, [integer_points], False, (0, 190, 230), 1, cv2.LINE_AA)
            for x, y in integer_points[::3]:
                cv2.rectangle(right, (x - 4, y - 4), (x + 4, y + 4), (0, 210, 255), 1)
        display = np.hstack((left, right))
        cv2.rectangle(
            display, (0, 0), (IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1), (180, 180, 180), 1
        )
        cv2.rectangle(
            display,
            (IMAGE_WIDTH, 0),
            (2 * IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1),
            (180, 180, 180),
            1,
        )
        cv2.putText(
            display,
            "REFERENCE / DESIRED",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 90, 170),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "CURRENT / ROBOT OUTPUT - drag here to erase",
            (IMAGE_WIDTH + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (50, 90, 170),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"{status}   E: execute   A: auto-repair   R/C: reset   Q: quit",
            (12, IMAGE_HEIGHT - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 130, 55) if status == "READY" else (40, 80, 190),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"MISSING / UNRECOVERED: {self._missing_percent:5.1f}%",
            (IMAGE_WIDTH + 12, IMAGE_HEIGHT - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            (
                "SIMULATED ROBOT OUTPUT"
            ),
            (IMAGE_WIDTH + 12, IMAGE_HEIGHT - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 60, 20),
            1,
            cv2.LINE_AA,
        )
        return display

    def mouse_callback(self, event: int, x: int, y: int, _flags: int, _data: object) -> None:
        on_reference = x < IMAGE_WIDTH
        point = (x, y)
        current_point = (x - IMAGE_WIDTH, y)

        if not on_reference:
            if event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
            if event == cv2.EVENT_LBUTTONDOWN:
                self.erasing = True
            if self.erasing and event in (
                cv2.EVENT_LBUTTONDOWN,
                cv2.EVENT_MOUSEMOVE,
                cv2.EVENT_LBUTTONUP,
            ):
                self.erase_points.append(current_point)
            if event == cv2.EVENT_LBUTTONUP:
                self.erasing = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.strokes.append([point])
            cv2.circle(self.image, point, 2, (25, 25, 25), -1, cv2.LINE_AA)
            self._version += 1
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            previous = self.strokes[-1][-1]
            if point != previous:
                self.strokes[-1].append(point)
                cv2.line(self.image, previous, point, (25, 25, 25), 3, cv2.LINE_AA)
                self._version += 1
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            previous = self.strokes[-1][-1]
            if point != previous:
                self.strokes[-1].append(point)
                cv2.line(self.image, previous, point, (25, 25, 25), 3, cv2.LINE_AA)
                self._version += 1
            self.drawing = False


def capture_strokes() -> list[np.ndarray] | None:
    """Return strokes after Execute, or None when the user cancels."""
    recorder = StrokeRecorder()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, recorder.mouse_callback)

    print("Draw with the left mouse button.")
    print("E = execute, C = clear, Q/Esc = cancel")

    while True:
        cv2.imshow(WINDOW_NAME, recorder.display_image())
        key = cv2.waitKey(16) & 0xFF

        if key in (ord("q"), 27):
            cv2.destroyWindow(WINDOW_NAME)
            return None
        if key == ord("c"):
            recorder.clear()
        if key == ord("e") and any(len(stroke) > 1 for stroke in recorder.strokes):
            cv2.destroyWindow(WINDOW_NAME)
            return recorder.as_arrays()
