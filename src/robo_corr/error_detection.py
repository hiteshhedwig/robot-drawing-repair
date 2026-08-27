"""Compare desired and current drawing canvases to find missing regions."""

from dataclasses import dataclass

import cv2
import numpy as np


INK_THRESHOLD = 200
MATCH_TOLERANCE_PIXELS = 6
PERPENDICULAR_TOLERANCE_PIXELS = 8.0
TANGENTIAL_TOLERANCE_PIXELS = 2.0
ENDPOINT_TANGENTIAL_TOLERANCE_PIXELS = 12.0
PATH_SAMPLE_SPACING_PIXELS = 2.0


@dataclass(frozen=True)
class MissingRegionResult:
    """Pixel-level result used by the UI and later repair planner."""

    desired_mask: np.ndarray
    current_mask: np.ndarray
    error_map: np.ndarray
    missing_percent: float
    desired_pixels: int
    missing_pixels: int


def detect_missing_regions(
    reference_canvas: np.ndarray,
    current_canvas: np.ndarray,
    tolerance_pixels: int = MATCH_TOLERANCE_PIXELS,
    reference_strokes: list[list[tuple[int, int]]] | None = None,
) -> MissingRegionResult:
    """Return desired ink not represented near the current robot output.

    Recorded strokes use directional matching: generous perpendicular tolerance
    handles tracking offset while tight tangential tolerance preserves gaps. The
    returned error_map stays aligned with the desired canvas for repair planning.
    """
    desired_gray = cv2.cvtColor(reference_canvas, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_canvas, cv2.COLOR_BGR2GRAY)
    desired_mask = desired_gray < INK_THRESHOLD
    current_mask = current_gray < INK_THRESHOLD

    if reference_strokes:
        error_map, desired_pixels, missing_pixels = _path_aware_error_map(
            reference_strokes, current_mask
        )
    else:
        kernel_size = 2 * tolerance_pixels + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        current_neighborhood = cv2.dilate(current_mask.astype(np.uint8), kernel) > 0
        error_map = desired_mask & ~current_neighborhood
        desired_pixels = int(np.count_nonzero(desired_mask))
        missing_pixels = int(np.count_nonzero(error_map))
    missing_percent = (
        100.0 * missing_pixels / desired_pixels if desired_pixels else 0.0
    )
    return MissingRegionResult(
        desired_mask=desired_mask,
        current_mask=current_mask,
        error_map=error_map,
        missing_percent=missing_percent,
        desired_pixels=desired_pixels,
        missing_pixels=missing_pixels,
    )


def _resample_path(points: np.ndarray) -> np.ndarray:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segments)))
    keep = np.concatenate(([True], np.diff(distance) > 1e-9))
    points = points[keep]
    distance = distance[keep]
    if len(points) < 2 or distance[-1] == 0:
        return points
    samples = np.arange(0.0, distance[-1], PATH_SAMPLE_SPACING_PIXELS)
    samples = np.append(samples, distance[-1])
    return np.column_stack(
        [np.interp(samples, distance, points[:, axis]) for axis in range(2)]
    )


def _path_aware_error_map(
    reference_strokes: list[list[tuple[int, int]]], current_mask: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Match ink with an oriented neighborhood around the desired path."""
    height, width = current_mask.shape
    error_map = np.zeros_like(current_mask, dtype=np.uint8)
    total_samples = 0
    missing_samples = 0
    search_radius = int(np.ceil(PERPENDICULAR_TOLERANCE_PIXELS))

    for recorded in reference_strokes:
        points = _resample_path(np.asarray(recorded, dtype=float))
        if len(points) < 2:
            continue
        missing_flags = np.zeros(len(points), dtype=bool)
        for index, point in enumerate(points):
            before = points[max(0, index - 1)]
            after = points[min(len(points) - 1, index + 1)]
            tangent = after - before
            tangent_length = np.linalg.norm(tangent)
            if tangent_length == 0:
                tangent = np.asarray([1.0, 0.0])
            else:
                tangent /= tangent_length
            normal = np.asarray([-tangent[1], tangent[0]])

            x0 = max(0, int(np.floor(point[0])) - search_radius)
            x1 = min(width, int(np.ceil(point[0])) + search_radius + 1)
            y0 = max(0, int(np.floor(point[1])) - search_radius)
            y1 = min(height, int(np.ceil(point[1])) + search_radius + 1)
            ys, xs = np.nonzero(current_mask[y0:y1, x0:x1])
            covered = False
            if len(xs):
                offsets = np.column_stack((xs + x0 - point[0], ys + y0 - point[1]))
                along = np.abs(offsets @ tangent)
                across = np.abs(offsets @ normal)
                along_tolerance = (
                    ENDPOINT_TANGENTIAL_TOLERANCE_PIXELS
                    if index < 4 or index >= len(points) - 4
                    else TANGENTIAL_TOLERANCE_PIXELS
                )
                covered = bool(
                    np.any(
                        (along <= along_tolerance)
                        & (across <= PERPENDICULAR_TOLERANCE_PIXELS)
                    )
                )
            missing_flags[index] = not covered

        total_samples += len(points)
        missing_samples += int(np.count_nonzero(missing_flags))
        for index, missing in enumerate(missing_flags):
            if not missing:
                continue
            point = tuple(np.rint(points[index]).astype(int))
            cv2.circle(error_map, point, 2, 1, -1)
            if index and missing_flags[index - 1]:
                previous = tuple(np.rint(points[index - 1]).astype(int))
                cv2.line(error_map, previous, point, 1, 3, cv2.LINE_AA)

    return error_map.astype(bool), total_samples, missing_samples


def dotted_error_overlay(error_map: np.ndarray, spacing: int = 7) -> np.ndarray:
    """Create a sparse dotted mask for displaying missing desired ink."""
    rows, columns = np.indices(error_map.shape)
    seeds = error_map & (rows % spacing == 0) & (columns % spacing == 0)
    dots = cv2.dilate(seeds.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return dots & error_map
