"""Convert a binary line-art image into ordered robot drawing strokes."""

from pathlib import Path

import cv2
import numpy as np


IMPORT_MARGIN = 24


def import_line_art(
    path: str | Path, canvas_width: int, canvas_height: int
) -> list[list[tuple[int, int]]]:
    """Load line art, fit it to the canvas, skeletonize it, and trace strokes."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read image: {path}")

    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(ink) > ink.size // 2:
        ink = cv2.bitwise_not(ink)

    ys, xs = np.nonzero(ink)
    if not len(xs):
        raise ValueError("The selected image contains no dark foreground")
    cropped = ink[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    available_width = canvas_width - 2 * IMPORT_MARGIN
    available_height = canvas_height - 2 * IMPORT_MARGIN
    scale = min(available_width / cropped.shape[1], available_height / cropped.shape[0])
    resized = cv2.resize(
        cropped,
        (max(1, round(cropped.shape[1] * scale)), max(1, round(cropped.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    resized = (resized > 127).astype(np.uint8)
    fitted = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    x0 = (canvas_width - resized.shape[1]) // 2
    y0 = (canvas_height - resized.shape[0]) // 2
    fitted[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized

    skeleton = _morphological_skeleton(fitted)
    return _trace_skeleton(skeleton)


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Connectivity-preserving Zhang-Suen thinning using only NumPy."""
    image = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for first_pass in (True, False):
            center = image[1:-1, 1:-1]
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]
            linked = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                (p2 == 0) & (p3 == 1)
            ).astype(np.uint8)
            for before, after in (
                (p3, p4), (p4, p5), (p5, p6), (p6, p7),
                (p7, p8), (p8, p9), (p9, p2),
            ):
                transitions += ((before == 0) & (after == 1)).astype(np.uint8)
            if first_pass:
                edge_condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                edge_condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (
                (center == 1)
                & (linked >= 2)
                & (linked <= 6)
                & (transitions == 1)
                & edge_condition
            )
            if np.any(remove):
                center[remove] = 0
                changed = True
    return image[1:-1, 1:-1] > 0


def _trace_skeleton(skeleton: np.ndarray) -> list[list[tuple[int, int]]]:
    """Create one continuous covering walk per connected ink component."""
    pixels = {tuple(point) for point in np.argwhere(skeleton)}  # (y, x)

    def neighbors(point: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = point
        return [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy or dx) and (y + dy, x + dx) in pixels
        ]

    adjacency = {point: neighbors(point) for point in pixels}
    unseen = set(pixels)
    strokes: list[list[tuple[int, int]]] = []
    while unseen:
        start = min(unseen)
        visited_component = {start}
        unseen.remove(start)
        walk = [start]
        stack: list[tuple[tuple[int, int], object]] = [
            (start, iter(adjacency[start]))
        ]
        while stack:
            point, linked_iterator = stack[-1]
            try:
                linked = next(linked_iterator)
            except StopIteration:
                stack.pop()
                if stack:
                    walk.append(stack[-1][0])
                continue
            if linked in visited_component:
                continue
            visited_component.add(linked)
            unseen.discard(linked)
            walk.append(linked)
            stack.append((linked, iter(adjacency[linked])))

        if len(walk) >= 2:
            strokes.append([(x, y) for y, x in walk])
    return strokes
