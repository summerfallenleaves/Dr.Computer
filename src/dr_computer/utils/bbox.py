"""Bounding-box math.

Pure functions on ``(x1, y1, x2, y2)`` tuples. No I/O, no Pydantic — used by
grounders, executors and tests.
"""

from __future__ import annotations

from collections.abc import Iterable

BBox = tuple[int, int, int, int]
"""``(x1, y1, x2, y2)`` in pixel coordinates."""


def center(bbox: BBox) -> tuple[int, int]:
    """Geometric center of a bbox."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def size(bbox: BBox) -> tuple[int, int]:
    """Width and height."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1, y2 - y1)


def area(bbox: BBox) -> int:
    w, h = size(bbox)
    return w * h


def is_valid(bbox: BBox) -> bool:
    """Whether a bbox satisfies ``x2 > x1`` and ``y2 > y1``."""
    x1, y1, x2, y2 = bbox
    return x2 > x1 and y2 > y1


def intersection(a: BBox, b: BBox) -> BBox | None:
    """Largest bbox contained in both ``a`` and ``b``, or ``None`` if disjoint."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two bboxes in ``[0, 1]``."""
    inter = intersection(a, b)
    if inter is None:
        return 0.0
    inter_area = area(inter)
    union_area = area(a) + area(b) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def clamp(bbox: BBox, bounds: BBox) -> BBox:
    """Clamp a bbox to lie entirely within ``bounds``.

    If the clamped result would be invalid (zero or negative area), returns
    a 1x1 box at the clamped center.
    """
    x1 = max(bbox[0], bounds[0])
    y1 = max(bbox[1], bounds[1])
    x2 = min(bbox[2], bounds[2])
    y2 = min(bbox[3], bounds[3])
    if x2 <= x1:
        cx = max(min(center(bounds)[0], bounds[2] - 1), bounds[0])
        x1 = x2 = cx
    if y2 <= y1:
        cy = max(min(center(bounds)[1], bounds[3] - 1), bounds[1])
        y1 = y2 = cy
    return (x1, y1, x2, y2)


def union_all(boxes: Iterable[BBox]) -> BBox | None:
    """Smallest bbox covering all input boxes, or ``None`` if empty."""
    boxes_list = list(boxes)
    if not boxes_list:
        return None
    x1 = min(b[0] for b in boxes_list)
    y1 = min(b[1] for b in boxes_list)
    x2 = max(b[2] for b in boxes_list)
    y2 = max(b[3] for b in boxes_list)
    return (x1, y1, x2, y2)


__all__ = [
    "BBox",
    "area",
    "center",
    "clamp",
    "intersection",
    "iou",
    "is_valid",
    "size",
    "union_all",
]
