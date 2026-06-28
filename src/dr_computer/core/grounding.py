"""Grounding: locating a semantic UI description on the screen."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class GroundedTarget(BaseModel):
    """A screen element located by a grounder.

    `bbox` is in screen pixel coordinates as ``(x1, y1, x2, y2)`` where
    ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` is the bottom-right.

    The geometric center is what executors typically click.
    """

    bbox: tuple[int, int, int, int]
    """``(x1, y1, x2, y2)`` in screen pixels."""

    label: str | None = None
    """Recognized text/label of the element, if any (e.g. "Submit")."""

    element_type: str | None = None
    """Detected element kind (button, input, link, ...), if known."""

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """Detector confidence in ``[0, 1]``."""

    @model_validator(mode="after")
    def _check_bbox(self) -> GroundedTarget:
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox must satisfy x2>x1 and y2>y1, got {self.bbox!r}")
        return self

    @property
    def center(self) -> tuple[int, int]:
        """Geometric center, suitable for clicking."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


__all__ = ["GroundedTarget"]
