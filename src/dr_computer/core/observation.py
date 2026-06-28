"""Observation: what the agent sees of the environment at one point in time."""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """A snapshot of the screen (and optionally the UI tree).

    MVP only carries a screenshot. Phase 2 will extend with an accessibility
    tree (list of `UIElement`) so grounders can use structured info in addition
    to pixels.
    """

    screenshot: bytes = Field(repr=False)
    """PNG-encoded screenshot bytes."""

    width: int
    """Screenshot width in physical pixels."""

    height: int
    """Screenshot height in physical pixels."""

    timestamp: float = Field(default_factory=time.time)
    """When the observation was captured (unix seconds)."""

    source: Literal["screenshot", "accessibility", "hybrid"] = "screenshot"
    """What was used to produce this observation."""

    @property
    def center(self) -> tuple[int, int]:
        return (self.width // 2, self.height // 2)


__all__ = ["Observation"]
