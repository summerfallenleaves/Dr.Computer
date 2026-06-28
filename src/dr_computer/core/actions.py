"""Action: a fully-resolved, executable operation.

Per decision B2 (coordinate-driven), an Action is self-contained — every
field needed to execute it is already populated. This makes actions:

- serializable (save to disk, replay later);
- recordable (a recorder just observes executed actions);
- skill-able (a skill is a sequence of actions).

Actions are produced by ``AgentLoop`` from an ``Intent`` after grounding.
They are consumed by ``Executor`` implementations.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .grounding import GroundedTarget


class _ActionCommon(BaseModel):
    """Marker base — every action has the literal ``type`` field."""

    type: str


class ClickAction(_ActionCommon):
    type: Literal["click"] = "click"
    target: GroundedTarget
    button: Literal["left", "right", "middle"] = "left"


class DoubleClickAction(_ActionCommon):
    type: Literal["double_click"] = "double_click"
    target: GroundedTarget
    button: Literal["left", "right", "middle"] = "left"


class RightClickAction(_ActionCommon):
    type: Literal["right_click"] = "right_click"
    target: GroundedTarget


class TypeAction(_ActionCommon):
    type: Literal["type"] = "type"
    text: str
    """Text to type. May contain newlines."""

    target: GroundedTarget | None = None
    """If set, click into this element first, then type."""


class HotkeyAction(_ActionCommon):
    type: Literal["hotkey"] = "hotkey"
    keys: list[str]
    """Key chord, e.g. ``["cmd", "s"]``. Order is the press order."""


class ScrollAction(_ActionCommon):
    type: Literal["scroll"] = "scroll"
    dx: int = 0
    dy: int = 0
    """Positive dy scrolls down. At least one of dx/dy must be non-zero."""

    anchor: tuple[int, int] | None = None
    """Optional ``(x, y)`` to scroll at; None means current cursor."""


class WaitAction(_ActionCommon):
    type: Literal["wait"] = "wait"
    seconds: float = Field(default=1.0, ge=0.0)


class DragAction(_ActionCommon):
    type: Literal["drag"] = "drag"
    source: GroundedTarget
    target: GroundedTarget
    duration: float = Field(default=0.5, ge=0.0)
    """Seconds taken to perform the drag motion."""


class DoneAction(_ActionCommon):
    """Terminal marker, not executed by Executor."""

    type: Literal["done"] = "done"
    summary: str = ""


ActionUnion = (
    ClickAction
    | DoubleClickAction
    | RightClickAction
    | TypeAction
    | HotkeyAction
    | ScrollAction
    | WaitAction
    | DragAction
    | DoneAction
)
"""Bare union of all action variants, without the discriminator annotation."""

Action = Annotated[ActionUnion, Field(discriminator="type")]
"""Discriminated action union. Use this in pydantic models for validation."""


__all__ = [
    "Action",
    "ActionUnion",
    "ClickAction",
    "DoneAction",
    "DoubleClickAction",
    "DragAction",
    "HotkeyAction",
    "RightClickAction",
    "ScrollAction",
    "TypeAction",
    "WaitAction",
]


SPATIAL_ACTION_TYPES: frozenset[str] = frozenset({"click", "double_click", "right_click", "drag"})
"""Action types that carry coordinate information."""
