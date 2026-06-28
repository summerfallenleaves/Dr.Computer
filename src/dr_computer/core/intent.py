"""Intent: what the LLM has decided to do, before grounding is resolved.

Per decision A3 (planner/grounder may be fused), a Provider returns an Intent
rather than a final Action. The AgentLoop is responsible for:

- if ``grounded_target`` is already set (fused provider like Qwen-VL native
  grounding), use it directly;
- otherwise, call ``Grounder.locate(target_description, observation)`` to
  resolve the description into a ``GroundedTarget``.

Once resolved, the Intent is converted to a fully-grounded ``Action``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .grounding import GroundedTarget

IntentType = Literal[
    "click", "double_click", "right_click", "type", "hotkey", "scroll", "wait", "drag", "done"
]


class Intent(BaseModel):
    """LLM-decided next step.

    Exactly one of the following must be true for a well-formed intent:

    - ``action_type == "done"`` → task complete, ``summary`` describes the outcome.
    - ``action_type`` is an action and the corresponding fields are populated
      (e.g. ``text`` for ``type``, ``keys`` for ``hotkey``).
    """

    action_type: IntentType
    """What the agent wants to do."""

    reasoning: str = ""
    """LLM's chain-of-thought (for debugging and trajectory logs)."""

    summary: str = ""
    """Only for ``action_type == "done"``: outcome description."""

    target_description: str | None = None
    """For spatial actions: a natural-language description of the target.
    Required when ``grounded_target`` is None and the action is spatial."""

    grounded_target: GroundedTarget | None = None
    """For spatial actions: an already-resolved target. Set by fused providers."""

    text: str | None = None
    """For ``type``: the text to type."""

    keys: list[str] | None = None
    """For ``hotkey``: the key chord (e.g. ``["cmd", "s"]``)."""

    scroll_dx: int = 0
    """For ``scroll``: horizontal delta in pixels (rarely used)."""

    scroll_dy: int = 0
    """For ``scroll``: vertical delta in pixels. Positive = down."""

    wait_seconds: float | None = None
    """For ``wait``: how long to wait. Defaults to a small idle."""

    def is_spatial(self) -> bool:
        """Whether this intent needs a target on screen."""
        return self.action_type in {"click", "double_click", "right_click", "drag"}

    def is_terminal(self) -> bool:
        return self.action_type == "done"


__all__ = ["Intent", "IntentType"]
