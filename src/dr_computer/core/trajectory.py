"""Trajectory: the complete history of a single agent task."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from .actions import Action
from .grounding import GroundedTarget
from .intent import Intent
from .observation import Observation


def new_task_id() -> str:
    """Generate a unique task identifier."""
    return uuid.uuid4().hex


class StepStatus:
    """Lifecycle markers for a single step."""

    PLANNED = "planned"
    GROUNDED = "grounded"
    APPROVED = "approved"
    EXECUTED = "executed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Step(BaseModel):
    """One iteration of the agent loop.

    Steps are append-only and form a trajectory. A step records everything
    needed to replay, debug, or learn from this single iteration.
    """

    num: int
    """Zero-based step index within the trajectory."""

    observation: Observation | None = None
    """What the agent saw. ``None`` only for synthetic/test steps."""

    intent: Intent | None = None
    """What the LLM decided. ``None`` if the step errored before planning."""

    grounding_query: str | None = None
    """Description sent to the grounder, if grounding happened."""

    grounded_target: GroundedTarget | None = None
    """What the grounder returned (or what fused provider supplied)."""

    action: Action | None = None
    """The fully-resolved action that was (or would be) executed."""

    status: str = StepStatus.PLANNED
    """Where this step is in its lifecycle."""

    error: str | None = None
    """Exception message, if ``status == FAILED``."""

    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    duration_ms: float | None = None


class Trajectory(BaseModel):
    """Complete record of one ``AgentLoop.run`` invocation."""

    task_id: str = Field(default_factory=new_task_id)
    goal: str
    steps: list[Step] = Field(default_factory=list)
    started_at: float = Field(default_factory=time.time)
    finished_at: float | None = None
    outcome: Literal["running", "done", "failed", "aborted", "max_steps"] = "running"
    summary: str = ""

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def last_step(self) -> Step | None:
        return self.steps[-1] if self.steps else None


# Pydantic v2 forward-ref resolution: ensure Action discriminated union sees Step.
Step.model_rebuild()


__all__ = [
    "Step",
    "StepStatus",
    "Trajectory",
    "new_task_id",
]
