"""Protocols (interfaces) for all pluggable components.

Every Protocol here is a contract — agents depend on Protocols, never on
concrete implementations. This is what lets users swap Qwen-VL for GLM-4V,
or OmniParser for PaddleOCR, without touching the AgentLoop.

Design rationale:

- All protocols are ``async`` (decision C3). The sync wrapper lives in
  ``AgentLoop.run`` and translates to/from the async runtime.
- Protocols are minimal: methods that the loop actually calls. Helpers and
  state belong on the implementation.
- ``Protocol`` (structural) is preferred over ABC. Implementations don't
  need to inherit, just match the shape.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from .actions import Action
from .grounding import GroundedTarget
from .intent import Intent
from .messages import Message
from .observation import Observation
from .trajectory import Step, Trajectory


@runtime_checkable
class Provider(Protocol):
    """LLM provider. Translates messages + screenshot into an Intent.

    Fused providers (decision A3) may also fill ``Intent.grounded_target``
    directly when the underlying model supports native visual grounding
    (e.g. Qwen2.5-VL). Non-fused providers leave it ``None`` and the loop
    resolves coordinates via :class:`Grounder`.
    """

    async def chat(
        self,
        messages: list[Message],
        observation: Observation,
    ) -> Intent:
        """Decide the next step given history and current observation.

        Args:
            messages: Conversation history (system + user + assistant turns,
                plus any tool results). The current screenshot is *not* in
                here — pass it via ``observation``.
            observation: Current screen state, including the screenshot.

        Returns:
            An :class:`Intent`. If ``intent.is_terminal()``, the loop stops.
        """
        ...


@runtime_checkable
class Perceiver(Protocol):
    """Captures an :class:`Observation` from the environment.

    MVP only has a screenshot perceiver. Phase 2 adds accessibility-tree
    perceptors; the protocol is unchanged — implementers decide what their
    ``Observation`` carries.
    """

    async def observe(self) -> Observation:
        """Capture the current state of the environment."""
        ...


@runtime_checkable
class Grounder(Protocol):
    """Resolves a natural-language target description to screen coordinates.

    Used only when the provider did not supply ``Intent.grounded_target``
    (non-fused path). Fused providers skip grounding entirely.
    """

    async def locate(
        self,
        description: str,
        observation: Observation,
    ) -> GroundedTarget:
        """Find a UI element matching ``description`` on ``observation``."""
        ...


@runtime_checkable
class Executor(Protocol):
    """Executes an :class:`Action` against the environment."""

    async def execute(self, action: Action) -> ActionResult:
        """Perform ``action`` and report what happened."""
        ...


@runtime_checkable
class Memory(Protocol):
    """Persists per-task context and (optionally) cross-task recall.

    Memory stores :class:`Step` objects as they are produced. Long-term
    recall across tasks is opt-in — Phase 1's InMemory implementation only
    keeps the current trajectory.
    """

    async def open(self, task_id: str, goal: str) -> Trajectory:
        """Start (or resume) a trajectory for ``task_id``."""
        ...

    async def append(self, task_id: str, step: Step) -> None:
        """Append a step to the trajectory identified by ``task_id``."""
        ...

    async def close(self, task_id: str, outcome: str, summary: str = "") -> Trajectory:
        """Finalize the trajectory."""
        ...

    async def load(self, task_id: str) -> Trajectory:
        """Load a trajectory by id."""
        ...


@runtime_checkable
class SafetyGuard(Protocol):
    """Inspects actions before execution; can deny or request confirmation.

    Per decision E2, the MVP ships a default blacklist guard. Custom guards
    implement this protocol — e.g. an interactive one that prompts the user
    or a policy-based one that reads a config file.
    """

    async def check(self, action: Action, observation: Observation) -> SafetyDecision:
        """Inspect ``action`` and decide whether to allow it."""
        ...


@runtime_checkable
class Verifier(Protocol):
    """Checks whether the task goal is satisfied after each step.

    Per decision D1, the MVP does not implement this — the protocol is
    defined so Phase 2 can add verification without changing the loop.
    """

    async def verify(
        self,
        goal: str,
        step: Step,
        next_observation: Observation,
    ) -> VerifyResult:
        """Return whether the goal appears satisfied after ``step``."""
        ...


# --- Return-type dataclasses (defined here to avoid forward-ref churn) ---


from pydantic import BaseModel, Field  # noqa: E402

SafetyVerdict = Literal["allow", "deny", "ask"]


class SafetyDecision(BaseModel):
    """Outcome of :meth:`SafetyGuard.check`."""

    verdict: SafetyVerdict = "allow"
    """``allow`` → proceed, ``deny`` → skip this action, ``ask`` → pause for human input."""

    reason: str = ""
    """Human-readable explanation, surfaced to the operator on ``ask``/``deny``."""


class ActionResult(BaseModel):
    """Outcome of :meth:`Executor.execute`."""

    success: bool = True
    error: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, object] = Field(default_factory=dict)
    """Executor-specific extra info (e.g. resolved coordinates, scroll amount)."""


class VerifyResult(BaseModel):
    """Outcome of :meth:`Verifier.verify`."""

    satisfied: bool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = ""


__all__ = [
    "ActionResult",
    "Executor",
    "Grounder",
    "Memory",
    "Perceiver",
    "Provider",
    "SafetyDecision",
    "SafetyGuard",
    "SafetyVerdict",
    "Verifier",
    "VerifyResult",
]
