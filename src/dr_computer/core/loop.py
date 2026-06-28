"""AgentLoop: the orchestrator.

The loop wires together the pluggable components and drives them through
``Observe → Think → Ground → Safety → Act → Memory``. All heavy lifting lives
in the components; this file is intentionally small and easy to audit.

Sync vs async (decision C3):

- ``AgentLoop.arun`` is the canonical, async implementation.
- ``AgentLoop.run`` is a thin sync wrapper that runs ``arun`` on a fresh
  event loop. Most users only need ``run``; advanced users (already inside
  an event loop, e.g. a FastAPI server) use ``arun`` directly.

Cancellation:

- ``arun`` checks ``self._cancel_event`` between steps, so a caller can
  cancel a run via ``loop.cancel()`` from another task.
- A single step is atomic — once ``executor.execute`` is called, it runs to
  completion. Cancellation only takes effect between steps.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..utils.events import EventEmitter
from .actions import (
    Action,
    ClickAction,
    DoneAction,
    DoubleClickAction,
    HotkeyAction,
    RightClickAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)
from .grounding import GroundedTarget
from .intent import Intent
from .messages import Message
from .observation import Observation
from .protocols import (
    Memory,
    Perceiver,
    Provider,
    SafetyDecision,
    SafetyGuard,
    Verifier,
)
from .safety import DefaultSafetyGuard
from .trajectory import Step, StepStatus, Trajectory, new_task_id

logger = logging.getLogger(__name__)


class AgentLoopCancelled(Exception):
    """Raised when ``loop.cancel()`` is called mid-run."""


class HumanConfirmationRequired(Exception):
    """Raised when a safety guard returns ``ask`` and no confirter is set.

    The exception carries the decision so callers (e.g. a CLI) can prompt
    the user and resume.
    """

    def __init__(self, decision: SafetyDecision, action: Action) -> None:
        super().__init__(f"Safety guard requires confirmation: {decision.reason}")
        self.decision = decision
        self.action = action


Confirmer = Any  # Callable[[Action, SafetyDecision], Awaitable[bool]] | None


class AgentLoop:
    """Composes a Provider, Perceiver, optional Grounder, Executor, Memory.

    Args:
        provider: LLM provider. Returns an :class:`Intent` per turn.
        perceiver: Captures each :class:`Observation`.
        executor: Runs the resolved :class:`Action`.
        memory: Stores the trajectory.
        grounder: Optional. Required only when the provider does not
            natively supply ``Intent.grounded_target`` (non-fused providers).
        safety_guard: Optional. Defaults to :class:`DefaultSafetyGuard`.
            Pass ``None`` to disable — *not recommended* outside tests.
        verifier: Optional. Per decision D1, MVP does not use it; provided
            so Phase 2 can plug one in without changing the constructor.
        max_steps: Safety bound. The loop stops after this many iterations
            even if the provider has not returned ``done``.
        default_wait_seconds: Used when an Intent's ``wait_seconds`` is None.
        confirmer: Optional async callback for ``SafetyDecision.verdict == "ask"``.
            Receives the action and decision; returns True to allow, False to deny.
            If None, the loop raises :class:`HumanConfirmationRequired`.
        events: Optional pre-existing :class:`EventEmitter`. If None, a fresh
            one is created and exposed as ``self.events``.
        loop_detection: Stop after this many consecutive identical actions
            (default 3). Prevents the agent from hammering the same target
            when the model keeps requesting the same action without ever
            declaring ``done``. Set to a large value to effectively disable.
    """

    def __init__(
        self,
        *,
        provider: Provider,
        perceiver: Perceiver,
        executor: Any,
        memory: Memory,
        grounder: Any | None = None,
        safety_guard: SafetyGuard | None = None,
        verifier: Verifier | None = None,
        max_steps: int = 50,
        default_wait_seconds: float = 1.0,
        confirmer: Confirmer = None,
        events: EventEmitter | None = None,
        loop_detection: int = 3,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if loop_detection < 2:
            raise ValueError("loop_detection must be at least 2")
        self.provider = provider
        self.perceiver = perceiver
        self.executor = executor
        self.memory = memory
        self.grounder = grounder
        self.safety_guard: SafetyGuard | None = (
            safety_guard if safety_guard is not None else DefaultSafetyGuard()
        )
        self.verifier = verifier
        self.max_steps = max_steps
        self.default_wait_seconds = default_wait_seconds
        self.confirmer = confirmer
        self.events = events or EventEmitter()
        self.loop_detection = loop_detection
        self._cancel_event: asyncio.Event | None = None

    # --- public API ---

    def cancel(self) -> None:
        """Request cancellation. Effective at the next step boundary."""
        if self._cancel_event is not None:
            self._cancel_event.set()

    def run(self, goal: str, *, task_id: str | None = None) -> Trajectory:
        """Sync entry point (decision C3). Runs ``arun`` on a fresh loop."""
        return asyncio.run(self.arun(goal, task_id=task_id))

    async def arun(self, goal: str, *, task_id: str | None = None) -> Trajectory:
        """Async entry point. Composable inside an existing event loop."""
        task_id = task_id or new_task_id()
        self._cancel_event = asyncio.Event()
        trajectory = await self.memory.open(task_id, goal)

        await self.events.emit("task.started", {"task_id": task_id, "goal": goal})

        try:
            for step_num in range(self.max_steps):
                if self._cancel_event.is_set():
                    trajectory.outcome = "aborted"
                    break

                step = await self._run_step(step_num, goal, trajectory)
                await self.memory.append(task_id, step)
                await self.events.emit("step.completed", step)

                if step.intent is not None and step.intent.is_terminal():
                    trajectory.outcome = "done"
                    trajectory.summary = step.intent.summary or ""
                    break

                if self._is_stuck_in_loop(trajectory):
                    trajectory.outcome = "loop_detected"
                    trajectory.summary = (
                        "Agent repeated the same action; treating as done. "
                        "Last action likely achieved the goal."
                    )
                    logger.info(
                        "AgentLoop detected repeated actions for task %s; stopping.",
                        task_id,
                    )
                    break

            else:
                trajectory.outcome = "max_steps"
                logger.warning("AgentLoop hit max_steps=%d for task %s", self.max_steps, task_id)
        except AgentLoopCancelled:
            trajectory.outcome = "aborted"
        except Exception as exc:
            logger.exception("AgentLoop failed for task %s", task_id)
            trajectory.outcome = "failed"
            trajectory.summary = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            trajectory.finished_at = time.time()
            await self.memory.close(
                task_id,
                outcome=trajectory.outcome,
                summary=trajectory.summary,
            )
            await self.events.emit(
                "task.done",
                {"task_id": task_id, "outcome": trajectory.outcome},
            )
            self._cancel_event = None

        return trajectory

    # --- single-step machinery ---

    async def _run_step(
        self,
        step_num: int,
        goal: str,
        trajectory: Trajectory,
    ) -> Step:
        step = Step(num=step_num)
        try:
            # 1. Observe
            observation = await self.perceiver.observe()
            step.observation = observation
            await self.events.emit("step.observed", step)

            # 2. Think (provider)
            messages = self._build_messages(goal, trajectory)
            intent = await self.provider.chat(messages, observation)
            step.intent = intent
            await self.events.emit("step.planned", step)

            # 3. Done?
            if intent.is_terminal():
                step.action = DoneAction(summary=intent.summary)
                step.status = StepStatus.EXECUTED
                return step

            # 4. Resolve into a fully-grounded Action
            action = await self._resolve_action(intent, observation, step)

            # 5. Safety
            if self.safety_guard is not None:
                decision = await self.safety_guard.check(action, observation)
                step.status = StepStatus.APPROVED
                if decision.verdict == "deny":
                    step.status = StepStatus.SKIPPED
                    step.error = f"Safety denied: {decision.reason}"
                    await self.events.emit(
                        "safety.denied", {"action": action, "decision": decision}
                    )
                    return step
                if decision.verdict == "ask":
                    allowed = await self._confirm(action, decision)
                    if not allowed:
                        step.status = StepStatus.SKIPPED
                        step.error = f"Safety denied by human: {decision.reason}"
                        return step

            step.action = action

            # 6. Act
            result = await self.executor.execute(action)
            if not result.success:
                step.status = StepStatus.FAILED
                step.error = result.error
            else:
                step.status = StepStatus.EXECUTED
            await self.events.emit("step.executed", step)

            # 7. Verify (optional, Phase 2)
            if self.verifier is not None:
                # Only verify after non-terminal actions
                next_obs = await self.perceiver.observe()
                verify = await self.verifier.verify(goal, step, next_obs)
                if not verify.satisfied:
                    await self.events.emit("verify.failed", verify)

            return step
        except Exception as exc:
            step.status = StepStatus.FAILED
            step.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            step.finished_at = time.time()
            if step.started_at:
                step.duration_ms = (step.finished_at - step.started_at) * 1000

    async def _resolve_action(
        self,
        intent: Intent,
        observation: Observation,
        step: Step,
    ) -> Action:
        """Convert an Intent into a fully-resolved Action.

        Spatial intents (click/drag/...) need a ``GroundedTarget``. If the
        provider already supplied one (fused path, decision A3), use it.
        Otherwise call the grounder.
        """
        match intent.action_type:
            case "click":
                target = await self._resolve_target(intent, observation, step)
                return ClickAction(target=target)
            case "double_click":
                target = await self._resolve_target(intent, observation, step)
                return DoubleClickAction(target=target)
            case "right_click":
                target = await self._resolve_target(intent, observation, step)
                return RightClickAction(target=target)
            case "drag":
                # Drag needs two targets — for MVP we only support fused
                # providers that supply both source and target. Non-fused
                # drag is a Phase 2 concern.
                if intent.grounded_target is None:
                    raise ValueError(
                        "Drag action requires a fused provider that supplies "
                        "grounded_target (source); non-fused drag is not yet "
                        "supported in MVP."
                    )
                raise NotImplementedError(
                    "Drag with separate source/target needs Phase 2 work; "
                    "the Intent model only carries a single target today."
                )
            case "type":
                target = (
                    await self._resolve_target(intent, observation, step)
                    if intent.is_spatial() and intent.target_description
                    else intent.grounded_target
                )
                if intent.text is None:
                    raise ValueError("TypeAction requires intent.text")
                return TypeAction(text=intent.text, target=target)
            case "hotkey":
                if not intent.keys:
                    raise ValueError("HotkeyAction requires intent.keys")
                return HotkeyAction(keys=list(intent.keys))
            case "scroll":
                if intent.scroll_dx == 0 and intent.scroll_dy == 0:
                    raise ValueError("ScrollAction requires non-zero dx or dy")
                return ScrollAction(dx=intent.scroll_dx, dy=intent.scroll_dy)
            case "wait":
                return WaitAction(
                    seconds=intent.wait_seconds
                    if intent.wait_seconds is not None
                    else self.default_wait_seconds
                )
            case _:
                raise ValueError(f"Unknown action_type: {intent.action_type!r}")

    async def _resolve_target(
        self,
        intent: Intent,
        observation: Observation,
        step: Step,
    ) -> GroundedTarget:
        if intent.grounded_target is not None:
            step.status = StepStatus.GROUNDED
            step.grounded_target = intent.grounded_target
            return intent.grounded_target

        if intent.target_description is None:
            raise ValueError(
                f"Intent action_type={intent.action_type!r} requires either "
                "grounded_target or target_description."
            )
        if self.grounder is None:
            raise RuntimeError(
                "Non-fused provider returned an intent without grounded_target "
                "but no grounder is configured. Pass a Grounder to AgentLoop."
            )
        step.grounding_query = intent.target_description
        target = await self.grounder.locate(intent.target_description, observation)
        step.grounded_target = target
        step.status = StepStatus.GROUNDED
        return target

    async def _confirm(self, action: Action, decision: SafetyDecision) -> bool:
        if self.confirmer is None:
            raise HumanConfirmationRequired(decision, action)
        result = self.confirmer(action, decision)
        if asyncio.iscoroutine(result):
            result = await result
        return bool(result)

    def _is_stuck_in_loop(self, trajectory: Trajectory) -> bool:
        """True if the last ``loop_detection`` steps took the same action.

        Two actions are "the same" if their type and target center match.
        This catches the common failure where a VLM keeps re-issuing the
        same click without ever declaring the task done.
        """
        steps = trajectory.steps
        if len(steps) < self.loop_detection:
            return False
        recent = steps[-self.loop_detection :]
        signatures = []
        for s in recent:
            if s.action is None or s.intent is None:
                return False  # incomplete step; can't decide
            a = s.action
            sig = [a.type]
            target = getattr(a, "target", None)
            if target is not None:
                sig.append(target.center)
            else:
                sig.append(None)
            signatures.append(tuple(sig))
        return len(set(signatures)) == 1

    def _build_messages(self, goal: str, trajectory: Trajectory) -> list[Message]:
        """Build the message history sent to the provider.

        Includes the last few steps' intents so the model can recognize
        when an action has already been performed and declare done. Without
        this context, the model would re-issue the same action forever.
        """
        system_prompt = (
            "You are a Desktop AI Agent operating a macOS computer. "
            "Decide the next single action that moves toward the user's goal. "
            "Reply with exactly one Intent. Stop with action_type='done' "
            "once the goal appears achieved.\n\n"
            "Important: if the screenshot already shows the goal has been met "
            "(e.g. a menu you were asked to open is now visible), return "
            "action_type='done' immediately. Do not repeat an action you "
            "have already taken in the history below unless it clearly failed."
        )
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Goal: {goal}"),
        ]
        # Replay the last few completed steps as assistant/user dialogue so
        # the model can see what it already tried. Cap to keep token cost
        # bounded — the screenshot already shows current state, history is
        # only here to break loops.
        recent = trajectory.steps[-5:]
        if recent:
            messages.append(
                Message(
                    role="user",
                    content="Recent actions you have taken (oldest first):",
                )
            )
            for step in recent:
                if step.intent is None:
                    continue
                intent = step.intent
                line = f"step {step.num}: {intent.action_type}"
                if intent.grounded_target is not None:
                    line += f" @ {intent.grounded_target.center}"
                if intent.target_description:
                    line += f" ({intent.target_description})"
                if intent.text:
                    line += f" text={intent.text[:40]!r}"
                if intent.reasoning:
                    line += f" — {intent.reasoning[:80]}"
                status_note = "" if step.status == "executed" else f" [{step.status}]"
                messages.append(Message(role="assistant", content=line + status_note))
            messages.append(
                Message(
                    role="user",
                    content=(
                        "Decide the next single action based on the current "
                        "screenshot. If the goal is already met, say done."
                    ),
                )
            )
        return messages


__all__ = [
    "AgentLoop",
    "AgentLoopCancelled",
    "HumanConfirmationRequired",
]
