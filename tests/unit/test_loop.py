"""AgentLoop tests using fakes for all pluggable components.

These tests cover the orchestration logic without touching the screen, the
mouse, or any LLM API.
"""

from __future__ import annotations

from typing import Any

import pytest

from dr_computer import (
    AgentLoop,
    ClickAction,
    DoneAction,
    GroundedTarget,
    InMemoryMemory,
    Intent,
    MacOSScreenshotPerceiver,
    Message,
    Observation,
    PyAutoGUIExecutor,
    SafetyPolicy,
)
from dr_computer.core.safety import DefaultSafetyGuard

# --- Fakes ---


class FakePerceiver:
    """Returns a deterministic 100x100 observation every time."""

    def __init__(self) -> None:
        self.call_count = 0

    async def observe(self) -> Observation:
        self.call_count += 1
        return Observation(
            screenshot=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
            width=100,
            height=100,
        )


class ScriptedProvider:
    """Provider that replays a pre-defined list of Intents."""

    def __init__(self, intents: list[Intent]) -> None:
        self.intents = list(intents)
        self.index = 0
        self.calls: list[tuple[list[Message], Observation]] = []

    async def chat(self, messages: list[Message], observation: Observation) -> Intent:
        self.calls.append((list(messages), observation))
        if self.index >= len(self.intents):
            return Intent(action_type="done", summary="script exhausted")
        intent = self.intents[self.index]
        self.index += 1
        return intent


class RecordingExecutor:
    """Stores every action dispatched; reports success."""

    def __init__(self, *, fail_on: int | None = None) -> None:
        self.executed: list[Any] = []
        self.fail_on = fail_on

    async def execute(self, action: Any) -> Any:
        from dr_computer.core.protocols import ActionResult

        self.executed.append(action)
        if self.fail_on is not None and len(self.executed) - 1 == self.fail_on:
            return ActionResult(success=False, error="fake failure")
        return ActionResult(success=True)


# --- Tests ---


async def test_loop_runs_fused_click_then_done() -> None:
    """Provider supplies grounded_target directly — fused path."""
    provider = ScriptedProvider(
        [
            Intent(
                action_type="click",
                grounded_target=GroundedTarget(bbox=(10, 10, 30, 30)),
                reasoning="click the icon",
            ),
            Intent(action_type="done", summary="notes opened"),
        ]
    )
    perceiver = FakePerceiver()
    executor = RecordingExecutor()
    memory = InMemoryMemory()

    loop = AgentLoop(
        provider=provider,
        perceiver=perceiver,
        executor=executor,
        memory=memory,
        safety_guard=None,  # tests bypass safety
        max_steps=5,
    )

    traj = await loop.arun("open notes")

    assert traj.outcome == "done"
    assert traj.step_count == 2
    assert len(executor.executed) == 1
    assert isinstance(executor.executed[0], ClickAction)
    # final step is a terminal DoneAction that the loop synthesizes internally
    assert isinstance(traj.steps[-1].action, DoneAction)


async def test_loop_hits_max_steps_without_done() -> None:
    """If the provider never returns done and actions vary, the loop bails at max_steps."""
    # Use distinct actions each step so loop_detection doesn't fire.
    provider = ScriptedProvider(
        [
            Intent(action_type="hotkey", keys=["cmd", "space"]),
            Intent(action_type="hotkey", keys=["cmd", "tab"]),
            Intent(action_type="hotkey", keys=["cmd", "c"]),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        perceiver=FakePerceiver(),
        executor=RecordingExecutor(),
        memory=InMemoryMemory(),
        safety_guard=None,
        max_steps=3,
        loop_detection=10,  # disable for this test
    )
    traj = await loop.arun("press different keys")
    assert traj.outcome == "max_steps"
    assert traj.step_count == 3


async def test_loop_detects_repeated_actions() -> None:
    """If the same action repeats ``loop_detection`` times, we stop early."""
    provider = ScriptedProvider(
        [
            Intent(
                action_type="click",
                grounded_target=GroundedTarget(bbox=(10, 10, 30, 30)),
            ),
            Intent(
                action_type="click",
                grounded_target=GroundedTarget(bbox=(10, 10, 30, 30)),
            ),
            Intent(
                action_type="click",
                grounded_target=GroundedTarget(bbox=(10, 10, 30, 30)),
            ),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        perceiver=FakePerceiver(),
        executor=RecordingExecutor(),
        memory=InMemoryMemory(),
        safety_guard=None,
        max_steps=10,
        loop_detection=3,
    )
    traj = await loop.arun("click the same spot")
    assert traj.outcome == "loop_detected"
    assert traj.step_count == 3


async def test_loop_safety_skips_dangerous_action() -> None:
    """A deny verdict from SafetyGuard skips execution but continues."""
    provider = ScriptedProvider(
        [
            Intent(action_type="hotkey", keys=["cmd", "q"]),  # blocked
            Intent(action_type="done", summary="ok"),
        ]
    )
    executor = RecordingExecutor()
    loop = AgentLoop(
        provider=provider,
        perceiver=FakePerceiver(),
        executor=executor,
        memory=InMemoryMemory(),
        safety_guard=DefaultSafetyGuard(),
        max_steps=5,
    )
    traj = await loop.arun("try to quit")
    assert traj.outcome == "done"
    # The cmd+q was skipped; no actions should have been executed.
    assert executor.executed == []


async def test_loop_non_fused_intent_requires_grounder() -> None:
    """If intent.target_description is set but grounded_target is None,
    and no grounder is provided, the loop should fail the step."""
    provider = ScriptedProvider(
        [
            Intent(
                action_type="click",
                target_description="the Notes icon",
                reasoning="need to ground",
            )
        ]
    )
    loop = AgentLoop(
        provider=provider,
        perceiver=FakePerceiver(),
        executor=RecordingExecutor(),
        memory=InMemoryMemory(),
        safety_guard=None,
        grounder=None,
        max_steps=3,
    )
    with pytest.raises(RuntimeError, match="grounder"):
        await loop.arun("open notes")


async def test_loop_emits_step_events() -> None:
    received: list[str] = []

    provider = ScriptedProvider(
        [
            Intent(action_type="type", text="hello"),
            Intent(action_type="done", summary="done"),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        perceiver=FakePerceiver(),
        executor=RecordingExecutor(),
        memory=InMemoryMemory(),
        safety_guard=None,
        max_steps=5,
    )

    async def collect(payload: Any) -> None:
        received.append(payload.__class__.__name__)

    loop.events.subscribe("step.completed", collect)
    await loop.arun("type hello")

    assert len(received) == 2
    assert received == ["Step", "Step"]


async def test_loop_cancellation_between_steps() -> None:
    """``loop.cancel()`` between steps aborts at the next iteration."""

    class CancelAfterFirstStep:
        def __init__(self) -> None:
            self._seen = False

        async def __call__(self, payload: Any) -> None:
            if not self._seen:
                self._seen = True
                # Cancel via a side-channel — we cannot reach the loop from
                # inside the subscriber cleanly, so we use a closure.
                _cancel_target[0].cancel()

    _cancel_target: list[AgentLoop] = []

    provider = ScriptedProvider(
        [
            Intent(action_type="type", text="hi"),
            Intent(action_type="type", text="there"),
            Intent(action_type="done", summary="never reached"),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        perceiver=FakePerceiver(),
        executor=RecordingExecutor(),
        memory=InMemoryMemory(),
        safety_guard=None,
        max_steps=5,
    )
    _cancel_target.append(loop)
    loop.events.subscribe("step.completed", CancelAfterFirstStep())
    traj = await loop.arun("type then cancel")
    assert traj.outcome == "aborted"
    assert traj.step_count == 1


def test_loop_run_sync_wrapper() -> None:
    """The sync wrapper must produce the same result as arun."""
    provider = ScriptedProvider([Intent(action_type="done", summary="instant success")])
    loop = AgentLoop(
        provider=provider,
        perceiver=FakePerceiver(),
        executor=RecordingExecutor(),
        memory=InMemoryMemory(),
        safety_guard=None,
        max_steps=3,
    )
    traj = loop.run("quick")
    assert traj.outcome == "done"
    assert traj.step_count == 1


# Smoke check: the public-API default implementations at least construct.
def test_default_components_construct() -> None:
    p = MacOSScreenshotPerceiver()
    assert p is not None
    e = PyAutoGUIExecutor()
    assert e is not None
    g = DefaultSafetyGuard(SafetyPolicy())
    assert g is not None
