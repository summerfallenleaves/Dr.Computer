"""End-to-end demo: drive macOS via Dr.Computer.

Loads `.env` from the project root on startup, then runs a single agent
task against Qwen-VL.

Run it:

    cp .env.example .env
    # edit .env and put your DASHSCOPE_API_KEY in it
    uv run python examples/01_open_notes.py

You can override the goal from the CLI or the environment:

    uv run python examples/01_open_notes.py "Click on the Apple menu"
    DR_COMPUTER_GOAL="Open Safari" uv run python examples/01_open_notes.py

Before running, also grant Accessibility permission to the terminal that
runs this script (System Settings → Privacy & Security → Accessibility),
otherwise PyAutoGUI can't move the mouse. Screen Recording permission is
needed too — see CLAUDE.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow `python examples/01_open_notes.py` from a non-installed checkout.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

# Load .env from project root before importing dr_computer, so providers
# that read API keys at construction time see them.
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from dr_computer import (  # noqa: E402
    AgentLoop,
    DefaultSafetyGuard,
    InMemoryMemory,
    MacOSScreenshotPerceiver,
    PyAutoGUIExecutor,
    QwenVLProvider,
    SafetyPolicy,
)

# Sensible default goal — kept deliberately small so first-time users can
# verify the pipeline end-to-end without a long multi-step task.
DEFAULT_GOAL = "Click on the Apple menu () in the top-left corner of the screen."


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_goal(argv: list[str]) -> str:
    """Goal precedence: CLI arg > $DR_COMPUTER_GOAL > DEFAULT_GOAL."""
    if len(argv) >= 2 and not argv[1].startswith("-"):
        return argv[1]
    env_goal = os.environ.get("DR_COMPUTER_GOAL")
    if env_goal:
        return env_goal
    return DEFAULT_GOAL


def build_loop() -> AgentLoop:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key or api_key == "sk-replace-me":
        sys.exit(
            "DASHSCOPE_API_KEY is not set. Either:\n"
            "  1. Copy .env.example to .env and fill in your key, or\n"
            "  2. export DASHSCOPE_API_KEY=sk-xxx in your shell, or\n"
            "  3. Get a key at https://bailian.console.aliyun.com/ → API-KEY."
        )

    provider = QwenVLProvider(
        api_key=api_key,
        model=os.environ.get("DR_COMPUTER_QWEN_MODEL", "qwen-vl-max"),
    )
    perceiver = MacOSScreenshotPerceiver(
        # Logical pixels — matches what Qwen-VL expects to receive.
        retina_physical_pixels=False,
    )
    executor = PyAutoGUIExecutor(
        move_duration=0.25,  # smooth cursor motion; 0 looks jarring
        type_interval=0.02,
    )
    memory = InMemoryMemory()
    safety = DefaultSafetyGuard(
        policy=SafetyPolicy(
            # Production deployments may want ask_when_unmatched_spatial=True.
            ask_when_unmatched_spatial=False,
        )
    )

    loop = AgentLoop(
        provider=provider,
        perceiver=perceiver,
        executor=executor,
        memory=memory,
        grounder=None,  # fused provider — no grounder needed (decision A3)
        safety_guard=safety,
        max_steps=12,
    )

    # Live progress logger.
    async def on_step_completed(step) -> None:
        intent = step.intent
        if intent is None:
            return
        action_desc = intent.action_type
        if intent.grounded_target is not None:
            action_desc += f" @ {intent.grounded_target.bbox}"
        elif intent.target_description:
            action_desc += f" '{intent.target_description}'"
        elif intent.text:
            action_desc += f" '{intent.text[:30]}'"
        print(
            f"  step {step.num:2d} [{step.status:8s}] {action_desc:40s} — {intent.reasoning[:60]}"
        )

    loop.events.subscribe("step.completed", on_step_completed)
    return loop


async def main(argv: list[str]) -> int:
    configure_logging(verbose="--verbose" in argv)
    goal = resolve_goal(argv)
    print(f"Goal: {goal}\n")
    loop = build_loop()
    trajectory = await loop.arun(goal)

    print(f"\nOutcome: {trajectory.outcome} after {trajectory.step_count} step(s).")
    if trajectory.summary:
        print(f"Summary: {trajectory.summary}")
    # Treat "done" and "loop_detected" as success — the latter means the agent
    # made progress but the model didn't explicitly say done, which is fine
    # for the demo's simple click-style goals.
    success_outcomes = {"done", "loop_detected"}
    return 0 if trajectory.outcome in success_outcomes else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv)))
