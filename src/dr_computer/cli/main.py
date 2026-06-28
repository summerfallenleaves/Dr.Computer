"""Command-line interface entry point.

Implemented with stdlib ``argparse`` to avoid a Typer/click dependency for
what is currently a small surface. When the CLI grows past ~3 subcommands,
consider migrating to Typer.

Subcommands:
- ``run``     : run a single agent task (the main use case)
- ``version`` : print the package version

Examples:
    dr-computer run "Open Safari"
    dr-computer run --provider qwen-vl --model qwen-vl-max --max-steps 20 \\
        --verbose "Open Notes"
    dr-computer version
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

from .. import __version__
from ..core.loop import AgentLoop
from ..core.safety import DefaultSafetyGuard, SafetyPolicy
from ..execution.pyautogui_exec import PyAutoGUIExecutor
from ..memory.in_memory import InMemoryMemory
from ..perception.macos import MacOSScreenshotPerceiver
from ..providers.qwen_vl import QwenVLProvider

# --- .env loading ---


def _find_env_file(start: Path) -> Path | None:
    """Walk up from ``start`` looking for a .env file."""
    for candidate in [start, *start.parents]:
        env = candidate / ".env"
        if env.is_file():
            return env
    return None


def _load_env() -> None:
    """Load .env from CWD (or nearest parent) into os.environ.

    Called before any provider is constructed so API keys are visible.
    """
    env_file = _find_env_file(Path.cwd())
    if env_file is not None:
        load_dotenv(env_file, override=False)


# --- progress logging ---


async def _on_step_completed(step: object) -> None:
    """Live progress line printed for each completed step."""
    intent = getattr(step, "intent", None)
    if intent is None:
        return
    action_desc = intent.action_type
    target = getattr(intent, "grounded_target", None)
    desc = getattr(intent, "target_description", None)
    text = getattr(intent, "text", None)
    if target is not None:
        action_desc += f" @ {target.center}"
    elif desc:
        action_desc += f" '{desc}'"
    elif text:
        action_desc += f" '{text[:30]}'"
    status = getattr(step, "status", "")
    reasoning = (intent.reasoning or "")[:60]
    num = getattr(step, "num", "?")
    print(f"  step {num:>2} [{status:<8}] {action_desc:<40} — {reasoning}")


# --- provider factory ---


_PROVIDER_FACTORIES: dict[str, Callable[..., QwenVLProvider]] = {
    "qwen-vl": QwenVLProvider,
}


def _build_provider(args: argparse.Namespace) -> QwenVLProvider:
    name = args.provider
    if name not in _PROVIDER_FACTORIES:
        raise SystemExit(
            f"Unknown provider: {name!r}. Available: {', '.join(sorted(_PROVIDER_FACTORIES))}"
        )
    factory = _PROVIDER_FACTORIES[name]

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key or api_key == "sk-replace-me":
        raise SystemExit(
            f"DASHSCOPE_API_KEY is not set for provider {name!r}.\n"
            "  Options:\n"
            "    1. Put it in .env (see .env.example)\n"
            "    2. export DASHSCOPE_API_KEY=sk-xxx\n"
            "    3. Get one at https://bailian.console.aliyun.com/ → API-KEY"
        )

    return factory(
        api_key=api_key,
        model=args.model or None,
    )


def _build_loop(args: argparse.Namespace) -> AgentLoop:
    provider = _build_provider(args)
    perceiver = MacOSScreenshotPerceiver(
        retina_physical_pixels=args.retina_physical,
    )
    executor = PyAutoGUIExecutor(
        move_duration=args.move_duration,
        type_interval=args.type_interval,
    )
    safety = DefaultSafetyGuard(
        policy=SafetyPolicy(
            ask_when_unmatched_spatial=args.ask_for_spatial,
        )
    )

    loop = AgentLoop(
        provider=provider,
        perceiver=perceiver,
        executor=executor,
        memory=InMemoryMemory(),
        grounder=None,
        safety_guard=safety,
        max_steps=args.max_steps,
        loop_detection=args.loop_detection,
    )
    loop.events.subscribe("step.completed", _on_step_completed)
    return loop


# --- subcommands ---


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_run(args: argparse.Namespace) -> int:
    """``dr-computer run "goal"`` — run a single agent task."""
    _load_env()
    _configure_logging(args.verbose)

    if not args.goal:
        raise SystemExit('Goal is required. Usage: dr-computer run "<your goal>"')

    print(f"Goal: {args.goal}")
    print()
    loop = _build_loop(args)

    trajectory = loop.run(args.goal)

    print()
    print(f"Outcome: {trajectory.outcome} after {trajectory.step_count} step(s).")
    if trajectory.summary:
        print(f"Summary: {trajectory.summary}")

    success = trajectory.outcome in {"done", "loop_detected"}
    return 0 if success else 1


def cmd_version(_: argparse.Namespace) -> int:
    """``dr-computer version``."""
    print(f"dr-computer {__version__}")
    return 0


# --- argparse wiring ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dr-computer",
        description=(
            "Dr.Computer — a Python framework for building Desktop AI Agents. "
            "Currently ships one subcommand: 'run'."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    # run
    run = sub.add_parser(
        "run",
        help="Run a single agent task.",
        description="Run a single agent task against the configured provider.",
    )
    run.add_argument("goal", nargs="?", help="Natural-language goal.")
    run.add_argument(
        "--provider",
        default="qwen-vl",
        choices=sorted(_PROVIDER_FACTORIES),
        help="LLM provider (default: qwen-vl).",
    )
    run.add_argument(
        "--model",
        default=None,
        help="Override the provider's default model (e.g. qwen-vl-max).",
    )
    run.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum iterations before bailing (default: 20).",
    )
    run.add_argument(
        "--loop-detection",
        type=int,
        default=3,
        help="Stop after N consecutive identical actions (default: 3).",
    )
    run.add_argument(
        "--move-duration",
        type=float,
        default=0.25,
        help="Cursor motion time in seconds (default: 0.25).",
    )
    run.add_argument(
        "--type-interval",
        type=float,
        default=0.02,
        help="Seconds between keystrokes when typing (default: 0.02).",
    )
    run.add_argument(
        "--retina-physical",
        action="store_true",
        help="Use physical pixels (2x) instead of logical pixels. "
        "Use only if you also configure the executor for physical pixels.",
    )
    run.add_argument(
        "--ask-for-spatial",
        action="store_true",
        help="Require human confirmation for every spatial action (click/drag).",
    )
    run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    run.set_defaults(func=cmd_run)

    # version
    ver = sub.add_parser("version", help="Print version.")
    ver.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        return cmd_version(args)

    if not getattr(args, "command", None):
        # No subcommand: print help.
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
