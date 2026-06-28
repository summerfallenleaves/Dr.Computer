"""Default safety guard (decision E2).

A blacklist-based guard that intercepts obviously dangerous actions. This is
intentionally conservative — the goal is to prevent catastrophic demos
(``Cmd+Q`` quitting the agent's terminal, ``rm -rf`` typed into Terminal,
clicking through a bank's transfer page) without becoming unusable.

Custom guards can implement :class:`SafetyGuard` directly for richer policy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from ..core.actions import Action, HotkeyAction, TypeAction
from ..core.observation import Observation
from ..core.protocols import SafetyDecision


class SafetyPolicy(BaseModel):
    """Configuration for :class:`DefaultSafetyGuard`.

    All fields have sensible defaults; users override only what they need.
    """

    block_hotkeys: list[tuple[str, ...]] = Field(
        default_factory=lambda: [
            # Quit any app
            ("cmd", "q"),
            ("ctrl", "q"),
            # Close window/tab
            ("cmd", "w"),
            # Force-quit / lock screen / shutdown
            ("cmd", "ctrl", "q"),  # macOS lock screen
            ("ctrl", "alt", "del"),  # Windows
            # Terminal: kill process, clear
            ("cmd", "k"),
            ("ctrl", "c"),
            # Browser dev tools (so the agent can't self-sabotage)
            ("cmd", "option", "i"),
        ]
    )
    """Key chords that always require confirmation."""

    block_text_patterns: list[str] = Field(
        default_factory=lambda: [
            # Destructive shell
            r"\brm\s+-rf?\b",
            r"\bmkfs\b",
            r"\bdd\s+if=",
            r">\s*/dev/sd[a-z]",
            r":\(\)\s*\{\s*:\|:&\s*\};",  # fork bomb
            # Privilege escalation
            r"\bsudo\b",
            r"\bchmod\s+\+x\b",
            # Git mayhem
            r"\bgit\s+push\s+(-f|--force)\b",
            r"\bgit\s+reset\s+--hard\b",
            r"\bgit\s+clean\s+-fd\b",
        ]
    )
    """Regex patterns; if any matches text being typed, action is blocked."""

    block_text_pattern_flags: int = re.IGNORECASE
    """Flags applied to every pattern in ``block_text_patterns``."""

    ask_when_unmatched_spatial: bool = False
    """If True, every spatial action (click/drag) prompts for confirmation.

    Off by default — interactive confirmation on every click makes demos
    unusable. Turn on for production deployments with audit trails.
    """


class DefaultSafetyGuard:
    """Reference :class:`SafetyGuard` implementation.

    Stateless apart from its policy. Safe to share across loops.
    """

    def __init__(self, policy: SafetyPolicy | None = None) -> None:
        self.policy = policy or SafetyPolicy()
        self._compiled_patterns = [
            re.compile(p, self.policy.block_text_pattern_flags)
            for p in self.policy.block_text_patterns
        ]
        self._block_hotkeys = {
            tuple(sorted(k.lower() for k in chord)) for chord in self.policy.block_hotkeys
        }

    async def check(
        self,
        action: Action,
        observation: Observation | None = None,
    ) -> SafetyDecision:
        if isinstance(action, HotkeyAction):
            chord = tuple(sorted(k.lower() for k in action.keys))
            if chord in self._block_hotkeys:
                return SafetyDecision(
                    verdict="deny",
                    reason=f"Blocked hotkey: {list(action.keys)}",
                )

        if isinstance(action, TypeAction):
            for pattern in self._compiled_patterns:
                if pattern.search(action.text):
                    return SafetyDecision(
                        verdict="deny",
                        reason=(
                            f"Blocked text matching pattern "
                            f"{pattern.pattern!r}: {action.text[:60]!r}"
                        ),
                    )

        if self.policy.ask_when_unmatched_spatial and _is_spatial(action):
            return SafetyDecision(
                verdict="ask",
                reason="Manual confirmation required for spatial action.",
            )

        return SafetyDecision(verdict="allow")


def _is_spatial(action: Action) -> bool:
    """Whether an action targets a screen location."""
    return action.type in {"click", "double_click", "right_click", "drag"}


def policy_from_blocklists(
    *,
    hotkeys: Iterable[tuple[str, ...]] | None = None,
    patterns: Iterable[str] | None = None,
) -> SafetyPolicy:
    """Convenience builder: merge extra entries into the default policy."""
    base = SafetyPolicy()
    if hotkeys is not None:
        base.block_hotkeys = [*base.block_hotkeys, *hotkeys]
    if patterns is not None:
        base.block_text_patterns = [*base.block_text_patterns, *patterns]
    return base


__all__ = [
    "DefaultSafetyGuard",
    "SafetyPolicy",
    "policy_from_blocklists",
]
