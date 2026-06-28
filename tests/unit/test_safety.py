"""Safety guard tests."""

from __future__ import annotations

import pytest

from dr_computer.core.actions import (
    ClickAction,
    DoubleClickAction,
    HotkeyAction,
    TypeAction,
)
from dr_computer.core.grounding import GroundedTarget
from dr_computer.core.safety import DefaultSafetyGuard, SafetyPolicy


@pytest.fixture
def guard() -> DefaultSafetyGuard:
    return DefaultSafetyGuard()


async def test_dangerous_hotkey_blocked(guard: DefaultSafetyGuard) -> None:
    decision = await guard.check(HotkeyAction(keys=["cmd", "q"]))
    assert decision.verdict == "deny"
    assert "cmd" in decision.reason.lower()


async def test_safe_hotkey_allowed(guard: DefaultSafetyGuard) -> None:
    decision = await guard.check(HotkeyAction(keys=["cmd", "c"]))
    assert decision.verdict == "allow"


async def test_hotkey_case_insensitive(guard: DefaultSafetyGuard) -> None:
    decision = await guard.check(HotkeyAction(keys=["CMD", "Q"]))
    assert decision.verdict == "deny"


async def test_hotkey_order_independent(guard: DefaultSafetyGuard) -> None:
    decision = await guard.check(HotkeyAction(keys=["q", "cmd"]))
    assert decision.verdict == "deny"


async def test_rm_rf_text_blocked(guard: DefaultSafetyGuard) -> None:
    decision = await guard.check(TypeAction(text="rm -rf /"))
    assert decision.verdict == "deny"


async def test_force_push_blocked(guard: DefaultSafetyGuard) -> None:
    decision = await guard.check(TypeAction(text="git push --force origin main"))
    assert decision.verdict == "deny"


async def test_safe_text_allowed(guard: DefaultSafetyGuard) -> None:
    decision = await guard.check(TypeAction(text="hello world"))
    assert decision.verdict == "allow"


async def test_spatial_action_default_allowed(guard: DefaultSafetyGuard) -> None:
    action = ClickAction(target=GroundedTarget(bbox=(10, 10, 50, 50)))
    decision = await guard.check(action)
    assert decision.verdict == "allow"


async def test_strict_policy_asks_for_spatial() -> None:
    guard = DefaultSafetyGuard(SafetyPolicy(ask_when_unmatched_spatial=True))
    action = DoubleClickAction(target=GroundedTarget(bbox=(10, 10, 50, 50)))
    decision = await guard.check(action)
    assert decision.verdict == "ask"


async def test_sudo_in_text_blocked(guard: DefaultSafetyGuard) -> None:
    decision = await guard.check(TypeAction(text="sudo apt install evil"))
    assert decision.verdict == "deny"


def test_safety_policy_defaults() -> None:
    p = SafetyPolicy()
    assert ("cmd", "q") in p.block_hotkeys
    assert any("rm" in pat for pat in p.block_text_patterns)
