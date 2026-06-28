"""PyAutoGUI-backed Executor.

PyAutoGUI is the most cross-platform "good-enough" library for mouse/keyboard
on macOS. It abstracts over pyobjc/quartz on macOS, Xlib on Linux, and
ctypes on Windows.

This executor is intentionally thin — every action is a one-shot call. State
(e.g. currently focused window) is left to the OS, mirroring how a human
operates.

PyAutoGUI's FAILSAFE (move mouse to corner to abort) is left on by default.
"""

from __future__ import annotations

import time

import pyautogui

from ..core.actions import (
    Action,
    ClickAction,
    DoubleClickAction,
    DragAction,
    HotkeyAction,
    RightClickAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)
from ..core.protocols import ActionResult

# Match PyAutoGUI's button names to our literal set.
_BUTTON_MAP = {"left": "left", "right": "right", "middle": "middle"}


class PyAutoGUIExecutor:
    """Reference :class:`Executor` implementation.

    Args:
        move_duration: Seconds spent moving the cursor between two points.
            0 means teleport. Small values look more natural and avoid
            FAILSAFE false-positives.
        type_interval: Seconds between keystrokes when typing.
        fail_safe: If True (default), slamming the mouse to a screen corner
            raises pyautogui.FailSafeException. Keep on.
    """

    def __init__(
        self,
        *,
        move_duration: float = 0.15,
        type_interval: float = 0.0,
        fail_safe: bool = True,
    ) -> None:
        self.move_duration = move_duration
        self.type_interval = type_interval
        pyautogui.FAILSAFE = fail_safe
        pyautogui.PAUSE = 0.0  # we manage pauses ourselves

    async def execute(self, action: Action) -> ActionResult:
        started = time.time()
        try:
            self._dispatch(action)
            return ActionResult(
                success=True,
                duration_ms=(time.time() - started) * 1000,
                metadata={"action_type": action.type},
            )
        except pyautogui.FailSafeException as exc:
            return ActionResult(
                success=False,
                error=f"Fail-safe triggered: {exc}",
                duration_ms=(time.time() - started) * 1000,
            )
        except Exception as exc:
            return ActionResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.time() - started) * 1000,
            )

    def _dispatch(self, action: Action) -> None:
        if isinstance(action, ClickAction):
            x, y = action.target.center
            pyautogui.moveTo(x, y, duration=self.move_duration)
            pyautogui.click(button=_BUTTON_MAP[action.button])
        elif isinstance(action, DoubleClickAction):
            x, y = action.target.center
            pyautogui.moveTo(x, y, duration=self.move_duration)
            pyautogui.doubleClick(button=_BUTTON_MAP[action.button])
        elif isinstance(action, RightClickAction):
            x, y = action.target.center
            pyautogui.moveTo(x, y, duration=self.move_duration)
            pyautogui.rightClick()
        elif isinstance(action, TypeAction):
            if action.target is not None:
                x, y = action.target.center
                pyautogui.click(x, y, duration=self.move_duration)
            pyautogui.write(action.text, interval=self.type_interval)
        elif isinstance(action, HotkeyAction):
            self._hotkey(action.keys)
        elif isinstance(action, ScrollAction):
            anchor = action.anchor
            if anchor is not None:
                pyautogui.moveTo(*anchor, duration=self.move_duration)
            # PyAutoGUI scroll takes "clicks" not pixels; 1 click ~= ~100 px.
            clicks = action.scroll_dy // 100 or action.scroll_dx // 100
            if clicks != 0:
                pyautogui.scroll(clicks)
        elif isinstance(action, DragAction):
            sx, sy = action.source.center
            tx, ty = action.target.center
            pyautogui.moveTo(sx, sy, duration=self.move_duration)
            pyautogui.dragTo(tx, ty, duration=action.duration, button="left")
        elif isinstance(action, WaitAction):
            time.sleep(action.seconds)
        else:
            raise ValueError(
                f"PyAutoGUIExecutor cannot dispatch action of type "
                f"{getattr(action, 'type', '<unknown>')!r}"
            )

    @staticmethod
    def _hotkey(keys: list[str]) -> None:
        """Press keys in order, releasing in reverse.

        PyAutoGUI's ``hotkey`` does this already; we wrap it to normalize
        key naming (``cmd`` → ``command``, ``opt`` → ``option``, ...).
        """
        normalized = [_normalize_key(k) for k in keys]
        pyautogui.hotkey(*normalized)


# PyAutoGUI uses long names; humans type short ones. Map common aliases.
_KEY_ALIASES: dict[str, str] = {
    "cmd": "command",
    "opt": "option",
    "ctrl": "ctrl",
    "shift": "shift",
    "esc": "escape",
    "return": "return",
    "enter": "return",
    "tab": "tab",
    "space": "space",
    "del": "delete",
    "backspace": "backspace",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


def _normalize_key(key: str) -> str:
    """Lower-case and map short aliases to PyAutoGUI's expected names."""
    lower = key.lower()
    return _KEY_ALIASES.get(lower, lower)


__all__ = ["PyAutoGUIExecutor"]
