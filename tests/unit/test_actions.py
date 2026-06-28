"""Action discriminated-union serialization tests."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from dr_computer.core.actions import (
    Action,
    ClickAction,
    DoneAction,
    HotkeyAction,
    TypeAction,
    WaitAction,
)
from dr_computer.core.grounding import GroundedTarget


def _action_adapter() -> TypeAdapter[Action]:
    return TypeAdapter(Action)


def test_click_action_round_trip() -> None:
    a = ClickAction(target=GroundedTarget(bbox=(10, 20, 110, 120)))
    dumped = a.model_dump()
    assert dumped["type"] == "click"
    # Pydantic v2 keeps tuple types intact.
    assert tuple(dumped["target"]["bbox"]) == (10, 20, 110, 120)


def test_action_discriminator_picks_click() -> None:
    payload = {
        "type": "click",
        "target": {"bbox": [0, 0, 10, 10]},
        "button": "right",
    }
    a = _action_adapter().validate_python(payload)
    assert isinstance(a, ClickAction)
    assert a.button == "right"


def test_action_discriminator_picks_type() -> None:
    payload = {"type": "type", "text": "hello"}
    a = _action_adapter().validate_python(payload)
    assert isinstance(a, TypeAction)
    assert a.text == "hello"


def test_action_discriminator_picks_hotkey() -> None:
    payload = {"type": "hotkey", "keys": ["cmd", "s"]}
    a = _action_adapter().validate_python(payload)
    assert isinstance(a, HotkeyAction)
    assert a.keys == ["cmd", "s"]


def test_action_discriminator_picks_done() -> None:
    payload = {"type": "done", "summary": "all good"}
    a = _action_adapter().validate_python(payload)
    assert isinstance(a, DoneAction)
    assert a.summary == "all good"


def test_invalid_action_type_rejected() -> None:
    payload = {"type": "explode", "force": 99}
    with pytest.raises(ValidationError):
        _action_adapter().validate_python(payload)


def test_click_requires_target() -> None:
    with pytest.raises(ValidationError):
        ClickAction()  # type: ignore[call-arg]


def test_wait_default_seconds() -> None:
    a = WaitAction()
    assert a.seconds == 1.0


def test_grounded_target_center() -> None:
    gt = GroundedTarget(bbox=(0, 0, 100, 50))
    assert gt.center == (50, 25)


def test_grounded_target_invalid_bbox_rejected() -> None:
    with pytest.raises(ValidationError):
        GroundedTarget(bbox=(10, 10, 5, 5))  # x2 < x1
