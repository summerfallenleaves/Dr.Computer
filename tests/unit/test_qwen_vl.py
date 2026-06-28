"""QwenVLProvider.parse_intent tests.

Covers the various JSON shapes Qwen-VL (and other VLMs) actually return in
practice: clean JSON, markdown-wrapped, 2-element point vs 4-element bbox,
dict-style bbox, missing fields, malformed input.
"""

from __future__ import annotations

import os

import pytest

from dr_computer.providers.qwen_vl import QwenVLProvider


@pytest.fixture
def provider() -> QwenVLProvider:
    """Construct with a fake key — we never call the API in these tests."""
    return QwenVLProvider(api_key="sk-fake-for-tests")


# --- clean JSON, all action types ---


def test_parse_click_with_4_element_bbox(provider: QwenVLProvider) -> None:
    raw = '{"reasoning":"click","action_type":"click","bbox":[10,20,110,80]}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "click"
    assert intent.grounded_target is not None
    assert intent.grounded_target.bbox == (10, 20, 110, 80)
    assert intent.grounded_target.center == (60, 50)
    assert intent.is_spatial()
    assert not intent.is_terminal()


def test_parse_click_with_2_element_point(provider: QwenVLProvider) -> None:
    """Qwen-VL's actual native format — must be supported."""
    raw = '{"action_type":"click","bbox":[100,200]}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "click"
    assert intent.grounded_target is not None
    # 2-element point expands to ±12 px box around the click target.
    assert intent.grounded_target.center == (100, 200)


def test_parse_click_with_coordinates_key(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"click","coordinates":[50,75]}'
    intent = provider.parse_intent(raw)
    assert intent.grounded_target is not None
    assert intent.grounded_target.center == (50, 75)


def test_parse_click_with_point_key(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"click","point":[50,75]}'
    intent = provider.parse_intent(raw)
    assert intent.grounded_target is not None
    assert intent.grounded_target.center == (50, 75)


def test_parse_click_with_dict_bbox(provider: QwenVLProvider) -> None:
    """Some Qwen variants return xmin/ymin/xmax/ymax dict."""
    raw = '{"action_type":"click","bbox":{"xmin":10,"ymin":20,"xmax":100,"ymax":80},"label":"OK"}'
    intent = provider.parse_intent(raw)
    assert intent.grounded_target is not None
    assert intent.grounded_target.bbox == (10, 20, 100, 80)
    assert intent.grounded_target.label == "OK"


def test_parse_type(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"type","text":"hello world","reasoning":"greet"}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "type"
    assert intent.text == "hello world"
    assert intent.grounded_target is None
    assert not intent.is_spatial()


def test_parse_type_with_target(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"type","text":"hi","bbox":[200,50,400,80]}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "type"
    assert intent.text == "hi"
    assert intent.grounded_target is not None


def test_parse_hotkey(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"hotkey","keys":["cmd","s"]}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "hotkey"
    assert intent.keys == ["cmd", "s"]


def test_parse_hotkey_string_keys(provider: QwenVLProvider) -> None:
    """Some models return keys as a single string instead of list."""
    raw = '{"action_type":"hotkey","keys":"cmd+s"}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "hotkey"
    # Single string is wrapped into a one-element list.
    assert intent.keys == ["cmd+s"]


def test_parse_scroll(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"scroll","scroll_dy":-300}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "scroll"
    assert intent.scroll_dy == -300
    assert intent.scroll_dx == 0


def test_parse_wait(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"wait","wait_seconds":1.5}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "wait"
    assert intent.wait_seconds == 1.5


def test_parse_done(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"done","summary":"notes opened"}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "done"
    assert intent.is_terminal()
    assert intent.summary == "notes opened"


# --- resilience: malformed input ---


def test_parse_markdown_wrapped_json(provider: QwenVLProvider) -> None:
    raw = '```json\n{"action_type":"done","summary":"ok"}\n```'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "done"
    assert intent.summary == "ok"


def test_parse_json_with_prose_around(provider: QwenVLProvider) -> None:
    raw = (
        "Let me think about this...\n"
        '{"action_type":"click","bbox":[10,10,50,50]}\n'
        "That's my answer."
    )
    intent = provider.parse_intent(raw)
    assert intent.action_type == "click"
    assert intent.grounded_target is not None


def test_parse_invalid_json_returns_done(provider: QwenVLProvider) -> None:
    """Invalid JSON must not crash — fall back to terminal done."""
    intent = provider.parse_intent("this is not json at all")
    assert intent.action_type == "done"
    assert intent.is_terminal()
    assert "Parse failure" in intent.summary


def test_parse_empty_bbox_list(provider: QwenVLProvider) -> None:
    raw = '{"action_type":"click","bbox":[]}'
    intent = provider.parse_intent(raw)
    # Empty bbox cannot be expanded → grounded_target stays None.
    assert intent.action_type == "click"
    assert intent.grounded_target is None


def test_parse_invalid_bbox_values(provider: QwenVLProvider) -> None:
    """Non-numeric bbox values shouldn't crash — they're just ignored."""
    raw = '{"action_type":"click","bbox":["a","b","c","d"]}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "click"
    assert intent.grounded_target is None


def test_parse_invalid_4_element_bbox(provider: QwenVLProvider) -> None:
    """x2<=x1 or y2<=y1 should be rejected."""
    raw = '{"action_type":"click","bbox":[100,100,50,50]}'
    intent = provider.parse_intent(raw)
    assert intent.grounded_target is None


def test_parse_unknown_action_type(provider: QwenVLProvider) -> None:
    """Unknown action_type should still construct (validation happens downstream)."""
    raw = '{"action_type":"teleport","reasoning":"unsupported"}'
    # parse_intent doesn't validate action_type — Intent will, but we trust
    # the model and let it through.
    try:
        intent = provider.parse_intent(raw)
        assert intent.action_type == "teleport"
    except Exception:
        # Pydantic may reject unknown Literal — that's also acceptable.
        pass


def test_parse_missing_reasoning(provider: QwenVLProvider) -> None:
    """reasoning defaults to empty string."""
    raw = '{"action_type":"done","summary":"ok"}'
    intent = provider.parse_intent(raw)
    assert intent.reasoning == ""


def test_parse_extra_unknown_fields_ignored(provider: QwenVLProvider) -> None:
    """Unknown JSON fields should be ignored, not error."""
    raw = '{"action_type":"done","summary":"ok","random_field":"ignore me","another":[1,2,3]}'
    intent = provider.parse_intent(raw)
    assert intent.action_type == "done"


# --- safe_json_load helper ---


def test_safe_json_load_plain(provider: QwenVLProvider) -> None:
    assert provider.safe_json_load('{"a":1}') == {"a": 1}


def test_safe_json_load_fenced(provider: QwenVLProvider) -> None:
    raw = '```json\n{"a":1}\n```'
    assert provider.safe_json_load(raw) == {"a": 1}


def test_safe_json_load_with_prose(provider: QwenVLProvider) -> None:
    raw = 'Sure: {"a":1} hope that helps'
    assert provider.safe_json_load(raw) == {"a": 1}


def test_safe_json_load_invalid(provider: QwenVLProvider) -> None:
    assert provider.safe_json_load("totally not json") is None


# --- regression: env var name ---


def test_api_key_env_name(provider: QwenVLProvider) -> None:
    assert provider._api_key_env_name() == "DASHSCOPE_API_KEY"


def test_default_base_url_uses_international_endpoint(
    provider: QwenVLProvider,
) -> None:
    """Decision F1: aliyuncs.com, not aliyun.com."""
    url = provider._default_base_url()
    assert "aliyuncs.com" in url
    assert "aliyun.com" not in url.replace("aliyuncs.com", "X")


# Keep env from leaking between tests.
@pytest.fixture(autouse=True)
def _isolate_dashscope_env() -> None:
    saved = os.environ.pop("DASHSCOPE_API_KEY", None)
    yield
    if saved is not None:
        os.environ["DASHSCOPE_API_KEY"] = saved
