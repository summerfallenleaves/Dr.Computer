"""Qwen-VL provider — fused (decision A3).

Qwen2.5-VL has native visual grounding: it can directly output pixel
coordinates for elements described in the prompt. This makes Qwen-VL a
*fused* provider — it plays both Provider (planner) and Grounder at once,
filling ``Intent.grounded_target`` directly. The AgentLoop detects this and
skips the separate Grounder call.

Endpoint:

- DashScope's OpenAI-compatible endpoint at
  ``https://dashscope.aliyuncs.com/compatible-mode/v1``
  (also reachable via the ``dashscope.aliyun.com`` alias, but ``aliyuncs.com``
  has more reliable DNS resolution outside China mainland)
- Set ``DASHSCOPE_API_KEY`` (or pass ``api_key``).

Response format expected from the model:

The system prompt instructs the model to emit a single JSON object:

    {
      "reasoning": "I need to click the Notes icon in the Dock",
      "action_type": "click",
      "bbox": [1230, 1180, 1290, 1240],   // optional, present for spatial
      "label": "Notes",                    // optional
      "text": "hello",                     // for type
      "keys": ["cmd", "s"],                // for hotkey
      "scroll_dx": 0, "scroll_dy": -300,   // for scroll
      "wait_seconds": 1.5,                 // for wait
      "summary": "task complete"           // for done
    }

bbox is in *logical* pixel coordinates matching what the perceiver returned.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from ..core.grounding import GroundedTarget
from ..core.intent import Intent
from ..core.messages import Message
from ..core.observation import Observation
from .base import OpenAICompatibleBase

logger = logging.getLogger(__name__)


QWEN_VL_SYSTEM_PROMPT = """\
You are a Desktop AI Agent operating a macOS computer.

You receive the current screenshot and the user's goal. You must decide the
single next action that moves toward the goal, then return JSON only.

Available action types:
- click:        click on a screen element (requires bbox)
- double_click: double-click (requires bbox)
- right_click:  right-click (requires bbox)
- type:         type text into the focused element (requires text). Optionally
                include bbox to click into a field first.
- hotkey:       press a key chord, e.g. cmd+s (requires keys)
- scroll:       scroll the page (requires non-zero scroll_dx and/or scroll_dy)
- wait:         wait briefly for UI to settle (optional wait_seconds)
- done:         the goal is achieved; include a summary

bbox format: [x1, y1, x2, y2] in pixel coordinates of the screenshot you see.
You may also return a click point as [x, y] (2 elements); both are accepted.

Respond with a single JSON object, no markdown, no prose:
{
  "reasoning": "<one short sentence>",
  "action_type": "<one of the above>",
  "bbox": [x1, y1, x2, y2],
  "label": "<optional element label>",
  "text": "<for type>",
  "keys": ["<key>", "..."],
  "scroll_dx": 0,
  "scroll_dy": 0,
  "wait_seconds": 1.0,
  "summary": "<for done>"
}

Only include the fields that apply to your chosen action_type.

## When to return "done"

Return action_type="done" as soon as the goal is achieved. Specifically:

- If the goal is "open X" and X is now visible/active → done.
- If the goal is "click Y" and Y has been clicked (you can see the result,
  e.g. a menu opened, a dialog appeared, the URL changed) → done IMMEDIATELY
  on the next turn. Do NOT click Y again.
- If the goal is "type Z" and Z is now in the input field → done.

Do not over-act. A single click that achieves the goal is success — the
next turn should be "done", not another click.

## Few-shot examples

Goal: "Open the Apple menu"
Step 1: screenshot shows desktop, Apple logo at top-left
  → {"reasoning":"click Apple logo to open menu","action_type":"click","bbox":[5,5]}
Step 2: screenshot shows Apple menu dropdown is now open
  → {"reasoning":"menu is open, goal achieved","action_type":"done","summary":"Apple menu opened"}

Goal: "Type 'hello' into the search box"
Step 1: screenshot shows a search box at coordinates [200, 50, 400, 80]
  → {"reasoning":"click into search box","action_type":"click","bbox":[200,50,400,80]}
Step 2: screenshot shows cursor is now in the search box
  → {"reasoning":"type the query","action_type":"type","text":"hello"}
Step 3: screenshot shows 'hello' is in the search box
  → {"reasoning":"text entered, goal achieved","action_type":"done","summary":"typed 'hello'"}

Goal: "Save the document"
Step 1: screenshot shows the document editor
  → {"reasoning":"trigger save via keyboard","action_type":"hotkey","keys":["cmd","s"]}
Step 2: screenshot shows save succeeded (no save dialog, or status changed)
  → {"reasoning":"save completed","action_type":"done","summary":"document saved"}

## Anti-patterns (do NOT do these)

- Re-issuing the same click that already happened.
- Returning "done" before checking the screenshot actually shows success.
- Returning an action when the goal is already met.
- Including markdown fences or prose around the JSON.
"""


class QwenVLProvider(OpenAICompatibleBase):
    """Fused Qwen-VL provider. Returns intents with coordinates already filled."""

    name = "qwen-vl"
    default_model = "qwen-vl-max"

    def _api_key_env_name(self) -> str:
        return "DASHSCOPE_API_KEY"

    def _default_base_url(self) -> str:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def build_messages(
        self, messages: list[Message], observation: Observation
    ) -> list[dict[str, Any]]:
        # Inject the Qwen-VL-specific system prompt at the head.
        out = super().build_messages(messages, observation)
        if not out or out[0].get("role") != "system":
            out.insert(0, {"role": "system", "content": QWEN_VL_SYSTEM_PROMPT})
        else:
            # Merge with whatever system text the loop passed in.
            existing = out[0].get("content", "")
            out[0]["content"] = QWEN_VL_SYSTEM_PROMPT + "\n\n" + str(existing)
        return out

    def parse_intent(self, raw_text: str, **_: Any) -> Intent:
        data = self.safe_json_load(raw_text)
        if data is None:
            logger.warning(
                "Qwen-VL response was not valid JSON. Falling back to a "
                "terminal intent. Raw text: %.200s",
                raw_text,
            )
            return Intent(
                action_type="done",
                reasoning="Could not parse model response as JSON.",
                summary=f"Parse failure: {raw_text[:120]}",
            )

        action_type = data.get("action_type", "done")
        kwargs: dict[str, Any] = {
            "reasoning": str(data.get("reasoning", ""))[:500],
        }

        bbox = data.get("bbox")
        coordinates = data.get("coordinates") or data.get("point") or data.get("position")
        # Many VLMs (qwen-vl-max included) return a click point [x, y]
        # rather than a 4-element box. If we only get a point, expand it
        # into a small box so the rest of the pipeline (which expects a
        # bbox) keeps working. Clicks land on the centre anyway.
        point_expand = 12  # half-size of the synthetic box, in pixels

        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = (int(v) for v in bbox)
            except (TypeError, ValueError):
                x1 = y1 = x2 = y2 = 0
            if x2 > x1 and y2 > y1:
                kwargs["grounded_target"] = GroundedTarget(
                    bbox=(x1, y1, x2, y2),
                    label=data.get("label"),
                )
        elif isinstance(bbox, list) and len(bbox) == 2:
            try:
                px, py = (int(v) for v in bbox)
                kwargs["grounded_target"] = GroundedTarget(
                    bbox=(
                        px - point_expand,
                        py - point_expand,
                        px + point_expand,
                        py + point_expand,
                    ),
                    label=data.get("label"),
                )
            except (TypeError, ValueError):
                pass
        elif isinstance(coordinates, list) and len(coordinates) == 2:
            try:
                px, py = (int(v) for v in coordinates)
                kwargs["grounded_target"] = GroundedTarget(
                    bbox=(
                        px - point_expand,
                        py - point_expand,
                        px + point_expand,
                        py + point_expand,
                    ),
                    label=data.get("label"),
                )
            except (TypeError, ValueError):
                pass
        elif isinstance(bbox, dict):
            # Some Qwen variants return {"xmin":..., "ymin":..., ...}.
            with contextlib.suppress(KeyError, TypeError, ValueError):
                kwargs["grounded_target"] = GroundedTarget(
                    bbox=(
                        int(bbox["xmin"]),
                        int(bbox["ymin"]),
                        int(bbox["xmax"]),
                        int(bbox["ymax"]),
                    ),
                    label=data.get("label"),
                )

        target_desc = data.get("target_description") or data.get("description")
        if target_desc and "grounded_target" not in kwargs:
            kwargs["target_description"] = str(target_desc)

        if action_type == "type" and data.get("text") is not None:
            kwargs["text"] = str(data["text"])
        if action_type == "hotkey" and data.get("keys"):
            keys = data["keys"]
            if isinstance(keys, str):
                keys = [keys]
            kwargs["keys"] = [str(k) for k in keys]
        if action_type == "scroll":
            kwargs["scroll_dx"] = int(data.get("scroll_dx", 0) or 0)
            kwargs["scroll_dy"] = int(data.get("scroll_dy", 0) or 0)
        if action_type == "wait":
            wait = data.get("wait_seconds")
            kwargs["wait_seconds"] = float(wait) if wait is not None else None
        if action_type == "done":
            kwargs["summary"] = str(data.get("summary", ""))

        return Intent(action_type=action_type, **kwargs)

    async def chat(self, messages: list[Message], observation: Observation) -> Intent:
        raw = await self._raw_chat(messages, observation)
        return self.parse_intent(raw)


__all__ = ["QwenVLProvider"]
