"""OpenAI-compatible HTTP client base.

Most Chinese VLM providers (Qwen-VL via DashScope's OpenAI-compatible mode,
GLM-4V via Zhipuai's, Moonshot, DeepSeek, ...) accept the OpenAI Chat
Completions payload format with ``image_url`` content blocks.

This base class wraps the OpenAI Python SDK so concrete providers only need
to fill in:

- endpoint URL and API key
- model name
- how to parse the model's response into an :class:`Intent`

Subclasses can also override ``_build_chat_request`` if the API deviates
from the OpenAI shape (e.g. requires a non-standard grounding field).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI

from ..core.intent import Intent
from ..core.messages import Message
from ..core.observation import Observation

logger = logging.getLogger(__name__)


class OpenAICompatibleBase:
    """Common scaffolding for OpenAI-protocol providers.

    Concrete providers (``QwenVLProvider``, ``GLM4VProvider``, ...) should
    inherit and override :meth:`parse_intent` (response → :class:`Intent`)
    at minimum.
    """

    #: Display name shown in logs.
    name: str = "openai-compatible"

    #: Default model id; subclasses set this.
    default_model: str = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        resolved_key = api_key or self._default_api_key()
        if not resolved_key:
            raise RuntimeError(
                f"No API key for {self.name}. Set it via constructor or the "
                f"environment variable {self._api_key_env_name()!r}."
            )

        self.api_key = resolved_key
        self.base_url = base_url or self._default_base_url()
        self.model = model or self.default_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_headers = extra_headers or {}
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            default_headers=self.extra_headers,
        )

    # --- subclasses override these ---

    def _default_api_key(self) -> str | None:
        return os.environ.get(self._api_key_env_name())

    def _api_key_env_name(self) -> str:
        return "DR_COMPUTER_API_KEY"

    def _default_base_url(self) -> str | None:
        return None

    def parse_intent(self, raw_text: str, **kwargs: Any) -> Intent:
        """Parse the model's text response into an :class:`Intent`.

        Default implementation expects a JSON object matching the Intent
        schema. Subclasses may override to accept a different format
        (e.g. XML, Markdown with fenced JSON, native tool-call args).
        """
        raise NotImplementedError

    # --- shared helpers ---

    def build_messages(
        self, messages: list[Message], observation: Observation
    ) -> list[dict[str, Any]]:
        """Convert internal messages + screenshot into OpenAI chat format.

        The current screenshot is appended as a final ``user`` turn with
        an ``image_url`` content block. If the last message is already from
        the user, we append to it; otherwise we add a new turn.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.content
            if isinstance(content, str):
                out.append({"role": msg.role, "content": content})
            else:
                # Non-string content blocks (images from history, tool calls).
                # For MVP we just dump text blocks; full fidelity is Phase 2.
                text = msg.as_text()
                out.append({"role": msg.role, "content": text})

        # Append the current screenshot as a final user turn.
        image_url = self._screenshot_data_url(observation)
        out.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Here is the current screen state. Decide the next single action."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
        return out

    def _screenshot_data_url(self, observation: Observation) -> str:
        """PNG → base64 data URL, with optional downscaling."""
        from ..utils.image import resize_bytes, to_base64

        # Cap to 1280px on the longest side — most VLM APIs downscale
        # anyway, and this keeps latency & token cost in check.
        data = resize_bytes(observation.screenshot, max_width=1280, max_height=1280)
        return to_base64(data, mime_type="image/png")

    async def _raw_chat(self, messages: list[Message], observation: Observation) -> str:
        """Send the chat request and return the assistant's raw text.

        Subclasses call this, then feed the result through :meth:`parse_intent`.
        """
        oai_messages = self.build_messages(messages, observation)
        logger.debug(
            "%s sending %d messages (screenshot %dx%d)",
            self.name,
            len(oai_messages),
            observation.width,
            observation.height,
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # Take the first choice's text. Tool-call handling is Phase 2.
        choice = resp.choices[0]
        return choice.message.content or ""

    @staticmethod
    def safe_json_load(text: str) -> dict[str, Any] | None:
        """Try to extract a JSON object from a model's text response.

        Models sometimes wrap JSON in ```json fences or include prose
        around it. This helper extracts the first balanced ``{...}`` blob.
        """
        text = text.strip()
        if text.startswith("```"):
            # Strip a leading code fence.
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Fallback: scan for the first {...} region.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


__all__ = ["OpenAICompatibleBase"]
