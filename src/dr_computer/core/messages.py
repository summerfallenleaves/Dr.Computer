"""LLM message types used across all providers.

Provider-agnostic so any implementation (OpenAI-compatible, Anthropic, local)
translates to/from these primitives.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class TextBlock(BaseModel):
    """Plain text content."""

    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    """Image attached to a message.

    `data` is raw bytes of an encoded image (PNG/JPEG). Providers handle base64
    encoding and MIME-type wrapping when calling their underlying API.
    """

    type: Literal["image"] = "image"
    data: bytes
    media_type: Literal["image/png", "image/jpeg"] = "image/png"


class ToolCall(BaseModel):
    """A tool/function call requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of a tool call, returned to the assistant."""

    tool_call_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ImageBlock | ToolCall | ToolResult


class Message(BaseModel):
    """A single chat message.

    `content` may be a string (typical for user/system) or a list of typed
    blocks (needed when attaching images or tool calls).
    """

    role: Role
    content: str | list[ContentBlock] = ""

    def as_text(self) -> str:
        """Flatten content to a plain string, ignoring non-text blocks."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for block in self.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolResult):
                parts.append(block.content)
        return "\n".join(parts)


__all__ = [
    "ContentBlock",
    "ImageBlock",
    "Message",
    "Role",
    "TextBlock",
    "ToolCall",
    "ToolResult",
]
