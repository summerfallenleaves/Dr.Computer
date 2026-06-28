"""Dr.Computer — A Python framework for building Desktop AI Agents."""

from __future__ import annotations

__version__ = "0.1.0"

# --- Core abstractions ---
from .core.actions import (
    Action,
    ClickAction,
    DoneAction,
    DoubleClickAction,
    DragAction,
    HotkeyAction,
    RightClickAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)
from .core.grounding import GroundedTarget
from .core.intent import Intent, IntentType
from .core.loop import AgentLoop, AgentLoopCancelled, HumanConfirmationRequired
from .core.messages import (
    ContentBlock,
    ImageBlock,
    Message,
    Role,
    TextBlock,
    ToolCall,
    ToolResult,
)
from .core.observation import Observation
from .core.protocols import (
    ActionResult,
    Executor,
    Grounder,
    Memory,
    Perceiver,
    Provider,
    SafetyDecision,
    SafetyGuard,
    SafetyVerdict,
    Verifier,
    VerifyResult,
)
from .core.safety import DefaultSafetyGuard, SafetyPolicy, policy_from_blocklists
from .core.trajectory import Step, StepStatus, Trajectory, new_task_id

# --- Default implementations ---
from .execution.pyautogui_exec import PyAutoGUIExecutor
from .memory.in_memory import InMemoryMemory
from .perception.macos import MacOSScreenshotPerceiver
from .providers.base import OpenAICompatibleBase
from .providers.qwen_vl import QwenVLProvider

__all__ = [
    "Action",
    "ActionResult",
    "AgentLoop",
    "AgentLoopCancelled",
    "ClickAction",
    "ContentBlock",
    "DefaultSafetyGuard",
    "DoneAction",
    "DoubleClickAction",
    "DragAction",
    "Executor",
    "GroundedTarget",
    "Grounder",
    "HotkeyAction",
    "HumanConfirmationRequired",
    "ImageBlock",
    "InMemoryMemory",
    "Intent",
    "IntentType",
    "MacOSScreenshotPerceiver",
    "Memory",
    "Message",
    "Observation",
    "OpenAICompatibleBase",
    "Perceiver",
    "Provider",
    "PyAutoGUIExecutor",
    "QwenVLProvider",
    "RightClickAction",
    "Role",
    "SafetyDecision",
    "SafetyGuard",
    "SafetyPolicy",
    "SafetyVerdict",
    "ScrollAction",
    "Step",
    "StepStatus",
    "TextBlock",
    "ToolCall",
    "ToolResult",
    "Trajectory",
    "TypeAction",
    "Verifier",
    "VerifyResult",
    "WaitAction",
    "__version__",
    "new_task_id",
    "policy_from_blocklists",
]
