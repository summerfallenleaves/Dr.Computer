"""Async event emitter for the AgentLoop.

The loop emits events at each lifecycle point (step started, action executed,
task done, ...). UIs, loggers, metrics collectors subscribe via
``EventEmitter.subscribe`` and get called with the payload.

Design:

- Async-first: subscribers are coroutines. They run sequentially; one slow
  subscriber delays the rest. This is intentional — it preserves ordering
  and lets a subscriber block the loop (e.g. human-in-the-loop confirm).
- Exceptions in subscribers propagate to the caller of ``emit``; the loop
  catches and surfaces them as step errors. Subscribers that should never
  break the loop must catch their own exceptions.
- No threading. The loop runs on a single event loop; subscribers share it.
"""

from __future__ import annotations

import contextlib
import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

Subscriber: TypeAlias = Callable[[Any], Awaitable[None] | None]


class EventEmitter:
    """Minimal async pub/sub keyed by event name.

    Event names are dotted strings (``"step.started"``, ``"task.done"``).
    Wildcards are not supported — keep the surface small.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[Subscriber]] = defaultdict(list)

    def subscribe(self, event: str, handler: Subscriber) -> Callable[[], None]:
        """Register ``handler`` for ``event``.

        Returns an unsubscribe function — call it to remove the handler.
        """
        self._subs[event].append(handler)

        def _unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subs[event].remove(handler)

        return _unsubscribe

    async def emit(self, event: str, payload: Any = None) -> None:
        """Notify all subscribers of ``event`` with ``payload``.

        Subscribers run sequentially in subscription order. A subscriber may
        be a coroutine or a plain function; both are awaited/invoked.
        """
        for handler in list(self._subs.get(event, ())):
            result = handler(payload)
            if inspect.isawaitable(result):
                await result


__all__ = ["EventEmitter", "Subscriber"]
