"""In-process Memory implementation.

No persistence — trajectories live as long as the Python process. Used by
MVP examples and tests. For production, drop in a SQLite- or Redis-backed
implementation that satisfies the same :class:`Memory` protocol.

The implementation is dict-based; ``open`` returns a fresh trajectory keyed
by ``task_id``. ``append`` mutates that trajectory. ``close`` marks outcome.
"""

from __future__ import annotations

import asyncio

from ..core.trajectory import Step, Trajectory


class InMemoryMemory:
    """Reference :class:`Memory` implementation. Thread-unsafe, single-process."""

    def __init__(self) -> None:
        # asyncio.Lock keeps the surface consistent with persistent backends
        # (SQLite, Redis) that will need real locking. The lock is reentrant
        # within one event loop.
        self._lock = asyncio.Lock()
        self._store: dict[str, Trajectory] = {}

    async def open(self, task_id: str, goal: str) -> Trajectory:
        async with self._lock:
            if task_id in self._store:
                # Resuming an existing trajectory — return as-is.
                return self._store[task_id]
            traj = Trajectory(task_id=task_id, goal=goal)
            self._store[task_id] = traj
            return traj

    async def append(self, task_id: str, step: Step) -> None:
        async with self._lock:
            traj = self._store.get(task_id)
            if traj is None:
                raise KeyError(f"Unknown task_id {task_id!r}; call open() before append().")
            traj.steps.append(step)

    async def close(self, task_id: str, outcome: str, summary: str = "") -> Trajectory:
        async with self._lock:
            traj = self._store.get(task_id)
            if traj is None:
                raise KeyError(f"Unknown task_id {task_id!r}")
            traj.outcome = outcome  # type: ignore[assignment]
            traj.summary = summary
            return traj

    async def load(self, task_id: str) -> Trajectory:
        async with self._lock:
            traj = self._store.get(task_id)
            if traj is None:
                raise KeyError(f"Unknown task_id {task_id!r}")
            return traj

    def all_trajectories(self) -> list[Trajectory]:
        """Test/debug helper — synchronous snapshot of all stored trajectories."""
        return list(self._store.values())


__all__ = ["InMemoryMemory"]
