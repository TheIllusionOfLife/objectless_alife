"""Typed domain model for agent state snapshots.

Provides ``AgentState`` (frozen dataclass) and ``Snapshot`` (type alias) to
replace the raw ``tuple[int, int, int, int]`` snapshot tuples used throughout
the codebase.  The field order matches the existing tuple convention:
``(agent_id, x, y, state)``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentState:
    """Immutable snapshot of a single agent at one point in time."""

    agent_id: int
    x: int
    y: int
    state: int
    clock: int = 0


Snapshot = tuple[AgentState, ...]
"""Ordered tuple of agent states capturing the full world at one step."""
