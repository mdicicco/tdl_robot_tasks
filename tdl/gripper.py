"""Carried gripper state timeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from tdl.schema import GripperCommand, Task


@dataclass(frozen=True)
class GripperEvent:
    time: float
    gripper: str
    closed: bool


def events_from_commands(commands: list[GripperCommand], time: float) -> list[GripperEvent]:
    out: list[GripperEvent] = []
    for cmd in commands:
        if cmd.name is None:
            continue
        out.append(GripperEvent(time, cmd.name, cmd.set == "closed"))
    return out


@dataclass
class GripperTimeline:
    initial: dict[str, bool] = field(default_factory=dict)
    events: list[GripperEvent] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: Task, event_lists: list[list[GripperEvent]]) -> GripperTimeline | None:
        if not task.grippers:
            return None
        merged = [ev for group in event_lists for ev in group]
        merged.sort(key=lambda e: (e.time, not e.closed))
        initial = {name: spec.initial == "closed" for name, spec in task.grippers.items()}
        return cls(initial=initial, events=merged)

    def state_at(self, time: float) -> dict[str, bool]:
        """True means closed."""
        state = dict(self.initial)
        for ev in self.events:
            if ev.time > time + 1e-9:
                break
            state[ev.gripper] = ev.closed
        return state
