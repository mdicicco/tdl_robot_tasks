"""Tower light timeline from trajectory segments."""

from __future__ import annotations

from dataclasses import dataclass, field

from tdl.schema import Task, TowerLight
from tdl.trajectory import Segment


@dataclass(frozen=True)
class TowerEvent:
    time: float
    light: str
    state: int | None


@dataclass
class TowerTimeline:
    lights: dict[str, TowerLight] = field(default_factory=dict)
    events: list[TowerEvent] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: Task, segments: list[Segment]) -> TowerTimeline | None:
        if not task.tower_lights:
            return None
        events: list[TowerEvent] = []
        for light_name, light in task.tower_lights.items():
            prev: int | None = None
            for i, seg in enumerate(segments):
                idx = light.state_for_task(seg.tower_task)
                if i == 0 or idx != prev:
                    events.append(TowerEvent(seg.t0, light_name, idx))
                    prev = idx
        return cls(lights=dict(task.tower_lights), events=events)

    def state_at(self, time: float) -> dict[str, int | None]:
        state: dict[str, int | None] = {name: None for name in self.lights}
        for ev in self.events:
            if ev.time > time + 1e-9:
                break
            state[ev.light] = ev.state
        return state

    def names(self) -> list[str]:
        return sorted(self.lights)
