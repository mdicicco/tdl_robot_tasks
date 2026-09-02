"""Per-system view of a task for sequence expansion and knot building."""

from __future__ import annotations

from dataclasses import dataclass

from tdl.schema import ForcePush, Gate, Keyhole, Location, Task


@dataclass
class SystemContext:
    name: str | None
    locations: dict[str, Location]
    force_pushes: dict[str, ForcePush]
    keyholes: dict[str, Keyhole]
    gates: dict[str, Gate]
    sequence: list

    @classmethod
    def from_task(cls, task: Task, system_name: str | None = None) -> SystemContext:
        if system_name is not None:
            spec = task.systems[system_name]
            return cls(
                name=system_name,
                locations=spec.locations,
                force_pushes=spec.force_pushes,
                keyholes=spec.keyholes,
                gates=task.gates,
                sequence=spec.sequence,
            )
        return cls(
            name=None,
            locations=task.locations,
            force_pushes=task.force_pushes,
            keyholes=task.keyholes,
            gates=task.gates,
            sequence=task.sequence,
        )

    def limits(self, task: Task):
        if self.name is not None:
            spec = task.systems[self.name]
            return spec.limits or task.limits
        return task.limits
