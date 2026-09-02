"""Digital I/O timeline from location commands."""

from __future__ import annotations

from dataclasses import dataclass, field

from tdl.schema import IoCommand, Task


@dataclass(frozen=True)
class IoEvent:
    time: float
    signal: str
    on: bool


def events_from_commands(commands: list[IoCommand], time: float) -> list[IoEvent]:
    out: list[IoEvent] = []
    for cmd in commands:
        if cmd.pulse is not None:
            out.append(IoEvent(time, cmd.signal, True))
            out.append(IoEvent(time + float(cmd.pulse), cmd.signal, False))
        elif cmd.set == "on":
            out.append(IoEvent(time, cmd.signal, True))
        elif cmd.set == "off":
            out.append(IoEvent(time, cmd.signal, False))
    return out


@dataclass
class IoTimeline:
    initial: dict[str, bool] = field(default_factory=dict)
    events: list[IoEvent] = field(default_factory=list)

    @classmethod
    def from_task(cls, task: Task, event_lists: list[list[IoEvent]]) -> IoTimeline:
        merged = [ev for group in event_lists for ev in group]
        merged.sort(key=lambda e: (e.time, not e.on))
        return cls(initial={name: sig.initial for name, sig in task.io.items()}, events=merged)

    def state_at(self, time: float) -> dict[str, bool]:
        state = dict(self.initial)
        for ev in self.events:
            if ev.time > time + 1e-9:
                break
            state[ev.signal] = ev.on
        return state

    def names(self) -> list[str]:
        seen: set[str] = set(self.initial)
        for ev in self.events:
            seen.add(ev.signal)
        return sorted(seen)
