"""Flatten nested sequence / repeat blocks into primitive steps."""

from __future__ import annotations

from typing import Any

from tdl.context import SystemContext
from tdl.schema import ForcePush, IoCommand, Location, RepeatBlock, Step, Task


def _tag_tower(steps: list[Step], task_name: str | None) -> list[Step]:
    return [step.model_copy(update={"tower_task": task_name}) for step in steps]

_KINDS = ("move", "approach", "target", "retract", "keyhole", "location", "pre", "post", "gate")


def parse_item(raw: Any) -> Step | RepeatBlock | str:
    if raw == "rest" or raw is True:
        return Step(kind="rest")
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"Sequence item must be a string or mapping, got {type(raw)}")
    if "repeat" in raw:
        block = raw["repeat"]
        if not isinstance(block, dict):
            raise ValueError("repeat must be a mapping with times and steps")
        return RepeatBlock.model_validate(block)
    if raw.get("type") == "rest" or "rest" in raw and len(raw) == 1:
        return Step(kind="rest")
    if "pause" in raw:
        return _parse_pause(raw["pause"])
    if "gate" in raw:
        ref = raw["gate"]
        if not isinstance(ref, str):
            raise ValueError("gate value must be a name string")
        return Step(kind="gate", ref=ref)
    for kind in _KINDS:
        if kind in raw:
            ref = raw[kind]
            if not isinstance(ref, str):
                raise ValueError(f"{kind} value must be a name string")
            if kind == "location":
                return ref
            return Step(kind=kind, ref=ref)
    raise ValueError(f"Unrecognized sequence item: {raw}")


def _parse_pause(val: Any) -> Step:
    if isinstance(val, (int, float)):
        hold = float(val)
    elif isinstance(val, dict):
        raw_hold = val.get("seconds", val.get("duration", val.get("hold")))
        if raw_hold is None:
            raise ValueError("pause mapping must include seconds")
        hold = float(raw_hold)
    else:
        raise ValueError("pause must be a number of seconds")
    if hold < 0:
        raise ValueError("pause duration must be >= 0")
    return Step(kind="pause", hold=hold)


def _parse_io(val: Any) -> IoCommand:
    if not isinstance(val, dict):
        raise ValueError("io step must be a mapping with signal and set or pulse")
    return IoCommand.model_validate(val)


def parse_location_item(raw: Any, loc_name: str, visit: int, loc: Location) -> Step:
    if isinstance(raw, str):
        if raw == "approach":
            return Step(kind="approach", ref=loc_name, index=0, visit=visit)
        if raw == "target":
            return Step(kind="target", ref=loc_name, visit=visit)
        if raw == "retract":
            n_post = len(loc.post_specs())
            if n_post == 0:
                raise ValueError(f"location {loc_name!r} has no retract/post strokes")
            return Step(kind="retract", ref=loc_name, index=n_post - 1, visit=visit)
        raise ValueError(f"Unrecognized location sequence item {raw!r} for {loc_name!r}")
    if not isinstance(raw, dict):
        raise ValueError(f"Location sequence item must be a string or mapping, got {type(raw)}")
    if "pause" in raw:
        step = _parse_pause(raw["pause"])
        return Step(kind="pause", hold=step.hold, ref=loc_name, visit=visit)
    if "io" in raw:
        return Step(kind="io", ref=loc_name, visit=visit, io_command=_parse_io(raw["io"]))
    if "approach" in raw:
        return Step(kind="approach", ref=loc_name, index=0, visit=visit)
    if "target" in raw:
        return Step(kind="target", ref=loc_name, visit=visit)
    if "retract" in raw:
        n_post = len(loc.post_specs())
        if n_post == 0:
            raise ValueError(f"location {loc_name!r} has no retract/post strokes")
        return Step(kind="retract", ref=loc_name, index=n_post - 1, visit=visit)
    if "pre" in raw:
        index = int(raw["pre"])
        return Step(kind="pre", ref=loc_name, index=index, visit=visit)
    if "post" in raw:
        index = int(raw["post"])
        return Step(kind="post", ref=loc_name, index=index, visit=visit)
    raise ValueError(f"Unrecognized location sequence item {raw!r} for {loc_name!r}")


def parse_force_push_item(raw: Any, name: str, fp: ForcePush) -> Step:
    if isinstance(raw, str):
        if raw == "approach":
            if fp.approach is None:
                raise ValueError(f"force push {name!r} has no approach stroke")
            return Step(kind="fp_approach", ref=name)
        if raw == "start":
            return Step(kind="fp_start", ref=name)
        if raw == "push":
            return Step(kind="fp_push", ref=name)
        if raw == "retract":
            return Step(kind="fp_retract", ref=name)
        raise ValueError(f"Unrecognized force push sequence item {raw!r} for {name!r}")
    if not isinstance(raw, dict):
        raise ValueError(f"Force push sequence item must be a string or mapping, got {type(raw)}")
    if "pause" in raw:
        step = _parse_pause(raw["pause"])
        return Step(kind="pause", hold=step.hold, ref=name)
    if "io" in raw:
        return Step(kind="io", ref=name, io_command=_parse_io(raw["io"]))
    if "approach" in raw:
        if fp.approach is None:
            raise ValueError(f"force push {name!r} has no approach stroke")
        return Step(kind="fp_approach", ref=name)
    if "start" in raw:
        return Step(kind="fp_start", ref=name)
    if "push" in raw:
        return Step(kind="fp_push", ref=name)
    if "retract" in raw:
        return Step(kind="fp_retract", ref=name)
    raise ValueError(f"Unrecognized force push sequence item {raw!r} for {name!r}")


def _expand_force_push(name: str, fp: ForcePush) -> list[Step]:
    if fp.sequence is not None:
        return [parse_force_push_item(raw, name, fp) for raw in fp.sequence]

    steps: list[Step] = []
    if fp.approach is not None:
        steps.append(Step(kind="fp_approach", ref=name))
    steps.append(Step(kind="fp_start", ref=name))
    steps.append(Step(kind="fp_push", ref=name))
    steps.append(Step(kind="fp_retract", ref=name))
    return steps


def expand_sequence(task: Task, ctx: SystemContext | None = None) -> list[Step]:
    ctx = ctx or SystemContext.from_task(task)
    visits: dict[str, int] = {}
    return _expand_list(ctx.sequence, task, ctx, visits)


def _expand_list(items: list[Any], task: Task, ctx: SystemContext, visits: dict[str, int]) -> list[Step]:
    out: list[Step] = []
    for raw in items:
        item = parse_item(raw)
        if isinstance(item, RepeatBlock):
            for _ in range(item.times):
                out.extend(_expand_list(item.steps, task, ctx, visits))
        elif isinstance(item, str):
            out.extend(_resolve_name(item, task, ctx, visits))
        else:
            _validate_step(item, task, ctx)
            out.append(item)
    return out


def _expand_location(name: str, loc: Location, visit: int) -> list[Step]:
    if loc.sequence is not None:
        steps: list[Step] = []
        n_pre = len(loc.pre_specs())
        n_post = len(loc.post_specs())
        for raw in loc.sequence:
            step = parse_location_item(raw, name, visit, loc)
            if step.kind in {"approach", "pre"} and step.index >= n_pre:
                raise IndexError(f"pre index {step.index} out of range for {name!r}")
            if step.kind in {"retract", "post"} and step.index >= n_post:
                raise IndexError(f"post index {step.index} out of range for {name!r}")
            steps.append(step)
        return steps

    steps: list[Step] = []
    n_pre = len(loc.pre_specs())
    for i in range(n_pre):
        kind = "approach" if i == 0 else "pre"
        steps.append(Step(kind=kind, ref=name, index=i, visit=visit))
    steps.append(Step(kind="target", ref=name, visit=visit))
    n_post = len(loc.post_specs())
    for i in range(n_post):
        kind = "retract" if i == n_post - 1 else "post"
        steps.append(Step(kind=kind, ref=name, index=i, visit=visit))
    return steps


def _resolve_name(name: str, task: Task, ctx: SystemContext, visits: dict[str, int]) -> list[Step]:
    if name == "rest":
        return [Step(kind="rest")]
    if name in ctx.locations:
        loc = ctx.locations[name]
        visit = visits.get(name, 0)
        visits[name] = visit + 1
        return _tag_tower(_expand_location(name, loc, visit), name)
    if name in ctx.force_pushes:
        return _tag_tower(_expand_force_push(name, ctx.force_pushes[name]), name)
    if name in ctx.keyholes:
        return [Step(kind="keyhole", ref=name)]
    raise KeyError(
        f"Unknown sequence name {name!r} (not a location, force push, or keyhole)"
    )


def _validate_step(step: Step, task: Task, ctx: SystemContext) -> None:
    if step.kind in {"rest", "pause"}:
        return
    if step.kind == "io":
        if step.io_command is None:
            raise ValueError("io step missing command")
        if step.io_command.signal not in task.io:
            raise KeyError(f"Unknown io signal {step.io_command.signal!r}")
        return
    if step.kind == "gate":
        if step.ref not in ctx.gates:
            raise KeyError(f"Unknown gate {step.ref!r}")
        gate = ctx.gates[step.ref]
        if task.sensors and gate.until not in task.sensors:
            raise KeyError(f"Unknown sensor {gate.until!r} for gate {step.ref!r}")
        return
    if step.kind.startswith("fp_"):
        if step.ref not in ctx.force_pushes:
            raise KeyError(f"Unknown force push {step.ref!r}")
        return
    if step.kind == "move":
        if step.ref not in task.free_space:
            raise KeyError(f"Unknown free-space path {step.ref!r}")
        return
    if step.kind == "keyhole":
        if step.ref not in ctx.keyholes:
            raise KeyError(f"Unknown keyhole {step.ref!r}")
        return
    if step.ref not in ctx.locations:
        raise KeyError(f"Unknown location {step.ref!r}")
