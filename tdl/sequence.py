"""Flatten nested sequence / repeat blocks into primitive steps."""

from __future__ import annotations

from typing import Any

from tdl.schema import RepeatBlock, Step, Task

_KINDS = ("move", "approach", "target", "retract", "keyhole", "location", "pre", "post")


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
    for kind in _KINDS:
        if kind in raw:
            ref = raw[kind]
            if not isinstance(ref, str):
                raise ValueError(f"{kind} value must be a name string")
            if kind == "location":
                return ref
            return Step(kind=kind, ref=ref)
    raise ValueError(f"Unrecognized sequence item: {raw}")


def expand_sequence(task: Task) -> list[Step]:
    visits: dict[str, int] = {}
    return _expand_list(task.sequence, task, visits)


def _expand_list(items: list[Any], task: Task, visits: dict[str, int]) -> list[Step]:
    out: list[Step] = []
    for raw in items:
        item = parse_item(raw)
        if isinstance(item, RepeatBlock):
            for _ in range(item.times):
                out.extend(_expand_list(item.steps, task, visits))
        elif isinstance(item, str):
            out.extend(_resolve_name(item, task, visits))
        else:
            _validate_step(item, task)
            out.append(item)
    return out


def _resolve_name(name: str, task: Task, visits: dict[str, int]) -> list[Step]:
    if name == "rest":
        return [Step(kind="rest")]
    if name in task.locations:
        loc = task.locations[name]
        visit = visits.get(name, 0)
        visits[name] = visit + 1
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
    if name in task.keyholes:
        return [Step(kind="keyhole", ref=name)]
    raise KeyError(f"Unknown sequence name {name!r} (not a location or keyhole)")


def _validate_step(step: Step, task: Task) -> None:
    if step.kind == "rest":
        return
    if step.kind == "move":
        if step.ref not in task.free_space:
            raise KeyError(f"Unknown free-space path {step.ref!r}")
        return
    if step.kind == "keyhole":
        if step.ref not in task.keyholes:
            raise KeyError(f"Unknown keyhole {step.ref!r}")
        return
    if step.ref not in task.locations:
        raise KeyError(f"Unknown location {step.ref!r}")
