"""Turn a task sequence into a time-parameterized Cartesian trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tdl import geometry as G
from tdl.context import SystemContext
from tdl.schema import GripperCommand, IoCommand, Limits, Task
from tdl.sequence import Step, expand_sequence
from tdl.signals import IoEvent, events_from_commands
from tdl.gripper import GripperEvent, events_from_commands as gripper_events_from_commands


@dataclass
class Knot:
    pose: np.ndarray
    kind: str
    label: str
    dwell: float = 0.0
    linear: bool = False
    gate_ref: str | None = None
    io_commands: list[IoCommand] = field(default_factory=list)
    gripper_commands: list[GripperCommand] = field(default_factory=list)
    tower_task: str | None = None


@dataclass
class Segment:
    t0: float
    t1: float
    kind: str
    label: str
    tower_task: str | None = None


@dataclass
class Trajectory:
    t: np.ndarray
    poses: np.ndarray
    kinds: list[str]
    labels: list[str]
    segments: list[Segment]
    duration: float
    knots: list[Knot] = field(default_factory=list)
    io_events: list[IoEvent] = field(default_factory=list)
    gripper_events: list[GripperEvent] = field(default_factory=list)

    def at_time(self, time: float) -> tuple[np.ndarray, str, str]:
        if len(self.t) == 0:
            return np.eye(4), "", ""
        time = float(np.clip(time, 0.0, self.duration))
        i = int(np.searchsorted(self.t, time, side="right") - 1)
        i = int(np.clip(i, 0, len(self.t) - 1))
        return self.poses[i], self.kinds[i], self.labels[i]


def knots_from_task(task: Task) -> list[Knot]:
    return knots_from_context(SystemContext.from_task(task), task)


def knots_from_context(ctx: SystemContext, task: Task) -> list[Knot]:
    steps = expand_sequence(task, ctx)
    rest = task.robot.rest.tool.matrix(task.degrees)
    prefix = f"{ctx.name}:" if ctx.name else ""
    knots: list[Knot] = []
    tt = lambda step: step.tower_task

    for step in steps:
        if step.kind == "rest":
            knots.append(Knot(rest, "rest", f"{prefix}rest", tower_task=tt(step)))
            continue
        if step.kind == "move":
            path = task.free_space[step.ref]
            for i, wp in enumerate(path.waypoints):
                knots.append(
                    Knot(
                        wp.matrix(task.degrees),
                        "transit",
                        f"{prefix}move:{step.ref}[{i}]",
                        tower_task=tt(step),
                    )
                )
            continue
        if step.kind == "keyhole":
            kh = ctx.keyholes[step.ref]
            knots.append(
                Knot(kh.matrix(task.degrees), "keyhole", f"{prefix}keyhole:{step.ref}", tower_task=tt(step))
            )
            continue
        if step.kind == "gate":
            gate = ctx.gates[step.ref]
            knots.append(
                Knot(
                    gate.matrix(task.degrees),
                    "gate",
                    f"{prefix}gate:{step.ref}",
                    linear=True,
                    gate_ref=step.ref,
                    tower_task=tt(step),
                )
            )
            continue
        if step.kind == "pause":
            if not knots:
                raise ValueError("pause cannot be the first sequence item")
            prev = knots[-1]
            label = f"{prefix}pause {step.hold:.2f}s"
            if step.ref:
                label = f"{prefix}{step.ref}:pause {step.hold:.2f}s"
            knots.append(Knot(prev.pose.copy(), "pause", label, dwell=step.hold, tower_task=tt(step)))
            continue
        if step.kind == "io":
            if not knots:
                raise ValueError("io cannot be the first sequence item")
            if step.io_command is None:
                raise ValueError("io step missing command")
            prev = knots[-1]
            cmd = step.io_command
            label = f"{prefix}{step.ref}:io:{cmd.signal}" if step.ref else f"{prefix}io:{cmd.signal}"
            knots.append(Knot(prev.pose.copy(), "io", label, io_commands=[cmd], tower_task=tt(step)))
            continue
        if step.kind == "gripper":
            if not knots:
                raise ValueError("gripper cannot be the first sequence item")
            if step.gripper_command is None:
                raise ValueError("gripper step missing command")
            prev = knots[-1]
            cmd = step.gripper_command
            gname = cmd.name or "gripper"
            label = f"{prefix}{step.ref}:gripper:{gname}" if step.ref else f"{prefix}gripper:{gname}"
            knots.append(
                Knot(prev.pose.copy(), "gripper", label, gripper_commands=[cmd], tower_task=tt(step))
            )
            continue
        if step.kind.startswith("fp_"):
            fp = ctx.force_pushes[step.ref]
            approach = fp.approach_pose(task.degrees)
            start = fp.start_pose(task.degrees)
            push_end = fp.push_end_pose(task.degrees)
            retract = fp.retract_pose(task.degrees)
            task_tag = tt(step)
            if step.kind == "fp_approach":
                knots.append(Knot(approach, "fp_approach", f"{prefix}approach:{step.ref}", tower_task=task_tag))
            elif step.kind == "fp_start":
                knots.append(
                    Knot(start, "fp_start", f"{prefix}start:{step.ref}", linear=fp.approach is not None, tower_task=task_tag)
                )
            elif step.kind == "fp_push":
                knots.append(Knot(push_end, "fp_push", f"{prefix}push:{step.ref}", linear=True, tower_task=task_tag))
            elif step.kind == "fp_retract":
                knots.append(Knot(retract, "fp_retract", f"{prefix}retract:{step.ref}", linear=True, tower_task=task_tag))
            continue
        if step.kind.startswith("gp_"):
            gp = ctx.grind_paths[step.ref]
            poses = gp.path_poses(task.degrees)
            task_tag = tt(step)
            if step.kind == "gp_approach":
                knots.append(Knot(gp.approach_pose(task.degrees), "gp_approach", f"{prefix}approach:{step.ref}", tower_task=task_tag))
            elif step.kind == "gp_start":
                knots.append(
                    Knot(
                        poses[0],
                        "gp_start",
                        f"{prefix}start:{step.ref}",
                        linear=gp.approach is not None,
                        tower_task=task_tag,
                    )
                )
            elif step.kind == "gp_path":
                if step.index >= len(poses):
                    raise IndexError(f"path index {step.index} out of range for {step.ref!r}")
                knots.append(
                    Knot(poses[step.index], "gp_path", f"{prefix}path:{step.ref}[{step.index}]", linear=True, tower_task=task_tag)
                )
            elif step.kind == "gp_retract":
                knots.append(Knot(gp.retract_pose(task.degrees), "gp_retract", f"{prefix}retract:{step.ref}", linear=True, tower_task=task_tag))
            continue
        loc = ctx.locations[step.ref]
        pre, tgt, post = loc.cartesian_poses(task.degrees, visit=step.visit)
        slot = loc.pattern.slot_suffix(step.visit) if loc.pattern is not None else ""
        n_pre = len(pre)
        n_post = len(post)
        task_tag = tt(step)
        if step.kind in {"approach", "pre"}:
            if step.index >= len(pre):
                raise IndexError(f"{step.kind} index {step.index} out of range for {step.ref!r}")
            n = len(pre)
            label = f"approach:{step.ref}{slot}" if n == 1 else f"pre:{step.ref}{slot}[{step.index}]"
            phase = "approach" if step.index == 0 else "pre"
            io_cmds = loc.io_at(phase, step.index, n_pre, n_post)
            knots.append(
                Knot(pre[step.index], step.kind, label, linear=step.index > 0, io_commands=io_cmds, tower_task=task_tag)
            )
        elif step.kind == "target":
            io_cmds = loc.io_at("target", 0, n_pre, n_post)
            knots.append(
                Knot(
                    tgt,
                    "target",
                    f"target:{step.ref}{slot}",
                    dwell=loc.dwell,
                    linear=len(pre) > 0,
                    io_commands=io_cmds,
                    tower_task=task_tag,
                )
            )
        elif step.kind in {"retract", "post"}:
            if step.index >= len(post):
                raise IndexError(f"{step.kind} index {step.index} out of range for {step.ref!r}")
            n = len(post)
            label = f"retract:{step.ref}{slot}" if n == 1 else f"post:{step.ref}{slot}[{step.index}]"
            phase = "retract" if step.index == n_post - 1 else "post"
            io_cmds = loc.io_at(phase, step.index, n_pre, n_post)
            knots.append(Knot(post[step.index], step.kind, label, linear=True, io_commands=io_cmds, tower_task=task_tag))
    return _dedupe_adjacent(knots)


def _segment_tower_task(a: Knot, b: Knot) -> str | None:
    if b.kind in {"transit", "keyhole", "rest", "gate"}:
        return None
    if a.tower_task != b.tower_task:
        return None
    return b.tower_task


def _dedupe_adjacent(knots: list[Knot]) -> list[Knot]:
    if not knots:
        return knots
    out = [knots[0]]
    for k in knots[1:]:
        lin, ang = G.geodesic_metrics(out[-1].pose, k.pose)
        if k.kind == "pause" or k.kind == "gate" or k.kind == "io" or k.kind == "gripper" or out[-1].kind in {"pause", "gate", "io", "gripper"}:
            out.append(k)
            continue
        if lin < 1e-9 and ang < 1e-9:
            out[-1].dwell = max(out[-1].dwell, k.dwell)
            out[-1].label = k.label
            out[-1].kind = k.kind
            out[-1].linear = k.linear
            out[-1].io_commands = list(out[-1].io_commands) + list(k.io_commands)
            out[-1].gripper_commands = list(out[-1].gripper_commands) + list(k.gripper_commands)
            out[-1].tower_task = k.tower_task
        else:
            out.append(k)
    return out


def time_parameterize(
    knots: list[Knot],
    limits: Limits,
    degrees: bool,
    dt: float = 1 / 60,
) -> Trajectory:
    if not knots:
        return Trajectory(
            t=np.array([0.0]),
            poses=np.eye(4)[None, ...],
            kinds=["idle"],
            labels=["idle"],
            segments=[],
            duration=0.0,
            knots=knots,
        )

    v_lin = max(limits.linear, 1e-6)
    v_ang = max(limits.angular_rad(degrees), 1e-6)
    v_app = max(limits.approach_linear, 1e-6)

    times: list[float] = []
    poses: list[np.ndarray] = []
    kinds: list[str] = []
    labels: list[str] = []
    segments: list[Segment] = []
    io_events: list[IoEvent] = []
    gripper_events: list[GripperEvent] = []

    t = 0.0
    times.append(t)
    poses.append(knots[0].pose.copy())
    kinds.append(knots[0].kind)
    labels.append(knots[0].label)

    def emit_io(commands: list[IoCommand], at: float) -> None:
        io_events.extend(events_from_commands(commands, at))

    def emit_gripper(commands: list[GripperCommand], at: float) -> None:
        gripper_events.extend(gripper_events_from_commands(commands, at))

    def sample_hold(pose: np.ndarray, kind: str, label: str, hold: float) -> None:
        nonlocal t
        if hold <= 0:
            return
        t_end = t + hold
        n = max(1, int(np.ceil(hold / dt)))
        for i in range(1, n + 1):
            ti = min(t + i * dt, t_end)
            times.append(ti)
            poses.append(pose.copy())
            kinds.append(kind)
            labels.append(label)
        t = t_end

    emit_io(knots[0].io_commands, t)
    emit_gripper(knots[0].gripper_commands, t)
    sample_hold(knots[0].pose, knots[0].kind, knots[0].label, knots[0].dwell)

    def sample_linear(a: Knot, b: Knot) -> None:
        nonlocal t
        lin, ang = G.geodesic_metrics(a.pose, b.pose)
        duration = max(lin / v_app, ang / v_ang, dt)
        t0 = t
        n = max(1, int(np.ceil(duration / dt)))
        for i in range(1, n + 1):
            s = i / n
            times.append(t0 + duration * s)
            poses.append(G.interpolate_pose(a.pose, b.pose, s))
            kinds.append(b.kind)
            labels.append(b.label)
        t = t0 + duration
        segments.append(Segment(t0, t, b.kind, b.label, tower_task=_segment_tower_task(a, b)))
        emit_io(b.io_commands, t)
        emit_gripper(b.gripper_commands, t)
        sample_hold(b.pose, b.kind, b.label, b.dwell)

    def sample_spline(chain: list[Knot]) -> None:
        nonlocal t
        spline = G.CubicPoseSpline([k.pose for k in chain])
        ang = sum(G.geodesic_metrics(a.pose, b.pose)[1] for a, b in zip(chain, chain[1:]))
        duration = max(spline.arc_length / v_lin, ang / v_ang, dt)
        t0 = t
        knot_frac = spline.knot_arc_fractions()
        n = max(1, int(np.ceil(duration / dt)))
        for i in range(1, n + 1):
            s = i / n
            times.append(t0 + duration * s)
            poses.append(spline.at(s))
            dest = chain[spline.interval_index(s) + 1]
            kinds.append(dest.kind)
            labels.append(dest.label)
        t = t0 + duration
        for i, b in enumerate(chain[1:], start=1):
            a = chain[i - 1]
            knot_t = t0 + duration * float(knot_frac[i - 1])
            emit_io(b.io_commands, knot_t)
            emit_gripper(b.gripper_commands, knot_t)
            segments.append(
                Segment(
                    knot_t,
                    t0 + duration * float(knot_frac[i]),
                    b.kind,
                    b.label,
                    tower_task=_segment_tower_task(a, b),
                )
            )
        sample_hold(chain[-1].pose, chain[-1].kind, chain[-1].label, chain[-1].dwell)

    i = 0
    while i < len(knots) - 1:
        if knots[i + 1].kind == "io":
            k = knots[i + 1]
            emit_io(k.io_commands, t)
            segments.append(Segment(t, t, "io", k.label, tower_task=k.tower_task))
            i += 1
            continue
        if knots[i + 1].kind == "gripper":
            k = knots[i + 1]
            emit_gripper(k.gripper_commands, t)
            segments.append(Segment(t, t, "gripper", k.label, tower_task=k.tower_task))
            i += 1
            continue
        if knots[i + 1].kind == "pause":
            t0 = t
            k = knots[i + 1]
            sample_hold(knots[i].pose, "pause", k.label, k.dwell)
            if t > t0:
                segments.append(Segment(t0, t, "pause", k.label, tower_task=k.tower_task))
            i += 1
            continue
        if knots[i + 1].linear:
            sample_linear(knots[i], knots[i + 1])
            i += 1
            continue
        j = i + 1
        while (
            j < len(knots) - 1
            and not knots[j + 1].linear
            and knots[j].dwell <= 0
            and knots[j + 1].kind not in {"pause", "io", "gripper"}
        ):
            j += 1
        sample_spline(knots[i : j + 1])
        i = j

    return Trajectory(
        t=np.asarray(times, dtype=float),
        poses=np.stack(poses, axis=0),
        kinds=kinds,
        labels=labels,
        segments=segments,
        duration=float(times[-1]),
        knots=knots,
        io_events=io_events,
        gripper_events=gripper_events,
    )


def build_trajectory(task: Task, limits: Limits | None = None, dt: float = 1 / 60) -> Trajectory:
    from tdl.multi import build_multi_trajectory

    return build_multi_trajectory(task, limits_override=limits, dt=dt).primary()
