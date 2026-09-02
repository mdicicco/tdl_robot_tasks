"""Turn a task sequence into a time-parameterized Cartesian trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tdl import geometry as G
from tdl.context import SystemContext
from tdl.schema import Limits, Task
from tdl.sequence import Step, expand_sequence


@dataclass
class Knot:
    pose: np.ndarray
    kind: str
    label: str
    dwell: float = 0.0
    linear: bool = False
    gate_ref: str | None = None


@dataclass
class Segment:
    t0: float
    t1: float
    kind: str
    label: str


@dataclass
class Trajectory:
    t: np.ndarray
    poses: np.ndarray
    kinds: list[str]
    labels: list[str]
    segments: list[Segment]
    duration: float
    knots: list[Knot] = field(default_factory=list)

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

    for step in steps:
        if step.kind == "rest":
            knots.append(Knot(rest, "rest", f"{prefix}rest"))
            continue
        if step.kind == "move":
            path = task.free_space[step.ref]
            for i, wp in enumerate(path.waypoints):
                knots.append(
                    Knot(
                        wp.matrix(task.degrees),
                        "transit",
                        f"{prefix}move:{step.ref}[{i}]",
                    )
                )
            continue
        if step.kind == "keyhole":
            kh = ctx.keyholes[step.ref]
            knots.append(Knot(kh.matrix(task.degrees), "keyhole", f"{prefix}keyhole:{step.ref}"))
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
                )
            )
            continue
        if step.kind == "pause":
            if not knots:
                raise ValueError("pause cannot be the first sequence item")
            prev = knots[-1]
            knots.append(
                Knot(prev.pose.copy(), "pause", f"{prefix}pause {step.hold:.2f}s", dwell=step.hold)
            )
            continue
        if step.kind.startswith("fp_"):
            fp = ctx.force_pushes[step.ref]
            approach = fp.approach_pose(task.degrees)
            start = fp.start_pose(task.degrees)
            push_end = fp.push_end_pose(task.degrees)
            retract = fp.retract_pose(task.degrees)
            if step.kind == "fp_approach":
                knots.append(Knot(approach, "fp_approach", f"{prefix}approach:{step.ref}"))
            elif step.kind == "fp_start":
                knots.append(Knot(start, "fp_start", f"{prefix}start:{step.ref}", linear=fp.approach is not None))
            elif step.kind == "fp_push":
                knots.append(Knot(push_end, "fp_push", f"{prefix}push:{step.ref}", linear=True))
            elif step.kind == "fp_retract":
                knots.append(Knot(retract, "fp_retract", f"{prefix}retract:{step.ref}", linear=True))
            continue
        loc = ctx.locations[step.ref]
        pre, tgt, post = loc.cartesian_poses(task.degrees, visit=step.visit)
        slot = loc.pattern.slot_suffix(step.visit) if loc.pattern is not None else ""
        if step.kind in {"approach", "pre"}:
            if step.index >= len(pre):
                raise IndexError(f"{step.kind} index {step.index} out of range for {step.ref!r}")
            n = len(pre)
            label = f"approach:{step.ref}{slot}" if n == 1 else f"pre:{step.ref}{slot}[{step.index}]"
            knots.append(
                Knot(pre[step.index], step.kind, label, linear=step.index > 0)
            )
        elif step.kind == "target":
            knots.append(
                Knot(tgt, "target", f"target:{step.ref}{slot}", dwell=loc.dwell, linear=len(pre) > 0)
            )
        elif step.kind in {"retract", "post"}:
            if step.index >= len(post):
                raise IndexError(f"{step.kind} index {step.index} out of range for {step.ref!r}")
            n = len(post)
            label = f"retract:{step.ref}{slot}" if n == 1 else f"post:{step.ref}{slot}[{step.index}]"
            knots.append(Knot(post[step.index], step.kind, label, linear=True))
    return _dedupe_adjacent(knots)


def _dedupe_adjacent(knots: list[Knot]) -> list[Knot]:
    if not knots:
        return knots
    out = [knots[0]]
    for k in knots[1:]:
        lin, ang = G.geodesic_metrics(out[-1].pose, k.pose)
        if k.kind == "pause" or k.kind == "gate" or out[-1].kind in {"pause", "gate"}:
            out.append(k)
            continue
        if lin < 1e-9 and ang < 1e-9:
            out[-1].dwell = max(out[-1].dwell, k.dwell)
            out[-1].label = k.label
            out[-1].kind = k.kind
            out[-1].linear = k.linear
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

    t = 0.0
    times.append(t)
    poses.append(knots[0].pose.copy())
    kinds.append(knots[0].kind)
    labels.append(knots[0].label)

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
        segments.append(Segment(t0, t, b.kind, b.label))
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
            segments.append(
                Segment(t0 + duration * float(knot_frac[i - 1]), t0 + duration * float(knot_frac[i]), b.kind, b.label)
            )
        sample_hold(chain[-1].pose, chain[-1].kind, chain[-1].label, chain[-1].dwell)

    i = 0
    while i < len(knots) - 1:
        if knots[i + 1].kind == "pause":
            t0 = t
            sample_hold(knots[i].pose, "pause", knots[i + 1].label, knots[i + 1].dwell)
            if t > t0:
                segments.append(Segment(t0, t, "pause", knots[i + 1].label))
            i += 1
            continue
        if knots[i + 1].linear:
            sample_linear(knots[i], knots[i + 1])
            i += 1
            continue
        j = i + 1
        while j < len(knots) - 1 and not knots[j + 1].linear and knots[j].dwell <= 0:
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
    )


def build_trajectory(task: Task, limits: Limits | None = None, dt: float = 1 / 60) -> Trajectory:
    from tdl.multi import build_multi_trajectory

    return build_multi_trajectory(task, limits_override=limits, dt=dt).primary()
