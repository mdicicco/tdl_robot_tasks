"""Coupled multi-system trajectories with gate / sensor synchronization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tdl.context import SystemContext
from tdl.schema import Limits, Task
from tdl.signals import IoEvent, IoTimeline
from tdl.tower import TowerTimeline
from tdl.trajectory import Segment, Trajectory, knots_from_context, time_parameterize


@dataclass
class MultiTrajectory:
    systems: dict[str, Trajectory]
    duration: float
    segments: list[Segment] = field(default_factory=list)
    io_timeline: IoTimeline | None = None
    tower_timeline: TowerTimeline | None = None

    @property
    def is_multi(self) -> bool:
        return len(self.systems) > 1

    def at_time(self, time: float) -> dict[str, tuple[np.ndarray, str, str]]:
        time = float(np.clip(time, 0.0, self.duration))
        return {name: traj.at_time(time) for name, traj in self.systems.items()}

    def primary(self) -> Trajectory:
        return next(iter(self.systems.values()))


def _gate_arrival(traj: Trajectory) -> tuple[float | None, str | None, np.ndarray | None]:
    for knot in traj.knots:
        if knot.kind != "gate":
            continue
        gate_ref = knot.gate_ref or knot.label.split("gate:")[-1]
        target = knot.pose[:3, 3]
        arrive_t: float | None = None
        for t, kind, pose in zip(traj.t, traj.kinds, traj.poses):
            if kind != "gate":
                continue
            if float(np.linalg.norm(pose[:3, 3] - target)) < 1e-3:
                arrive_t = float(t)
        if arrive_t is not None:
            return arrive_t, gate_ref, knot.pose.copy()
    return None, None, None


def _handshake_ready(task: Task, held_gates: set[str]) -> bool:
    if not task.sensors:
        return True
    return all(sensor.active({}, held_gates) for sensor in task.sensors.values())


def _simulate_gated(
    task: Task,
    uncoupled: dict[str, Trajectory],
    dt: float,
) -> tuple[dict[str, Trajectory], list[IoEvent]]:
    names = list(uncoupled)
    gate_arrivals = {n: _gate_arrival(uncoupled[n]) for n in names}
    released = {n: gate_arrivals[n][0] is None for n in names}
    local_t = {n: 0.0 for n in names}
    done = {n: False for n in names}
    last_pose: dict[str, np.ndarray] = {}
    last_kind: dict[str, str] = {}
    last_label: dict[str, str] = {}

    pending_io: dict[str, list[IoEvent]] = {
        n: sorted(uncoupled[n].io_events, key=lambda e: e.time) for n in names
    }
    io_merged: list[IoEvent] = []
    times: dict[str, list[float]] = {n: [] for n in names}
    poses: dict[str, list[np.ndarray]] = {n: [] for n in names}
    kinds: dict[str, list[str]] = {n: [] for n in names}
    labels: dict[str, list[str]] = {n: [] for n in names}

    def _flush_io(n: str, global_t: float) -> None:
        queue = pending_io[n]
        while queue and local_t[n] >= queue[0].time - 1e-9:
            ev = queue.pop(0)
            io_merged.append(IoEvent(global_t, ev.signal, ev.on))

    global_t = 0.0
    max_t = max(traj.duration for traj in uncoupled.values()) + 120.0
    while global_t <= max_t:
        frame_poses: dict[str, np.ndarray] = {}
        frame_kinds: dict[str, str] = {}
        frame_labels: dict[str, str] = {}
        blocked: dict[str, bool] = {}

        for n in names:
            if done[n]:
                if n in last_pose:
                    frame_poses[n] = last_pose[n]
                    frame_kinds[n] = last_kind[n]
                    frame_labels[n] = last_label[n]
                continue
            arrive_t, gate_ref, gate_pose = gate_arrivals[n]
            is_blocked = arrive_t is not None and local_t[n] >= arrive_t - 1e-9 and not released[n]
            blocked[n] = is_blocked
            if is_blocked:
                pose = gate_pose if gate_pose is not None else uncoupled[n].at_time(arrive_t)[0]
                kind, label = "gate", f"{n}:gate:{gate_ref}"
            else:
                pose, kind, label = uncoupled[n].at_time(local_t[n])
                if kind == "gate":
                    kind, label = "transit", f"{n}:goto:{gate_ref}"
            frame_poses[n] = pose
            frame_kinds[n] = kind
            frame_labels[n] = label
            last_pose[n] = pose
            last_kind[n] = kind
            last_label[n] = label

        held_gates = {frame_labels[n].split("gate:")[-1] for n in blocked if blocked.get(n)}
        if _handshake_ready(task, held_gates):
            for n in blocked:
                if blocked[n]:
                    released[n] = True

        for n in names:
            if n not in frame_poses:
                continue
            times[n].append(global_t)
            poses[n].append(frame_poses[n].copy())
            kinds[n].append(frame_kinds[n])
            labels[n].append(frame_labels[n])

        for n in names:
            if done[n]:
                continue
            arrive_t, _, _ = gate_arrivals[n]
            if arrive_t is not None and local_t[n] >= arrive_t - 1e-9 and not released[n]:
                _flush_io(n, global_t)
                continue
            prev_local = local_t[n]
            if local_t[n] >= uncoupled[n].duration - 1e-9:
                _flush_io(n, global_t)
                done[n] = True
                continue
            next_t = local_t[n] + dt
            if arrive_t is not None and local_t[n] < arrive_t <= next_t:
                next_t = arrive_t
            local_t[n] = min(next_t, uncoupled[n].duration)
            if local_t[n] > prev_local + 1e-9:
                _flush_io(n, global_t)

        if all(done[n] for n in names):
            break
        global_t += dt

    out: dict[str, Trajectory] = {}
    for n in names:
        segs = _segments_from_samples(times[n], kinds[n], labels[n])
        out[n] = Trajectory(
            t=np.asarray(times[n], dtype=float),
            poses=np.stack(poses[n], axis=0),
            kinds=kinds[n],
            labels=labels[n],
            segments=segs,
            duration=float(times[n][-1]) if times[n] else 0.0,
        )
    return out, io_merged


def _segments_from_samples(times: list[float], kinds: list[str], labels: list[str]) -> list[Segment]:
    if not times:
        return []
    segments: list[Segment] = []
    t0 = times[0]
    cur_kind = kinds[0]
    cur_label = labels[0]
    for i in range(1, len(times)):
        if kinds[i] != cur_kind or labels[i] != cur_label:
            segments.append(Segment(t0, times[i], cur_kind, cur_label))
            t0 = times[i]
            cur_kind = kinds[i]
            cur_label = labels[i]
    segments.append(Segment(t0, times[-1], cur_kind, cur_label))
    return segments


def build_multi_trajectory(
    task: Task,
    limits_override: Limits | None = None,
    dt: float = 1 / 60,
) -> MultiTrajectory:
    if not task.systems:
        limits = limits_override or task.limits
        traj = time_parameterize(knots_from_context(SystemContext.from_task(task), task), limits, task.degrees, dt=dt)
        io_timeline = IoTimeline.from_task(task, [traj.io_events]) if task.io else None
        tower_timeline = TowerTimeline.from_task(task, traj.segments)
        return MultiTrajectory(
            systems={"": traj},
            duration=traj.duration,
            segments=traj.segments,
            io_timeline=io_timeline,
            tower_timeline=tower_timeline,
        )

    uncoupled: dict[str, Trajectory] = {}
    for name in task.systems:
        ctx = SystemContext.from_task(task, name)
        limits = limits_override or ctx.limits(task)
        uncoupled[name] = time_parameterize(knots_from_context(ctx, task), limits, task.degrees, dt=dt)

    io_event_lists: list[list[IoEvent]] = []
    if task.gates and task.sensors:
        coupled, io_merged = _simulate_gated(task, uncoupled, dt=dt)
        io_event_lists = [io_merged]
    else:
        coupled = uncoupled
        io_event_lists = [traj.io_events for traj in uncoupled.values()]

    duration = max(traj.duration for traj in coupled.values())
    merged: list[Segment] = []
    for name, traj in coupled.items():
        prefix = f"{name}: " if len(coupled) > 1 else ""
        for seg in traj.segments:
            merged.append(Segment(seg.t0, seg.t1, seg.kind, prefix + seg.label))
    merged.sort(key=lambda s: s.t0)
    io_timeline = IoTimeline.from_task(task, io_event_lists) if task.io else None
    tower_source = next(iter(uncoupled.values()))
    tower_timeline = TowerTimeline.from_task(task, tower_source.segments)
    return MultiTrajectory(
        systems=coupled,
        duration=duration,
        segments=merged,
        io_timeline=io_timeline,
        tower_timeline=tower_timeline,
    )
