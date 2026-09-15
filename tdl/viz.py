"""PyVista helpers for frames, search volumes, and paths."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from tdl.schema import ForcePush, Gate, GrindPath, Keyhole, Location, SearchArea, Sensor, Task
from tdl.trajectory import Trajectory

try:
    from tdl.multi import MultiTrajectory
except ImportError:
    MultiTrajectory = None  # type: ignore

SYSTEM_TCP_COLORS = {
    "line_a": "#4FC3F7",
    "line_b": "#FF7043",
    "": "#FFECB3",
}

KIND_COLORS = {
    "rest": "#9E9E9E",
    "transit": "#4FC3F7",
    "keyhole": "#FFCA28",
    "approach": "#FFA726",
    "pre": "#FFA726",
    "target": "#66BB6A",
    "post": "#AB47BC",
    "retract": "#AB47BC",
    "pause": "#80CBC4",
    "io": "#AED581",
    "gripper": "#FFCC80",
    "fp_approach": "#FFA726",
    "fp_start": "#66BB6A",
    "fp_push": "#EF5350",
    "fp_retract": "#AB47BC",
    "gp_approach": "#FFA726",
    "gp_start": "#66BB6A",
    "gp_path": "#BA68C8",
    "gp_retract": "#AB47BC",
    "gate": "#FFD54F",
    "idle": "#BDBDBD",
}

# Friendly legend labels; only kinds present in the trajectory are shown.
_LEGEND_LABELS = {
    "rest": "rest",
    "transit": "transit",
    "keyhole": "keyhole",
    "approach": "approach",
    "pre": "approach",
    "target": "target",
    "post": "retract",
    "retract": "retract",
    "pause": "pause",
    "io": "io",
    "gripper": "gripper",
    "fp_approach": "approach",
    "fp_start": "force start",
    "fp_push": "force push",
    "fp_retract": "retract",
    "gp_approach": "approach",
    "gp_start": "path start",
    "gp_path": "grind path",
    "gp_retract": "retract",
    "gate": "gate",
}

_LEGEND_ORDER = [
    "rest",
    "transit",
    "keyhole",
    "approach",
    "pre",
    "fp_approach",
    "gp_approach",
    "target",
    "fp_start",
    "fp_push",
    "gp_start",
    "gp_path",
    "gate",
    "io",
    "gripper",
    "pause",
    "post",
    "retract",
    "fp_retract",
    "gp_retract",
]

AXIS_COLORS = ("#E53935", "#43A047", "#1E88E5")


def frame_axes_mesh(scale: float = 0.07, radius: float | None = None) -> list[pv.PolyData]:
    radius = 0.0032 if radius is None else radius
    meshes = []
    for direction in np.eye(3):
        height = scale
        cyl = pv.Cylinder(
            center=(0.0, 0.0, 0.0),
            direction=tuple(direction),
            radius=radius * (scale / 0.07),
            height=height,
        )
        cyl.translate(direction * (height / 2.0), inplace=True)
        meshes.append(cyl)
    return meshes


def add_frame(plotter: pv.Plotter, T: np.ndarray, scale: float = 0.07, name: str = "frame", opacity: float = 1.0) -> None:
    for i, (mesh, color) in enumerate(zip(frame_axes_mesh(scale), AXIS_COLORS)):
        m = mesh.copy()
        m.transform(T, inplace=True)
        plotter.add_mesh(
            m,
            color=color,
            name=f"{name}-{i}",
            opacity=opacity,
            reset_camera=False,
            smooth_shading=True,
        )


def search_mesh(search: SearchArea) -> pv.PolyData:
    if search.shape == "box":
        sx, sy, sz = search.extents or (0.05, 0.05, 0.02)
        return pv.Box(bounds=(-sx / 2, sx / 2, -sy / 2, sy / 2, -sz / 2, sz / 2))
    if search.shape == "sphere":
        return pv.Sphere(radius=search.radius or 0.04, theta_resolution=24, phi_resolution=16)
    return pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, 1.0),
        radius=search.radius or 0.04,
        height=search.height or 0.04,
        resolution=32,
        capping=True,
    )


def add_location(plotter: pv.Plotter, name: str, loc: Location, degrees: bool) -> None:
    n_slots = loc.pattern.n_slots() if loc.pattern is not None else 1
    label_pts = []
    label_text = []
    for visit in range(n_slots):
        T_nom = loc.slot_nominal(degrees, visit)
        T_found = loc.found_target(degrees, visit)
        volume = search_mesh(loc.search)
        volume.transform(T_nom, inplace=True)
        plotter.add_mesh(
            volume,
            color="#FFEE58",
            opacity=0.18 if visit == 0 else 0.10,
            style="wireframe",
            line_width=2,
            name=f"search-{name}-{visit}",
            reset_camera=False,
        )
        add_frame(plotter, T_nom, scale=0.06, name=f"nominal-{name}-{visit}", opacity=0.45 if visit == 0 else 0.25)
        add_frame(plotter, T_found, scale=0.07 if visit == 0 else 0.05, name=f"found-{name}-{visit}")
        suffix = loc.pattern.slot_suffix(visit) if loc.pattern is not None else ""
        label_pts.append(T_found[:3, 3] + np.array([0.0, 0.0, 0.08]))
        label_text.append(f"{name}{suffix}")
        if visit == 0:
            pre, _, post = loc.cartesian_poses(degrees, visit=visit)
            for i, T in enumerate(pre):
                add_frame(plotter, T, scale=0.05, name=f"pre-{name}-{i}", opacity=0.8 if i == 0 else 0.55)
            for i, T in enumerate(post):
                add_frame(plotter, T, scale=0.05, name=f"post-{name}-{i}", opacity=0.8 if i == len(post) - 1 else 0.55)
    if n_slots > 1:
        pts = np.stack([loc.found_target(degrees, v)[:3, 3] for v in range(n_slots)])
        order = pv.lines_from_points(pts)
        plotter.add_mesh(
            order,
            color="#FFEE58",
            line_width=2,
            name=f"pattern-{name}",
            reset_camera=False,
        )
    plotter.add_point_labels(
        label_pts,
        label_text,
        font_size=14 if n_slots == 1 else 11,
        name=f"label-{name}",
        text_color="white",
        shape_opacity=0.35,
        reset_camera=False,
        always_visible=True,
    )


def add_keyhole(plotter: pv.Plotter, name: str, kh: Keyhole, degrees: bool) -> None:
    T = kh.matrix(degrees)
    r = float(kh.radius)
    theta = np.linspace(0.0, 2.0 * np.pi, 72)
    circle = np.column_stack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)])
    ring = pv.lines_from_points(circle, close=True).tube(radius=max(0.003, r * 0.07))
    ring.transform(T, inplace=True)
    plotter.add_mesh(
        ring,
        color=KIND_COLORS["keyhole"],
        name=f"keyhole-ring-{name}",
        reset_camera=False,
        smooth_shading=True,
    )
    disc = pv.Disc(inner=0.0, outer=r * 0.92, normal=(0.0, 0.0, 1.0), c_res=48)
    disc.transform(T, inplace=True)
    plotter.add_mesh(
        disc,
        color=KIND_COLORS["keyhole"],
        opacity=0.14,
        name=f"keyhole-disc-{name}",
        reset_camera=False,
    )
    add_frame(plotter, T, scale=0.05, name=f"keyhole-frame-{name}")
    label_pt = T[:3, 3] + T[:3, 2] * (r + 0.04)
    plotter.add_point_labels(
        [label_pt],
        [f"keyhole:{name}"],
        font_size=13,
        name=f"label-keyhole-{name}",
        text_color=KIND_COLORS["keyhole"],
        shape_opacity=0.35,
        reset_camera=False,
        always_visible=True,
    )


def add_force_push(plotter: pv.Plotter, name: str, fp: ForcePush, degrees: bool) -> None:
    start = fp.start_pose(degrees)
    push_end = fp.push_end_pose(degrees)
    retract = fp.retract_pose(degrees)
    add_frame(plotter, start, scale=0.06, name=f"fp-start-{name}")
    if fp.approach is not None:
        add_frame(plotter, fp.approach_pose(degrees), scale=0.05, name=f"fp-approach-{name}", opacity=0.75)
    add_frame(plotter, retract, scale=0.05, name=f"fp-retract-{name}", opacity=0.8)

    push_line = pv.lines_from_points(np.stack([start[:3, 3], push_end[:3, 3]]))
    plotter.add_mesh(
        push_line.tube(radius=0.004),
        color=KIND_COLORS["fp_push"],
        name=f"fp-push-range-{name}",
        reset_camera=False,
        smooth_shading=True,
    )
    arrow = pv.Arrow(
        start=push_end[:3, 3],
        direction=fp.push.direction(degrees),
        scale=0.05,
    )
    plotter.add_mesh(
        arrow,
        color=KIND_COLORS["fp_push"],
        name=f"fp-push-arrow-{name}",
        reset_camera=False,
    )
    label_pt = start[:3, 3] + np.array([0.0, 0.0, 0.09])
    plotter.add_point_labels(
        [label_pt],
        [f"{name}\n≤{fp.push.force_limit:.0f} N, {fp.push.max_travel:.2f} m max"],
        font_size=12,
        name=f"label-fp-{name}",
        text_color=KIND_COLORS["fp_push"],
        shape_opacity=0.35,
        reset_camera=False,
        always_visible=True,
    )


def add_grind_path(plotter: pv.Plotter, name: str, gp: GrindPath, degrees: bool) -> None:
    poses = gp.path_poses(degrees)
    pts = np.stack([T[:3, 3] for T in poses])
    add_frame(plotter, poses[0], scale=0.06, name=f"gp-start-{name}")
    if gp.approach is not None:
        add_frame(plotter, gp.approach_pose(degrees), scale=0.05, name=f"gp-approach-{name}", opacity=0.75)
    add_frame(plotter, gp.retract_pose(degrees), scale=0.05, name=f"gp-retract-{name}", opacity=0.8)
    if len(pts) >= 2:
        line = pv.lines_from_points(pts)
        plotter.add_mesh(
            line.tube(radius=0.0025),
            color=KIND_COLORS["gp_path"],
            name=f"gp-path-{name}",
            reset_camera=False,
            smooth_shading=True,
        )
    plotter.add_mesh(
        pv.PolyData(pts),
        color=KIND_COLORS["gp_path"],
        point_size=8,
        render_points_as_spheres=True,
        name=f"gp-pts-{name}",
        reset_camera=False,
    )
    label_pt = poses[0][:3, 3] + np.array([0.0, 0.0, 0.08])
    plotter.add_point_labels(
        [label_pt],
        [f"{name}\n{len(poses)} pts"],
        font_size=12,
        name=f"label-gp-{name}",
        text_color=KIND_COLORS["gp_path"],
        shape_opacity=0.35,
        reset_camera=False,
        always_visible=True,
    )


def add_sensor(plotter: pv.Plotter, name: str, sensor: Sensor) -> None:
    center = np.asarray(sensor.xyz, dtype=float)
    sphere = pv.Sphere(radius=float(sensor.radius), center=center, theta_resolution=20, phi_resolution=16)
    plotter.add_mesh(
        sphere,
        color="#4FC3F7",
        opacity=0.15,
        style="wireframe",
        line_width=2,
        name=f"sensor-{name}",
        reset_camera=False,
    )
    plotter.add_point_labels(
        [center + np.array([0.0, 0.0, sensor.radius + 0.03])],
        [f"sensor:{name}"],
        font_size=12,
        name=f"label-sensor-{name}",
        text_color="#4FC3F7",
        shape_opacity=0.35,
        reset_camera=False,
        always_visible=True,
    )


def add_gate(plotter: pv.Plotter, name: str, gate: Gate, degrees: bool) -> None:
    T = gate.matrix(degrees)
    add_frame(plotter, T, scale=0.055, name=f"gate-{name}", opacity=0.85)
    plotter.add_point_labels(
        [T[:3, 3] + np.array([0.0, 0.0, 0.07])],
        [f"gate:{name}\nuntil {gate.until}"],
        font_size=11,
        name=f"label-gate-{name}",
        text_color=KIND_COLORS["gate"],
        shape_opacity=0.35,
        reset_camera=False,
        always_visible=True,
    )


def add_world(plotter: pv.Plotter) -> None:
    plotter.set_background("#1B1E24")
    plane = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0, 0, 1), i_size=1.2, j_size=1.0)
    plotter.add_mesh(plane, color="#2C313C", opacity=0.9, name="ground", reset_camera=False)
    grid = pv.Plane(center=(0.0, 0.0, 0.0), direction=(0, 0, 1), i_size=1.2, j_size=1.0, i_resolution=12, j_resolution=10)
    plotter.add_mesh(grid, style="wireframe", color="#3E4654", line_width=1, name="grid", reset_camera=False)
    add_frame(plotter, np.eye(4), scale=0.12, name="world")
    plotter.add_point_labels(
        [[0.0, 0.0, 0.13]],
        ["world"],
        font_size=12,
        name="label-world",
        text_color="#B0BEC5",
        shape_opacity=0.3,
        reset_camera=False,
        always_visible=True,
    )


def add_rest(plotter: pv.Plotter, task: Task) -> None:
    T = task.robot.rest.tool.matrix(task.degrees)
    add_frame(plotter, T, scale=0.08, name="rest")
    p = T[:3, 3] + np.array([0.0, 0.0, 0.09])
    plotter.add_point_labels(
        [p],
        ["rest"],
        font_size=14,
        name="label-rest",
        text_color="#BDBDBD",
        shape_opacity=0.3,
        reset_camera=False,
        always_visible=True,
    )


def add_path(plotter: pv.Plotter, traj: Trajectory, path_id: str = "") -> None:
    tag = f"{path_id}-" if path_id else ""
    for i, seg in enumerate(traj.segments):
        mask = (traj.t >= seg.t0 - 1e-9) & (traj.t <= seg.t1 + 1e-9)
        pts = traj.poses[mask, :3, 3]
        if len(pts) < 2 or seg.kind in {"pause", "gate", "io", "gripper"}:
            continue
        line = pv.lines_from_points(pts)
        plotter.add_mesh(
            line,
            color=KIND_COLORS.get(seg.kind, "white"),
            line_width=4,
            name=f"path-{tag}{i}",
            reset_camera=False,
        )


def _legend_entries(traj: Trajectory | MultiTrajectory) -> list[tuple[str, str]]:
    """Build legend rows from segment kinds actually used in this trajectory."""
    kinds: set[str] = set()
    if MultiTrajectory is not None and isinstance(traj, MultiTrajectory):
        for sys_traj in traj.systems.values():
            kinds.update(seg.kind for seg in sys_traj.segments)
    else:
        kinds.update(seg.kind for seg in traj.segments)  # type: ignore[union-attr]

    entries: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for kind in _LEGEND_ORDER:
        if kind not in kinds:
            continue
        label = _LEGEND_LABELS.get(kind, kind)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        entries.append((label, KIND_COLORS.get(kind, "#FFFFFF")))
    return entries


def add_static_scene(plotter: pv.Plotter, task: Task, traj: Trajectory | MultiTrajectory) -> None:
    add_world(plotter)
    add_rest(plotter, task)
    if task.systems:
        for sys_name, spec in task.systems.items():
            for loc_name, loc in spec.locations.items():
                add_location(plotter, f"{sys_name}/{loc_name}", loc, task.degrees)
            for fp_name, fp in spec.force_pushes.items():
                add_force_push(plotter, f"{sys_name}/{fp_name}", fp, task.degrees)
            for gp_name, gp in spec.grind_paths.items():
                add_grind_path(plotter, f"{sys_name}/{gp_name}", gp, task.degrees)
            for kh_name, kh in spec.keyholes.items():
                add_keyhole(plotter, f"{sys_name}/{kh_name}", kh, task.degrees)
    else:
        for name, loc in task.locations.items():
            add_location(plotter, name, loc, task.degrees)
        for name, fp in task.force_pushes.items():
            add_force_push(plotter, name, fp, task.degrees)
        for name, gp in task.grind_paths.items():
            add_grind_path(plotter, name, gp, task.degrees)
        for name, kh in task.keyholes.items():
            add_keyhole(plotter, name, kh, task.degrees)
    for name, sensor in task.sensors.items():
        add_sensor(plotter, name, sensor)
    for name, gate in task.gates.items():
        add_gate(plotter, name, gate, task.degrees)
    if MultiTrajectory is not None and isinstance(traj, MultiTrajectory):
        for sys_name, sys_traj in traj.systems.items():
            add_path(plotter, sys_traj, path_id=sys_name or "main")
    else:
        add_path(plotter, traj)  # type: ignore[arg-type]
    legend = _legend_entries(traj)
    if legend:
        plotter.add_legend(
            legend,
            bcolor="#1B1E24",
            face="rectangle",
            loc="upper right",
        )


class TcpActor:
    """Tool triad that follows the trajectory via VTK user_matrix."""

    def __init__(self, plotter: pv.Plotter, scale: float = 0.09, name: str = "tcp", ball_color: str = "#FFECB3"):
        self.plotter = plotter
        self.actors = []
        for i, (mesh, color) in enumerate(zip(frame_axes_mesh(scale, radius=0.004), AXIS_COLORS)):
            actor = plotter.add_mesh(
                mesh,
                color=color,
                name=f"{name}-{i}",
                reset_camera=False,
                smooth_shading=True,
            )
            self.actors.append(actor)
        ball = pv.Sphere(radius=0.008)
        self.origin = plotter.add_mesh(ball, color=ball_color, name=f"{name}-origin", reset_camera=False)

    def set_pose(self, T: np.ndarray) -> None:
        M = np.asarray(T, dtype=float)
        for actor in self.actors:
            actor.user_matrix = M
        self.origin.user_matrix = M
