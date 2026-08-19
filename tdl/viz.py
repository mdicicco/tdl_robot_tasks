"""PyVista helpers for frames, search volumes, and paths."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from tdl.schema import Keyhole, Location, SearchArea, Task
from tdl.trajectory import Trajectory

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
    "idle": "#BDBDBD",
}

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


def add_world(plotter: pv.Plotter) -> None:
    plotter.set_background("#1B1E24")
    plane = pv.Plane(center=(0.25, 0.0, 0.0), direction=(0, 0, 1), i_size=1.2, j_size=1.0)
    plotter.add_mesh(plane, color="#2C313C", opacity=0.9, name="ground", reset_camera=False)
    grid = pv.Plane(center=(0.25, 0.0, 0.0), direction=(0, 0, 1), i_size=1.2, j_size=1.0, i_resolution=12, j_resolution=10)
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


def add_path(plotter: pv.Plotter, traj: Trajectory) -> None:
    for i, seg in enumerate(traj.segments):
        mask = (traj.t >= seg.t0 - 1e-9) & (traj.t <= seg.t1 + 1e-9)
        pts = traj.poses[mask, :3, 3]
        if len(pts) < 2 or seg.kind == "pause":
            continue
        line = pv.lines_from_points(pts)
        plotter.add_mesh(
            line,
            color=KIND_COLORS.get(seg.kind, "white"),
            line_width=4,
            name=f"path-{i}",
            reset_camera=False,
        )


def add_static_scene(plotter: pv.Plotter, task: Task, traj: Trajectory) -> None:
    add_world(plotter)
    add_rest(plotter, task)
    for name, loc in task.locations.items():
        add_location(plotter, name, loc, task.degrees)
    for name, kh in task.keyholes.items():
        add_keyhole(plotter, name, kh, task.degrees)
    add_path(plotter, traj)
    plotter.add_legend(
        [
            ("transit", KIND_COLORS["transit"]),
            ("keyhole", KIND_COLORS["keyhole"]),
            ("approach", KIND_COLORS["approach"]),
            ("target", KIND_COLORS["target"]),
            ("retract", KIND_COLORS["retract"]),
            ("pause", KIND_COLORS["pause"]),
            ("rest", KIND_COLORS["rest"]),
        ],
        bcolor="#1B1E24",
        face="rectangle",
        loc="upper right",
    )


class TcpActor:
    """Tool triad that follows the trajectory via VTK user_matrix."""

    def __init__(self, plotter: pv.Plotter, scale: float = 0.09):
        self.plotter = plotter
        self.actors = []
        for i, (mesh, color) in enumerate(zip(frame_axes_mesh(scale, radius=0.004), AXIS_COLORS)):
            actor = plotter.add_mesh(
                mesh,
                color=color,
                name=f"tcp-{i}",
                reset_camera=False,
                smooth_shading=True,
            )
            self.actors.append(actor)
        ball = pv.Sphere(radius=0.008)
        self.origin = plotter.add_mesh(ball, color="#FFECB3", name="tcp-origin", reset_camera=False)

    def set_pose(self, T: np.ndarray) -> None:
        M = np.asarray(T, dtype=float)
        for actor in self.actors:
            actor.user_matrix = M
        self.origin.user_matrix = M
