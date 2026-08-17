"""SE(3) helpers built on numpy and scipy.spatial.transform."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

Vec3 = tuple[float, float, float]


def _as3(v: np.ndarray | Vec3) -> np.ndarray:
    a = np.asarray(v, dtype=float).reshape(3)
    return a


def rpy_to_matrix(rpy: Vec3, degrees: bool) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    return Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_matrix()


def quat_xyzw_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    return Rotation.from_quat(q).as_matrix()


def make_pose(xyz: Vec3, rot: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = _as3(xyz)
    return T


def pose_xyz_rpy(xyz: Vec3, rpy: Vec3, degrees: bool) -> np.ndarray:
    return make_pose(xyz, rpy_to_matrix(rpy, degrees=degrees))


def invert(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def compose(*Ts: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def translate(offset: Vec3) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = _as3(offset)
    return T


def spherical_direction(azimuth: float, elevation: float, degrees: bool) -> np.ndarray:
    """Unit vector in a frame: azimuth about Z from +X, elevation from the XY plane."""
    az = np.deg2rad(azimuth) if degrees else azimuth
    el = np.deg2rad(elevation) if degrees else elevation
    ce = np.cos(el)
    return np.array([ce * np.cos(az), ce * np.sin(az), np.sin(el)], dtype=float)


def normalize(v: np.ndarray | Vec3) -> np.ndarray:
    a = _as3(v)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return a / n


def approach_pose(target: np.ndarray, direction_target: np.ndarray, distance: float) -> np.ndarray:
    """Same orientation as target; origin offset along `direction` (target frame)."""
    d = normalize(direction_target) * float(distance)
    return target @ translate(d)


def rotation_angle(R0: np.ndarray, R1: np.ndarray) -> float:
    rel = Rotation.from_matrix(R0.T @ R1)
    return float(rel.magnitude())


def interpolate_pose(T0: np.ndarray, T1: np.ndarray, s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    p = (1.0 - s) * T0[:3, 3] + s * T1[:3, 3]
    key = Rotation.from_matrix([T0[:3, :3], T1[:3, :3]])
    slerp = Slerp([0.0, 1.0], key)
    R = slerp([s]).as_matrix()[0]
    return make_pose(p, R)


class CubicPoseSpline:
    """Natural cubic spline on translation; piecewise SLERP on rotation.

    Query parameter `s` is arc-length fraction in ``[0, 1]``.
    """

    def __init__(self, poses: list[np.ndarray], dense: int = 256):
        mats = [np.asarray(T, dtype=float) for T in poses]
        if len(mats) < 2:
            raise ValueError("need at least two poses")
        pts = np.stack([T[:3, 3] for T in mats], axis=0)
        rots = Rotation.from_matrix([T[:3, :3] for T in mats])

        chords = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        chords = np.maximum(chords, 1e-9)
        u = np.concatenate([[0.0], np.cumsum(chords)])
        self._u = u / u[-1]
        self._pos = CubicSpline(self._u, pts, bc_type="natural")
        self._slerp = Slerp(self._u, rots)

        n = max(int(dense), 8 * len(mats))
        self._u_dense = np.linspace(0.0, 1.0, n)
        seglens = np.linalg.norm(np.diff(self._pos(self._u_dense), axis=0), axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(seglens)])
        self.arc_length = float(self._cum[-1])
        self._knot_arcs = np.interp(self._u, self._u_dense, self._cum)

    def knot_arc_fractions(self) -> np.ndarray:
        if self.arc_length < 1e-12:
            return self._u.copy()
        return self._knot_arcs / self.arc_length

    def _u_at(self, s: np.ndarray | float) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        if self.arc_length < 1e-12:
            return np.zeros_like(s, dtype=float)
        return np.interp(s * self.arc_length, self._cum, self._u_dense)

    def at(self, s: float) -> np.ndarray:
        u = float(np.reshape(self._u_at(s), ()))
        return make_pose(self._pos(u), self._slerp([u]).as_matrix()[0])

    def interval_index(self, s: float) -> int:
        frac = self.knot_arc_fractions()
        i = int(np.searchsorted(frac, float(np.clip(s, 0.0, 1.0)), side="right") - 1)
        return int(np.clip(i, 0, len(frac) - 2))


def geodesic_metrics(T0: np.ndarray, T1: np.ndarray) -> tuple[float, float]:
    linear = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
    angular = rotation_angle(T0[:3, :3], T1[:3, :3])
    return linear, angular
