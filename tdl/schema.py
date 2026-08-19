"""Pydantic models for the Task Definition Language."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tdl import geometry as G


class Units(BaseModel):
    length: Literal["m", "meters"] = "m"
    angle: Literal["deg", "degree", "degrees", "rad", "radian", "radians"] = "deg"

    @property
    def degrees(self) -> bool:
        return self.angle in {"deg", "degree", "degrees"}


class Frame(BaseModel):
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] | None = None
    quat_xyzw: tuple[float, float, float, float] | None = None

    def matrix(self, degrees: bool) -> np.ndarray:
        if self.quat_xyzw is not None:
            rot = G.quat_xyzw_to_matrix(self.quat_xyzw)
        elif self.rpy is not None:
            rot = G.rpy_to_matrix(self.rpy, degrees=degrees)
        else:
            rot = np.eye(3)
        return G.make_pose(self.xyz, rot)


class RestPose(BaseModel):
    joints: list[float] = Field(default_factory=list)
    tool: Frame = Field(default_factory=Frame)


class Robot(BaseModel):
    rest: RestPose = Field(default_factory=RestPose)


class SearchArea(BaseModel):
    shape: Literal["box", "sphere", "cylinder"] = "box"
    extents: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    found_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _check_dims(self) -> SearchArea:
        if self.shape == "box" and self.extents is None:
            self.extents = (0.05, 0.05, 0.02)
        if self.shape == "sphere" and self.radius is None:
            self.radius = 0.04
        if self.shape == "cylinder":
            if self.radius is None:
                self.radius = 0.04
            if self.height is None:
                self.height = 0.04
        return self


class ApproachSpec(BaseModel):
    """A Cartesian offset in the found-target frame (orientation unchanged)."""

    distance: float = 0.1
    axis: tuple[float, float, float] | None = None
    azimuth: float | None = None
    elevation: float | None = None
    xyz: tuple[float, float, float] | None = None

    def offset(self, degrees: bool) -> np.ndarray:
        if self.xyz is not None:
            return np.asarray(self.xyz, dtype=float).reshape(3)
        return self.direction(degrees) * float(self.distance)

    def direction(self, degrees: bool) -> np.ndarray:
        if self.axis is not None:
            return G.normalize(self.axis)
        if self.azimuth is None and self.elevation is None:
            return np.array([0.0, 0.0, -1.0])
        az = 0.0 if self.azimuth is None else self.azimuth
        if self.elevation is None:
            el = -90.0 if degrees else -np.pi / 2
        else:
            el = self.elevation
        return G.spherical_direction(az, el, degrees=degrees)


class GridSpec(BaseModel):
    """Raster of translations. `counts: [nx, ny]` with X varying fastest."""

    counts: list[int] = Field(min_length=1, max_length=3)
    step: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @model_validator(mode="after")
    def _positive_counts(self) -> GridSpec:
        if any(c < 1 for c in self.counts):
            raise ValueError("grid counts must be >= 1")
        return self

    def _counts3(self) -> tuple[int, int, int]:
        c = list(self.counts) + [1, 1, 1]
        return int(c[0]), int(c[1]), int(c[2])

    def generate(self) -> list[np.ndarray]:
        nx, ny, nz = self._counts3()
        sx, sy, sz = self.step
        ox, oy, oz = self.origin
        out: list[np.ndarray] = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    out.append(np.array([ox + i * sx, oy + j * sy, oz + k * sz], dtype=float))
        return out

    def unravel(self, visit: int) -> tuple[int, ...]:
        nx, ny, _nz = self._counts3()
        n = nx * ny * _nz
        v = int(visit) % max(n, 1)
        k, rem = divmod(v, nx * ny)
        j, i = divmod(rem, nx)
        dim = len(self.counts)
        if dim == 1:
            return (i,)
        if dim == 2:
            return (i, j)
        return (i, j, k)


class LocationPattern(BaseModel):
    """Offsets applied to a location, advancing one slot each visit and wrapping."""

    grid: GridSpec | None = None
    offsets: list[tuple[float, float, float]] | None = None
    frame: Literal["target", "world"] = "target"

    @model_validator(mode="after")
    def _one_source(self) -> LocationPattern:
        if self.grid is not None and self.offsets is not None:
            raise ValueError("pattern: use grid or offsets, not both")
        if self.grid is None and self.offsets is None:
            raise ValueError("pattern needs grid or offsets")
        return self

    def all_offsets(self) -> list[np.ndarray]:
        if self.offsets is not None:
            return [np.asarray(o, dtype=float).reshape(3) for o in self.offsets]
        assert self.grid is not None
        return self.grid.generate()

    def n_slots(self) -> int:
        return len(self.all_offsets())

    def offset_at(self, visit: int) -> np.ndarray:
        offs = self.all_offsets()
        return offs[int(visit) % len(offs)]

    def slot_suffix(self, visit: int) -> str:
        n = self.n_slots()
        v = int(visit) % n
        if self.grid is not None:
            idx = self.grid.unravel(v)
            return "[" + ",".join(str(i) for i in idx) + "]"
        return f"[{v}]"


class Location(BaseModel):
    description: str = ""
    target: Frame
    search: SearchArea = Field(default_factory=SearchArea)
    # Single-stroke aliases. Prefer `pre` / `post` when there are several.
    approach: ApproachSpec | None = None
    retract: ApproachSpec | None = None
    pre: list[ApproachSpec] | None = None
    post: list[ApproachSpec] | None = None
    pattern: LocationPattern | None = None
    dwell: float = 0.25

    @model_validator(mode="after")
    def _pre_post_exclusive(self) -> Location:
        if self.pre is not None and self.approach is not None:
            raise ValueError("specify pre or approach, not both")
        if self.post is not None and self.retract is not None:
            raise ValueError("specify post or retract, not both")
        return self

    def pre_specs(self) -> list[ApproachSpec]:
        if self.pre is not None:
            return list(self.pre)
        return [self.approach if self.approach is not None else ApproachSpec()]

    def post_specs(self) -> list[ApproachSpec]:
        if self.post is not None:
            return list(self.post)
        return [self.retract if self.retract is not None else ApproachSpec()]

    def nominal_target(self, degrees: bool) -> np.ndarray:
        return self.target.matrix(degrees)

    def _apply_pattern(self, T: np.ndarray, visit: int) -> np.ndarray:
        if self.pattern is None:
            return T
        off = self.pattern.offset_at(visit)
        if self.pattern.frame == "world":
            return G.translate(off) @ T
        return T @ G.translate(off)

    def slot_nominal(self, degrees: bool, visit: int = 0) -> np.ndarray:
        return self._apply_pattern(self.nominal_target(degrees), visit)

    def found_target(self, degrees: bool, visit: int = 0) -> np.ndarray:
        T = self.nominal_target(degrees) @ G.translate(self.search.found_offset)
        return self._apply_pattern(T, visit)

    def cartesian_poses(
        self, degrees: bool, visit: int = 0
    ) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
        """Pre poses (outermost first), found target, post poses (leaving the target).

        Each spec is applied incrementally: pre composes outward from the target,
        post composes from the target through each stroke in order.
        """
        target = self.found_target(degrees, visit=visit)
        pre: list[np.ndarray] = []
        cur = target
        for spec in reversed(self.pre_specs()):
            cur = cur @ G.translate(spec.offset(degrees))
            pre.append(cur)
        pre.reverse()

        post: list[np.ndarray] = []
        cur = target
        for spec in self.post_specs():
            cur = cur @ G.translate(spec.offset(degrees))
            post.append(cur)
        return pre, target, post

    def approach_pose(self, degrees: bool, visit: int = 0) -> np.ndarray:
        pre, target, _ = self.cartesian_poses(degrees, visit=visit)
        return pre[0] if pre else target

    def retract_pose(self, degrees: bool, visit: int = 0) -> np.ndarray:
        _, target, post = self.cartesian_poses(degrees, visit=visit)
        return post[-1] if post else target


class Keyhole(Frame):
    """Via pose that free-space motion must pass through.

    Z of the frame is the aperture axis (the direction you go through the hole).
    """

    description: str = ""
    radius: float = Field(default=0.05, gt=0.0)


class FreeSpacePath(BaseModel):
    waypoints: list[Frame] = Field(default_factory=list)


class Limits(BaseModel):
    linear: float = 0.25
    angular: float = 1.5
    approach_linear: float = 0.08

    def angular_rad(self, degrees: bool) -> float:
        return float(np.deg2rad(self.angular) if degrees else self.angular)


class RepeatBlock(BaseModel):
    times: int = Field(ge=1)
    steps: list[Any]


class Step(BaseModel):
    kind: Literal["rest", "move", "approach", "target", "retract", "keyhole", "pre", "post", "pause"]
    ref: str | None = None
    index: int = 0
    visit: int = 0
    hold: float = 0.0


class Task(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: int = 1
    name: str = "task"
    units: Units = Field(default_factory=Units)
    robot: Robot = Field(default_factory=Robot)
    limits: Limits = Field(default_factory=Limits)
    locations: dict[str, Location] = Field(default_factory=dict)
    keyholes: dict[str, Keyhole] = Field(default_factory=dict)
    free_space: dict[str, FreeSpacePath] = Field(default_factory=dict)
    sequence: list[Any] = Field(default_factory=list)

    @property
    def degrees(self) -> bool:
        return self.units.degrees

    @model_validator(mode="after")
    def _unique_names(self) -> Task:
        overlap = set(self.locations) & set(self.keyholes)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"name used as both location and keyhole: {names}")
        return self
