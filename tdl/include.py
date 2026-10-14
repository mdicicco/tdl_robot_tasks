"""Resolve nested task files (`include`) before schema validation.

An include step splices a child task into the parent sequence. Child world poses
are left-multiplied by an SE(3) `offset` so the subtask's enter/exit (approach /
retract) line up with the parent free-space path at a relative placement.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tdl import geometry as G
from tdl.schema import Frame

# Catalogs whose names are prefixed with `as:` and whose sequence refs are rewritten.
_PREFIXED_POOLS = (
    "locations",
    "force_pushes",
    "grind_paths",
    "keyholes",
    "free_space",
    "sensors",
    "gates",
)

# Shared cell resources: merge by name; identical defs ok; do not prefix.
_SHARED_POOLS = ("io", "grippers")

# Sequence keys whose string value is a prefixed motion / via / gate name.
_REF_KEYS = frozenset(
    {
        "approach",
        "target",
        "retract",
        "pre",
        "post",
        "keyhole",
        "move",
        "gate",
        "force_push",
        "grind_path",
        "location",
    }
)

# Bare sequence tokens that are phases / keywords, not catalog names.
_PHASE_TOKENS = frozenset(
    {
        "rest",
        "approach",
        "target",
        "retract",
        "start",
        "push",
        "path",
        "pre",
        "post",
    }
)


def load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported task file type: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Task file must contain a mapping/object: {path}")
    return data


def resolve_includes(data: dict[str, Any], path: Path, *, _stack: frozenset[Path] | None = None) -> dict[str, Any]:
    """Recursively flatten `include` steps into catalogs + spliced sequences."""
    path = path.resolve()
    stack = _stack or frozenset()
    if path in stack:
        chain = " → ".join(str(p) for p in sorted(stack | {path}))
        raise ValueError(f"Circular task include: {chain}")
    stack = stack | {path}

    out = copy.deepcopy(data)
    if "sequence" in out:
        out["sequence"] = _resolve_sequence(out["sequence"], out, out, path, stack)
    systems = out.get("systems")
    if isinstance(systems, dict):
        for spec in systems.values():
            if isinstance(spec, dict) and "sequence" in spec:
                # Motion catalogs merge into the system; io/grippers/gates stay on the task root.
                spec["sequence"] = _resolve_sequence(spec["sequence"], out, spec, path, stack)
    return out


def _resolve_sequence(
    items: list[Any],
    root: dict[str, Any],
    motion_target: dict[str, Any],
    parent_path: Path,
    stack: frozenset[Path],
) -> list[Any]:
    if not isinstance(items, list):
        raise ValueError("sequence must be a list")
    resolved: list[Any] = []
    for raw in items:
        if isinstance(raw, dict) and "include" in raw:
            if len(raw) != 1:
                raise ValueError("include step must be a single-key mapping: {include: {...}}")
            resolved.extend(_expand_include(raw["include"], root, motion_target, parent_path, stack))
            continue
        if isinstance(raw, dict) and "repeat" in raw:
            block = raw["repeat"]
            if not isinstance(block, dict):
                raise ValueError("repeat must be a mapping with times and steps")
            block = dict(block)
            block["steps"] = _resolve_sequence(
                block.get("steps") or [], root, motion_target, parent_path, stack
            )
            resolved.append({"repeat": block})
            continue
        resolved.append(copy.deepcopy(raw))
    return resolved


def _expand_include(
    spec: Any,
    root: dict[str, Any],
    motion_target: dict[str, Any],
    parent_path: Path,
    stack: frozenset[Path],
) -> list[Any]:
    if not isinstance(spec, dict):
        raise ValueError("include must be a mapping with file, as, and optional offset")
    file_field = spec.get("file")
    if not isinstance(file_field, str) or not file_field.strip():
        raise ValueError("include requires file: <path>")
    alias = spec.get("as")
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("include requires as: <namespace prefix>")
    alias = alias.strip()
    strip_rest = bool(spec.get("strip_rest", True))

    child_path = (parent_path.parent / file_field).resolve()
    if not child_path.is_file():
        raise FileNotFoundError(f"Included task not found: {child_path} (from {parent_path})")

    child = resolve_includes(load_raw(child_path), child_path, _stack=stack)
    _check_units(root, child, child_path)
    if child.get("systems"):
        raise ValueError(f"Included task may not define systems: {child_path}")

    degrees = _parent_degrees(root)
    T_offset = _offset_matrix(spec.get("offset"), degrees)
    child = _transform_child(child, T_offset, degrees)
    child = _prefix_child(child, alias)

    _merge_catalogs(root, motion_target, child, child_path)
    seq = list(child.get("sequence") or [])
    if strip_rest:
        seq = _strip_outer_rest(seq)
    return seq


def _parent_degrees(parent: dict[str, Any]) -> bool:
    units = parent.get("units") or {}
    angle = units.get("angle", "deg") if isinstance(units, dict) else "deg"
    return angle in {"deg", "degree", "degrees"}


def _check_units(parent: dict[str, Any], child: dict[str, Any], child_path: Path) -> None:
    pu = parent.get("units") or {}
    cu = child.get("units") or {}
    if not isinstance(pu, dict):
        pu = {}
    if not isinstance(cu, dict):
        cu = {}
    for key in ("length", "angle"):
        pv = pu.get(key)
        cv = cu.get(key)
        if pv is not None and cv is not None and pv != cv:
            raise ValueError(f"Included task units.{key} mismatch ({cv!r} vs parent {pv!r}): {child_path}")


def _offset_matrix(offset: Any, degrees: bool) -> np.ndarray:
    if offset is None:
        return np.eye(4)
    if not isinstance(offset, dict):
        raise ValueError("include offset must be a frame mapping (xyz / rpy or quat_xyzw)")
    return Frame.model_validate(offset).matrix(degrees)


def _transform_child(child: dict[str, Any], T_offset: np.ndarray, degrees: bool) -> dict[str, Any]:
    out = copy.deepcopy(child)
    for name, loc in list((out.get("locations") or {}).items()):
        if isinstance(loc, dict) and "target" in loc:
            loc["target"] = _transform_frame(loc["target"], T_offset, degrees)
            _transform_world_pattern(loc.get("pattern"), T_offset, degrees)
    for name, fp in list((out.get("force_pushes") or {}).items()):
        if isinstance(fp, dict) and "target" in fp:
            fp["target"] = _transform_frame(fp["target"], T_offset, degrees)
    for name, gp in list((out.get("grind_paths") or {}).items()):
        if isinstance(gp, dict) and "target" in gp:
            gp["target"] = _transform_frame(gp["target"], T_offset, degrees)
    for name, kh in list((out.get("keyholes") or {}).items()):
        if isinstance(kh, dict):
            out["keyholes"][name] = _transform_frame(kh, T_offset, degrees, keep_extra=True)
    for name, path in list((out.get("free_space") or {}).items()):
        if isinstance(path, dict) and isinstance(path.get("waypoints"), list):
            path["waypoints"] = [_transform_frame(wp, T_offset, degrees) for wp in path["waypoints"]]
    for name, sensor in list((out.get("sensors") or {}).items()):
        if isinstance(sensor, dict) and "xyz" in sensor:
            sensor["xyz"] = _transform_point(sensor["xyz"], T_offset)
    for name, gate in list((out.get("gates") or {}).items()):
        if isinstance(gate, dict) and "pose" in gate:
            gate["pose"] = _transform_frame(gate["pose"], T_offset, degrees)
    return out


def _transform_frame(frame: Any, T_offset: np.ndarray, degrees: bool, *, keep_extra: bool = False) -> dict[str, Any]:
    if not isinstance(frame, dict):
        raise ValueError("frame must be a mapping")
    base = {k: frame[k] for k in ("xyz", "rpy", "quat_xyzw") if k in frame}
    T = T_offset @ Frame.model_validate(base).matrix(degrees)
    xyz, rpy = G.matrix_to_xyz_rpy(T, degrees=degrees)
    out: dict[str, Any] = {"xyz": list(xyz), "rpy": list(rpy)}
    if keep_extra:
        for k, v in frame.items():
            if k not in {"xyz", "rpy", "quat_xyzw"}:
                out[k] = copy.deepcopy(v)
    return out


def _transform_point(xyz: Any, T_offset: np.ndarray) -> list[float]:
    p = np.asarray(xyz, dtype=float).reshape(3)
    q = T_offset[:3, :3] @ p + T_offset[:3, 3]
    return [float(v) for v in q]


def _transform_world_pattern(pattern: Any, T_offset: np.ndarray, degrees: bool) -> None:
    if not isinstance(pattern, dict) or pattern.get("frame") != "world":
        return
    R = T_offset[:3, :3]
    t = T_offset[:3, 3]
    if "offsets" in pattern and isinstance(pattern["offsets"], list):
        pattern["offsets"] = [
            [float(v) for v in (R @ np.asarray(o, dtype=float).reshape(3) + t)] for o in pattern["offsets"]
        ]
    grid = pattern.get("grid")
    if isinstance(grid, dict):
        if "origin" in grid:
            grid["origin"] = [float(v) for v in (R @ np.asarray(grid["origin"], dtype=float).reshape(3) + t)]
        if "step" in grid:
            grid["step"] = [float(v) for v in (R @ np.asarray(grid["step"], dtype=float).reshape(3))]


def _prefix_child(child: dict[str, Any], alias: str) -> dict[str, Any]:
    out = copy.deepcopy(child)
    rename: dict[str, str] = {}
    for pool in _PREFIXED_POOLS:
        mapping = out.get(pool)
        if not isinstance(mapping, dict):
            continue
        new_map: dict[str, Any] = {}
        for name, value in mapping.items():
            new_name = f"{alias}/{name}"
            rename[name] = new_name
            new_map[new_name] = value
        out[pool] = new_map

    # Gate.until and Sensor.gate reference sensor/gate names (already in rename).
    for sensor in (out.get("sensors") or {}).values():
        if isinstance(sensor, dict) and isinstance(sensor.get("gate"), str):
            g = sensor["gate"]
            if g in rename:
                sensor["gate"] = rename[g]
    for gate in (out.get("gates") or {}).values():
        if isinstance(gate, dict) and isinstance(gate.get("until"), str):
            s = gate["until"]
            if s in rename:
                gate["until"] = rename[s]

    if "sequence" in out:
        out["sequence"] = _rewrite_sequence_refs(out["sequence"], rename, alias)
    for pool in ("locations", "force_pushes", "grind_paths"):
        for item in (out.get(pool) or {}).values():
            if isinstance(item, dict) and isinstance(item.get("sequence"), list):
                # Location-internal phases stay as-is; only rewrite explicit named refs.
                item["sequence"] = _rewrite_inner_sequence(item["sequence"], rename)

    lights = out.get("tower_lights")
    if isinstance(lights, dict):
        for light in lights.values():
            if isinstance(light, dict) and isinstance(light.get("tasks"), dict):
                light["tasks"] = {rename.get(task, f"{alias}/{task}"): idx for task, idx in light["tasks"].items()}
    return out


def _rewrite_sequence_refs(items: list[Any], rename: dict[str, str], alias: str) -> list[Any]:
    out: list[Any] = []
    for raw in items:
        if isinstance(raw, str):
            if raw in _PHASE_TOKENS:
                out.append(raw)
            else:
                out.append(rename.get(raw, f"{alias}/{raw}"))
            continue
        if isinstance(raw, dict):
            if "repeat" in raw:
                block = dict(raw["repeat"])
                block["steps"] = _rewrite_sequence_refs(block.get("steps") or [], rename, alias)
                out.append({"repeat": block})
                continue
            if "include" in raw:
                raise ValueError("internal error: unresolved include after resolve_includes")
            rewritten = copy.deepcopy(raw)
            for key in _REF_KEYS:
                if key in rewritten and isinstance(rewritten[key], str):
                    name = rewritten[key]
                    rewritten[key] = rename.get(name, f"{alias}/{name}")
            out.append(rewritten)
            continue
        out.append(copy.deepcopy(raw))
    return out


def _rewrite_inner_sequence(items: list[Any], rename: dict[str, str]) -> list[Any]:
    """Location / FP / grind sequences: keep phase tokens; rewrite explicit named refs only."""
    out: list[Any] = []
    for raw in items:
        if isinstance(raw, str):
            out.append(raw)
            continue
        if isinstance(raw, dict):
            rewritten = copy.deepcopy(raw)
            for key in _REF_KEYS:
                if key in rewritten and isinstance(rewritten[key], str):
                    name = rewritten[key]
                    if name in rename:
                        rewritten[key] = rename[name]
            out.append(rewritten)
            continue
        out.append(copy.deepcopy(raw))
    return out


_MOTION_POOLS = ("locations", "force_pushes", "grind_paths", "keyholes", "free_space")
_ROOT_PREFIXED_POOLS = ("sensors", "gates")


def _merge_catalogs(
    root: dict[str, Any],
    motion_target: dict[str, Any],
    child: dict[str, Any],
    child_path: Path,
) -> None:
    for pool in _MOTION_POOLS:
        _merge_dict_pool(motion_target, pool, child.get(pool) or {}, child_path, allow_overlap=False)
    for pool in _ROOT_PREFIXED_POOLS:
        _merge_dict_pool(root, pool, child.get(pool) or {}, child_path, allow_overlap=False)
    for pool in _SHARED_POOLS:
        _merge_shared_pool(root, pool, child.get(pool) or {}, child_path)
    _merge_tower_lights(root, child.get("tower_lights") or {})


def _merge_dict_pool(
    parent: dict[str, Any],
    pool: str,
    incoming: dict[str, Any],
    child_path: Path,
    *,
    allow_overlap: bool,
) -> None:
    if not incoming:
        return
    dest = parent.setdefault(pool, {})
    if not isinstance(dest, dict):
        raise ValueError(f"parent {pool} must be a mapping")
    for name, value in incoming.items():
        if name in dest and not allow_overlap:
            raise ValueError(f"Included name {name!r} collides with parent {pool}: {child_path}")
        dest[name] = copy.deepcopy(value)


def _merge_shared_pool(parent: dict[str, Any], pool: str, incoming: dict[str, Any], child_path: Path) -> None:
    if not incoming:
        return
    dest = parent.setdefault(pool, {})
    if not isinstance(dest, dict):
        raise ValueError(f"parent {pool} must be a mapping")
    for name, value in incoming.items():
        if name not in dest:
            dest[name] = copy.deepcopy(value)
            continue
        if dest[name] != value:
            raise ValueError(
                f"Included {pool} {name!r} conflicts with parent definition: {child_path}"
            )


def _merge_tower_lights(parent: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Add child lights that are new. If a light name already exists on the parent,
    the parent owns the stack — map prefixed child tasks there yourself.
    """
    if not incoming:
        return
    dest = parent.setdefault("tower_lights", {})
    if not isinstance(dest, dict):
        raise ValueError("parent tower_lights must be a mapping")
    for name, light in incoming.items():
        if name in dest:
            continue
        dest[name] = copy.deepcopy(light)


def _strip_outer_rest(items: list[Any]) -> list[Any]:
    seq = list(items)
    while seq and seq[0] == "rest":
        seq.pop(0)
    while seq and seq[-1] == "rest":
        seq.pop()
    return seq
