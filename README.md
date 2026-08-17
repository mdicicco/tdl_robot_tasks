# Task Definition Language

A small Python tool for **specifying pick-and-place motions** without a robot model, then inspecting the resulting Cartesian motion in 3D.

You describe rest joints, named locations (target frame + search region + approach/retract), optional keyhole via poses, free-space waypoints, and a sequence that can repeat. The viewer draws the frames, lets you scrub time, and reapplies velocity limits.

## Install

```bash
cd task-definition-language
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the viewer

```bash
tdl-viewer examples/pick_and_place.yaml
```

JSON files with the same schema also work.

## Language

Top-level document (`version: 1`):

| Field | Meaning |
| --- | --- |
| `robot.rest.joints` | Rest pose in joint space (stored for later execution) |
| `robot.rest.tool` | Cartesian tool frame used to visualize rest |
| `locations` | Named pick/place sites |
| `keyholes` | Named via poses; free-space goto between locations passes through them |
| `free_space` | Named via-point paths (optional; keyholes usually replace these) |
| `sequence` | Ordered steps, including `repeat` blocks |
| `limits` | Default linear / angular / approach speeds |

Each **location** has:

- **target** — tool pose at the nominal contact
- **search** — box, sphere, or cylinder in the target frame; optional `found_offset` for the actual pick inside that region
- **approach / retract** — shorthand for a single Cartesian stroke before / after the target. Direction is either `axis: [x, y, z]` in the target frame, `xyz` offset, or spherical `azimuth` / `elevation` (deg or rad per `units.angle`). Offset is from the *current* pose along that chain: `pre` composes outward from the target, `post` composes from the target through each stroke. Orientation stays the target orientation.
- **pre / post** — lists of the same Cartesian specs, any length. `approach` is one `pre` stroke; `retract` is one `post` stroke. Pickup and dropoff use one of each; a process station can insert, then hop up and retract with two `post` moves.
- **dwell** — hold time at the target
- **pattern** — optional visit sequence. Each time the location is used, the target shifts to the next slot and wraps. `grid` with `counts` and `step` (target-frame XYZ by default, or `frame: world`) rasters a tray; `offsets` is an explicit list of translations.

A **keyhole** is a pose the tool must pass through during free-space motion (a doorway, fixture clearance, etc.). Put one or more `{keyhole: <name>}` steps between two locations; the cubic spline from the previous retract (or rest) to the next approach interpolates through each keyhole. `radius` is the visualized aperture; frame **Z** is the pass-through axis.

Sequence steps:

- `rest`
- `<location name>` — shorthand for every `pre` stroke, the target, then every `post` stroke
- `{keyhole: <name>}` — via pose on the free-space goto to the next destination
- `{move: <free_space name>}` — extra via points from the current pose
- `{approach: <location>}` / `{target: <location>}` / `{retract: <location>}`
- `{repeat: {times: N, steps: [...]}}`

Geometry uses 4×4 homogeneous transforms (`scipy.spatial.transform` for orientation). Free-space / transit motion (including into the first approach pose) follows a cubic spline through waypoints; every Cartesian stroke at a location (into the target and each post-target move) stays straight-line. Orientation is SLERP, time-scaled by the active velocity limits.

## Layout

```
tdl/           language, geometry, trajectory, GUI
examples/      sample task files
```
