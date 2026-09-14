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
tdl-viewer examples/gated_dual_pick_place.yaml
```

JSON files with the same schema also work.

## Language

Top-level document (`version: 1`):

| Field | Meaning |
| --- | --- |
| `robot.rest.joints` | Rest pose in joint space (stored for later execution) |
| `robot.rest.tool` | Cartesian tool frame used to visualize rest |
| `locations` | Named pick/place sites |
| `io` | Digital outputs for external equipment; each has an `initial` on/off state |
| `grippers` | Carried gripper state (`initial: open` or `closed`); persists across cycle repeats |
| `tower_lights` | Stack lights with indexed `states` and per-task color mapping |
| `force_pushes` | Compliant push operations (approach, push until force or max travel, retract) |
| `grind_paths` | Contact paths: approach, follow a list of Cartesian points, retract |
| `sensors` | Spherical presence regions tied to a gate; active while that gate is held |
| `gates` | Wait poses; hold until every sensor reads true (mutual rendezvous) |
| `systems` | Parallel lines, each with its own locations, limits, and sequence |
| `keyholes` | Named via poses; free-space goto between locations passes through them |
| `free_space` | Named via-point paths (optional; keyholes usually replace these) |
| `sequence` | Ordered steps, including `repeat` blocks |
| `limits` | Default linear / angular / approach speeds |

Each **location** has:

- **tool** — which end-effector these frames belong to (`name`, default `tool1`). Stored for later robot binding; the viewer does not use it.
- **target** — tool pose at the nominal contact
- **search** — box, sphere, or cylinder in the target frame; optional `found_offset` for the actual pick inside that region
- **approach / retract** — shorthand for a single Cartesian stroke before / after the target. Direction is either `axis: [x, y, z]` in the target frame, `xyz` offset, or spherical `azimuth` / `elevation` (deg or rad per `units.angle`). Offset is from the *current* pose along that chain: `pre` composes outward from the target, `post` composes from the target through each stroke. Orientation stays the target orientation.
- **pre / post** — lists of the same Cartesian specs, any length. `approach` is one `pre` stroke; `retract` is one `post` stroke. Pickup and dropoff use one of each; a process station can insert, then hop up and retract with two `post` moves.
- **dwell** — hold time at the target
- **pattern** — optional visit sequence. Each time the location is used, the target shifts to the next slot and wraps. `grid` with `counts` and `step` (target-frame XYZ by default, or `frame: world`) rasters a tray; `offsets` is an explicit list of translations.
- **io** — optional list of commands fired when a location phase is reached (legacy). Prefer an explicit **sequence** on the location instead.
- **sequence** — optional ordered sub-steps for a location visit. Interleave `approach`, `target`, `retract`, `pre: <i>`, `post: <i>`, `{pause: <seconds>}`, `{io: {signal, set/pulse}}`, and `{gripper: open/close}` anywhere in the visit. When omitted, the default is all pre strokes, target, then all post strokes.

A **keyhole** is a pose the tool must pass through during free-space motion (a doorway, fixture clearance, etc.). Put one or more `{keyhole: <name>}` steps between two locations; the cubic spline from the previous retract (or rest) to the next approach interpolates through each keyhole. `radius` is the visualized aperture; frame **Z** is the pass-through axis.

A **force push** moves to a start pose (with optional approach), then pushes along a prescribed direction until a `force_limit` is reached or `max_travel` is exceeded, then retracts. Fields:

- **target** — pose where the push begins
- **tool** — selected end-effector (`name`, default `tool1`)
- **approach** — optional pre-stroke (same spec as locations)
- **push** — `axis` or `azimuth` / `elevation`, plus `max_travel` and `force_limit` (Newtons)
- **retract** — stroke leaving the contact point
- **sequence** — optional ordered sub-steps: `approach`, `start`, `push`, `retract`, plus `{pause: ...}` and `{io: ...}` (same as locations). Default is approach (if any), start, push, retract.

The viewer currently travels the full `max_travel` distance (no force feedback yet). A real controller would stop early when resistance exceeds `force_limit`.

A **grind path** is a sanding / grinding stroke: move to an approach pose, drop to the start of a contact path, follow a list of points, then retract. Fields:

- **target** — tool pose that defines the path frame (points are XYZ offsets in this frame; orientation stays the target orientation)
- **tool** — selected end-effector (`name`, default `tool1`)
- **approach** — optional pre-stroke (same spec as locations)
- **points** — ordered list of `[x, y, z]` offsets. The first point is the start; remaining points are followed with straight-line Cartesian moves
- **retract** — stroke leaving the last path point
- **sequence** — optional ordered sub-steps: `approach`, `start`, `path`, `retract`, plus `{pause: ...}` and `{io: ...}`. Default is approach (if any), start, every remaining path point, retract.

A **sensor** is a sphere (`xyz` + `radius`) tied to a `gate`. It reads **true** while that gate is held. A **gate** moves to a wait pose and holds until every sensor reads true (mutual rendezvous). Use `{gate: <name>}` in a sequence.

**systems** run in parallel in the viewer. Each has its own `locations`, optional `limits`, and `sequence`. See `examples/gated_dual_pick_place.yaml` for a two-line handshake demo.

**io** defines named digital outputs (`initial: true/false`). Attach `{io: ...}` steps inside a location or force-push **sequence**. Use quoted `"on"` / `"off"` for discrete sets in YAML (`on` and `off` parse as booleans). The viewer shows a simulated LED per signal (dull off, bright green on).

**grippers** are internal carried state, not cell I/O. Define them at the top level with `initial: open` or `closed`. Issue `{gripper: close}` / `{gripper: open}` (or `{gripper: {name, set}}` when there are several) inside a location sequence. The state holds until the next command, including across `repeat` cycles. The viewer shows an amber LED when closed and a dim indicator when open.

**tower_lights** are stack indicators with an ordered `states` list (`name` + `color` per lamp). Map sequence sub-tasks to state indices with `tasks:`. During transport between tasks none of the lamps are lit; each lamp shows its color dimly until its task is active, then brightens. The viewer shows a vertical stack with one lamp per state.

Sequence steps:

- `rest`
- `<location name>` — shorthand for every `pre` stroke, the target, then every `post` stroke
- `<force_push name>` — approach (if any), push start, compliant push to max travel, retract
- `<grind_path name>` — approach (if any), start, follow `points`, retract
- `{gate: <name>}` — move to the gate pose and wait for mutual rendezvous (all sensors true)
- `{keyhole: <name>}` — via pose on the free-space goto to the next destination
- `{move: <free_space name>}` — extra via points from the current pose
- `{approach: <location>}` / `{target: <location>}` / `{retract: <location>}`
- `{pause: <seconds>}` — hold the previous pose (no extra frame)
- `{repeat: {times: N, steps: [...]}}`

Geometry uses 4×4 homogeneous transforms (`scipy.spatial.transform` for orientation). Free-space / transit motion (including into the first approach pose) follows a cubic spline through waypoints; every Cartesian stroke at a location (into the target and each post-target move) stays straight-line. Orientation is SLERP, time-scaled by the active velocity limits.

## Layout

```
tdl/           language, geometry, trajectory, GUI
examples/      sample task files
```
