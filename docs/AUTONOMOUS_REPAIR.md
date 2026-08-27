# Autonomous Drawing Repair: Technical Design

## Purpose

The autonomous mode demonstrates closed-loop self-monitoring and error
correction. It is not framed as consciousness or true self-awareness.

The robot repeatedly performs this loop:

```text
desired drawing
      ↓
execute motion
      ↓
observe actual drawing
      ↓
detect missing regions
      ↓
plan the shortest-first correction
      ↓
repair and observe again
```

The current implementation uses a fixed simulated RGB camera as its autonomous
observation channel. Marker-tip projection remains only as a privileged oracle
for debugging. See `CAMERA_PERCEPTION.md` for the strict boundary and pipeline.

## Main components

| File | Responsibility |
|---|---|
| `drawing_input.py` | Holds the desired and current canvases, handles drawing and erasing, and renders status overlays. |
| `error_detection.py` | Compares desired and current drawings and produces the missing percentage and Boolean error map. |
| `perception.py` | Renders fixed-camera RGB, rectifies the canvas, and segments visible ink. |
| `manual.py` | Maps pixels to the robot workspace, builds drawing and repair paths, controls execution, and handles live replanning. |
| `kinematics.py` | Solves marker-tip position and orientation targets using damped least-squares inverse kinematics. |
| `scene.py` | Builds the Panda, marker, physical canvas, table, lighting, and MuJoCo model. |

## Canvas state

The system maintains two separate images.

### Reference canvas

`reference_canvas` is the drawing created on the left OpenCV panel. It describes
what should exist. The program also retains every mouse stroke as an ordered list
of 2D points.

Keeping the ordered trajectory is important. A pixel-only image says where ink
exists, but not the order or direction in which the marker should traverse it.

### Current canvas

`current_canvas` is reconstructed from a fixed RGB camera frame using calibrated
perspective rectification and classical dark-ink segmentation. Commanded motion,
marker-tip history, and oracle ink pixels do not update it. Dragging on the
current panel removes simulated physical ink; the canvas changes only after the
camera observes that consequence.

## Coordinate mapping

The OpenCV canvas is mapped into a conservative `28 × 24 cm` area at the center
of the larger physical canvas.

For normalized image coordinates `u` and `v`:

```text
robot_x = center_x + half_x × (1 - 2v)
robot_y = center_y + half_y × (1 - 2u)
```

The mapping is intentionally inverted in places so that the visual direction of
the robot drawing matches the OpenCV input from the inspection camera angle.

The inverse mapping converts marker positions into pixels only for the
debug-only oracle, simulated eraser geometry, and route-start selection. It does
not update the camera-derived current canvas. Camera-to-canvas registration uses
the independently calibrated homography described in `CAMERA_PERCEPTION.md`.

## Missing-region detection

The detector produces:

```text
desired_mask
current_mask
error_map
missing_percent
```

### Why a simple image difference was insufficient

A direct pixel subtraction marks ordinary robot tracking offset as damage. A
uniformly dilated current mask avoids that false error, but creates another
problem: ink on both ends of a short gap can overlap after dilation and falsely
cover the empty middle.

The implementation therefore uses the ordered reference trajectory for
directional matching.

### Directional matching

The reference trajectory is resampled approximately every two pixels. At each
sample, the local tangent is estimated from neighboring reference points. A
normal vector is perpendicular to this tangent.

Nearby current-ink pixels are expressed in this local coordinate frame:

```text
along  = absolute(offset · tangent)
across = absolute(offset · normal)
```

A sample is covered when current ink is sufficiently close in both directions.
The tolerances are deliberately asymmetric:

- Larger perpendicular tolerance accepts normal sideways tracking error.
- Tight tangential tolerance ensures the two sides of a gap cannot cover its
  empty center.
- A slightly larger tangential tolerance is allowed near the true beginning and
  end of a complete stroke, where small endpoint lag is harmless.

Uncovered samples are rasterized into the Boolean `error_map`. The UI renders a
sparse red dotted overlay from this map without modifying the underlying current
canvas.

The missing metric is:

```text
missing_percent = 100 × missing_path_samples / desired_path_samples
```

It measures missing trajectory length rather than raw antialiased pixel area.

## Building a repair path

Pressing `A` starts autonomous repair.

### 1. Extract missing trajectory runs

The detector's error map is sampled along each original mouse stroke. Consecutive
missing points are grouped into repair runs.

The repair path follows the original mouse trajectory. The robot does not trace
arbitrary red pixels or invent a geometrically unrelated connection.

### 2. Overlap intact ink

Each repair run is extended by approximately 12 pixels into healthy trajectory
at both ends. This deliberate overlap ensures the physical repair reconnects the
existing line instead of stopping at the exact detection boundary and leaving a
small seam.

### 3. Choose a short execution order

There may be several disconnected damaged regions. The planner uses a greedy
nearest-endpoint strategy:

1. Start at the marker's current projected canvas position.
2. Measure travel to both endpoints of every remaining repair run.
3. Select the closest endpoint.
4. Reverse that repair run when its far endpoint is closer.
5. Repeat from the selected run's finishing point.

This reduces marker-up travel and is easy to inspect. It is a nearest-neighbor
heuristic, not a guaranteed globally optimal travelling-salesperson solution.

Yellow lines and square waypoint boxes on the current panel show the selected
repair order before and during execution.

## Inverse kinematics

Each repair point becomes a Cartesian marker-tip target above the canvas.

The IK solver uses the MuJoCo site Jacobian:

```text
J = [position Jacobian]
    [rotation Jacobian]
```

For Cartesian error `e`, a damped least-squares update is computed as:

```text
Δq = Jᵀ (J Jᵀ + λI)⁻¹ e
```

Damping prevents unstable joint updates near singular configurations. Joint
updates are clipped, and Panda joint limits are enforced at every iteration.

Orientation error is weighted below position error. The marker must stay nearly
vertical, but accurate placement of the tip on the drawing trajectory is the
primary objective.

Disconnected runs use explicit marker-up and marker-down targets.

## Motion execution and ink recording

Joint targets are connected with cosine interpolation. This gives zero commanded
velocity at the beginning and end of each segment and avoids abrupt changes.

MuJoCo model bias forces are applied as compensation during execution. This
primarily cancels gravity so the position servos do not retain a significant
endpoint offset.

Before drawing begins, the controller holds the pen-down waypoint briefly so the
arm can settle. Ink is explicitly anchored at pen-down and explicitly closed at
pen-up. These details prevent catch-up motion and unsampled endpoint tails from
appearing as hooks or false damage.

The simulation retains physical ink as world-space segments so it can render the
scene. Autonomous perception receives only the resulting RGB pixels. A separate
marker-tip-projected oracle is retained for diagnostic comparison only.

## Live disturbance handling

The eraser increments a disturbance version whenever it requests a change to
the simulated physical ink. It does not modify the camera-derived current image.

If erasing occurs during autonomous repair:

1. Active repair motion is cancelled at the robot's present pose.
2. The controller waits until the eraser mouse button is released.
3. The arm retreats to the fixed observation pose.
4. A new RGB frame is rendered, rectified, and segmented.
5. The camera-derived current canvas and error map are recomputed.
6. Missing trajectory runs are extracted again.
7. The shortest-first ordering is recalculated from the current marker position.
8. A new yellow preview replaces the obsolete plan and repair resumes.

This is replanning, not merely appending the new damage to an old trajectory.

## Repair termination

After a repair path finishes, the current canvas is observed again. If actionable
missing runs remain, another correction cycle is planned.

The controller does not declare success merely because the percentage is small.
It finishes when no missing trajectory run remains.

To prevent endless no-op motion, progress is also measured across repair cycles.
If a cycle improves the metric by less than `0.05%`, execution stops with an
explicit `REPAIR STALLED (...% UNRESOLVED)` status. This reports failure honestly
instead of claiming success or repeatedly touching the same region.

## Current limitations

- Ink is a visualization layer rather than deposited deformable material.
- Erasing deletes simulated ink geometry through a debug interaction; camera
  perception is not directly told which pixels were removed.
- Camera calibration and canvas geometry are exact and fixed.
- Observation occurs after a retreat pose instead of continuously under robot
  occlusion.
- Repair ordering uses a greedy shortest-first heuristic and is not globally
  optimal.
- The controller has no collision-aware Cartesian planner yet.
- The marker has no limited-ink model yet.
- The error detector relies on the stored reference trajectory. Arbitrary target
  images without ordered strokes will require skeletonization or another path
  extraction method.

## Natural next steps

1. Add calibration noise and a calibration-validation procedure.
2. Replace debug ink capsules with a more physical deposition model.
3. Add collision-aware approach and retreat motion.
4. Replace the greedy ordering with an exact or approximate route optimizer when
   many repair regions exist.
5. Add paint usage and resource-constrained repair priorities.
