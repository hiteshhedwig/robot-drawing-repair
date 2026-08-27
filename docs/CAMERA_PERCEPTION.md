# Camera-Derived Canvas Observation

## Architectural boundary

Autonomous repair now uses this information flow:

```text
SIMULATION
  fixed RGB camera + visible simulated ink
                 │
                 │ RGB pixels only
                 ▼
PERCEPTION (`perception.py`)
  homography rectification + dark-ink segmentation
                 │
                 │ observed_canvas / ink_mask
                 ▼
ERROR DETECTION
  desired trajectory vs camera-derived mask
                 ▼
REPAIR PLANNING → ROBOT CONTROL → SIMULATION
```

The desired mouse trajectory remains known exactly. The current drawing used by
autonomy is reconstructed from RGB pixels.

## Strict autonomy rule

`manual.py::observe_camera()` is the only function that updates the autonomous
`current_image`. It stores the output of `CameraCanvasObserver.observe()`.

Repair planning, missing percentage, the error map, progress checking, and
post-repair verification all read this camera-derived image.

They do not read:

- marker-tip history;
- commanded trajectory history as evidence of success;
- world-space simulated ink segments;
- erased OpenCV pixels; or
- the oracle canvas.

Marker-tip pose remains available for IK, motion control, and choosing the
nearest repair route. This is control state, not visual evidence.

## Fixed MuJoCo camera

`scene.py` adds `canvas_rgb_camera`, a fixed overhead camera centered on the
physical canvas. Its pose does not depend on the drawing or robot state.

The ordinary Panda home pose blocks the overhead view. The controller therefore
uses a fixed `OBSERVATION_QPOS` retreat pose that moves the arm beside the canvas.
Camera evidence is accepted after manual execution, repair, or a disturbance
retreat reaches this pose.

This avoids interpreting the dark marker or robot body as ink and avoids trying
to infer drawing hidden behind an occluder.

## Simulation-side ink rendering

The simulator retains ink as world-space segments because MuJoCo has no built-in
deformable marker-deposition model. Before an RGB frame is rendered, these
segments are inserted into the camera render scene as thin, overlapping
world-space capsules. Capsules produce continuous perspective geometry more
reliably than screen-space debug lines at curved joins.

This segment list is a simulation implementation detail. It is allowed to affect
the rendered physical scene, just as object geometry affects any simulated
camera. It does not cross the RGB boundary into perception or error detection.

## Deterministic calibration and rectification

The camera and canvas geometry are fixed and known. `CanvasPerception` projects
the four physical corners of the usable `28 × 24 cm` drawing region through the
known camera extrinsics and vertical field of view.

Those four image points define a homography to the existing `640 × 512`
reference-canvas coordinates:

```text
raw camera trapezoid/rectangle
          │
          │ cv2.getPerspectiveTransform
          ▼
640 × 512 top-down canvas
```

The output coordinate convention is identical to mouse trajectory coordinates,
so the existing directional error detector and repair-path extraction can be
reused.

This is a calibration shortcut, not privileged drawing-state access: exact fixed
camera and canvas geometry are known, but no knowledge of where ink exists is
used to compute the homography.

## Classical perception pipeline

One observation performs:

1. Render a `640 × 480` RGB frame from `canvas_rgb_camera`.
2. Convert RGB to BGR for OpenCV.
3. Rectify the known drawing ROI with `cv2.warpPerspective`.
4. Convert the rectified image to grayscale.
5. Threshold dark ink at grayscale value `60`.
6. Apply a small `2 × 2` morphological opening.
7. Construct a white observed canvas with dark segmented ink.

No learned model, neural network, or simulator object/segmentation ID is used.

## Observation timing

Perception is intentionally discrete rather than trusted during occlusion:

- After a manual drawing, the arm retreats and a new camera frame is processed.
- Pressing `A` refreshes the camera before planning.
- After every repair cycle, the arm retreats, a new frame is processed, and only
  then is success or another repair decided.
- If erasing occurs during repair, repair motion is cancelled, the arm retreats,
  the camera observes the modified scene, and a replacement plan is generated.

The commanded repair is never assumed to have succeeded.

## Erasing boundary

Dragging over the camera-current panel queues eraser locations for the simulator.
The simulator removes intersecting physical ink segments. The OpenCV camera
canvas is not painted white directly.

Consequently, immediately after simulation-side erasure the autonomous metric is
still stale. It changes only after a new RGB frame is rendered and processed.
This behavior is explicitly covered by validation.

The disturbance event is allowed to trigger cancellation/retreat, but it does
not tell error detection which desired pixels are missing.

## Oracle/debug channel

The previous marker-tip projection remains as `oracle_image`. It is updated from
exact simulated marker motion and is privileged.

It is used only for:

- the `ORACLE` diagnostic percentage;
- the camera/oracle percentage difference;
- the `ORACLE DEBUG` perception panel; and
- debugging simulation ink retention.

No autonomous planning or termination decision reads the oracle result.

## Inspectable views

The `robo_corr camera perception debug` window contains:

1. raw RGB camera frame;
2. rectified camera ROI;
3. segmented ink mask;
4. desired/reference canvas;
5. camera-derived canvas with error overlay; and
6. privileged oracle canvas, clearly labeled as debug-only.

The raw RGB panel refreshes at about 10 Hz while the robot moves or the eraser
is active, so robot motion and drawing are directly inspectable. These live,
possibly occluded preview frames never update autonomous state. Rectification,
segmentation, error detection, and repair decisions accept a frame only at the
unobstructed observation pose.

The main window reports:

```text
CAMERA  x.x%   ORACLE  y.y%   DIFF  z.z%
```

The main `MISSING / UNRECOVERED` value is the camera value.

## Validated scenarios

Headless offscreen tests using the real MuJoCo renderer validated:

| Scenario | Result |
|---|---|
| Blank canvas | Zero segmented ink pixels in the observation pose. |
| Straight line | Camera missing percentage reached `0.0%`. |
| Arbitrary squiggle | Camera missing percentage reached `0.0%`. |
| Complete figure-eight | Camera and oracle both reached `0.0%`. |
| Middle section erased | Camera stayed stale before rendering, then detected `1.91%` missing. |
| Autonomous repair | A new camera frame verified the repaired result at `0.0%`. |
| Second disturbance | A separate gap was independently detected and repaired through the same RGB loop. |

## Remaining shortcuts and limitations

- Ink is simulated as renderable world-space capsules, not deposited material.
- The debug eraser deletes those simulation ink segments using known geometry.
  Perception is not told the deletion result.
- Camera extrinsics, intrinsics, canvas geometry, and homography are known exactly.
  There is no calibration noise or automatic calibration procedure yet.
- Observation occurs after retreat rather than continuously through occlusion.
- Lighting and canvas appearance are controlled, and the grayscale threshold is
  fixed.
- The desired ordered mouse trajectory is known exactly and is used to turn a
  camera-derived error map into an executable path.
- The oracle still exists in memory for diagnostics, but it is not an autonomy
  input.
