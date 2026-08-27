"""Manual or imported-image-to-Panda trajectory demonstration."""

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from robo_corr.drawing_input import (
    ERASER_RADIUS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    WINDOW_NAME,
    StrokeRecorder,
)
from robo_corr.kinematics import MarkerIK
from robo_corr.image_import import import_line_art
from robo_corr.scene import (
    CANVAS_CENTER,
    CANVAS_HALF_SIZE,
    build_model,
    make_home_data,
)


SURFACE_Z = CANVAS_CENTER[2] + CANVAS_HALF_SIZE[2]
DRAW_Z = SURFACE_Z + 0.002
HOVER_Z = SURFACE_Z + 0.045
PIXEL_SPACING = 12.0
DRAW_JOINT_SPEED = 1.20
TRAVEL_JOINT_SPEED = 1.80
MIN_SEGMENT_DURATION = 0.05
PEN_DOWN_SETTLE_DURATION = 0.05
REPAIR_MIN_PROGRESS_PERCENT = 0.05
AUTO_REPAIR_DELAY_SECONDS = 2.0
# Approximate image-space execution rates used only to decide whether two
# damaged runs on the same reference stroke should be joined. Lifting and
# lowering includes the two vertical moves and their settling overhead.
ESTIMATED_DRAW_PIXELS_PER_SECOND = 85.0
ESTIMATED_AIR_PIXELS_PER_SECOND = 135.0
PEN_LIFT_LOWER_OVERHEAD_SECONDS = 0.55
DRAWING_HALF_X = CANVAS_HALF_SIZE[0] - 0.110
DRAWING_HALF_Y = CANVAS_HALF_SIZE[1] - 0.080


def choose_image_file() -> str | None:
    """Open a small native file picker without adding a project dependency."""
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Import binary line-art reference",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*"),
            ],
        )
        root.destroy()
        return selected or None
    except Exception as error:
        print(f"Could not open the image picker: {error}")
        print("Use: python -m robo_corr.manual --image /path/to/image.png")
        return None


def resample_stroke(stroke: np.ndarray, spacing: float = PIXEL_SPACING) -> np.ndarray:
    """Resample a mouse stroke at near-uniform spacing without changing its order."""
    segments = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
    distance = np.concatenate(([0.0], np.cumsum(segments)))
    keep = np.concatenate(([True], np.diff(distance) > 1e-9))
    stroke = stroke[keep]
    distance = distance[keep]
    if len(stroke) < 2 or distance[-1] == 0:
        return stroke
    samples = np.arange(0.0, distance[-1], spacing)
    samples = np.append(samples, distance[-1])
    return np.column_stack(
        [np.interp(samples, distance, stroke[:, axis]) for axis in range(2)]
    )


def pixel_to_canvas(points: np.ndarray, z: float) -> np.ndarray:
    """Map the OpenCV tablet into the canvas with a small safety margin."""
    # Use the central 28 x 24 cm as the first validated drawing workspace.
    # The visible physical canvas is larger, leaving room to expand after
    # collision-aware reachability testing.
    u = points[:, 0] / (IMAGE_WIDTH - 1)
    v = points[:, 1] / (IMAGE_HEIGHT - 1)

    # Screen top is farther from the robot; screen right is robot -Y.
    robot_x = CANVAS_CENTER[0] + DRAWING_HALF_X * (1.0 - 2.0 * v)
    robot_y = CANVAS_CENTER[1] + DRAWING_HALF_Y * (1.0 - 2.0 * u)
    return np.column_stack((robot_x, robot_y, np.full(len(points), z)))


def canvas_to_pixel(point: np.ndarray) -> tuple[int, int]:
    """Map an actual robot marker position into current_canvas pixels."""
    u = 0.5 * (1.0 - (point[1] - CANVAS_CENTER[1]) / DRAWING_HALF_Y)
    v = 0.5 * (1.0 - (point[0] - CANVAS_CENTER[0]) / DRAWING_HALF_X)
    x = int(np.clip(round(u * (IMAGE_WIDTH - 1)), 0, IMAGE_WIDTH - 1))
    y = int(np.clip(round(v * (IMAGE_HEIGHT - 1)), 0, IMAGE_HEIGHT - 1))
    return x, y


def build_shortest_repair_strokes(
    recorded_strokes: list[list[tuple[int, int]]],
    error_map: np.ndarray,
    start_pixel: tuple[int, int],
) -> list[np.ndarray]:
    """Extract missing stroke runs and order them by shortest pen-up travel.

    Each repair run follows the original mouse trajectory. A nearest-neighbor
    choice selects the closest remaining endpoint and reverses the run when its
    far end is closer. This keeps repair travel short while preserving the
    reference curve itself.
    """
    expanded_error = cv2.dilate(
        error_map.astype(np.uint8), np.ones((9, 9), np.uint8)
    ) > 0
    candidates: list[np.ndarray] = []

    for recorded in recorded_strokes:
        if len(recorded) < 2:
            continue
        points = resample_stroke(np.asarray(recorded, dtype=float), spacing=4.0)
        pixels = np.rint(points).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, IMAGE_WIDTH - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, IMAGE_HEIGHT - 1)
        missing = expanded_error[pixels[:, 1], pixels[:, 0]]

        raw_runs: list[tuple[int, int]] = []
        index = 0
        while index < len(points):
            if not missing[index]:
                index += 1
                continue
            run_start = index
            while index + 1 < len(points) and missing[index + 1]:
                index += 1
            run_end = index

            raw_runs.append((run_start, run_end))
            index += 1

        # A short healthy bridge is cheaper to draw over than to perform a
        # marker-up, faster air move, and marker-down cycle. Merge such runs so
        # the repair stays on the original trajectory without needless lifts.
        merged_runs: list[list[int]] = []
        for run_start, run_end in raw_runs:
            if not merged_runs:
                merged_runs.append([run_start, run_end])
                continue

            previous_end = merged_runs[-1][1]
            bridge = points[previous_end : run_start + 1]
            bridge_length = float(
                np.linalg.norm(np.diff(bridge, axis=0), axis=1).sum()
            )
            direct_distance = float(np.linalg.norm(points[run_start] - points[previous_end]))
            draw_time = bridge_length / ESTIMATED_DRAW_PIXELS_PER_SECOND
            air_time = (
                direct_distance / ESTIMATED_AIR_PIXELS_PER_SECOND
                + PEN_LIFT_LOWER_OVERHEAD_SECONDS
            )
            if draw_time <= air_time:
                merged_runs[-1][1] = run_end
            else:
                merged_runs.append([run_start, run_end])

        for run_start, run_end in merged_runs:
            # Twelve pixels of overlap at each end makes the physical repair
            # cross into healthy ink instead of stopping at a detected boundary.
            run_start = max(0, run_start - 3)
            run_end = min(len(points) - 1, run_end + 3)
            repair = points[run_start : run_end + 1]
            if len(repair) >= 2:
                candidates.append(repair)

    ordered: list[np.ndarray] = []
    current = np.asarray(start_pixel, dtype=float)
    while candidates:
        best_index = 0
        reverse = False
        best_distance = float("inf")
        for index, candidate in enumerate(candidates):
            distance_to_start = float(np.linalg.norm(candidate[0] - current))
            distance_to_end = float(np.linalg.norm(candidate[-1] - current))
            if min(distance_to_start, distance_to_end) < best_distance:
                best_index = index
                reverse = distance_to_end < distance_to_start
                best_distance = min(distance_to_start, distance_to_end)
        selected = candidates.pop(best_index)
        if reverse:
            selected = selected[::-1].copy()
        ordered.append(selected)
        current = selected[-1]
    return ordered


def plan_joint_path(
    model: mujoco.MjModel,
    start_qpos: np.ndarray,
    strokes: list[np.ndarray],
    return_qpos: np.ndarray | None = None,
    orientation_qpos: np.ndarray | None = None,
) -> list[tuple[np.ndarray, bool]]:
    """Convert strokes into joint targets; bool indicates marker-down motion."""
    ik = MarkerIK(
        model, start_qpos if orientation_qpos is None else orientation_qpos
    )
    seed = start_qpos.copy()
    path: list[tuple[np.ndarray, bool]] = []

    for stroke_number, raw_stroke in enumerate(strokes, start=1):
        pixels = resample_stroke(raw_stroke)
        draw_points = pixel_to_canvas(pixels, DRAW_Z)
        hover_start = draw_points[0].copy()
        hover_start[2] = HOVER_Z
        hover_end = draw_points[-1].copy()
        hover_end[2] = HOVER_Z

        targets = [(hover_start, False), (draw_points[0], False)]
        targets.extend((point, True) for point in draw_points[1:])
        targets.append((hover_end, False))

        for target, marker_down in targets:
            result = ik.solve(target, seed)
            seed = result.qpos
            path.append((seed.copy(), marker_down))
        print(f"Planned stroke {stroke_number}: {len(draw_points)} drawing waypoints")

    path.append(((start_qpos if return_qpos is None else return_qpos).copy(), False))
    return path


def add_ink_segment(
    viewer: mujoco.viewer.Handle, start: np.ndarray, end: np.ndarray
) -> None:
    """Add one persistent line to the viewer's user scene."""
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return
    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray([0.03, 0.03, 0.04, 1.0], dtype=np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 3.0, start, end)
    viewer.user_scn.ngeom += 1


class ManualExecutor:
    """Advance robot execution one simulation step without blocking either GUI."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        recorder: StrokeRecorder,
    ) -> None:
        self.model = model
        self.data = data
        self.recorder = recorder
        self.home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.arm_qpos_ids = np.asarray(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"
                    )
                ]
                for i in range(1, 8)
            ]
        )
        self.tip_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "marker_tip"
        )
        self.path: list[tuple[np.ndarray, bool]] = []
        self.waypoint = 0
        self.segment_step = 0
        self.segment_steps = 0
        self.segment_move_steps = 0
        self.segment_start = np.zeros(7)
        self.segment_goal = np.zeros(7)
        self.marker_down = False
        self.last_ink_point: np.ndarray | None = None
        self.ink_segments: list[tuple[np.ndarray, np.ndarray]] = []
        self.status = "READY"
        self.completion_count = 0

    @property
    def executing(self) -> bool:
        return bool(self.path)

    def start(
        self, path: list[tuple[np.ndarray, bool]], status: str = "EXECUTING"
    ) -> None:
        self.path = path
        self.waypoint = 0
        self.segment_step = 0
        self.last_ink_point = None
        self.status = status

    def cancel_motion(self) -> None:
        """Stop the active plan at the current pose, ready for replanning."""
        self.path = []
        self.segment_step = 0
        self.last_ink_point = None
        self.data.ctrl[:7] = self.data.qpos[self.arm_qpos_ids]
        self.data.ctrl[7] = 255.0

    def reset(self) -> None:
        self.path = []
        self.segment_step = 0
        self.last_ink_point = None
        self.ink_segments.clear()
        self.status = "READY"
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _point_segment_distance(
        point: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> float:
        segment = end - start
        length_squared = float(segment @ segment)
        if length_squared == 0:
            return float(np.linalg.norm(point - start))
        fraction = float(np.clip(((point - start) @ segment) / length_squared, 0, 1))
        return float(np.linalg.norm(point - (start + fraction * segment)))

    def apply_eraser(self, viewer: mujoco.viewer.Handle) -> bool:
        """Remove touched physical ink and report whether anything changed."""
        if not self.recorder.erase_points:
            return False
        erase_points = [np.asarray(point, dtype=float) for point in self.recorder.erase_points]
        self.recorder.erase_points.clear()

        previous_count = len(self.ink_segments)
        kept: list[tuple[np.ndarray, np.ndarray]] = []
        for start, end in self.ink_segments:
            pixel_start = np.asarray(canvas_to_pixel(start), dtype=float)
            pixel_end = np.asarray(canvas_to_pixel(end), dtype=float)
            touched = any(
                self._point_segment_distance(point, pixel_start, pixel_end)
                <= ERASER_RADIUS + 6
                for point in erase_points
            )
            if not touched:
                kept.append((start, end))
        self.ink_segments = kept
        changed = len(kept) != previous_count
        if not changed:
            return False

        self.recorder.disturbance_version += 1
        self.recorder.rebuild_current(
            [
                (canvas_to_pixel(start), canvas_to_pixel(end))
                for start, end in self.ink_segments
            ]
        )

        viewer.user_scn.ngeom = 0
        for start, end in self.ink_segments:
            add_ink_segment(viewer, start, end)
        return True

    def ensure_ink_visible(self, viewer: mujoco.viewer.Handle) -> None:
        """Restore persistent ink if the viewer's transient scene was cleared."""
        expected = min(len(self.ink_segments), viewer.user_scn.maxgeom)
        if viewer.user_scn.ngeom == expected:
            return
        viewer.user_scn.ngeom = 0
        for start, end in self.ink_segments:
            add_ink_segment(viewer, start, end)

    def _store_ink_segment(
        self, viewer: mujoco.viewer.Handle, start: np.ndarray, end: np.ndarray
    ) -> None:
        add_ink_segment(viewer, start, end)
        self.ink_segments.append((start.copy(), end.copy()))
        self.recorder.draw_robot_segment(canvas_to_pixel(start), canvas_to_pixel(end))

    def _begin_segment(self, viewer: mujoco.viewer.Handle) -> None:
        was_marker_down = self.marker_down
        target_qpos, self.marker_down = self.path[self.waypoint]

        if was_marker_down and not self.marker_down and self.last_ink_point is not None:
            endpoint = self.data.site_xpos[self.tip_id].copy()
            endpoint[2] = SURFACE_Z + 0.006
            if np.linalg.norm(endpoint[:2] - self.last_ink_point[:2]) > 1e-6:
                self._store_ink_segment(viewer, self.last_ink_point, endpoint)
            self.last_ink_point = None
        elif self.marker_down and not was_marker_down:
            # The preceding marker-down waypoint has settled. Anchor the ink at
            # that actual position before moving toward the next drawing point.
            self.last_ink_point = self.data.site_xpos[self.tip_id].copy()
            self.last_ink_point[2] = SURFACE_Z + 0.006

        self.segment_start = self.data.ctrl[:7].copy()
        self.segment_goal = target_qpos[self.arm_qpos_ids]
        max_change = float(np.max(np.abs(self.segment_goal - self.segment_start)))
        speed = DRAW_JOINT_SPEED if self.marker_down else TRAVEL_JOINT_SPEED
        duration = max(MIN_SEGMENT_DURATION, max_change / speed)
        self.segment_move_steps = max(2, int(duration / self.model.opt.timestep))
        settle_steps = (
            int(PEN_DOWN_SETTLE_DURATION / self.model.opt.timestep)
            if not self.marker_down
            else 0
        )
        self.segment_steps = self.segment_move_steps + settle_steps

    def step(self, viewer: mujoco.viewer.Handle) -> None:
        if self.path:
            if self.segment_step == 0:
                self._begin_segment(viewer)

            self.segment_step += 1
            move_fraction = min(1.0, self.segment_step / self.segment_move_steps)
            blend = 0.5 - 0.5 * np.cos(np.pi * move_fraction)
            self.data.ctrl[:7] = self.segment_start + blend * (
                self.segment_goal - self.segment_start
            )
            self.data.ctrl[7] = 255.0

        # Cancel model bias forces (primarily gravity) so the Panda's position
        # servos reach small drawing targets without a persistent endpoint lag.
        self.data.qfrc_applied[:] = self.data.qfrc_bias
        mujoco.mj_step(self.model, self.data)

        if self.path and self.marker_down:
            ink_point = self.data.site_xpos[self.tip_id].copy()
            ink_point[2] = SURFACE_Z + 0.006
            if self.last_ink_point is None:
                self.last_ink_point = ink_point
            elif np.linalg.norm(ink_point[:2] - self.last_ink_point[:2]) >= 0.002:
                self._store_ink_segment(viewer, self.last_ink_point, ink_point)
                self.last_ink_point = ink_point
        else:
            self.last_ink_point = None

        if self.path and self.segment_step >= self.segment_steps:
            self.waypoint += 1
            self.segment_step = 0
            if self.waypoint >= len(self.path):
                self.path = []
                self.status = "COMPLETE - R/C TO RESET"
                self.completion_count += 1
                print("Execution complete. Draw again or press R/C to reset both views.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, help="binary/line-art image to load as the reference"
    )
    args = parser.parse_args()

    model = build_model()
    data = make_home_data(model)
    home_qpos = data.qpos.copy()
    recorder = StrokeRecorder()
    executor = ManualExecutor(model, data, recorder)
    auto_mode_enabled = False
    auto_repair_due: float | None = None
    execution_purpose: str | None = None
    last_handled_disturbance = recorder.disturbance_version
    last_handled_completion = executor.completion_count
    repair_start_percent: float | None = None

    def load_image_reference(
        path: str | Path, viewer: mujoco.viewer.Handle | None = None
    ) -> bool:
        nonlocal auto_mode_enabled, auto_repair_due, execution_purpose
        nonlocal last_handled_disturbance, repair_start_percent
        try:
            strokes = import_line_art(path, IMAGE_WIDTH, IMAGE_HEIGHT)
        except ValueError as error:
            executor.status = "IMAGE IMPORT FAILED"
            print(error)
            return False
        if not strokes:
            executor.status = "IMAGE HAS NO DRAWABLE STROKES"
            print("The image produced no drawable line strokes.")
            return False

        executor.reset()
        if viewer is not None:
            viewer.user_scn.ngeom = 0
        recorder.set_reference_strokes(strokes)
        auto_mode_enabled = False
        auto_repair_due = None
        execution_purpose = None
        repair_start_percent = None
        last_handled_disturbance = recorder.disturbance_version
        executor.status = f"IMPORTED {len(strokes)} STROKES - PRESS E"
        point_count = sum(len(stroke) for stroke in strokes)
        print(f"Imported {path}: {len(strokes)} drawable strokes, {point_count} control points.")
        print("Press E to plan and draw the imported reference.")
        return True

    if args.image is not None:
        load_image_reference(args.image)

    def start_repair_plan(check_progress: bool = False) -> bool:
        nonlocal repair_start_percent, execution_purpose
        missing_percent = recorder.missing_percent
        if (
            check_progress
            and repair_start_percent is not None
            and repair_start_percent - missing_percent < REPAIR_MIN_PROGRESS_PERCENT
        ):
            recorder.repair_preview.clear()
            executor.status = f"REPAIR STALLED ({missing_percent:.1f}% UNRESOLVED)"
            print(
                f"Repair stopped with {missing_percent:.1f}% unresolved: the last "
                "simulated output showed no measurable improvement."
            )
            return False

        tip_position = data.site_xpos[executor.tip_id]
        repair_strokes = build_shortest_repair_strokes(
            recorder.strokes, recorder.error_map, canvas_to_pixel(tip_position)
        )
        if not repair_strokes:
            recorder.repair_preview.clear()
            executor.status = "REPAIR COMPLETE"
            print("No missing reference regions remain.")
            repair_start_percent = None
            return False

        recorder.repair_preview = [stroke.copy() for stroke in repair_strokes]
        executor.status = "PLANNING AUTONOMOUS REPAIR"
        print(
            f"Planning {len(repair_strokes)} shortest-first repair segment(s) "
            f"for {recorder.missing_percent:.1f}% missing ink..."
        )
        try:
            repair_path = plan_joint_path(
                model,
                data.qpos.copy(),
                repair_strokes,
                return_qpos=home_qpos,
                orientation_qpos=home_qpos,
            )
        except RuntimeError as error:
            executor.status = "REPAIR IK FAILED"
            print(error)
            return False
        executor.start(repair_path, status="AUTONOMOUS REPAIR")
        execution_purpose = "repair"
        repair_start_percent = missing_percent
        print(f"Repair IK succeeded for {len(repair_path)} robot waypoints.")
        return True

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, recorder.mouse_callback)
    print(
        "Window remains open. I = import image, E = execute, "
        "A = toggle persistent auto-repair, "
        "R/C = reset both, Q/Esc = quit."
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.48, 0.0, 0.43]
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -32.0

        while viewer.is_running():
            loop_start = time.monotonic()
            cv2.imshow(WINDOW_NAME, recorder.display_image(executor.status))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("c")):
                recorder.clear()
                executor.reset()
                viewer.user_scn.ngeom = 0
                auto_mode_enabled = False
                auto_repair_due = None
                execution_purpose = None
                repair_start_percent = None
                last_handled_disturbance = recorder.disturbance_version
                print("Reference and simulated robot-output canvases reset.")
            elif key == ord("i"):
                if executor.executing:
                    print("Wait for execution to finish or reset before importing.")
                else:
                    selected = choose_image_file()
                    if selected:
                        load_image_reference(selected, viewer)
            elif key == ord("e"):
                strokes = recorder.as_arrays()
                if executor.executing:
                    print("Robot is already executing; press R/C to cancel and reset.")
                elif strokes:
                    auto_mode_enabled = False
                    auto_repair_due = None
                    recorder.repair_preview.clear()
                    executor.status = "PLANNING"
                    cv2.imshow(WINDOW_NAME, recorder.display_image(executor.status))
                    cv2.waitKey(1)
                    print(f"Planning {len(strokes)} stroke(s)...")
                    try:
                        path = plan_joint_path(
                            model,
                            data.qpos.copy(),
                            strokes,
                            return_qpos=home_qpos,
                            orientation_qpos=home_qpos,
                        )
                    except RuntimeError as error:
                        executor.status = "IK FAILED - R/C TO RESET"
                        print(error)
                    else:
                        executor.start(path)
                        execution_purpose = "manual"
                        print(f"IK succeeded for {len(path)} robot waypoints.")
            elif key == ord("a"):
                if auto_mode_enabled:
                    auto_mode_enabled = False
                    auto_repair_due = None
                    executor.status = "AUTO MODE OFF"
                    print("Persistent auto-repair mode disabled.")
                elif executor.executing:
                    print("Wait for the current manual execution, then press A.")
                else:
                    auto_mode_enabled = True
                    last_handled_completion = executor.completion_count
                    if recorder.missing_percent > 0.0:
                        auto_repair_due = time.monotonic() + AUTO_REPAIR_DELAY_SECONDS
                        executor.status = "AUTO MODE ON - REPAIR IN 2.0s"
                    else:
                        executor.status = "AUTO MODE ON - WATCHING"
                    print("Persistent auto-repair mode enabled. Press A again to disable.")

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

            executor.apply_eraser(viewer)

            if recorder.disturbance_version != last_handled_disturbance:
                if (
                    auto_mode_enabled
                    and executor.executing
                    and execution_purpose == "repair"
                ):
                    executor.cancel_motion()
                    execution_purpose = None
                    auto_repair_due = None
                    executor.status = "AUTO MODE - WAITING FOR ERASER"
                if not recorder.erasing and not executor.executing:
                    last_handled_disturbance = recorder.disturbance_version
                    if auto_mode_enabled:
                        repair_start_percent = None
                        auto_repair_due = time.monotonic() + AUTO_REPAIR_DELAY_SECONDS
                        executor.status = "AUTO MODE ON - REPAIR IN 2.0s"

            if (
                auto_mode_enabled
                and auto_repair_due is not None
                and not recorder.erasing
                and not executor.executing
            ):
                remaining = auto_repair_due - time.monotonic()
                if remaining > 0.0:
                    executor.status = f"AUTO MODE ON - REPAIR IN {remaining:.1f}s"
                else:
                    auto_repair_due = None
                    if not start_repair_plan():
                        executor.status = "AUTO MODE ON - WATCHING"

            if executor.completion_count != last_handled_completion:
                last_handled_completion = executor.completion_count
                completed_purpose = execution_purpose
                execution_purpose = None
                if auto_mode_enabled and completed_purpose == "repair":
                    if not start_repair_plan(check_progress=True):
                        if recorder.missing_percent <= 0.0:
                            executor.status = "AUTO MODE ON - WATCHING"

            executor.step(viewer)
            executor.ensure_ink_visible(viewer)
            viewer.sync()
            delay = model.opt.timestep - (time.monotonic() - loop_start)
            if delay > 0:
                time.sleep(delay)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
