"""RGB-camera rendering and classical canvas perception."""

from dataclasses import dataclass

import cv2
import mujoco
import numpy as np

from robo_corr.scene import CANVAS_CAMERA_NAME, CANVAS_CENTER


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INK_GRAY_THRESHOLD = 60


@dataclass(frozen=True)
class CameraObservation:
    """Inspectable products of one camera-only perception pass."""

    raw_rgb: np.ndarray
    rectified_bgr: np.ndarray
    ink_mask: np.ndarray
    observed_canvas: np.ndarray


class SimulatedRGBCamera:
    """Simulation-side RGB sensor that renders visible ink geometry."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, CANVAS_CAMERA_NAME
        )
        if self.camera_id < 0:
            raise RuntimeError(f"Model has no camera named {CANVAS_CAMERA_NAME!r}")
        self.renderer = mujoco.Renderer(
            model, height=CAMERA_HEIGHT, width=CAMERA_WIDTH
        )

    def render(
        self,
        data: mujoco.MjData,
        visible_ink_segments: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Return RGB pixels; physical ink is injected before rasterization."""
        self.renderer.update_scene(data, camera=self.camera_id)
        scene = self.renderer.scene
        for start, end in visible_ink_segments:
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.eye(3).ravel(),
                np.asarray([0.025, 0.025, 0.025, 1.0], dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0018, start, end
            )
            scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


class CanvasPerception:
    """Rectify a fixed RGB camera frame and segment dark visible ink."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_width: int,
        output_height: int,
        drawing_half_x: float,
        drawing_half_y: float,
    ) -> None:
        self.output_size = (output_width, output_height)
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, CANVAS_CAMERA_NAME
        )
        source = np.asarray(
            [
                self._project(model, data, camera_id, x, y)
                for x, y in (
                    (CANVAS_CENTER[0] + drawing_half_x, CANVAS_CENTER[1] + drawing_half_y),
                    (CANVAS_CENTER[0] + drawing_half_x, CANVAS_CENTER[1] - drawing_half_y),
                    (CANVAS_CENTER[0] - drawing_half_x, CANVAS_CENTER[1] - drawing_half_y),
                    (CANVAS_CENTER[0] - drawing_half_x, CANVAS_CENTER[1] + drawing_half_y),
                )
            ],
            dtype=np.float32,
        )
        destination = np.asarray(
            [
                (0, 0),
                (output_width - 1, 0),
                (output_width - 1, output_height - 1),
                (0, output_height - 1),
            ],
            dtype=np.float32,
        )
        self.homography = cv2.getPerspectiveTransform(source, destination)

    @staticmethod
    def _project(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_id: int,
        world_x: float,
        world_y: float,
    ) -> tuple[float, float]:
        camera_position = data.cam_xpos[camera_id]
        camera_rotation = data.cam_xmat[camera_id].reshape(3, 3)
        world_point = np.asarray([world_x, world_y, CANVAS_CENTER[2] + 0.011])
        camera_point = camera_rotation.T @ (world_point - camera_position)
        depth = -camera_point[2]
        if depth <= 0:
            raise RuntimeError("Canvas corner lies behind the RGB camera")
        focal = 0.5 * CAMERA_HEIGHT / np.tan(
            np.deg2rad(model.cam_fovy[camera_id]) / 2.0
        )
        pixel_x = CAMERA_WIDTH / 2.0 + focal * camera_point[0] / depth
        pixel_y = CAMERA_HEIGHT / 2.0 - focal * camera_point[1] / depth
        return float(pixel_x), float(pixel_y)

    def process(self, raw_rgb: np.ndarray) -> CameraObservation:
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        rectified = cv2.warpPerspective(
            raw_bgr,
            self.homography,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        ink_mask = gray < INK_GRAY_THRESHOLD
        ink_mask = cv2.morphologyEx(
            ink_mask.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        ) > 0
        observed = np.full_like(rectified, 255)
        observed[ink_mask] = (25, 25, 25)
        return CameraObservation(
            raw_rgb=raw_rgb,
            rectified_bgr=rectified,
            ink_mask=ink_mask,
            observed_canvas=observed,
        )


class CameraCanvasObserver:
    """Boundary object: simulation produces RGB; perception consumes RGB only."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_width: int,
        output_height: int,
        drawing_half_x: float,
        drawing_half_y: float,
    ) -> None:
        self.camera = SimulatedRGBCamera(model)
        self.perception = CanvasPerception(
            model,
            data,
            output_width,
            output_height,
            drawing_half_x,
            drawing_half_y,
        )

    def observe(
        self,
        data: mujoco.MjData,
        simulation_ink: list[tuple[np.ndarray, np.ndarray]],
    ) -> CameraObservation:
        rgb = self.camera.render(data, simulation_ink)
        # Strict perception boundary: only RGB crosses into CanvasPerception.
        return self.perception.process(rgb)

    def render_preview(
        self,
        data: mujoco.MjData,
        simulation_ink: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Render live RGB for inspection without accepting it as evidence."""
        return self.camera.render(data, simulation_ink)

    def close(self) -> None:
        self.camera.close()
