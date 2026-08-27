"""Build the first Panda, marker, and drawing-canvas inspection scene."""

from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANDA_XML = (
    PROJECT_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "panda.xml"
)

# MuJoCo box sizes are half-extents: this is a 50 x 40 cm drawing area.
CANVAS_CENTER = (0.55, 0.0, 0.405)
CANVAS_HALF_SIZE = (0.25, 0.20, 0.005)
CANVAS_CAMERA_NAME = "canvas_rgb_camera"
OBSERVATION_QPOS = (1.2, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785)


def build_model() -> mujoco.MjModel:
    """Load Menagerie's Panda and add only this project's scene geometry."""
    if not PANDA_XML.is_file():
        raise FileNotFoundError(
            f"Panda model not found at {PANDA_XML}. "
            "Follow the Menagerie setup command in README.md."
        )

    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    spec.modelname = "Panda drawing workspace"

    hand = spec.body("hand")
    if hand is None:
        raise RuntimeError("The Menagerie Panda model has no 'hand' body.")

    # The hand's local +Z axis points through the gripper. The marker starts
    # beyond the fingertips; later motion planning will use marker_tip.
    marker = hand.add_body(name="marker", pos=[0.0, 0.0, 0.10])
    marker.add_geom(
        name="marker_barrel",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.035],
        size=[0.009, 0.035, 0.0],
        rgba=[0.10, 0.16, 0.23, 1.0],
        contype=0,
        conaffinity=0,
        mass=0.02,
    )
    marker.add_geom(
        name="marker_nib",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[0.0, 0.0, 0.073],
        size=[0.004, 0.005, 0.0],
        rgba=[0.02, 0.02, 0.02, 1.0],
        contype=0,
        conaffinity=0,
        mass=0.002,
    )
    marker.add_site(
        name="marker_tip",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[0.0, 0.0, 0.078],
        size=[0.006, 0.0, 0.0],
        rgba=[0.90, 0.12, 0.08, 1.0],
    )

    world = spec.worldbody
    world.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[2.0, 2.0, 0.05],
        rgba=[0.16, 0.19, 0.23, 1.0],
        contype=1,
        conaffinity=1,
    )

    # A low table supports the thin white drawing surface.
    world.add_geom(
        name="canvas_support",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[CANVAS_CENTER[0], CANVAS_CENTER[1], 0.20],
        size=[0.29, 0.24, 0.20],
        rgba=[0.36, 0.25, 0.17, 1.0],
        contype=1,
        conaffinity=1,
    )
    world.add_geom(
        name="canvas",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=CANVAS_CENTER,
        size=CANVAS_HALF_SIZE,
        rgba=[0.94, 0.94, 0.90, 1.0],
        contype=1,
        conaffinity=1,
    )
    world.add_site(
        name="canvas_center",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[CANVAS_CENTER[0], CANVAS_CENTER[1], CANVAS_CENTER[2] + 0.011],
        size=[0.008, 0.0, 0.0],
        rgba=[0.12, 0.45, 0.95, 0.0],
    )
    world.add_light(
        name="workspace_light",
        pos=[0.4, -0.2, 1.8],
        dir=[0.0, 0.1, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.85, 0.85, 0.85],
    )
    # Fixed overhead camera. Its calibrated projection is rectified in
    # perception.py; it never moves in response to the drawing.
    world.add_camera(
        name=CANVAS_CAMERA_NAME,
        pos=[CANVAS_CENTER[0], CANVAS_CENTER[1], 1.35],
        quat=[1.0, 0.0, 0.0, 0.0],
        fovy=42.0,
    )

    return spec.compile()


def make_home_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Create simulation data at Menagerie's stable, controlled home pose."""
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise RuntimeError("The Menagerie Panda model has no 'home' keyframe.")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    return data
