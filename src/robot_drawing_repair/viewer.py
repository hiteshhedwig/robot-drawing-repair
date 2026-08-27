"""Open an interactive MuJoCo window for mechanical inspection."""

import time

import mujoco
import mujoco.viewer

from robot_drawing_repair.scene import build_model, make_home_data


def main() -> None:
    model = build_model()
    data = make_home_data(model)

    print("Opening Panda drawing-workspace inspection scene.")
    print("Red site: marker tip. Blue site: canvas center.")
    print("Close the MuJoCo window or press Ctrl+C here to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.38, 0.0, 0.45]
        viewer.cam.distance = 1.65
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -25.0

        while viewer.is_running():
            step_start = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = model.opt.timestep - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()

