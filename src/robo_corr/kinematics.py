"""Beginner-readable inverse kinematics for the Panda marker tip."""

from dataclasses import dataclass

import mujoco
import numpy as np


ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray
    position_error: float
    orientation_error: float
    iterations: int


class MarkerIK:
    """Damped least-squares IK for marker position and orientation."""

    def __init__(self, model: mujoco.MjModel, initial_qpos: np.ndarray) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.tip_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "marker_tip"
        )
        if self.tip_id < 0:
            raise RuntimeError("Model has no marker_tip site")

        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ARM_JOINT_NAMES
        ]
        self.qpos_ids = np.asarray([model.jnt_qposadr[jid] for jid in joint_ids])
        self.dof_ids = np.asarray([model.jnt_dofadr[jid] for jid in joint_ids])
        self.lower = model.jnt_range[joint_ids, 0] + 1e-4
        self.upper = model.jnt_range[joint_ids, 1] - 1e-4

        self.data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, self.data)
        self.target_rotation = self.data.site_xmat[self.tip_id].reshape(3, 3).copy()
        self.rest_qpos = initial_qpos[self.qpos_ids].copy()

    @staticmethod
    def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Small-angle orientation error expressed in world coordinates."""
        return 0.5 * sum(
            np.cross(current[:, axis], target[:, axis]) for axis in range(3)
        )

    def solve(self, target_position: np.ndarray, seed_qpos: np.ndarray) -> IKResult:
        self.data.qpos[:] = seed_qpos
        damping = 5e-4
        max_iterations = 120

        for iteration in range(1, max_iterations + 1):
            mujoco.mj_forward(self.model, self.data)
            current_rotation = self.data.site_xmat[self.tip_id].reshape(3, 3)
            position_error = target_position - self.data.site_xpos[self.tip_id]
            rotation_error = self._rotation_error(current_rotation, self.target_rotation)

            if np.linalg.norm(position_error) < 1.5e-3 and np.linalg.norm(rotation_error) < 6e-2:
                return IKResult(
                    self.data.qpos.copy(),
                    float(np.linalg.norm(position_error)),
                    float(np.linalg.norm(rotation_error)),
                    iteration,
                )

            jac_position = np.zeros((3, self.model.nv))
            jac_rotation = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(
                self.model, self.data, jac_position, jac_rotation, self.tip_id
            )
            orientation_weight = 0.08
            jacobian = np.vstack(
                (
                    jac_position[:, self.dof_ids],
                    orientation_weight * jac_rotation[:, self.dof_ids],
                )
            )
            error = np.concatenate((position_error, orientation_weight * rotation_error))
            system = jacobian @ jacobian.T + damping * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(system, error)

            delta = np.clip(delta, -0.08, 0.08)
            self.data.qpos[self.qpos_ids] = np.clip(
                self.data.qpos[self.qpos_ids] + delta, self.lower, self.upper
            )

        mujoco.mj_forward(self.model, self.data)
        position_error = target_position - self.data.site_xpos[self.tip_id]
        rotation_error = self._rotation_error(
            self.data.site_xmat[self.tip_id].reshape(3, 3), self.target_rotation
        )
        raise RuntimeError(
            "IK did not converge: "
            f"position error={np.linalg.norm(position_error):.4f} m, "
            f"orientation error={np.linalg.norm(rotation_error):.4f} rad. "
            "Try a smaller drawing nearer the window center."
        )
