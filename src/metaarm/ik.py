"""Full-joint reference IK for inspecting the simulator's Cartesian geometry."""

import time

import mujoco
import numpy as np
from scipy.optimize import LinearConstraint, minimize


def solve_ik(
    model: mujoco.MjModel,
    kinematics: mujoco.MjData,
    target_pose: np.ndarray,
    q_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve one pose with seven joint variables; never substitute for reduced IK."""
    # Validate the explicit base-frame target and local initial configuration.
    target_pose = np.asarray(target_pose, dtype=np.float64)
    q = np.asarray(q_seed, dtype=np.float64).copy()
    if target_pose.shape != (7,) or not np.isfinite(target_pose).all():
        raise ValueError("Target pose must contain finite xyz and unit wxyz coordinates.")
    if not np.isclose(np.linalg.norm(target_pose[3:]), 1.0, atol=1e-6):
        raise ValueError("The target quaternion must have unit norm.")
    joints = np.array([model.joint(f"fr3_joint{i}").id for i in range(1, 8)])
    q_indices = model.jnt_qposadr[joints]
    v_indices = model.jnt_dofadr[joints]
    bounds = model.jnt_range[joints]
    if (
        q.shape != (7,)
        or not np.isfinite(q).all()
        or not np.all((q >= bounds[:, 0]) & (q <= bounds[:, 1]))
    ):
        raise ValueError("The IK seed must be a finite configuration inside the joint limits.")
    site = model.site("attachment_site").id
    jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    quaternion, rotation_error = np.empty(4), np.empty(3)
    mujoco.mj_resetDataKeyframe(model, kinematics, model.key("home").id)

    # Use position in metres and the base-aligned principal rotation vector in radians.
    for _ in range(150):
        kinematics.qpos[q_indices] = q
        mujoco.mj_forward(model, kinematics)
        mujoco.mju_mat2Quat(quaternion, kinematics.site_xmat[site])
        mujoco.mju_subQuat(rotation_error, target_pose[3:], quaternion)
        rotation_matrix = kinematics.site_xmat[site].reshape(3, 3)
        error = np.r_[
            target_pose[:3] - kinematics.site_xpos[site], rotation_matrix @ rotation_error
        ]
        residual = np.array([np.linalg.norm(error[:3]), np.linalg.norm(error[3:])])
        if residual[0] < 1e-4 and residual[1] < 1e-3:
            return q.copy(), residual
        mujoco.mj_jacSite(model, kinematics, jac_p, jac_r, site)
        jacobian = np.vstack((jac_p[:, v_indices], jac_r[:, v_indices]))

        # Damping regularizes the declared reference method; increments stay feasible.
        increment = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), error)
        increment_norm = np.linalg.norm(increment)
        if increment_norm < 1e-12:
            raise RuntimeError(f"Reference IK stalled; residuals {residual}.")
        increment *= min(1.0, 0.12 / increment_norm)
        direction_limits = np.full(7, np.inf)
        positive, negative = increment > 0, increment < 0
        direction_limits[positive] = (bounds[positive, 1] - q[positive]) / increment[positive]
        direction_limits[negative] = (bounds[negative, 0] - q[negative]) / increment[negative]
        fraction = min(1.0, 0.99 * np.min(direction_limits))
        if fraction <= 1e-10:
            raise RuntimeError(f"Reference IK reached a joint limit; residuals {residual}.")
        q += fraction * increment
    raise RuntimeError(f"Reference IK did not converge; position/rotation residuals {residual}.")


def solve_position(
    model: mujoco.MjModel,
    kinematics: mujoco.MjData,
    target_position: np.ndarray,
    q_seed: np.ndarray,
) -> dict:
    """Construct planar labels with explicit full-joint, orientation-free SLSQP."""
    # Full-joint coordinates are the identity-basis comparison, not a reduced rescue.
    target = np.asarray(target_position, dtype=np.float64)
    seed = np.asarray(q_seed, dtype=np.float64)
    joints = np.array([model.joint(f"fr3_joint{i}").id for i in range(1, 8)])
    q_indices, v_indices = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    bounds = model.jnt_range[joints]
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("The position target must be a finite base-frame three-vector.")
    if (
        seed.shape != (7,)
        or not np.isfinite(seed).all()
        or not np.all((seed >= bounds[:, 0]) & (seed <= bounds[:, 1]))
    ):
        raise ValueError("The position-reference seed must satisfy the joint limits.")
    site = model.site("attachment_site").id
    jacobian = np.empty((3, model.nv))
    mujoco.mj_resetDataKeyframe(model, kinematics, model.key("home").id)

    def objective(q):
        # Normalize metres by the declared tolerance and differentiate exact position FK.
        if not np.isfinite(q).all():
            raise FloatingPointError("Non-finite position-reference iterate.")
        kinematics.qpos[q_indices] = q
        mujoco.mj_forward(model, kinematics)
        mujoco.mj_jacSite(model, kinematics, jacobian, None, site)
        error = (kinematics.site_xpos[site] - target) / 1e-3
        gradient = (jacobian[:, v_indices] / 1e-3).T @ error
        return 0.5 * error @ error, gradient

    # Keep the original optimizer result and task feasibility as separate evidence.
    started = time.perf_counter()
    result = minimize(
        objective,
        seed.copy(),
        jac=True,
        method="SLSQP",
        constraints=LinearConstraint(np.eye(7), bounds[:, 0], bounds[:, 1]),
        options={"maxiter": 200, "ftol": 1e-10},
    )
    seconds = time.perf_counter() - started
    value, _ = objective(result.x)
    if not np.isfinite(value) or not np.isfinite(result.x).all():
        raise FloatingPointError("Non-finite position-reference result.")
    return {
        "q": result.x.copy(),
        "alpha": result.x - seed,
        "errors": np.array([np.sqrt(2 * value) * 1e-3]),
        "feasible": bool(np.all((result.x >= bounds[:, 0]) & (result.x <= bounds[:, 1]))),
        "optimizer": result,
        "seconds": seconds,
    }
