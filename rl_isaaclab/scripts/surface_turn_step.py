"""One explicit step from FREE, with mass, inertia, and friction left out.

FREE's ExplicitModel uses
    b_object = mass * gravity,
    Q_object = inertia,
    contact rows = normal + mu * tangent.
Those three inputs are not passed in. The object block of Q is the identity,
so the inverse stays defined and no inertia number enters. Each contact keeps
one normal row. The gap on that row is the Gaussian-process surface value.

The joint command is an increment, as in FREE's env.step (current joints + u).
Quaternion integration follows FREE's 1/2 factor, in the world frame, because
the Sharpa body Jacobian is a world Jacobian. FREE's own matrix is body-frame
and wxyz; this file converts from the xyzw layout used in the Sharpa logs.
"""

import numpy as np
from scipy.optimize import minimize

SIGMA = 0.5
H = 0.1
BETA = 100.0
FORCE_SCALE = 0.1
TURN_STEP = 0.05


def _normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_dot_world_xyzw(quat, omega_dt):
    """World-frame quaternion increment for an angular step omega * dt. xyzw."""
    x, y, z, w = (float(v) for v in quat)
    wx, wy, wz = (float(v) for v in omega_dt)
    return 0.5 * np.array(
        [
            w * wx + y * wz - z * wy,
            w * wy + z * wx - x * wz,
            w * wz + x * wy - y * wx,
            -x * wx - y * wy - z * wz,
        ]
    )


def quat_mul_xyzw(a, b):
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def axis_from_normals(normals):
    """Axis most perpendicular to the contact normals."""
    stacked = np.stack([n / np.linalg.norm(n) for n in normals], axis=0)
    moment = stacked.T @ stacked
    values, vectors = np.linalg.eigh(moment)
    axis = vectors[:, 0]
    if axis[2] < 0.0:
        axis = -axis
    return axis, float(values[0])


def turn_target(quat, axis):
    half = 0.5 * TURN_STEP
    dq = np.array([*(np.sin(half) * axis), np.cos(half)])
    return _normalize(quat_mul_xyzw(dq, quat))


def yaw_about(quat, q0, axis):
    x, y, z, w = (float(v) for v in q0)
    conj = np.array([-x, -y, -z, w])
    rel = quat_mul_xyzw(quat, conj)
    xyz = rel[:3]
    norm_v = float(np.linalg.norm(xyz))
    if norm_v < 1e-8:
        return 0.0
    angle = 2.0 * np.arctan2(norm_v, float(rel[3]))
    return float(angle * np.dot(xyz, axis) / norm_v)


def explicit_step(obj_pos, quat, joints, u, phi, jac):
    ju = jac[:, 6:] @ u
    raw = -SIGMA * (ju + phi) - FORCE_SCALE * SIGMA * ju / H
    force = np.log1p(np.exp(np.clip(BETA * raw, -60.0, 60.0))) / BETA
    next_pos = obj_pos + jac[:, 0:3].T @ force
    next_quat = _normalize(quat + quat_dot_world_xyzw(quat, jac[:, 3:6].T @ force))
    next_joints = joints + u + jac[:, 6:].T @ force
    return next_pos, next_quat, next_joints


def plan_once(obj_pos, quat, joints, phi, jac, axis, lower, upper):
    """One planning step. Target position stays where the object is now."""
    target_q = turn_target(quat, axis)
    target_p = np.array(obj_pos, dtype=np.float64)

    def cost(u):
        pos, pred_q, _ = explicit_step(obj_pos, quat, joints, u, phi, jac)
        position_cost = float(np.sum((pos - target_p) ** 2))
        quat_cost = 1.0 - float(np.dot(_normalize(pred_q), target_q) ** 2)
        return 10.0 * (100.0 * position_cost + 5.0 * quat_cost) + 0.1 * float(np.sum(u ** 2))

    guess = np.zeros(len(joints))
    bounds = list(zip(lower.tolist(), upper.tolist()))
    solved = minimize(cost, guess, method="L-BFGS-B", bounds=bounds, options={"maxiter": 40})
    return solved.x, float(solved.fun), target_q
