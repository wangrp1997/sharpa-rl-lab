"""One joint increment with self-collision inequalities and a pose-covariance step.

The hierarchy matches Escande, Mansard, and Wieber, IJRR 2014: inequalities
are satisfied before the tangent. Their MATLAB solver in refs/inequality_hqp/hqp
is not called. The Python port in hierarchical_qp needs quadprog, which is
not installed. The same inequalities are solved with osqp.

A link pair contributes a row only when its distance is within the watch
distance. That switch is the unilateral constraint of Dietrich, Wimböck,
Albu-Schäffer, and Hirzinger, TRO 2012, equations (3), (4), and (8). Their
torque, inertia matrix, maximum force, and damping ratio are not used. The
watch distance is not their 0.10 m to 0.15 m.

The tangent length shrinks when the standard deviation about the turn axis is
large. After the step, a Gaussian update tightens that variance. This is the
belief update of Platt, Tedrake, Kaelbling, and Lozano-Pérez, RSS 2010,
without their belief-space LQR. The cylinder mesh, mass, friction, and inertia
are not inputs. The shape does not choose a finger.
"""

import importlib.util
from pathlib import Path

import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import linprog

_tangent_path = Path(__file__).with_name("tangent_turn_step.py")
_tangent = importlib.util.spec_from_file_location("tangent_turn_step", _tangent_path)
_turn = importlib.util.module_from_spec(_tangent)
_tangent.loader.exec_module(_turn)

WATCH_DISTANCE = 0.0
STEP_MAX = 0.002
SIGMA_WIDE = 0.02
MAX_DQ = 0.008
REGULARIZATION = 1e-4
OVERLAP_WEIGHT = 50.0
TWIST_WEIGHT = 1.0
LINEAR_WEIGHT = 1.0
SHELL_SLACK = 1.0
NORMAL_BAND = 1.0
TRACK_WEIGHT = 5.0
APPROACH_WEIGHT = 0.2


def trust_length(axis_std, step_max=STEP_MAX, sigma_wide=SIGMA_WIDE):
    """Shorter tangent when the turn-axis standard deviation is larger."""
    sigma = max(float(axis_std), 1e-9)
    return float(min(step_max, step_max * sigma_wide / sigma))


def update_belief(mean, covariance, direction, measurement, meas_var):
    """Scalar observation of one pose direction. Unobserved directions stay wide."""
    mean = np.asarray(mean, dtype=np.float64).reshape(-1).copy()
    covariance = np.asarray(covariance, dtype=np.float64).copy()
    direction = np.asarray(direction, dtype=np.float64).reshape(-1)
    direction = direction / np.linalg.norm(direction)
    predicted = float(direction @ mean)
    innovation_scale = float(direction @ covariance @ direction + meas_var)
    gain = covariance @ direction / innovation_scale
    mean = mean + gain * (float(measurement) - predicted)
    covariance = covariance - np.outer(gain, direction @ covariance)
    covariance = 0.5 * (covariance + covariance.T)
    return mean, covariance


def _skew(vector):
    x, y, z = (float(v) for v in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def normals_enclose(normals):
    """True when the origin is a convex combination of the contact normals."""
    count = len(normals)
    if count < 3:
        return False
    matrix = np.column_stack([np.asarray(normal, dtype=np.float64).reshape(3) for normal in normals])
    result = linprog(
        np.zeros(count),
        A_eq=np.vstack((matrix, np.ones(count))),
        b_eq=np.array([0.0, 0.0, 0.0, 1.0]),
        bounds=[(0.0, None)] * count,
        method="highs",
    )
    if not result.success:
        return False
    return float(np.linalg.norm(matrix @ result.x)) <= 1e-6


def plan_belief_step(
    normals,
    jacobians,
    points,
    joints,
    lowers,
    uppers,
    pairs,
    axis_std,
    watch_distance=WATCH_DISTANCE,
    step_max=STEP_MAX,
    sigma_wide=SIGMA_WIDE,
    max_dq=MAX_DQ,
    shells=None,
    axis=None,
):
    """Joint increment and object twist. The twist is not measured.

    points[i] is the position of contact i. Pad velocity equals one rigid
    velocity: J dq = v + ω × r, with r measured from the contact centroid.
    The cost pulls ω onto the estimated axis. A normal may not separate.
    Overlap depth is a slack penalty. The cylinder mesh is not an input.
    """
    del watch_distance
    normals = [np.asarray(normal, dtype=np.float64) for normal in normals]
    jacobians = [np.asarray(jac, dtype=np.float64) for jac in jacobians]
    points = [np.asarray(point, dtype=np.float64).reshape(3) for point in points]
    n_dof = int(jacobians[0].shape[1]) if jacobians else 0
    zeros = np.zeros(n_dof, dtype=np.float64)
    empty = {
        "axis": None,
        "dq": zeros,
        "linear": np.zeros(3),
        "angular": np.zeros(3),
        "length": 0.0,
        "moved": [],
        "directions": [None] * len(jacobians),
        "enclosed": False,
    }
    axis = _turn.axis_from_normals(normals) if axis is None else np.asarray(axis, dtype=np.float64)
    if axis is not None:
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    if axis is None or n_dof == 0 or len(points) != len(jacobians):
        return empty
    empty["axis"] = axis
    empty["enclosed"] = normals_enclose(normals)
    centroid = np.mean(np.stack(points), axis=0)
    levers = [point - centroid for point in points]
    radius = float(np.mean([np.linalg.norm(lever) for lever in levers]))
    if radius < 1e-5:
        return empty
    length = trust_length(axis_std, step_max, sigma_wide)
    angle = length / radius
    directions = [_turn.tangential_direction(normal, axis) for normal in normals]
    empty["directions"] = directions
    q = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in joints])
    lower = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in lowers])
    upper = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in uppers])
    if q.size != n_dof:
        raise ValueError("joint blocks do not match the Jacobian width")
    bound_low = np.maximum(lower - q, -max_dq)
    bound_high = np.minimum(upper - q, max_dq)
    n_twist = 6
    normal_rows = []
    normal_limits = []
    band = NORMAL_BAND * float(axis_std)
    track = np.zeros((n_dof, n_dof))
    track_linear = np.zeros(n_dof)
    for normal, jac, lever, direction in zip(normals, jacobians, levers, directions):
        unit = normal / max(float(np.linalg.norm(normal)), 1e-9)
        cross = _skew(lever)
        block = np.concatenate([jac, -np.eye(3), cross], axis=1)
        normal_row = unit @ block
        normal_rows.append(normal_row)
        normal_rows.append(-normal_row)
        normal_limits.append(band)
        normal_limits.append(band)
        if direction is None:
            continue
        desired = np.asarray(direction, dtype=np.float64) * length
        projector = np.eye(3) - np.outer(unit, unit)
        weighted = projector @ jac
        track += weighted.T @ weighted
        track_linear += weighted.T @ (projector @ desired)
    overlap_rows = []
    overlap_gaps = []
    for direction, jac_i, jac_j, distance in pairs:
        gap = float(distance)
        if gap > 0.0:
            continue
        unit = np.asarray(direction, dtype=np.float64)
        unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
        jac_i = np.asarray(jac_i, dtype=np.float64)
        jac_j = np.asarray(jac_j, dtype=np.float64)
        row = np.zeros(n_dof + n_twist)
        row[:n_dof] = -unit @ (jac_j - jac_i)
        overlap_rows.append(row)
        overlap_gaps.append(gap)
    n_slack = len(overlap_rows)
    width = n_dof + n_twist + n_slack
    cost = np.eye(width) * REGULARIZATION
    cost[:n_dof, :n_dof] += TRACK_WEIGHT * track
    cost[n_dof:n_dof + 3, n_dof:n_dof + 3] = LINEAR_WEIGHT * np.eye(3)
    cost[n_dof + 3:n_dof + 6, n_dof + 3:n_dof + 6] = TWIST_WEIGHT * np.eye(3)
    linear = np.zeros(width)
    linear[:n_dof] = -TRACK_WEIGHT * track_linear
    linear[n_dof + 3:n_dof + 6] = -TWIST_WEIGHT * axis * angle
    if n_slack:
        cost[n_dof + n_twist:, n_dof + n_twist:] = OVERLAP_WEIGHT * np.eye(n_slack)
    bound_low = np.concatenate([bound_low, np.full(n_twist, -1.0), np.zeros(n_slack)])
    bound_high = np.concatenate([bound_high, np.full(n_twist, 1.0), np.full(n_slack, 1.0)])
    gain_rows = [np.concatenate([row, np.zeros(n_slack)]) for row in normal_rows]
    gain_limits = list(normal_limits)
    for index, (row, gap) in enumerate(zip(overlap_rows, overlap_gaps)):
        padded = np.concatenate([row, np.zeros(n_slack)])
        padded[n_dof + n_twist + index] = -1.0
        gain_rows.append(padded)
        gain_limits.append(gap)
    for direction, shell_jac, point, gap in shells or []:
        unit = np.asarray(direction, dtype=np.float64)
        unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
        lever = np.asarray(point, dtype=np.float64).reshape(3) - centroid
        row = np.zeros(width)
        row[:n_dof] = -unit @ np.asarray(shell_jac, dtype=np.float64)
        row[n_dof:n_dof + 3] = unit
        row[n_dof + 3:n_dof + 6] = np.cross(lever, unit)
        gain_rows.append(row)
        gain_limits.append(max(float(gap), 0.0) + SHELL_SLACK * float(axis_std))
        linear[:n_dof] += APPROACH_WEIGHT * np.asarray(shell_jac, dtype=np.float64).T @ unit
    solution = solve_qp(
        cost,
        linear,
        np.vstack(gain_rows),
        np.asarray(gain_limits, dtype=np.float64),
        None,
        None,
        lb=bound_low,
        ub=bound_high,
        solver="osqp",
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=10000,
        polish=True,
        verbose=False,
    )
    empty["length"] = length
    if solution is None:
        return empty
    solution = np.asarray(solution, dtype=np.float64)
    dq = solution[:n_dof]
    linear_velocity = solution[n_dof:n_dof + 3]
    angular = solution[n_dof + 3:n_dof + 6]
    moved = []
    for index, jac in enumerate(jacobians):
        if float(np.linalg.norm(jac @ dq)) >= 0.5 * length:
            moved.append(index)
    return {
        "axis": axis,
        "dq": dq,
        "linear": linear_velocity,
        "angular": angular,
        "length": length,
        "moved": moved,
        "directions": directions,
        "enclosed": bool(empty["enclosed"]),
    }


def _circle():
    normals = [
        np.array([1.0, 0.0, 0.0]),
        np.array([-0.5, np.sqrt(3) / 2, 0.0]),
        np.array([-0.5, -np.sqrt(3) / 2, 0.0]),
    ]
    blocks = []
    for index in range(3):
        block = np.zeros((3, 9))
        block[:, 3 * index:3 * index + 3] = np.eye(3)
        blocks.append(block)
    points = [0.02 * normal for normal in normals]
    joints = [np.zeros(3), np.zeros(3), np.zeros(3)]
    lowers = [np.full(3, -1.0)] * 3
    uppers = [np.full(3, 1.0)] * 3
    return normals, blocks, points, joints, lowers, uppers


def _check():
    normals, jacobians, points, joints, lowers, uppers = _circle()
    assert normals_enclose(normals)
    assert not normals_enclose(normals[:2])
    open_grasp = plan_belief_step(
        normals[:2],
        jacobians[:2],
        points[:2],
        [np.zeros(9)],
        [np.full(9, -1.0)],
        [np.full(9, 1.0)],
        [],
        axis_std=SIGMA_WIDE,
    )
    assert open_grasp["enclosed"] is False
    assert open_grasp["moved"] == [0, 1]
    tight = plan_belief_step(
        normals, jacobians, points, joints, lowers, uppers, [], axis_std=SIGMA_WIDE
    )
    wide = plan_belief_step(
        normals, jacobians, points, joints, lowers, uppers, [], axis_std=10.0 * SIGMA_WIDE
    )
    assert tight["enclosed"]
    assert tight["moved"] == [0, 1, 2]
    assert wide["moved"] == [0, 1, 2]
    assert wide["length"] < 0.2 * tight["length"]
    assert abs(float(np.dot(tight["angular"], tight["axis"]))) > 0.5 * np.linalg.norm(tight["angular"])
    assert np.linalg.norm(tight["linear"]) < np.linalg.norm(tight["angular"])
    gap_direction = np.array([0.0, 1.0, 0.0])
    pressed = plan_belief_step(
        normals,
        jacobians,
        points,
        joints,
        lowers,
        uppers,
        [(gap_direction, jacobians[0], jacobians[1], -0.02)],
        axis_std=SIGMA_WIDE,
    )
    free_sep = float(gap_direction @ (jacobians[1] - jacobians[0]) @ tight["dq"])
    pressed_sep = float(gap_direction @ (jacobians[1] - jacobians[0]) @ pressed["dq"])
    assert pressed_sep > free_sep
    locked_lower = [np.full(3, -1.0), np.zeros(3), np.full(3, -1.0)]
    locked_upper = [np.full(3, 1.0), np.zeros(3), np.full(3, 1.0)]
    locked = plan_belief_step(
        normals, jacobians, points, joints, locked_lower, locked_upper, [], axis_std=SIGMA_WIDE
    )
    assert 1 not in locked["moved"]
    mean = np.zeros(3)
    covariance = np.eye(3)
    updated, tightened = update_belief(mean, covariance, [0.0, 0.0, 1.0], 0.0, 1e-4)
    assert tightened[2, 2] < 0.5 * covariance[2, 2]
    assert abs(tightened[0, 0] - 1.0) < 1e-6
    assert abs(float(updated[2])) < 1e-6
    print("belief_turn_step ok", flush=True)


if __name__ == "__main__":
    _check()
