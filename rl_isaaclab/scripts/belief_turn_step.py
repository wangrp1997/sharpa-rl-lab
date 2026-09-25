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
LEAVE_ROOM = 0.004
PUSH_FRACTION = 0.5
PUSH_WEIGHT = 30.0
FORCE_WEIGHT = 0.02


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


def _tangent_basis(normal):
    unit = np.asarray(normal, dtype=np.float64)
    unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(unit[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(unit, helper)
    first = first / max(float(np.linalg.norm(first)), 1e-9)
    second = np.cross(unit, first)
    return first, second


def _overlap_rows(pairs):
    rows = []
    gaps = []
    for pair_dir, jac_i, jac_j, distance in pairs:
        gap = float(distance)
        if gap > 0.0:
            continue
        unit = np.asarray(pair_dir, dtype=np.float64)
        unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
        rows.append(-unit @ (np.asarray(jac_j, dtype=np.float64) - np.asarray(jac_i, dtype=np.float64)))
        gaps.append(gap)
    return rows, gaps


def support_holds(points, center, radius):
    """True when remaining contacts still surround the object center.

    Sundaralingam and Hermans, ICRA 2018, Section III, move one finger only
    while the other fingers keep the object. The paper requires the object
    mesh, the hand kinematics, an initial grasp, and a desired grasp, and
    it chooses the finger in an outer sequence. This test uses the mesh
    radius and the pose center. It does not run that sequence.
    """
    pts = [np.asarray(point, dtype=np.float64) for point in points]
    if len(pts) < 2:
        return False
    center = np.asarray(center, dtype=np.float64)
    flat = np.array([[point[0] - center[0], point[1] - center[1]] for point in pts])
    if len(pts) == 2:
        start, end = flat
        edge = end - start
        weight = float(np.clip(-(start @ edge) / (float(edge @ edge) + 1e-12), 0.0, 1.0))
        return float(np.linalg.norm(start + weight * edge)) <= float(radius)
    result = linprog(
        np.zeros(len(pts)),
        A_eq=np.vstack((flat.T, np.ones(len(pts)))),
        b_eq=np.zeros(3),
        bounds=[(0.0, None)] * len(pts),
        method="highs",
    )
    return bool(result.success)


def plan_rigid_step(
    jacobians,
    radii,
    normals,
    stay,
    directions,
    axis,
    joints,
    lowers,
    uppers,
    pairs,
    max_dq=MAX_DQ,
    object_shift=None,
):
    """One damped step whose contacting fingers share a rigid twist.

    Escande, Mansard, and Wieber, IJRR 2014, solve a hierarchy: a lower
    task is optimized only inside the set that keeps higher tasks at the
    value already achieved. Their MATLAB solver and the quadprog port are
    not used. Two osqp solves follow that order. The first minimizes the
    contact mismatch. The second minimizes the joint increment without
    letting that mismatch grow. Inequalities stay in both solves.

    The paper requires a stack of quadratic tasks and linear inequalities,
    and it assumes the higher-priority optimum can be held exactly. Joint
    torque, a full inverse-dynamics model, and their equality-cascade
    solver are not required here.

    Sundaralingam and Hermans, ICRA 2018, keep the current contact points
    fixed in the object frame while the object takes one rigid velocity.
    That in-grasp condition is J dq = v + ω × r at each holding pad.
    v and ω are decision variables and are not applied to the simulated
    object. A pad that is not holding falls back to a point on the mesh
    one small angle ahead; the solver does not name that finger.

    The paper requires the object mesh, the hand kinematics, an initial
    grasp, and a desired grasp. It also plans collision-free joint paths
    and moves one finger at a time while the others stay fixed. Those
    gait sequences and the desired grasp are not inputs. The public
    in-grasp code keeps every contact and does not break contact.
    """
    jacobians = [np.asarray(jac, dtype=np.float64) for jac in jacobians]
    n_dof = int(jacobians[0].shape[1]) if jacobians else 0
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    zeros = np.zeros(n_dof, dtype=np.float64)
    empty = {
        "dq": zeros,
        "directions": list(directions),
        "moved": [],
        "stay": [bool(v) for v in stay],
        "linear": np.zeros(3),
        "angular": np.zeros(3),
    }
    if n_dof == 0 or not jacobians:
        return empty
    q = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in joints])
    lower = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in lowers])
    upper = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in uppers])
    bound_low = np.maximum(lower - q, -max_dq)
    bound_high = np.minimum(upper - q, max_dq)
    overlap_rows, overlap_gaps = _overlap_rows(pairs)
    n_fingers = len(jacobians)
    n_overlap = len(overlap_rows)
    # dq | v | ω | leave slack per finger | overlap slack
    width = n_dof + 6 + n_fingers + n_overlap
    twist_v = n_dof
    twist_w = n_dof + 3
    leave0 = n_dof + 6
    overlap0 = leave0 + n_fingers

    def pack():
        cost = np.eye(width) * 0.05
        cost[twist_v:leave0, twist_v:leave0] = np.eye(6)
        span = np.maximum(upper - lower, 1e-6)
        margin = np.minimum(q - lower, upper - q) / span
        for index, room in enumerate(margin):
            # Fan, Tang, Lin, Zhao, and Tomizuka, arXiv:1710.10350, treat a
            # joint near its limit as less able to keep the grasp. Their
            # planner then picks which finger to lift. That choice is not
            # made here; the same limit distance only increases damping.
            # The paper requires the remaining contacts' convex hull and a
            # finger-lift search outside the solver.
            cost[index, index] += 1.0 / max(float(room), 0.05)
        linear = np.zeros(width)
        if object_shift is None:
            linear[twist_w:twist_w + 3] = -0.05 * axis
        else:
            # Move the cylinder toward pads that are not yet on the mesh.
            # The pose supplies the center and the mesh supplies the surface
            # point. Sundaralingam and Hermans, ICRA 2018, require the mesh
            # and the hand kinematics before a finger is lifted. Contact is
            # established first; v is not written onto the simulated object.
            shift = np.asarray(object_shift, dtype=np.float64)
            linear[twist_v:twist_v + 3] = -shift
        rows = []
        limits = []
        match_rows = []
        for finger, (jac, radius, normal, holding, direction) in enumerate(
            zip(jacobians, radii, normals, stay, directions)
        ):
            unit = np.asarray(normal, dtype=np.float64)
            unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
            if holding:
                lever = np.asarray(radius, dtype=np.float64)
                for tangent in _tangent_basis(unit):
                    match = np.zeros(width)
                    match[:n_dof] = tangent @ jac
                    match[twist_v:twist_v + 3] = -tangent
                    match[twist_w:twist_w + 3] = tangent @ _skew(lever)
                    match_rows.append(match)
                    cost += 800.0 * np.outer(match, match)
                penetrate = np.zeros(width)
                penetrate[:n_dof] = -unit @ jac
                rows.append(penetrate)
                limits.append(4e-4)
                separate = np.zeros(width)
                separate[:n_dof] = unit @ jac
                separate[leave0 + finger] = -1.0
                rows.append(separate)
                limits.append(4e-4)
                cost[leave0 + finger, leave0 + finger] = 40.0
            elif direction is not None:
                tangent = np.asarray(direction, dtype=np.float64)
                tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
                gain = tangent @ jac
                linear[:n_dof] -= 0.02 * gain
                cap = np.zeros(width)
                cap[:n_dof] = gain
                rows.append(cap)
                limits.append(0.002)
        for index, (row, gap) in enumerate(zip(overlap_rows, overlap_gaps)):
            padded = np.zeros(width)
            padded[:n_dof] = row
            padded[overlap0 + index] = -1.0
            rows.append(padded)
            limits.append(gap)
            cost[overlap0 + index, overlap0 + index] = OVERLAP_WEIGHT
        spin = 0.0 if object_shift is not None else 0.2
        low = np.concatenate([
            bound_low,
            np.full(3, -0.005),
            np.full(3, -spin),
            np.zeros(n_fingers + n_overlap),
        ])
        high = np.concatenate([
            bound_high,
            np.full(3, 0.005),
            np.full(3, spin),
            np.full(n_fingers, 0.01),
            np.ones(n_overlap),
        ])
        return cost, linear, rows, limits, low, high, match_rows

    cost, linear, rows, limits, low, high, match_rows = pack()
    if not rows and not match_rows:
        return empty

    def solve(cost_matrix, linear_term, extra_rows, extra_limits):
        ineq = rows + extra_rows
        bound = limits + extra_limits
        if not ineq:
            return None
        return solve_qp(
            cost_matrix,
            linear_term,
            np.vstack(ineq),
            np.asarray(bound, dtype=np.float64),
            None,
            None,
            lb=low,
            ub=high,
            solver="osqp",
            eps_abs=1e-8,
            eps_rel=1e-8,
            max_iter=10000,
            polish=True,
            verbose=False,
        )

    first = solve(cost, linear, [], [])
    if first is None:
        return empty
    solution = np.asarray(first, dtype=np.float64)
    dq = solution[:n_dof]
    linear_v = solution[twist_v:twist_v + 3]
    angular = solution[twist_w:twist_w + 3]
    moved = []
    for index, (jac, radius, normal, holding, direction) in enumerate(
        zip(jacobians, radii, normals, stay, directions)
    ):
        if holding:
            unit = np.asarray(normal, dtype=np.float64)
            unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
            tangent = np.cross(axis, unit)
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm < 1e-8:
                continue
            tangent = tangent / tangent_norm
            along = float(tangent @ (jac @ dq))
            wanted = abs(float(tangent @ np.cross(angular, radius)))
            if along >= max(0.3 * wanted, 1e-5):
                moved.append(index)
        elif direction is not None:
            tangent = np.asarray(direction, dtype=np.float64)
            tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
            if float(tangent @ (jac @ dq)) > 2e-4:
                moved.append(index)
    empty["dq"] = dq
    empty["moved"] = moved
    empty["linear"] = linear_v
    empty["angular"] = angular
    return empty


def plan_mesh_step(jacobians, directions, normals, stay, joints, lowers, uppers, pairs, step=0.002, max_dq=MAX_DQ):
    """Joint increment along mesh tangents. No palm target and no unused twist."""
    jacobians = [np.asarray(jac, dtype=np.float64) for jac in jacobians]
    n_dof = int(jacobians[0].shape[1]) if jacobians else 0
    zeros = np.zeros(n_dof, dtype=np.float64)
    empty = {"dq": zeros, "directions": list(directions), "moved": [], "stay": [bool(v) for v in stay]}
    if n_dof == 0:
        return empty
    q = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in joints])
    lower = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in lowers])
    upper = np.concatenate([np.asarray(block, dtype=np.float64).reshape(-1) for block in uppers])
    bound_low = np.maximum(lower - q, -max_dq)
    bound_high = np.minimum(upper - q, max_dq)
    overlap_rows = []
    overlap_gaps = []
    for pair_dir, jac_i, jac_j, distance in pairs:
        gap = float(distance)
        if gap > 0.0:
            continue
        unit = np.asarray(pair_dir, dtype=np.float64)
        unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
        row = -unit @ (np.asarray(jac_j, dtype=np.float64) - np.asarray(jac_i, dtype=np.float64))
        overlap_rows.append(row)
        overlap_gaps.append(gap)
    n_overlap = len(overlap_rows)
    width = n_dof + n_overlap
    cost = np.eye(width) * REGULARIZATION
    linear = np.zeros(width)
    gain_rows = []
    gain_limits = []
    if np.ndim(step) == 0:
        targets = [float(step)] * len(jacobians)
    else:
        targets = [float(value) for value in step]
    for jac, direction, normal, holding, target in zip(jacobians, directions, normals, stay, targets):
        if direction is None:
            continue
        tangent = np.asarray(direction, dtype=np.float64)
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
        gain = tangent @ jac
        cost[:n_dof, :n_dof] += PUSH_WEIGHT * np.outer(gain, gain)
        linear[:n_dof] -= PUSH_WEIGHT * target * gain
        if not holding:
            continue
        unit = np.asarray(normal, dtype=np.float64)
        unit = unit / max(float(np.linalg.norm(unit)), 1e-9)
        normal_gain = unit @ jac
        positive = np.zeros(width)
        positive[:n_dof] = normal_gain
        gain_rows.append(positive)
        gain_rows.append(-positive)
        gain_limits.extend((4e-4, 4e-4))
    for index, (row, gap) in enumerate(zip(overlap_rows, overlap_gaps)):
        padded = np.zeros(width)
        padded[:n_dof] = row
        padded[n_dof + index] = -1.0
        gain_rows.append(padded)
        gain_limits.append(gap)
        cost[n_dof + index, n_dof + index] = OVERLAP_WEIGHT
    bound_low = np.concatenate([bound_low, np.zeros(n_overlap)])
    bound_high = np.concatenate([bound_high, np.ones(n_overlap)])
    if not gain_rows:
        return empty
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
    if solution is None:
        return empty
    dq = np.asarray(solution[:n_dof], dtype=np.float64)
    moved = []
    for index, (jac, direction) in enumerate(zip(jacobians, directions)):
        if direction is None:
            continue
        tangent = np.asarray(direction, dtype=np.float64)
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-9)
        asked = targets[index] if index < len(targets) else step
        if float(tangent @ (jac @ dq)) >= 0.3 * float(asked):
            moved.append(index)
    empty["dq"] = dq
    empty["moved"] = moved
    return empty


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

    points[i] is the position of contact i. Each contact has a nonnegative
    force. Only a contact that can move along axis × n keeps that force and
    pushes. A zero-force contact may separate along its normal by LEAVE_ROOM.
    The planned twist is not applied to the object. Overlap depth is a slack
    penalty. The cylinder mesh is not an input.
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
        "forces": [0.0] * len(jacobians),
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
    band = NORMAL_BAND * float(axis_std)
    pushers = []
    for index, (normal, jac, lever, direction) in enumerate(zip(normals, jacobians, levers, directions)):
        unit = normal / max(float(np.linalg.norm(normal)), 1e-9)
        if direction is None:
            pushers.append(None)
            continue
        tangent = np.asarray(direction, dtype=np.float64)
        gain = tangent @ jac
        achievable = 0.0
        for column, component in enumerate(gain):
            achievable += max(component * bound_low[column], component * bound_high[column])
        step_i = min(length, max(achievable, 0.0))
        pushers.append((index, unit, jac, lever, tangent, gain, step_i))
    n_force = sum(item is not None for item in pushers)
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
    n_overlap = len(overlap_rows)
    width = n_dof + n_twist + n_overlap + n_force
    cost = np.eye(width) * REGULARIZATION
    cost[n_dof:n_dof + 3, n_dof:n_dof + 3] = LINEAR_WEIGHT * np.eye(3)
    cost[n_dof + 3:n_dof + 6, n_dof + 3:n_dof + 6] = TWIST_WEIGHT * np.eye(3)
    linear = np.zeros(width)
    linear[n_dof + 3:n_dof + 6] = -TWIST_WEIGHT * axis * angle
    if n_overlap:
        cost[n_dof + n_twist:n_dof + n_twist + n_overlap, n_dof + n_twist:n_dof + n_twist + n_overlap] = (
            OVERLAP_WEIGHT * np.eye(n_overlap)
        )
    bound_low = np.concatenate([bound_low, np.full(n_twist, -1.0), np.zeros(n_overlap), np.zeros(n_force)])
    bound_high = np.concatenate([bound_high, np.full(n_twist, 1.0), np.ones(n_overlap), np.ones(n_force)])
    gain_rows = []
    gain_limits = []
    force_slot = 0
    for item, normal, jac, lever in zip(pushers, normals, jacobians, levers):
        unit = normal / max(float(np.linalg.norm(normal)), 1e-9)
        block = np.concatenate([jac, -np.eye(3), _skew(lever)], axis=1)
        normal_gain = unit @ block
        penetrate = np.zeros(width)
        penetrate[:n_dof + n_twist] = -normal_gain
        gain_rows.append(penetrate)
        gain_limits.append(band)
        separate = np.zeros(width)
        separate[:n_dof + n_twist] = normal_gain
        if item is None:
            gain_rows.append(separate)
            gain_limits.append(band)
            continue
        _index, _unit, _jac, _lever, _tangent, gain, step_i = item
        separate[n_dof + n_twist + n_overlap + force_slot] = LEAVE_ROOM
        gain_rows.append(separate)
        gain_limits.append(band + LEAVE_ROOM)
        upper = np.zeros(width)
        upper[:n_dof] = gain
        upper[n_dof + n_twist + n_overlap + force_slot] = -step_i
        gain_rows.append(upper)
        gain_limits.append(0.0)
        lower_push = np.zeros(width)
        lower_push[:n_dof] = -gain
        lower_push[n_dof + n_twist + n_overlap + force_slot] = PUSH_FRACTION * step_i
        gain_rows.append(lower_push)
        gain_limits.append(0.0)
        linear[:n_dof] -= PUSH_WEIGHT * gain
        linear[n_dof + n_twist + n_overlap + force_slot] = FORCE_WEIGHT
        force_slot += 1
    for index, (row, gap) in enumerate(zip(overlap_rows, overlap_gaps)):
        padded = np.zeros(width)
        padded[:n_dof + n_twist] = row
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
    force_values = np.zeros(len(jacobians))
    force_slot = 0
    moved = []
    for item in pushers:
        if item is None:
            continue
        index, _unit, jac, _lever, tangent, _gain, step_i = item
        force = float(solution[n_dof + n_twist + n_overlap + force_slot])
        force_values[index] = force
        force_slot += 1
        along = float(tangent @ (jac @ dq))
        if force > 0.2 and along >= 0.3 * max(step_i, 1e-6):
            moved.append(index)
    return {
        "axis": axis,
        "dq": dq,
        "linear": linear_velocity,
        "angular": angular,
        "length": length,
        "moved": moved,
        "directions": directions,
        "forces": force_values.tolist(),
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
    assert max(wide["forces"]) < 0.2
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
    jac_a = np.zeros((3, 4))
    jac_a[1, 0] = 1.0
    jac_a[0, 1] = 1.0
    jac_b = np.zeros((3, 4))
    jac_b[1, 2] = -1.0
    jac_b[0, 3] = 1.0
    radius = 0.02
    rigid = plan_rigid_step(
        [jac_a, jac_b],
        [np.array([radius, 0.0, 0.0]), np.array([-radius, 0.0, 0.0])],
        [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])],
        [True, True],
        [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])],
        np.array([0.0, 0.0, 1.0]),
        [np.zeros(4)],
        [np.full(4, -1.0)],
        [np.full(4, 1.0)],
        [],
    )
    assert float(rigid["dq"][0]) > 1e-4
    assert float(rigid["dq"][2]) > 1e-4
    assert abs(float(rigid["dq"][1])) < 0.5 * float(rigid["dq"][0])
    assert abs(float(rigid["dq"][3])) < 0.5 * float(rigid["dq"][2])
    assert abs(float(rigid["angular"][2])) > abs(float(rigid["angular"][0]))
    held = [np.array([0.02, 0.0, 0.0]), np.array([-0.02, 0.0, 0.0])]
    assert support_holds(held, np.zeros(3), 0.02)
    assert not support_holds([np.array([0.02, 0.0, 0.0])], np.zeros(3), 0.02)
    assert not support_holds(
        [np.array([0.02, 0.02, 0.0]), np.array([0.02, -0.02, 0.0])],
        np.zeros(3),
        0.005,
    )
    mean = np.zeros(3)
    covariance = np.eye(3)
    updated, tightened = update_belief(mean, covariance, [0.0, 0.0, 1.0], 0.0, 1e-4)
    assert tightened[2, 2] < 0.5 * covariance[2, 2]
    assert abs(tightened[0, 0] - 1.0) < 1e-6
    assert abs(float(updated[2])) < 1e-6
    print("belief_turn_step ok", flush=True)


if __name__ == "__main__":
    _check()
