"""Closed-loop kinematic regrasp. One step is one joint increment.

OPT1 moves one unloaded finger onto the cylinder. The other joints stay
fixed. OPT2 runs when that finger cannot reach: loaded fingers carry the
cylinder, and one loaded fingertip writes the object pose. Friction, mass,
and inertia are not inputs.

Sundaralingam and Hermans, ICRA 2018, arXiv:1804.04292, solve the same
split with SNOPT. The costs E_pos and E_or are the relaxed-rigidity terms
from their RSS 2017 paper, arXiv:1806.00942. This file uses SLSQP.
"""

import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog, minimize

from hand_fk import PAD_LINKS, URDF, HandFK
from reach_shift_step import MAX_DQ, closest_on_cylinder

ETA = 0.01
BETA = 0.001
K_POS = 1000.0
K_OR = 10.0
W_GAP = 1.0e4
LAND = 2.0e-3
YAW_STEP = 0.15
W_YAW = 100.0
W_RIGID = 10.0

FINGER_JOINTS = (
    (
        "right_thumb_CMC_FE",
        "right_thumb_CMC_AA",
        "right_thumb_MCP_FE",
        "right_thumb_MCP_AA",
        "right_thumb_IP",
    ),
    (
        "right_index_MCP_FE",
        "right_index_MCP_AA",
        "right_index_PIP",
        "right_index_DIP",
    ),
    (
        "right_middle_MCP_FE",
        "right_middle_MCP_AA",
        "right_middle_PIP",
        "right_middle_DIP",
    ),
    (
        "right_ring_MCP_FE",
        "right_ring_MCP_AA",
        "right_ring_PIP",
        "right_ring_DIP",
    ),
    (
        "right_pinky_CMC",
        "right_pinky_MCP_FE",
        "right_pinky_MCP_AA",
        "right_pinky_PIP",
        "right_pinky_DIP",
    ),
)


def joint_limits(path=URDF):
    limits = {}
    for joint in ET.parse(path).getroot().findall("joint"):
        if joint.get("type") != "revolute":
            continue
        node = joint.find("limit")
        limits[joint.get("name")] = (float(node.get("lower")), float(node.get("upper")))
    return limits


def signed_distance(point, center, axis, radius, half_length):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    rel = np.asarray(point, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    along = float(np.dot(rel, axis))
    radial = float(np.linalg.norm(rel - along * axis))
    if abs(along) <= half_length and radial <= radius:
        return -min(radius - radial, half_length - abs(along))
    return float(np.linalg.norm(point - closest_on_cylinder(point, center, axis, radius, half_length)))


def _vector(positions, names):
    return np.array([float(positions.get(name, 0.0)) for name in names], dtype=np.float64)


def _write(positions, names, values):
    updated = dict(positions)
    for name, value in zip(names, values):
        updated[name] = float(value)
    return updated


def _pad_poses(fk, positions):
    poses = fk.transforms(positions)
    return [poses[name] for name in PAD_LINKS]


def _gap(fk, positions, finger, center, axis, radius, half_length):
    pad = _pad_poses(fk, positions)[finger][:3, 3]
    return signed_distance(pad, center, axis, radius, half_length)


def _scale(dq, max_dq):
    dq = np.asarray(dq, dtype=np.float64)
    peak = float(np.max(np.abs(dq))) if dq.size else 0.0
    if peak > max_dq > 0.0:
        dq = dq * (max_dq / peak)
    return dq


def _opt1(fk, positions, limits, finger, center, axis, radius, half_length):
    """Land one finger on the mesh. Other joints stay at `positions`."""
    names = FINGER_JOINTS[finger]
    q0 = _vector(positions, names)
    start = _pad_poses(fk, positions)[finger][:3, 3]
    bounds = [limits[name] for name in names]

    def pack(values):
        return _write(positions, names, values)

    def cost(values):
        return float(np.sum((values - q0) ** 2))

    def surface(values):
        return _gap(fk, pack(values), finger, center, axis, radius, half_length)

    def travel(values):
        pad = _pad_poses(fk, pack(values))[finger][:3, 3]
        return ETA - float(np.linalg.norm(pad - start))

    def clearance(values):
        poses = fk.transforms(pack(values))
        gaps = []
        for link in poses:
            if link == PAD_LINKS[finger] or not link.startswith(PAD_LINKS[finger].split("_elastomer")[0].rsplit("_", 1)[0]):
                continue
            if PAD_LINKS[finger].split("_")[1] not in link:
                continue
            gaps.append(signed_distance(poses[link][:3, 3], center, axis, radius, half_length) - BETA)
        return np.array(gaps if gaps else [ETA], dtype=np.float64)

    result = minimize(
        cost,
        q0,
        method="SLSQP",
        bounds=bounds,
        constraints=(
            {"type": "eq", "fun": surface},
            {"type": "ineq", "fun": travel},
            {"type": "ineq", "fun": clearance},
        ),
        options={"maxiter": 80, "ftol": 1e-12},
    )
    if not result.success and abs(surface(result.x)) > LAND:
        return None
    if abs(surface(result.x)) > LAND or travel(result.x) < -1e-4:
        return None
    solved = pack(result.x)
    order = fk.revolute
    dq = _scale(_vector(solved, order) - _vector(positions, order), MAX_DQ)
    return {"mode": "opt1", "finger": finger, "dq": dq, "names": order, "gap": surface(result.x)}


def _object_from_reference(pose0, pose1, center, axis):
    rotation0 = pose0[:3, :3]
    rotation1 = pose1[:3, :3]
    delta = rotation1 @ rotation0.T
    moved_center = rotation1 @ (rotation0.T @ (center - pose0[:3, 3])) + pose1[:3, 3]
    return moved_center, delta @ axis


def _opt2(fk, positions, limits, free, loaded, center, axis, radius, half_length):
    """Move the cylinder with the loaded fingers until the free finger is closer."""
    reference = loaded[0]
    names = [name for finger in loaded for name in FINGER_JOINTS[finger]]
    q0 = _vector(positions, names)
    poses0 = _pad_poses(fk, positions)
    bounds = [limits[name] for name in names]
    held = [finger for finger in loaded if finger != reference]

    def pack(values):
        return _write(positions, names, values)

    def moved_cylinder(values):
        poses = _pad_poses(fk, pack(values))
        return _object_from_reference(poses0[reference], poses[reference], center, axis)

    def cost(values):
        poses = _pad_poses(fk, pack(values))
        center1, axis1 = moved_cylinder(values)
        gap = signed_distance(poses[free][:3, 3], center1, axis1, radius, half_length)
        position = 0.0
        orientation = 0.0
        rotation0 = poses0[reference][:3, :3]
        rotation1 = poses[reference][:3, :3]
        for finger in held:
            expected = rotation1 @ (rotation0.T @ (poses0[finger][:3, 3] - poses0[reference][:3, 3])) + poses[reference][:3, 3]
            position += float(np.sum((poses[finger][:3, 3] - expected) ** 2))
            expected_rot = rotation1 @ rotation0.T @ poses0[finger][:3, :3]
            orientation += float(np.sum((poses[finger][:3, :3] - expected_rot) ** 2))
        return W_GAP * gap * gap + K_POS * position + K_OR * orientation

    def travel(values):
        pad = _pad_poses(fk, pack(values))[reference][:3, 3]
        return ETA - float(np.linalg.norm(pad - poses0[reference][:3, 3]))

    result = minimize(
        cost,
        q0,
        method="SLSQP",
        bounds=bounds,
        constraints=({"type": "ineq", "fun": travel},),
        options={"maxiter": 80, "ftol": 1e-12},
    )
    solved = pack(result.x)
    poses = _pad_poses(fk, solved)
    center1, axis1 = _object_from_reference(poses0[reference], poses[reference], center, axis)
    gap1 = signed_distance(poses[free][:3, 3], center1, axis1, radius, half_length)
    gap0 = signed_distance(poses0[free][:3, 3], center, axis, radius, half_length)
    if gap1 > gap0 - 1e-4:
        return None
    order = fk.revolute
    dq = _scale(_vector(solved, order) - _vector(positions, order), MAX_DQ)
    return {"mode": "opt2", "finger": free, "reference": reference, "dq": dq, "names": order, "gap": gap1}


def center_enclosed(points, center, axis, radius):
    """True when the center lies in the contacts' hull, in the plane normal to the axis."""
    pts = [np.asarray(point, dtype=np.float64) for point in points]
    if len(pts) < 2:
        return False
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(axis, seed)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    center = np.asarray(center, dtype=np.float64)
    flat = np.array([[np.dot(point - center, x_axis), np.dot(point - center, y_axis)] for point in pts])
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


def _yaw_about(rotation, axis):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    base = np.cross(axis, seed)
    base = base / np.linalg.norm(base)
    turned = rotation @ base
    turned = turned - float(np.dot(turned, axis)) * axis
    turned = turned / max(float(np.linalg.norm(turned)), 1e-9)
    return float(np.arctan2(np.dot(axis, np.cross(base, turned)), np.dot(base, turned)))


def _object_delta(pose0, pose1, center, axis):
    rotation0 = pose0[:3, :3]
    rotation1 = pose1[:3, :3]
    delta = rotation1 @ rotation0.T
    moved_center = delta @ (center - pose0[:3, 3]) + pose1[:3, 3]
    return moved_center, delta @ axis, delta


def _rotate(fk, positions, limits, loaded, center, axis, radius):
    """One in-grasp yaw. Object pose is the reference fingertip pose times the initial offset."""
    reference = 0 if 0 in loaded else loaded[0]
    names = [name for finger in loaded for name in FINGER_JOINTS[finger]]
    q0 = _vector(positions, names)
    poses0 = _pad_poses(fk, positions)
    relative0 = []
    rotation0 = poses0[reference][:3, :3]
    origin0 = poses0[reference][:3, 3]
    for finger in loaded:
        if finger == reference:
            continue
        relative0.append(rotation0.T @ (poses0[finger][:3, 3] - origin0))
    bounds = [limits[name] for name in names]

    def pack(values):
        return _write(positions, names, values)

    def measure(values):
        poses = _pad_poses(fk, pack(values))
        center1, axis1, delta = _object_delta(poses0[reference], poses[reference], center, axis)
        yaw = _yaw_about(delta, axis)
        rotation1 = poses[reference][:3, :3]
        origin1 = poses[reference][:3, 3]
        rigid = 0.0
        slot = 0
        for finger in loaded:
            if finger == reference:
                continue
            relative = rotation1.T @ (poses[finger][:3, 3] - origin1)
            rigid += float(np.sum((relative - relative0[slot]) ** 2))
            slot += 1
        pads = [poses[finger][:3, 3] for finger in loaded]
        return yaw, rigid, center_enclosed(pads, center1, axis1, radius)

    def cost(values):
        yaw, rigid, enclosed = measure(values)
        outside = 0.0 if enclosed else 100.0
        return W_YAW * (YAW_STEP - yaw) ** 2 + W_RIGID * rigid + outside + 1e-3 * float(np.sum((values - q0) ** 2))

    result = minimize(cost, q0, method="SLSQP", bounds=bounds, options={"maxiter": 60, "ftol": 1e-10})
    yaw, rigid, enclosed = measure(result.x)
    if not enclosed or yaw < 0.02:
        return None
    order = fk.revolute
    dq = _scale(_vector(pack(result.x), order) - _vector(positions, order), MAX_DQ)
    return {"mode": "rotate", "finger": reference, "dq": dq, "names": order, "yaw": yaw, "rigid": rigid}


def _release(fk, positions, limits, loaded, center, axis, radius):
    """Move one loaded finger outward when the others still enclose the center."""
    poses = _pad_poses(fk, positions)
    pads = [pose[:3, 3] for pose in poses]
    candidates = []
    for finger in loaded:
        others = [pads[other] for other in loaded if other != finger]
        if len(others) < 2 or not center_enclosed(others, center, axis, radius):
            continue
        margin = min(min(positions[name] - limits[name][0], limits[name][1] - positions[name]) for name in FINGER_JOINTS[finger])
        candidates.append((margin, finger))
    if not candidates:
        return None
    finger = min(candidates)[1]
    names = FINGER_JOINTS[finger]
    q0 = _vector(positions, names)
    outward = pads[finger] - np.asarray(center, dtype=np.float64)
    outward = outward / max(float(np.linalg.norm(outward)), 1e-9)

    def radial(values):
        pad = _pad_poses(fk, _write(positions, names, values))[finger][:3, 3]
        return float(np.dot(pad - pads[finger], outward))

    bounds = [limits[name] for name in names]
    result = minimize(lambda values: -radial(values), q0, method="SLSQP", bounds=bounds, options={"maxiter": 40, "ftol": 1e-10})
    if radial(result.x) < 1e-4:
        return None
    order = fk.revolute
    dq = _scale(_vector(_write(positions, names, result.x), order) - _vector(positions, order), MAX_DQ)
    return {"mode": "release", "finger": finger, "dq": dq, "names": order}


def plan_cycle_step(fk, positions, limits, loaded, center, axis, radius, half_length):
    """Rotate, release, reach, or shift. One call is one joint increment."""
    loaded = [int(finger) for finger in loaded]
    poses = _pad_poses(fk, positions)
    pads = [pose[:3, 3] for pose in poses]
    if len(loaded) >= 2 and center_enclosed([pads[finger] for finger in loaded], center, axis, radius):
        turned = _rotate(fk, positions, limits, loaded, center, axis, radius)
        if turned is not None:
            return turned
        released = _release(fk, positions, limits, loaded, center, axis, radius)
        if released is not None:
            return released
    return plan_regrasp_step(fk, positions, limits, loaded, center, axis, radius, half_length)


def plan_regrasp_step(fk, positions, limits, loaded, center, axis, radius, half_length):
    """Re-solve from the current pose and return only the first joint increment."""
    loaded = [int(finger) for finger in loaded]
    free = [finger for finger in range(5) if finger not in loaded]
    if not free or not loaded:
        return None
    gaps = [(_gap(fk, positions, finger, center, axis, radius, half_length), finger) for finger in free]
    finger = min(gaps)[1]
    planned = _opt1(fk, positions, limits, finger, center, axis, radius, half_length)
    if planned is not None:
        return planned
    return _opt2(fk, positions, limits, finger, loaded, center, axis, radius, half_length)


def _check():
    fk = HandFK()
    limits = joint_limits()
    positions = {
        "right_thumb_CMC_FE": np.deg2rad(95.12771),
        "right_thumb_CMC_AA": np.deg2rad(-3.11244),
        "right_thumb_MCP_FE": np.deg2rad(14.81626),
        "right_thumb_MCP_AA": np.deg2rad(-1.03493),
        "right_thumb_IP": np.deg2rad(12.23986),
        "right_index_MCP_FE": np.deg2rad(40.0),
        "right_index_MCP_AA": np.deg2rad(6.1133),
        "right_index_PIP": np.deg2rad(15.58495),
        "right_index_DIP": np.deg2rad(5.90325),
        "right_middle_MCP_FE": np.deg2rad(31.74149),
        "right_middle_MCP_AA": np.deg2rad(-0.95812),
        "right_middle_PIP": np.deg2rad(41.88173),
        "right_middle_DIP": np.deg2rad(12.844),
        "right_ring_MCP_FE": np.deg2rad(31.72383),
        "right_ring_MCP_AA": np.deg2rad(9.84458),
        "right_ring_PIP": np.deg2rad(35.22366),
        "right_ring_DIP": np.deg2rad(18.02839),
        "right_pinky_CMC": np.deg2rad(10.9712),
        "right_pinky_MCP_FE": np.deg2rad(68.30895),
        "right_pinky_MCP_AA": np.deg2rad(7.99151),
        "right_pinky_PIP": np.deg2rad(5.89626),
        "right_pinky_DIP": np.deg2rad(5.89875),
    }
    pads = fk.pads(positions)
    inward = pads[2] - pads[1]
    inward = inward / np.linalg.norm(inward)
    radius, half = 0.02, 0.016
    center = pads[1] - inward * (radius + 0.003)
    axis = np.array([0.0, 0.0, 1.0])
    gap0 = _gap(fk, positions, 1, center, axis, radius, half)
    planned = plan_regrasp_step(fk, positions, limits, [0, 2, 3], center, axis, radius, half)
    assert planned is not None and planned["mode"] == "opt1" and planned["finger"] == 1
    assert float(np.max(np.abs(planned["dq"]))) <= MAX_DQ + 1e-9
    assert abs(planned["gap"]) <= LAND
    middle = [fk.revolute.index(name) for name in FINGER_JOINTS[2]]
    assert float(np.max(np.abs(planned["dq"][middle]))) < 1e-8
    assert planned["gap"] < gap0
    far = pads[1] - inward * (radius + 0.03)
    far_plan = plan_regrasp_step(fk, positions, limits, [2, 3], far, axis, radius, half)
    assert far_plan is not None and far_plan["mode"] == "opt2"
    assert float(np.max(np.abs(far_plan["dq"]))) <= MAX_DQ + 1e-9
    assert far_plan["gap"] < _gap(fk, positions, 1, far, axis, radius, half)
    held = [0, 2, 3]
    held_center = np.mean(pads[held], axis=0)
    held_axis = np.array([0.0, 0.0, 1.0])
    assert center_enclosed(pads[held], held_center, held_axis, radius)
    turned = plan_cycle_step(fk, positions, limits, held, held_center, held_axis, radius, half)
    assert turned is not None and turned["mode"] == "rotate"
    assert turned["yaw"] > 0.02
    assert float(np.max(np.abs(turned["dq"]))) <= MAX_DQ + 1e-9
    index = [fk.revolute.index(name) for name in FINGER_JOINTS[1]]
    assert float(np.max(np.abs(turned["dq"][index]))) < 1e-8
    print("regrasp_step ok", flush=True)


if __name__ == "__main__":
    _check()
