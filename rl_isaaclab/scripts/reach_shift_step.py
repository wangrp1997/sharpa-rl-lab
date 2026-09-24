"""Search a small rigid motion of a known cylinder.

Contacting fingertips must be able to follow their surface points inside the
joint limits. After that motion, a finger that cannot reach the cylinder now
must have a joint increment inside its limits that lands on the surface.
"""

import numpy as np

URDF_RADIUS = 0.04
URDF_LENGTH = 0.064
MAX_DQ = 0.008
LAND_TOL = 2e-3


def cylinder_size(scale: float) -> tuple[float, float]:
    return URDF_RADIUS * scale, 0.5 * URDF_LENGTH * scale


def cylinder_axis(quat) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat)
    axis = np.array(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    axis = rotvec / theta
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)


def closest_on_cylinder(point, center, axis, radius, half_length) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    rel = np.asarray(point, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    along = float(np.dot(rel, axis))
    along_c = float(np.clip(along, -half_length, half_length))
    radial = rel - along * axis
    norm = float(np.linalg.norm(radial))
    if norm < 1e-9:
        seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        radial_hat = np.cross(axis, seed)
        radial_hat = radial_hat / np.linalg.norm(radial_hat)
    else:
        radial_hat = radial / norm
    return np.asarray(center, dtype=np.float64) + along_c * axis + radius * radial_hat


def joint_increment(block, delta, q, lower, upper, max_dq=None):
    """Least-squares increment. Returns the step and the cartesian residual."""
    block = np.asarray(block, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    gram = block @ block.T + 1e-4 * np.eye(3)
    step = block.T @ np.linalg.solve(gram, delta)
    room_lo = lower - q
    room_hi = upper - q
    if max_dq is not None:
        room_lo = np.maximum(room_lo, -max_dq)
        room_hi = np.minimum(room_hi, max_dq)
    step = np.clip(step, room_lo, room_hi)
    achieved = block @ step
    return step, float(np.linalg.norm(delta - achieved))


def _mapped(point, center, rotation, shift):
    return rotation @ (np.asarray(point, dtype=np.float64) - center) + center + shift


def _toward_free(center, axis, free_pad):
    radial = np.asarray(free_pad, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    radial = radial - float(np.dot(radial, axis)) * axis
    norm = float(np.linalg.norm(radial))
    if norm < 1e-8:
        return []
    radial_hat = radial / norm
    tangent = np.cross(axis, radial_hat)
    tangent = tangent / np.linalg.norm(tangent)
    motions = []
    for dist in (0.002, 0.004, 0.006, 0.008):
        motions.append((np.eye(3), radial_hat * dist, dist))
    for angle in (0.1, -0.1, 0.2, -0.2):
        motions.append((_rodrigues(tangent * angle), np.zeros(3), abs(angle) * 0.02))
    for dist in (0.003, 0.006):
        for angle in (0.1, -0.1):
            motions.append((_rodrigues(tangent * angle), radial_hat * dist, dist + abs(angle) * 0.02))
    return motions


def diagnose(pads, center, axis, radius, half_length, loaded, blocks, q, lower, upper):
    """Residual of each unloaded finger before and after the searched motions."""
    pads = np.asarray(pads, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    loaded = list(loaded)
    rows = []
    for finger in range(len(pads)):
        if finger in loaded:
            continue
        surface_now = closest_on_cylinder(pads[finger], center, axis, radius, half_length)
        delta_now = surface_now - pads[finger]
        _, residual_now = joint_increment(blocks[finger], delta_now, q[finger], lower[finger], upper[finger])
        best = None
        for rotation, shift, cost in _toward_free(center, axis, pads[finger]):
            new_center = center + shift
            new_axis = rotation @ axis
            loaded_residual = 0.0
            loaded_ok = True
            for held in loaded:
                contact = closest_on_cylinder(pads[held], center, axis, radius, half_length)
                target = _mapped(contact, center, rotation, shift)
                _, residual = joint_increment(blocks[held], target - pads[held], q[held], lower[held], upper[held])
                loaded_residual = max(loaded_residual, residual)
                if residual > LAND_TOL:
                    loaded_ok = False
                    break
            surface = closest_on_cylinder(pads[finger], new_center, new_axis, radius, half_length)
            _, free_residual = joint_increment(
                blocks[finger], surface - pads[finger], q[finger], lower[finger], upper[finger]
            )
            item = {
                "cost": cost,
                "loaded_ok": loaded_ok,
                "loaded_residual": loaded_residual,
                "free_residual": free_residual,
                "gap": float(np.linalg.norm(surface - pads[finger])),
            }
            if best is None or item["free_residual"] < best["free_residual"]:
                best = item
        rows.append({
            "finger": finger,
            "gap_before": float(np.linalg.norm(delta_now)),
            "residual_before": residual_now,
            "best": best,
        })
    return rows


def plan_shift(pads, center, axis, radius, half_length, loaded, blocks, q, lower, upper):
    """Return a motion that makes one unloaded finger reachable, or None."""
    pads = np.asarray(pads, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    loaded = list(loaded)
    free = [i for i in range(len(pads)) if i not in loaded]
    contacts = {
        finger: closest_on_cylinder(pads[finger], center, axis, radius, half_length) for finger in loaded
    }
    best = None
    for finger in free:
        surface_now = closest_on_cylinder(pads[finger], center, axis, radius, half_length)
        _, residual_now = joint_increment(
            blocks[finger], surface_now - pads[finger], q[finger], lower[finger], upper[finger]
        )
        if residual_now <= LAND_TOL:
            continue
        for rotation, shift, cost in _toward_free(center, axis, pads[finger]):
            new_center = center + shift
            new_axis = rotation @ axis
            tracked = {}
            reachable_loaded = True
            for held in loaded:
                target = _mapped(contacts[held], center, rotation, shift)
                delta = target - pads[held]
                step, residual = joint_increment(blocks[held], delta, q[held], lower[held], upper[held])
                if residual > LAND_TOL:
                    reachable_loaded = False
                    break
                tracked[held] = {"target": target, "step": step, "residual": residual}
            if not reachable_loaded:
                continue
            surface = closest_on_cylinder(pads[finger], new_center, new_axis, radius, half_length)
            delta = surface - pads[finger]
            step, residual = joint_increment(blocks[finger], delta, q[finger], lower[finger], upper[finger])
            if residual > LAND_TOL or float(np.linalg.norm(step)) < 1e-5:
                continue
            if best is None or cost < best["cost"]:
                best = {
                    "finger": finger,
                    "rotation": rotation,
                    "shift": shift,
                    "cost": cost,
                    "new_center": new_center,
                    "new_axis": new_axis,
                    "loaded": tracked,
                    "free_target": surface,
                    "free_step": step,
                    "free_residual": residual,
                    "residual_before": residual_now,
                }
    return best
