"""One tangential step of the pads that already read force.

Pairwise cross products of the contact normals estimate an axis. Each pad
moves along axis × normal, which is the direction that turns about that axis.
The cylinder mesh is not an input. A fixed positive joint increment is not used.
"""

import importlib.util
from pathlib import Path

import numpy as np

_step_path = Path(__file__).with_name("reach_shift_step.py")
_step = importlib.util.spec_from_file_location("reach_shift_step", _step_path)
_reach = importlib.util.module_from_spec(_step)
_step.loader.exec_module(_reach)
joint_increment = _reach.joint_increment
MAX_DQ = _reach.MAX_DQ

STEP_LENGTH = 0.002


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def axis_from_normals(normals):
    """Axis most consistent with pairwise cross products. None if undefined."""
    normals = [_unit(normal) for normal in np.asarray(normals, dtype=np.float64)]
    normals = [normal for normal in normals if normal is not None]
    crosses = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            direction = _unit(np.cross(normals[i], normals[j]))
            if direction is not None:
                crosses.append(direction)
    if not crosses:
        return None
    total = crosses[0].copy()
    for direction in crosses[1:]:
        if float(np.dot(direction, total)) < 0.0:
            direction = -direction
        total = total + direction
    return _unit(total)


def tangential_direction(normal, axis):
    """Unit direction axis × normal. None when the normal is along the axis."""
    normal = _unit(normal)
    axis = _unit(axis)
    if normal is None or axis is None:
        return None
    return _unit(np.cross(axis, normal))


def plan_tangent_steps(normals, blocks, joints, lowers, uppers, step_length=STEP_LENGTH, max_dq=MAX_DQ):
    """Joint increments that move each loaded pad along axis × normal.

    blocks[i], joints[i], lowers[i], uppers[i] belong to the same loaded finger.
    A finger that cannot move at least half the requested distance is left at zero.
    """
    axis = axis_from_normals(normals)
    steps = [np.zeros(np.asarray(block).shape[1], dtype=np.float64) for block in blocks]
    directions = [None] * len(blocks)
    if axis is None:
        return {"axis": None, "directions": directions, "steps": steps, "moved": []}
    moved = []
    for index, normal in enumerate(normals):
        direction = tangential_direction(normal, axis)
        directions[index] = direction
        if direction is None:
            continue
        delta = direction * step_length
        step, _residual = joint_increment(
            blocks[index],
            delta,
            joints[index],
            lowers[index],
            uppers[index],
            max_dq,
        )
        achieved = float(np.linalg.norm(np.asarray(blocks[index], dtype=np.float64) @ step))
        if achieved < 0.5 * step_length:
            continue
        steps[index] = step
        moved.append(index)
    return {"axis": axis, "directions": directions, "steps": steps, "moved": moved}


def quat_xyzw_to_matrix(quat):
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def point_in_object(point, center, quat):
    rotation = quat_xyzw_to_matrix(quat)
    return rotation.T @ (np.asarray(point, dtype=np.float64) - np.asarray(center, dtype=np.float64))


def _check():
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3) / 2, 0.0],
            [-0.5, -np.sqrt(3) / 2, 0.0],
        ]
    )
    axis = axis_from_normals(normals)
    assert abs(abs(axis[2]) - 1.0) < 1e-6
    if axis[2] < 0.0:
        axis = -axis
    direction = tangential_direction(normals[0], axis)
    assert np.allclose(direction, [0.0, 1.0, 0.0], atol=1e-6)
    blocks = [np.eye(3) for _ in range(3)]
    joints = [np.zeros(3) for _ in range(3)]
    lowers = [np.full(3, -1.0) for _ in range(3)]
    uppers = [np.full(3, 1.0) for _ in range(3)]
    plan = plan_tangent_steps(normals, blocks, joints, lowers, uppers)
    assert plan["moved"] == [0, 1, 2]
    assert float(np.dot(plan["directions"][0], normals[0])) < 1e-6
    lowers[1][:] = 0.0
    uppers[1][:] = 0.0
    blocked = plan_tangent_steps(normals, blocks, joints, lowers, uppers)
    assert 1 not in blocked["moved"]
    assert axis_from_normals(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])) is None
    print("tangent_turn_step ok", flush=True)


if __name__ == "__main__":
    _check()
