"""One exploration step inside an enveloping grasp.

This is the kinematic piece of Sommer and Billard, Robotics and Autonomous
Systems, 2016, equation (18). Their torque controller, inertia matrix, and
0.5 N contact force are not used. Desired contacts are the five elastomer
pads. The cylinder mesh is not an input.

A pad that already reads force keeps a zero joint increment. Each pad that
does not read force is assigned the nearest loaded pad. Its direction is the
sum of that contact's normal and the vector from the free pad to the contact,
then normalized. The free pad closest to a loaded pad, and still able to move
along that direction inside the joint limits, takes one small step.
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


def approach_direction(pad, contact, normal):
    """Unit direction of equation (18). None when the mix vanishes."""
    normal = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    mix = normal / norm + (np.asarray(contact, dtype=np.float64) - np.asarray(pad, dtype=np.float64))
    mix_norm = float(np.linalg.norm(mix))
    if mix_norm < 1e-8:
        return None
    return mix / mix_norm


def plan_envelop_step(
    pads,
    loaded,
    normals,
    blocks,
    joints,
    lowers,
    uppers,
    step_length=STEP_LENGTH,
    max_dq=MAX_DQ,
):
    """Return the single free finger to move, or finger=None when none can.

    pads, normals: (n, 3). loaded: indices that already read force.
    blocks[i]: jacobian of pad i, shape (3, n_i).
    joints, lowers, uppers: per-finger joint position and limits, matching blocks.
    """
    pads = np.asarray(pads, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    loaded = [int(i) for i in loaded]
    steps = [np.zeros(np.asarray(block).shape[1], dtype=np.float64) for block in blocks]
    empty = {
        "finger": None,
        "nearest": None,
        "distance": None,
        "direction": None,
        "residual": None,
        "steps": steps,
    }
    usable = [i for i in loaded if float(np.linalg.norm(normals[i])) >= 1e-8]
    if not usable:
        return empty

    best = None
    for finger in range(len(pads)):
        if finger in loaded:
            continue
        nearest = min(usable, key=lambda i: float(np.linalg.norm(pads[i] - pads[finger])))
        distance = float(np.linalg.norm(pads[nearest] - pads[finger]))
        direction = approach_direction(pads[finger], pads[nearest], normals[nearest])
        if direction is None:
            continue
        delta = direction * step_length
        step, residual = joint_increment(
            blocks[finger],
            delta,
            joints[finger],
            lowers[finger],
            uppers[finger],
            max_dq,
        )
        achieved = float(np.linalg.norm(np.asarray(blocks[finger], dtype=np.float64) @ step))
        if achieved < 0.5 * step_length:
            continue
        candidate = (distance, finger, nearest, direction, residual, step)
        if best is None or candidate[0] < best[0] or (candidate[0] == best[0] and finger < best[1]):
            best = candidate

    if best is None:
        return empty
    distance, finger, nearest, direction, residual, step = best
    steps[finger] = step
    return {
        "finger": finger,
        "nearest": nearest,
        "distance": distance,
        "direction": direction,
        "residual": residual,
        "steps": steps,
    }


def _check():
    pads = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.01, 0.03, 0.00],
            [0.05, 0.04, 0.00],
            [0.008, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    normals = np.zeros((5, 3))
    normals[0] = np.array([0.0, 0.0, 1.0])
    normals[2] = np.array([0.0, 1.0, 0.0])
    blocks = [np.eye(3) for _ in range(5)]
    joints = [np.zeros(3) for _ in range(5)]
    lowers = [np.full(3, -1.0) for _ in range(5)]
    uppers = [np.full(3, 1.0) for _ in range(5)]
    plan = plan_envelop_step(pads, [0, 2], normals, blocks, joints, lowers, uppers)
    assert plan["finger"] == 4
    assert plan["nearest"] == 0
    assert np.allclose(plan["steps"][0], 0.0)
    assert np.allclose(plan["steps"][2], 0.0)
    assert float(np.linalg.norm(plan["steps"][4])) > 0.0
    direction = approach_direction(pads[4], pads[0], normals[0])
    assert np.allclose(plan["direction"], direction)

    uppers[4][:] = 0.0
    blocked = plan_envelop_step(pads, [0, 2], normals, blocks, joints, lowers, uppers)
    assert blocked["finger"] == 1
    assert blocked["nearest"] == 0

    none = plan_envelop_step(pads, [], normals, blocks, joints, lowers, uppers)
    assert none["finger"] is None
    print("envelop_explore_step ok", flush=True)


if __name__ == "__main__":
    _check()
