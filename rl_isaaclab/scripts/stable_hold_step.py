"""Shape record and a grasp-holding joint increment.

Joint positions place the pads. Pads that already read force supply contact
points and normals. Those observations condition a Gaussian-process implicit
surface, so the object is a coarse shape with a covariance. This follows the
same observation as contact_gpis.py. It is not 3D Gaussian Splatting and it
does not choose a finger.

The increment keeps the loaded pads still: it is the proposal projected onto
the kinematic null space of their Jacobians, J dq = 0. Sommer and Billard,
Robotics and Autonomous Systems, 2016, put other motions in the null space of
contact forces with the robot inertia matrix. The inertia matrix, their
torque controller, the 0.5 N force, and equation (18) are not used. The
covariance is not an input of the projection.
"""

import importlib.util
from pathlib import Path

import numpy as np

_gp_path = Path(__file__).with_name("contact_gpis.py")
_gp = importlib.util.spec_from_file_location("contact_gpis", _gp_path)
_contact = importlib.util.module_from_spec(_gp)
_gp.loader.exec_module(_contact)

CLOSE_STEP = 0.0012


class ShapeRecord:
    """Posterior of the implicit surface. Querying it does not return a finger."""

    def __init__(self, points, normals):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        if len(points) == 0:
            raise ValueError("shape record needs at least one contact")
        self.center = points.mean(axis=0)
        span = float(np.max(np.linalg.norm(points - self.center, axis=1)))
        self.scale = span if span > 1e-6 else 1.0
        local = (points - self.center) / self.scale
        if len(points) == 1:
            self.length = 1.0
        else:
            dist = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1)
            self.length = 1.1 * float(np.max(dist))
        self.points = local
        self.gram, self.weight = _contact.fit(local, normals, self.length)

    def query(self, position):
        local = (np.asarray(position, dtype=np.float64) - self.center) / self.scale
        return _contact.predict(local, self.points, self.length, self.gram, self.weight)


def closing_proposal(n_dof, free_columns, step=CLOSE_STEP):
    """Small flexion on joints the caller already marked as free fingers.

    Which columns are free is proprioception plus which pads currently read
    force. The shape record is not read.
    """
    proposal = np.zeros(int(n_dof), dtype=np.float64)
    for column in free_columns:
        proposal[int(column)] = step
    return proposal


def hold_increment(proposal, loaded_blocks, loaded_columns):
    """Project proposal so loaded pad Jacobians produce no translation."""
    proposal = np.asarray(proposal, dtype=np.float64).copy()
    if len(loaded_blocks) == 0:
        return proposal
    rows = []
    for block, columns in zip(loaded_blocks, loaded_columns):
        block = np.asarray(block, dtype=np.float64)
        row = np.zeros((block.shape[0], proposal.size), dtype=np.float64)
        row[:, list(columns)] = block
        rows.append(row)
    jacobian = np.vstack(rows)
    gram = jacobian @ jacobian.T + 1e-8 * np.eye(jacobian.shape[0])
    correction = jacobian.T @ np.linalg.solve(gram, jacobian @ proposal)
    held = proposal - correction
    return held


def _check():
    points = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=np.float64)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    shape = ShapeRecord(points, normals)
    near_mean, near_var = shape.query(points[0])
    far_mean, far_var = shape.query(np.array([0.2, 0.2, 0.2]))
    assert near_var < far_var
    assert abs(near_mean) < abs(far_mean) or near_var < 0.05

    proposal = np.array([0.01, 0.0, 0.0, 0.0012], dtype=np.float64)
    block = np.eye(3)
    held = hold_increment(proposal, [block], [[0, 1, 2]])
    assert np.allclose(held[:3], 0.0, atol=1e-8)
    assert abs(held[3] - 0.0012) < 1e-8

    free = closing_proposal(4, [3])
    assert np.allclose(free, [0.0, 0.0, 0.0, CLOSE_STEP])
    untouched = hold_increment(free, [block], [[0, 1, 2]])
    assert np.allclose(untouched, free)
    print("stable_hold_step ok", flush=True)


if __name__ == "__main__":
    _check()
