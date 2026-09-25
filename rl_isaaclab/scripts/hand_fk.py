"""Forward kinematics of the right Sharpa Wave URDF.

Joint origins and axes come from assets/SharpaWave/right_sharpa_wave.urdf.
Link positions are in the right_hand_C_MC frame. Mesh files are not loaded.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = "right_hand_C_MC"
PAD_LINKS = (
    "right_thumb_elastomer",
    "right_index_elastomer",
    "right_middle_elastomer",
    "right_ring_elastomer",
    "right_pinky_elastomer",
)
URDF = Path(__file__).resolve().parents[2] / "assets" / "SharpaWave" / "right_sharpa_wave.urdf"


def rpy_matrix(roll, pitch, yaw):
    """URDF fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def axis_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    v = 1.0 - c
    return np.array(
        [
            [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
        ]
    )


def _origin(joint):
    origin = joint.find("origin")
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if origin is not None:
        if origin.get("xyz"):
            xyz = np.array([float(v) for v in origin.get("xyz").split()], dtype=np.float64)
        if origin.get("rpy"):
            rpy = np.array([float(v) for v in origin.get("rpy").split()], dtype=np.float64)
    axis_node = joint.find("axis")
    axis = np.array([0.0, 0.0, 1.0])
    if axis_node is not None and axis_node.get("xyz"):
        axis = np.array([float(v) for v in axis_node.get("xyz").split()], dtype=np.float64)
    return xyz, rpy, axis


class HandFK:
    """Product of joint transforms from the palm link."""

    def __init__(self, path=URDF):
        tree = ET.parse(path)
        self.joints = []
        self.revolute = []
        for joint in tree.getroot().findall("joint"):
            xyz, rpy, axis = _origin(joint)
            record = {
                "name": joint.get("name"),
                "type": joint.get("type"),
                "parent": joint.find("parent").get("link"),
                "child": joint.find("child").get("link"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
            }
            self.joints.append(record)
            if record["type"] == "revolute":
                self.revolute.append(record["name"])
        self.children = {}
        for record in self.joints:
            self.children.setdefault(record["parent"], []).append(record)

    def transforms(self, positions):
        """4x4 pose of every link in the palm frame. Missing joints stay at 0."""
        poses = {ROOT: np.eye(4)}
        pending = list(self.children.get(ROOT, []))
        while pending:
            joint = pending.pop()
            parent = poses.get(joint["parent"])
            if parent is None:
                pending.append(joint)
                continue
            angle = float(positions.get(joint["name"], 0.0)) if joint["type"] == "revolute" else 0.0
            local = np.eye(4)
            local[:3, :3] = rpy_matrix(*joint["rpy"]) @ axis_matrix(joint["axis"], angle)
            local[:3, 3] = joint["xyz"]
            poses[joint["child"]] = parent @ local
            pending.extend(self.children.get(joint["child"], []))
        return poses

    def pads(self, positions):
        """Five elastomer origins in the palm frame, thumb to pinky."""
        poses = self.transforms(positions)
        return np.stack([poses[name][:3, 3] for name in PAD_LINKS])


def _check():
    hand = HandFK()
    assert len(hand.revolute) == 22
    pads = hand.pads({})
    assert pads.shape == (5, 3)
    assert float(np.linalg.norm(pads[1] - pads[2])) > 0.01
    moved = hand.pads({"right_index_MCP_FE": 0.4})
    assert float(np.linalg.norm(moved[1] - pads[1])) > 1e-3
    assert float(np.linalg.norm(moved[2] - pads[2])) < 1e-8
    print("hand_fk ok", flush=True)


if __name__ == "__main__":
    _check()
