"""Call FREE's MPCExplicit server with a 22-DoF Sharpa state.

This module does not import Isaac. solve_increment is the one-shot bridge:
a 22-vector goes in, and the action vector from MPCExplicit.plan_once comes out.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

FREE_REPO = Path("/home/rw/Documents/Complementarity-Free-Dexterous-Manipulation")
FREE_PYTHON = Path("/home/rw/miniconda3/envs/free/bin/python")
N_ROBOT = 22


class FreeExplicitClient:
    def __init__(self, repo=FREE_REPO, python=FREE_PYTHON, log_file=None):
        self.repo = Path(repo)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo)
        env["PYTHONUNBUFFERED"] = "1"
        self._log = open(log_file, "w") if log_file is not None else None
        self.proc = subprocess.Popen(
            [str(python), "-u", "examples/mpc/sharpa/cylinder/server.py"],
            cwd=str(self.repo),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log if self._log is not None else None,
            text=True,
        )

    def request(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError("FREE MPCExplicit server exited before RESULT")
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
                if not result.get("ok", False):
                    raise RuntimeError(result.get("error", "FREE server failed"))
                return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        if self._log is not None:
            self._log.close()


def solve_increment(q, obj_pos, obj_quat_xyzw, hand_pos, hand_quat_xyzw, repo=FREE_REPO, python=FREE_PYTHON):
    """One MPCExplicit.plan_once call. q is the fake or measured 22-vector."""
    q = np.asarray(q, dtype=float).reshape(N_ROBOT)
    client = FreeExplicitClient(repo=repo, python=python)
    try:
        client.request({
            "cmd": "init",
            "hand_pos": np.asarray(hand_pos, dtype=float).reshape(3).tolist(),
            "hand_quat_xyzw": np.asarray(hand_quat_xyzw, dtype=float).reshape(4).tolist(),
            "obj_pos": np.asarray(obj_pos, dtype=float).reshape(3).tolist(),
            "obj_quat_xyzw": np.asarray(obj_quat_xyzw, dtype=float).reshape(4).tolist(),
            "q": q.tolist(),
        })
        result = client.request({
            "cmd": "step",
            "obj_pos": np.asarray(obj_pos, dtype=float).reshape(3).tolist(),
            "obj_quat_xyzw": np.asarray(obj_quat_xyzw, dtype=float).reshape(4).tolist(),
            "q": q.tolist(),
        })
    finally:
        client.close()
    action = np.asarray(result["action"], dtype=float)
    if action.shape != (N_ROBOT,) or not np.all(np.isfinite(action)):
        raise RuntimeError(f"FREE action is not a finite 22-vector: {action}")
    result["action"] = action
    return result


def _smoke():
    sys.path.insert(0, str(FREE_REPO))
    from models.sharpa.kinematics import SharpaKinematics, midpoint_q

    q = midpoint_q()
    hand_pos = np.array([0.1, -0.02, 0.62])
    hand_quat_xyzw = np.array([0.0, 0.0, 0.3826834, 0.9238795])
    hand_quat_wxyz = np.array([0.9238795, 0.0, 0.0, 0.3826834])
    centers = SharpaKinematics().pad_centers(q, hand_pos, hand_quat_wxyz)
    obj_pos = centers.mean(axis=0)
    result = solve_increment(q, obj_pos, np.array([0.0, 0.0, 0.0, 1.0]), hand_pos, hand_quat_xyzw)
    action = result["action"]
    print(
        f"status={result['status']} cost={result['cost']:.6f} "
        f"action_norm={result['action_norm']:.6f} action0={float(action[0]):.6f} "
        f"ncon={result['ncon']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
