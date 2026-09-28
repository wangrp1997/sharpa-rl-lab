"""Run Jiang's qsim DDP and apply its joint increments on the Sharpa grasp.

The planner is sharpa_qsim_server.py inside the existing inhand-jiang Docker.
Isaac only freezes the current grasp search and tracks the returned joint targets.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Jiang qsim joint targets on one Sharpa grasp.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=1999979387)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--exec_steps", type=int, default=8)
parser.add_argument(
    "--jiang_repo",
    type=str,
    default="/home/rw/Documents/in_hand_manipulation_2",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import carb

from isaaclab.envs import DirectRLEnvCfg
import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

LOG_PATH = Path("logs/live_diag/jiang_qsim_turn.jsonl")
SERVER_LOG = Path("logs/live_diag/jiang_qsim_server.log")
DROP = 0.02
CYLINDER_RADIUS = 0.02
PAD_NAMES = [
    "right_thumb_elastomer",
    "right_index_elastomer",
    "right_middle_elastomer",
    "right_ring_elastomer",
    "right_pinky_elastomer",
]


def as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def pad_loads(base, env_id):
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.normal_force_matrix_w[:, 0, 0, :]
        columns.append(torch.linalg.norm(force, dim=-1))
    return torch.stack(columns, dim=0)[:, env_id]


def joint_limits(base, env_id):
    lower = base.hand_dof_lower_limits[env_id]
    upper = base.hand_dof_upper_limits[env_id]
    if lower.numel() != base.prev_targets.shape[-1]:
        index = base.actuated_dof_indices
        lower, upper = lower[index], upper[index]
    return lower, upper


def actuated_position(base, env_id):
    position = base.hand_dof_pos[env_id]
    if position.numel() != base.prev_targets.shape[-1]:
        position = position[base.actuated_dof_indices]
    return position


# Planners index joints in URDF order. Isaac's articulation order is different.
PLANNER_JOINTS = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]


def planner_slots(joint_names):
    names = list(joint_names)
    return [names.index(name) for name in PLANNER_JOINTS]


def to_planner(q_isaac, slots):
    return np.asarray(q_isaac, dtype=float)[np.asarray(slots, dtype=int)]


def to_isaac(q_planner, slots, size):
    out = np.zeros(size, dtype=float)
    out[np.asarray(slots, dtype=int)] = np.asarray(q_planner, dtype=float)
    return out


def targets_to_actions(q_des, prev, scale, lower, upper):
    q_des = torch.maximum(torch.minimum(q_des, upper), lower)
    return ((q_des - prev) / scale).clamp(-1.0, 1.0)


# Positive flexion closes on this hand. Indices are the planner's URDF order.
FLEXION = {
    0: (0, 2, 4),
    1: (5, 7, 8),
    2: (9, 11, 12),
    3: (13, 15, 16),
    4: (18, 20, 21),
}
ABDUCTION = {
    0: (1, 3),
    1: (6,),
    2: (10,),
    3: (14,),
    4: (17, 19),
}
SWING_LIMIT = 0.04


def tangential_roll(q, hand_pos, hand_quat_wxyz, obj_pos, axis, loads, radii, lower, upper, stuck):
    """Joint step that slides freeze-loaded pads along +yaw at the grasp radius."""
    sys.path.insert(0, "/home/rw/Documents/Complementarity-Free-Dexterous-Manipulation")
    from models.sharpa.kinematics import SharpaKinematics

    if not hasattr(tangential_roll, "kin"):
        tangential_roll.kin = SharpaKinematics()
    kin = tangential_roll.kin
    q = np.asarray(q, dtype=float)
    centers = kin.pad_centers(q, hand_pos, hand_quat_wxyz)
    eps = 1e-4
    flat = centers.reshape(-1)
    jacobian = np.zeros((15, 22))
    for joint in range(22):
        bumped = q.copy()
        bumped[joint] += eps
        jacobian[:, joint] = (kin.pad_centers(bumped, hand_pos, hand_quat_wxyz).reshape(-1) - flat) / eps
    desired = np.zeros(15)
    columns = []
    for finger, joints in FLEXION.items():
        if float(loads[finger]) <= 0.5:
            continue
        columns.extend(joints)
        columns.extend(ABDUCTION[finger])
        delta = centers[finger] - np.asarray(obj_pos, dtype=float)
        radial = delta - float(np.dot(delta, axis)) * axis
        distance = float(np.linalg.norm(radial))
        if distance < 1e-6:
            continue
        normal = radial / distance
        tangent = np.cross(axis, normal)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-6:
            continue
        tangent = tangent / tangent_norm
        desired[3 * finger: 3 * finger + 3] = 0.0015 * tangent + 0.3 * (float(radii[finger]) - distance) * normal
    # A joint already against its URDF stop cannot keep sliding the pad.
    # Isaac's action limits are wider than those stops, so use the URDF.
    urdf_limits = kin.limits()
    free = []
    for joint in dict.fromkeys(columns):
        if joint in stuck:
            continue
        low, high = urdf_limits[joint]
        if q[joint] <= float(low) + 0.03 or q[joint] >= float(high) - 0.03:
            continue
        free.append(joint)
    columns = free
    command = np.zeros(22)
    if not columns:
        return command
    jac = jacobian[:, columns]
    gram = jac.T @ jac + 1e-4 * np.eye(len(columns))
    command[columns] = np.linalg.solve(gram, jac.T @ desired)
    return np.clip(command, -0.02, 0.02)


def grasp_radii(q, hand_pos, hand_quat_wxyz, obj_pos, axis):
    sys.path.insert(0, "/home/rw/Documents/Complementarity-Free-Dexterous-Manipulation")
    from models.sharpa.kinematics import SharpaKinematics

    centers = SharpaKinematics().pad_centers(q, hand_pos, hand_quat_wxyz)
    radii = []
    for finger in range(5):
        delta = centers[finger] - np.asarray(obj_pos, dtype=float)
        radial = delta - float(np.dot(delta, axis)) * axis
        radii.append(float(np.linalg.norm(radial)))
    return radii


def block_opening(increment, loads0):
    """Remove the part of a command that opens a finger loaded at the freeze."""
    command = np.asarray(increment, dtype=float).copy()
    for finger, joints in FLEXION.items():
        if float(loads0[finger]) <= 0.5:
            continue
        for joint in joints:
            if command[joint] < 0.0:
                command[joint] = 0.0
    return command


def keep_loaded_curl(increment, q, q0, loads0, loads_now):
    """Keep freeze-loaded pads on the cylinder while executing the planner increment.

    Opening curl is removed. Abduction of those fingers stays within SWING_LIMIT
    of the grasp, and a pad that has already left is walked back and closed.
    """
    command = np.asarray(increment, dtype=float).copy()
    current = np.asarray(q, dtype=float)
    origin = np.asarray(q0, dtype=float)
    for finger, joints in FLEXION.items():
        if float(loads0[finger]) <= 0.5:
            continue
        for joint in joints:
            if command[joint] < 0.0:
                command[joint] = 0.0
        for joint in ABDUCTION[finger]:
            proposed = current[joint] + command[joint]
            low = origin[joint] - SWING_LIMIT
            high = origin[joint] + SWING_LIMIT
            proposed = min(max(proposed, low), high)
            command[joint] = proposed - current[joint]
    return command


def apply_increment(env, base, env_id, increment, lower, upper, scale, num_envs):
    q_des = base.prev_targets[env_id].detach().cpu().numpy().copy() + increment
    q_des_t = torch.as_tensor(q_des, dtype=lower.dtype, device=lower.device)
    gap = float(np.max(np.abs(increment)))
    for _ in range(max(int(math.ceil(gap / scale)), 1)):
        env.step(targets_to_actions(
            q_des_t.view(1, -1).expand(num_envs, -1),
            base.prev_targets,
            scale,
            lower,
            upper,
        ))


def xyzw_to_matrix(quat):
    x, y, z, w = [float(v) for v in quat]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def xyzw_to_wxyz(quat):
    x, y, z, w = [float(v) for v in quat]
    return [w, x, y, z]


def object_axis(quat_xyzw):
    axis = xyzw_to_matrix(quat_xyzw)[:, 2]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def loaded_sphere_centers(base, env_id, origin, obj_pos, axis, loads):
    """World centers that put a 10.5 mm sphere 1.5 mm into the cylinder."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from link_spheres import SPHERES

    press = 0.0015
    radius = 0.0105
    positions = as_torch(base.hand.data.body_pos_w)[env_id, base.elastomer_ids]
    positions = positions.detach().cpu().numpy() - origin
    quats = as_torch(base.hand.data.body_quat_w)[env_id, base.elastomer_ids].detach().cpu().numpy()
    centers = {}
    for finger, name in enumerate(PAD_NAMES):
        if float(loads[finger]) <= 0.5:
            continue
        rotation = xyzw_to_matrix(quats[finger])
        best_gap = None
        best_world = None
        for center, patch_radius in SPHERES[name]:
            world = positions[finger] + rotation @ np.asarray(center, dtype=float)
            offset = world - obj_pos
            radial = offset - float(np.dot(offset, axis)) * axis
            gap = float(np.linalg.norm(radial) - patch_radius - CYLINDER_RADIUS)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_world = world
        offset = best_world - obj_pos
        radial = offset - float(np.dot(offset, axis)) * axis
        outward = radial / np.linalg.norm(radial)
        center_w = best_world - radial + (CYLINDER_RADIUS + radius - press) * outward
        centers[name] = center_w
        print(
            f"press {name} patch_gap={best_gap:.4f} center_w={np.array2string(center_w, precision=4)}",
            flush=True,
        )
    return {name: center.tolist() for name, center in centers.items()}


def yaw_about_z(quat_xyzw):
    x, y, z, w = [float(v) for v in quat_xyzw]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class QsimServer:
    def __init__(self, repo):
        SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
        self.err = open(SERVER_LOG, "w")
        self.proc = subprocess.Popen(
            [
                "docker", "run", "-i", "--rm", "--network", "host",
                "-v", f"{repo}:/workspace",
                "-e", "MPLBACKEND=Agg",
                "-e", "INHAND_HOME=/workspace/high_level",
                "-w", "/workspace/high_level/planner/ddp/tasks",
                "inhand-jiang:20.04",
                "python", "-u", "sharpa_qsim_server.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.err,
            text=True,
        )

    def request(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"qsim server exited, see {SERVER_LOG}")
            if line.startswith("RESULT "):
                result = json.loads(line[len("RESULT "):])
                if not result.get("ok", False):
                    raise RuntimeError(result.get("error", "qsim server failed"))
                return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        self.err.close()


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree("outputs/", ignore_errors=True)
    env_cfg.hold_pose = False
    env_cfg.freeze_on_success = True
    env_cfg.replay_cache = None
    env_cfg.randomize_mass = False
    env_cfg.gravity_curriculum = False
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    print(f"jiang qsim seed {env_cfg.seed} envs={args_cli.num_envs}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()
    base = env.unwrapped
    while simulation_app.is_running() and not base.frozen:
        env.step(env.zero_actions())
    if not base.frozen:
        print("screen stopped before a grasp held", flush=True)
        return

    base.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
    env_id = int(base.frozen_env_id)
    base.cfg.hold_pose = True
    base._focus_env(env_id)
    print(f"grasp env {env_id}", flush=True)
    for _ in range(args_cli.settle_steps):
        if not simulation_app.is_running():
            return
        env.step(env.zero_actions())

    lower, upper = joint_limits(base, env_id)
    scale = float(base.cfg.action_scale)
    origin = base.scene.env_origins[env_id].detach().cpu().numpy()
    hand_pos = (as_torch(base.hand.data.root_pos_w)[env_id].detach().cpu().numpy() - origin)
    hand_quat = as_torch(base.hand.data.root_quat_w)[env_id].detach().cpu().numpy()
    obj_pos = base.object_pos[env_id].detach().cpu().numpy()
    slots = planner_slots(base.hand.joint_names)
    q_isaac = actuated_position(base, env_id).detach().cpu().numpy()
    q = to_planner(q_isaac, slots)
    q0 = q.copy()
    hand_quat_wxyz = xyzw_to_wxyz(hand_quat)
    yaw = yaw_about_z(as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy())
    z0 = float(obj_pos[2])
    loads0 = pad_loads(base, env_id).detach().cpu().numpy()
    axis = object_axis(as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy())
    radii = grasp_radii(q, hand_pos, hand_quat_wxyz, obj_pos, axis)
    print(f"grasp radii {['%.4f' % v for v in radii]}", flush=True)
    centers_w = loaded_sphere_centers(base, env_id, origin, obj_pos, axis, loads0)
    server = QsimServer(args_cli.jiang_repo)
    records = []
    try:
        ready = server.request({
            "cmd": "init",
            "hand_pos": hand_pos.tolist(),
            "hand_quat_wxyz": xyzw_to_wxyz(hand_quat),
            "obj_pos": obj_pos.tolist(),
            "q": q.tolist(),
            "yaw": yaw,
            "sphere_centers_w": centers_w,
        })
        print(f"qsim planner ready centers={ready.get('local_centers')}", flush=True)
        print(f"joint order {list(base.hand.joint_names)}", flush=True)
        print(f"planner slots {slots}", flush=True)
        good = None
        last_yaw = yaw
        q_prev = q.copy()
        stuck = {}
        phase = "roll"
        phase_left = 0
        yaw_window = []
        for step_id in range(args_cli.exec_steps):
            if not simulation_app.is_running():
                break
            base._refresh_lab()
            q_isaac = actuated_position(base, env_id).detach().cpu().numpy()
            q = to_planner(q_isaac, slots)
            quat = as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy()
            yaw = yaw_about_z(quat)
            loads_now = pad_loads(base, env_id).detach().cpu().numpy()
            obj_now = base.object_pos[env_id].detach().cpu().numpy()
            axis_now = object_axis(quat)
            result = server.request({"cmd": "step", "q": q.tolist(), "yaw": yaw})
            lower_p = to_planner(lower.detach().cpu().numpy(), slots)
            upper_p = to_planner(upper.detach().cpu().numpy(), slots)
            yaw_window.append(yaw)
            if len(yaw_window) > 8:
                yaw_window.pop(0)
            plateau = (
                phase == "roll"
                and len(yaw_window) == 8
                and yaw_window[-1] - yaw_window[0] < 0.03
                and step_id > 12
                and float(loads_now[0]) > 0.5
                and float(loads_now[2]) > 0.5
            )
            if plateau:
                phase = "release"
                phase_left = 4
                print(f"gait release at step {step_id}", flush=True)
            if phase == "roll":
                command = block_opening(
                    tangential_roll(
                        q, hand_pos, hand_quat_wxyz, obj_now, axis_now, loads0, radii, lower_p, upper_p,
                        set(stuck),
                    ),
                    loads0,
                )
            elif phase == "release":
                command = np.zeros(22)
                for joint in FLEXION[2]:
                    command[joint] = -0.015
                phase_left -= 1
                if phase_left <= 0 or float(loads_now[2]) < 0.2:
                    phase = "swing"
                    phase_left = 4
                    print(f"gait swing at step {step_id}", flush=True)
            elif phase == "swing":
                # Slide the lifted middle finger further along +yaw, not back to the original pinch.
                command = tangential_roll(
                    q, hand_pos, hand_quat_wxyz, obj_now, axis_now, loads0,
                    grasp_radii(q, hand_pos, hand_quat_wxyz, obj_now, axis_now),
                    lower_p, upper_p, set(),
                )
                command[:9] = 0.0
                command[13:] = 0.0
                phase_left -= 1
                if phase_left <= 0:
                    phase = "regrasp"
                    phase_left = 8
                    print(f"gait regrasp at step {step_id}", flush=True)
            else:
                command = np.zeros(22)
                for joint in FLEXION[2]:
                    command[joint] = 0.02
                phase_left -= 1
                if float(loads_now[2]) > 0.5 or phase_left <= 0:
                    phase = "roll"
                    yaw_window = []
                    stuck = {}
                    print(f"gait roll at step {step_id}", flush=True)
            last_yaw = yaw
            prev = base.prev_targets[env_id].detach().cpu().numpy()
            increment = to_isaac(command, slots, q_isaac.shape[0])
            lead = prev - q_isaac
            for joint in range(increment.shape[0]):
                if increment[joint] * lead[joint] > 0.0 and abs(lead[joint]) > 0.05:
                    increment[joint] = 0.0
            if phase in ("release", "swing"):
                for planner_joint in (0, 1, 2, 3, 4):
                    increment[slots[planner_joint]] = 0.0
            apply_increment(env, base, env_id, increment, lower, upper, scale, args_cli.num_envs)
            base._refresh_lab()
            q_after = to_planner(actuated_position(base, env_id).detach().cpu().numpy(), slots)
            moved = q_after - q
            for joint in range(22):
                if abs(float(command[joint])) > 0.008 and abs(float(moved[joint])) < 0.002:
                    stuck[joint] = step_id
            stuck = {joint: seen for joint, seen in stuck.items() if step_id - seen < 12}
            q_prev = q_after
            if step_id % 10 == 0:
                print(f"stuck {sorted(stuck)}", flush=True)
            z = float(base.object_pos[env_id, 2].item())
            loads = pad_loads(base, env_id).detach().cpu().numpy()
            yaw_after = yaw_about_z(as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy())
            xy = base.object_pos[env_id, :2].detach().cpu().numpy() - obj_pos[:2]
            row = {
                "event": "step",
                "step": step_id,
                "yaw": yaw_after,
                "yaw_pred": result["yaw_next"],
                "yaw_in": result.get("yaw_in"),
                "z": z,
                "xy": [float(v) for v in xy],
                "du": [float(v) for v in increment],
                "du_norm": result["du_norm"],
                "loads": loads.tolist(),
                "sdists": result.get("sdists"),
                "geom_names": result.get("geom_names"),
                "force_norms": result.get("force_norms"),
            }
            records.append(row)
            print(
                f"exec {step_id} yaw={yaw_after:.4f} pred={result['yaw_next']:.4f} "
                f"z={z:.4f} loads={[round(float(v), 3) for v in loads]} |du|={result['du_norm']:.4f}",
                flush=True,
            )
            pads_left = phase == "roll" and bool(np.all(loads < 0.05))
            if z < z0 - DROP or float(np.linalg.norm(xy)) > 0.03 or pads_left:
                reason = "dropped" if z < z0 - DROP or float(np.linalg.norm(xy)) > 0.03 else "pads left the cylinder"
                records.append({"event": "result", "stop": reason, "yaw": yaw_after, "z": z})
                print(f"stop {reason} z={z:.4f} yaw={yaw_after:.4f}", flush=True)
                break
        else:
            records.append({"event": "result", "stop": "step limit", "yaw": yaw_after, "z": z})
            print(f"result stop=step limit yaw={yaw_after:.4f} z={z:.4f}", flush=True)
    finally:
        server.close()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("".join(json.dumps(row) + "\n" for row in records))
        print(f"log {LOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
