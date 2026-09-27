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


def targets_to_actions(q_des, prev, scale, lower, upper):
    q_des = torch.maximum(torch.minimum(q_des, upper), lower)
    return ((q_des - prev) / scale).clamp(-1.0, 1.0)


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
    q = actuated_position(base, env_id).detach().cpu().numpy()
    yaw = yaw_about_z(as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy())
    z0 = float(obj_pos[2])
    loads0 = pad_loads(base, env_id).detach().cpu().numpy()
    axis = object_axis(as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy())
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
        for step_id in range(args_cli.exec_steps):
            if not simulation_app.is_running():
                break
            base._refresh_lab()
            q = actuated_position(base, env_id).detach().cpu().numpy()
            quat = as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy()
            yaw = yaw_about_z(quat)
            result = server.request({"cmd": "step", "q": q.tolist(), "yaw": yaw})
            increment = np.asarray(result["du"], dtype=float)
            apply_increment(env, base, env_id, increment, lower, upper, scale, args_cli.num_envs)
            base._refresh_lab()
            z = float(base.object_pos[env_id, 2].item())
            loads = pad_loads(base, env_id).detach().cpu().numpy()
            yaw_after = yaw_about_z(as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy())
            row = {
                "event": "step",
                "step": step_id,
                "yaw": yaw_after,
                "yaw_pred": result["yaw_next"],
                "yaw_in": result.get("yaw_in"),
                "z": z,
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
            xy = base.object_pos[env_id, :2].detach().cpu().numpy() - obj_pos[:2]
            if z < z0 - DROP or float(np.linalg.norm(xy)) > 0.03 or bool(np.all(loads < 0.05)):
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
