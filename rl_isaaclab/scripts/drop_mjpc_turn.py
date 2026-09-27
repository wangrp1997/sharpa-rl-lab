"""Run DROP's cross-entropy planner on the Sharpa grasp.

The planner is the leap-hardware MJPC binary. Isaac freezes the current grasp
and tracks the position targets that planner returns. The Sharpa model is a
planner-side copy, not a change to the grasp search.
"""

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DROP cross-entropy targets on one Sharpa grasp.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=1999979387)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--exec_steps", type=int, default=12)
parser.add_argument("--mjpc_image", type=str, default="mjpc-drop-tooled")
parser.add_argument("--mjpc_repo", type=str, default="/home/rw/Documents/mujoco_mpc")
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

LOG_PATH = Path("logs/live_diag/drop_mjpc_turn.jsonl")
SERVER_LOG = Path("logs/live_diag/drop_mjpc_server.log")
DROP = 0.02


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


def apply_target(env, base, env_id, target, lower, upper, scale, num_envs):
    q_des = torch.as_tensor(target, dtype=lower.dtype, device=lower.device)
    current = base.prev_targets[env_id].detach().cpu().numpy()
    gap = float(np.max(np.abs(np.asarray(target) - current)))
    for _ in range(max(int(math.ceil(gap / scale)), 1)):
        env.step(targets_to_actions(
            q_des.view(1, -1).expand(num_envs, -1),
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


def quat_multiply(lhs, rhs):
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def yaw_about_z(quat_xyzw):
    x, y, z, w = [float(v) for v in quat_xyzw]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def cylinder_in_hand(hand_pos, hand_quat_xyzw, obj_pos, obj_quat_xyzw):
    rotation = xyzw_to_matrix(hand_quat_xyzw)
    local = rotation.T @ (np.asarray(obj_pos, dtype=float) - np.asarray(hand_pos, dtype=float))
    hand_wxyz = xyzw_to_wxyz(hand_quat_xyzw)
    hand_inv = [hand_wxyz[0], -hand_wxyz[1], -hand_wxyz[2], -hand_wxyz[3]]
    quat = quat_multiply(hand_inv, xyzw_to_wxyz(obj_quat_xyzw))
    return np.concatenate([local, np.asarray(quat, dtype=float)])


class DropServer:
    def __init__(self, image, repo):
        SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
        self.err = open(SERVER_LOG, "w")
        self.proc = subprocess.Popen(
            [
                "docker", "run", "-i", "--rm",
                "-v", f"{repo}:/src",
                "-w", "/src/build",
                image,
                "./bin/sharpa_drop",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.err,
            text=True,
        )

    def request(self, command, pose, joints):
        values = " ".join(f"{float(v):.8g}" for v in list(pose) + list(joints))
        self.proc.stdin.write(f"{command} {values}\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"drop server exited, see {SERVER_LOG}")
            if line.startswith("RESULT"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "error":
                    raise RuntimeError(line.strip())
                return np.asarray([float(v) for v in parts[1:]], dtype=float)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
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
    print(f"drop mjpc seed {env_cfg.seed} envs={args_cli.num_envs}", flush=True)

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
    hand_pos = as_torch(base.hand.data.root_pos_w)[env_id].detach().cpu().numpy() - origin
    hand_quat = as_torch(base.hand.data.root_quat_w)[env_id].detach().cpu().numpy()
    obj_pos0 = base.object_pos[env_id].detach().cpu().numpy().copy()
    z0 = float(obj_pos0[2])
    server = DropServer(args_cli.mjpc_image, args_cli.mjpc_repo)
    records = []
    try:
        for step_id in range(args_cli.exec_steps):
            if not simulation_app.is_running():
                break
            base._refresh_lab()
            q = actuated_position(base, env_id).detach().cpu().numpy()
            obj_pos = base.object_pos[env_id].detach().cpu().numpy()
            obj_quat = as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy()
            pose = cylinder_in_hand(hand_pos, hand_quat, obj_pos, obj_quat)
            command = "init" if step_id == 0 else "step"
            ctrl = server.request(command, pose, q)
            if ctrl.shape != (22,):
                raise RuntimeError(f"expected 22 targets, got {ctrl.shape}")
            apply_target(env, base, env_id, ctrl, lower, upper, scale, args_cli.num_envs)
            base._refresh_lab()
            z = float(base.object_pos[env_id, 2].item())
            loads = pad_loads(base, env_id).detach().cpu().numpy()
            yaw = yaw_about_z(as_torch(base.object.data.root_quat_w)[env_id].detach().cpu().numpy())
            move = float(np.max(np.abs(ctrl - q)))
            row = {
                "event": "step",
                "step": step_id,
                "yaw": yaw,
                "z": z,
                "move": move,
                "loads": loads.tolist(),
            }
            records.append(row)
            print(
                f"exec {step_id} yaw={yaw:.4f} z={z:.4f} "
                f"loads={[round(float(v), 3) for v in loads]} |dq|={move:.4f}",
                flush=True,
            )
            xy = base.object_pos[env_id, :2].detach().cpu().numpy() - obj_pos0[:2]
            if z < z0 - DROP or float(np.linalg.norm(xy)) > 0.03 or bool(np.all(loads < 0.05)):
                reason = "dropped" if z < z0 - DROP or float(np.linalg.norm(xy)) > 0.03 else "pads left the cylinder"
                records.append({"event": "result", "stop": reason, "yaw": yaw, "z": z})
                print(f"stop {reason} z={z:.4f} yaw={yaw:.4f}", flush=True)
                break
        else:
            records.append({"event": "result", "stop": "step limit", "yaw": yaw, "z": z})
            print(f"result stop=step limit yaw={yaw:.4f} z={z:.4f}", flush=True)
    finally:
        server.close()
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json
        LOG_PATH.write_text("".join(json.dumps(row) + "\n" for row in records))
        print(f"log {LOG_PATH}", flush=True)


if __name__ == "__main__":
    main()
