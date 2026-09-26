"""DROP-style cross-entropy planning on the existing Sharpa grasp.

The plant is Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0. The grasp is the same
live search used by the current Sharpa scripts: keep the first downward grasp
that stays in contact. The planner is the CEM setup from Li et al., DROP
(arXiv:2409.14562), as configured in the LEAP cube task of mujoco_mpc
(leap-hardware): planner 5, horizon 1 s, 4 spline knots, exploration 0.5,
std floor 0.5, 4 elites. Costs follow that task's two active residuals. Sampled values are joint
targets in radians, with exploration 0.5 rad, not the policy's unit residual.
The position term uses their slope of 250, and a downward move past 2 mm is
outside the tube. The goal is a yaw about this environment's rot_axis,
advanced by 90 degrees once the error falls under 0.4 rad.
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

RECORD = "--record" in sys.argv
sys.argv = [arg for arg in sys.argv if arg not in ("--headless", "--record")]
if RECORD:
    sys.argv += ["--viz", "kit"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CEM in-hand reorientation on one cached Sharpa grasp.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=1999979387)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--exec_steps", type=int, default=20)
parser.add_argument("--horizon_s", type=float, default=1.0)
parser.add_argument("--knots", type=int, default=4)
parser.add_argument("--cem_iters", type=int, default=3)
parser.add_argument("--elites", type=int, default=4)
parser.add_argument("--explore", type=float, default=0.5)
parser.add_argument("--std_min", type=float, default=0.5)
parser.add_argument("--goal_yaw", type=float, default=math.pi / 2)
parser.add_argument("--goal_tol", type=float, default=0.4)
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
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_angle_axis, quat_mul
import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

LOG_PATH = Path("logs/live_diag/drop_turn.jsonl")
VIDEO_PATH = Path("logs/to_delete/drop_turn.mp4")
DROP = 0.02
POS_WEIGHT = 2.5
ORI_WEIGHT = 1.0
RECTIFY_P = 0.05
POS_SLOPE = 250.0
POS_BOX_XY = 0.01
POS_DROP = 0.002
POS_LIFT = 0.01


class FrameWriter:
    def __init__(self, path, fps=10):
        self.path = Path(path)
        self.fps = fps
        self.proc = None
        self.count = 0

    def add(self, frame):
        image = np.ascontiguousarray(frame[:, :, :3])
        if self.proc is None:
            height, width = image.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(self.fps),
                    "-i", "-",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(self.path),
                ],
                stdin=subprocess.PIPE,
            )
        self.proc.stdin.write(image.tobytes())
        self.count += 1

    def close(self):
        if self.proc is None:
            return
        self.proc.stdin.close()
        self.proc.wait()
        self.proc = None


def grab_frame(base):
    import os
    import tempfile

    import imageio.v2 as imageio
    import omni.kit.app
    from omni.kit.viewport.utility import capture_viewport_to_file

    for viz in getattr(base.sim, "visualizers", []):
        viewport = getattr(viz, "_viewport_api", None)
        if viewport is None:
            continue
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        path = handle.name
        handle.close()
        os.remove(path)
        capture_viewport_to_file(viewport, path)
        for _ in range(30):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                image = imageio.imread(path)
                os.remove(path)
                return np.ascontiguousarray(image[:, :, :3])
            omni.kit.app.get_app().update()
        return None
    return None


def record_frame(video, base):
    if video is None:
        return
    frame = grab_frame(base)
    if frame is not None:
        video.add(frame)


def as_torch(value):
    return value.torch if hasattr(value, "torch") else value


def pad_loads(base, env_id=0):
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.normal_force_matrix_w[:, 0, 0, :]
        columns.append(torch.linalg.norm(force, dim=-1))
    return torch.stack(columns, dim=0)[:, env_id]


def yaw_about(axis, angle, device):
    axis = torch.as_tensor(axis, dtype=torch.float32, device=device)
    axis = axis / torch.linalg.norm(axis)
    return quat_from_angle_axis(torch.tensor([angle], device=device), axis.view(1, 3))[0]


def rectified_position(pos, origin):
    delta = pos - origin
    outside = torch.stack(
        (
            delta[:, 0].abs() - POS_BOX_XY,
            delta[:, 1].abs() - POS_BOX_XY,
            -delta[:, 2] - POS_DROP,
            delta[:, 2] - POS_LIFT,
        ),
        dim=-1,
    ).clamp(min=0.0)
    dist = torch.linalg.norm(outside, dim=-1)
    return RECTIFY_P * torch.nn.functional.softplus(POS_SLOPE * dist / RECTIFY_P)


def joint_limits(base, env_id):
    lower = base.hand_dof_lower_limits[env_id]
    upper = base.hand_dof_upper_limits[env_id]
    if lower.numel() != base.prev_targets.shape[-1]:
        index = base.actuated_dof_indices
        lower, upper = lower[index], upper[index]
    return lower, upper


def targets_to_actions(q_des, prev, scale, lower, upper):
    q_des = torch.maximum(torch.minimum(q_des, upper), lower)
    return ((q_des - prev) / scale).clamp(-1.0, 1.0)


def orientation_cost(quat, goal):
    error = axis_angle_from_quat(quat_mul(goal.expand_as(quat), quat_conjugate(quat)))
    return torch.sum(error * error, dim=-1), torch.linalg.norm(error, dim=-1)


def interpolate(knots, horizon):
    count = knots.shape[1]
    grid = torch.linspace(0, count - 1, horizon, device=knots.device)
    left = grid.long().clamp(max=count - 2)
    alpha = (grid - left.float()).view(1, horizon, 1)
    return (1.0 - alpha) * knots[:, left] + alpha * knots[:, left + 1]


def shift_knots(knots, horizon):
    count = knots.shape[0]
    step = (count - 1) / max(horizon - 1, 1)
    grid = torch.arange(count, device=knots.device, dtype=knots.dtype) + step
    grid = grid.clamp(0, count - 1)
    left = grid.long().clamp(max=count - 2)
    alpha = (grid - left.float()).unsqueeze(-1)
    return (1.0 - alpha) * knots[left] + alpha * knots[left + 1]


def capture(base, env_id):
    base._refresh_lab()
    origin = base.scene.env_origins[env_id]
    return {
        "obj_pos": (as_torch(base.object.data.root_pos_w)[env_id] - origin).detach().clone(),
        "obj_quat": as_torch(base.object.data.root_quat_w)[env_id].detach().clone(),
        "obj_vel": as_torch(base.object.data.root_vel_w)[env_id].detach().clone(),
        "hand_pos": (as_torch(base.hand.data.root_pos_w)[env_id] - origin).detach().clone(),
        "hand_quat": as_torch(base.hand.data.root_quat_w)[env_id].detach().clone(),
        "hand_vel": as_torch(base.hand.data.root_vel_w)[env_id].detach().clone(),
        "q": base.hand.data.joint_pos[env_id].detach().clone(),
        "qd": base.hand.data.joint_vel[env_id].detach().clone(),
        "prev": base.prev_targets[env_id].detach().clone(),
        "cur": base.cur_targets[env_id].detach().clone(),
    }


def restore(base, snap):
    count = base.num_envs
    origins = base.scene.env_origins
    obj_pose = snap["obj_quat"].view(1, 4).repeat(count, 1)
    obj_pose = torch.cat([(snap["obj_pos"] + origins), obj_pose], dim=-1)
    hand_pose = torch.cat([(snap["hand_pos"] + origins), snap["hand_quat"].view(1, 4).repeat(count, 1)], dim=-1)
    base.object.write_root_pose_to_sim_index(root_pose=obj_pose)
    base.object.write_root_velocity_to_sim_index(root_velocity=snap["obj_vel"].view(1, -1).repeat(count, 1))
    base.hand.write_root_pose_to_sim_index(root_pose=hand_pose)
    base.hand.write_root_velocity_to_sim_index(root_velocity=snap["hand_vel"].view(1, -1).repeat(count, 1))
    base.hand.write_joint_position_to_sim_index(position=snap["q"].view(1, -1).repeat(count, 1))
    base.hand.write_joint_velocity_to_sim_index(velocity=snap["qd"].view(1, -1).repeat(count, 1))
    base.prev_targets[:] = snap["prev"]
    base.cur_targets[:] = snap["cur"]
    base.hand.set_joint_position_target(base.cur_targets)
    base.sim.forward()
    base._refresh_lab()
    base.object_pos_prev[:] = base.object_pos
    base.object_rot_prev[:] = base.object_rot
    base.reset_buf[:] = 0


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
    print(f"drop seed {env_cfg.seed} envs={args_cli.num_envs}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()
    base = env.unwrapped
    if base.num_envs > 1:
        center = base.scene.env_origins.detach().float().mean(dim=0).cpu()
        eye = center + torch.tensor([0.0, -7.0, 6.0])
        base.sim.set_camera_view(
            tuple(float(v) for v in eye.tolist()),
            tuple(float(v) for v in center.tolist()),
        )
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
    video = FrameWriter(VIDEO_PATH) if RECORD else None
    for _ in range(args_cli.settle_steps):
        if not simulation_app.is_running():
            if video is not None:
                video.close()
            return
        env.step(env.zero_actions())
        record_frame(video, base)

    device = base.device
    horizon = max(int(round(args_cli.horizon_s / float(base.step_dt))), 2)
    action_dim = int(base.cfg.action_space)
    axis = torch.tensor(base.cfg.rot_axis, dtype=torch.float32, device=device)
    snap = capture(base, env_id)
    restore(base, snap)
    home = base.object_pos[env_id].detach().clone()
    z0 = float(home[2].item())
    goal = quat_mul(yaw_about(axis, args_cli.goal_yaw, device).view(1, 4), base.object_rot[env_id].view(1, 4))[0]
    lower, upper = joint_limits(base, env_id)
    action_scale = float(base.cfg.action_scale)
    mean = snap["prev"].detach().clone().view(1, -1).repeat(args_cli.knots, 1)
    std = torch.full_like(mean, args_cli.explore)
    yaw = 0.0
    rotations = 0
    records = [{"event": "start", "z": z0, "horizon": horizon, "home": home.tolist()}]
    loads = pad_loads(base, env_id).detach().cpu().tolist()
    print(
        f"settled z={z0:.4f} horizon_steps={horizon} goal_yaw={args_cli.goal_yaw:.3f} "
        f"loads={[round(v, 3) for v in loads]}",
        flush=True,
    )

    for step_id in range(args_cli.exec_steps):
        if not simulation_app.is_running():
            break
        for _ in range(args_cli.cem_iters):
            restore(base, snap)
            noise = torch.randn((args_cli.num_envs, args_cli.knots, action_dim), device=device)
            knots = mean + noise * std
            q_spline = interpolate(knots, horizon)
            pos_cost = torch.zeros(args_cli.num_envs, device=device)
            ori_cost = torch.zeros(args_cli.num_envs, device=device)
            for knot_step in range(horizon):
                env.step(targets_to_actions(q_spline[:, knot_step], base.prev_targets, action_scale, lower, upper))
                base._refresh_lab()
                pos_cost = pos_cost + POS_WEIGHT * rectified_position(base.object_pos, home)
                ori_term, _ = orientation_cost(base.object_rot, goal)
                ori_cost = ori_cost + ORI_WEIGHT * ori_term
            total = pos_cost + ori_cost
            elite_ids = torch.topk(total, args_cli.elites, largest=False).indices
            mean = knots[elite_ids].mean(dim=0)
            std = knots[elite_ids].std(dim=0, unbiased=False).clamp_min(args_cli.std_min)

        restore(base, snap)
        q_first = interpolate(mean.unsqueeze(0), horizon)[0, 0]
        q_before = base.object_rot[env_id].detach().clone()
        env.step(targets_to_actions(q_first.view(1, -1).expand(args_cli.num_envs, -1), base.prev_targets, action_scale, lower, upper))
        base._refresh_lab()
        turned = axis_angle_from_quat(quat_mul(base.object_rot[env_id], quat_conjugate(q_before)))
        yaw += float(torch.dot(turned, axis).item())
        _, goal_angle = orientation_cost(base.object_rot[env_id].view(1, 4), goal)
        angle = float(goal_angle[0].item())
        z = float(base.object_pos[env_id, 2].item())
        if angle < args_cli.goal_tol:
            rotations += 1
            goal = quat_mul(
                yaw_about(axis, args_cli.goal_yaw, device).view(1, 4),
                base.object_rot[env_id].view(1, 4),
            )[0]
            print(f"goal advance {rotations} at step {step_id}", flush=True)
        row = {
            "event": "step",
            "step": step_id,
            "yaw": yaw,
            "goal_angle": angle,
            "rotations": rotations,
            "z": z,
            "elite_cost": float(total[elite_ids].mean().item()),
        }
        records.append(row)
        loads = pad_loads(base, env_id).detach().cpu().tolist()
        row["loads"] = loads
        print(
            f"exec {step_id} yaw={yaw:.4f} goal_angle={angle:.3f} z={z:.4f} "
            f"loads={[round(v, 3) for v in loads]} elite_pos={float(pos_cost[elite_ids].mean()):.2f} "
            f"elite_ori={float(ori_cost[elite_ids].mean()):.2f}",
            flush=True,
        )
        record_frame(video, base)
        if z < z0 - DROP:
            records.append({"event": "result", "stop": "dropped", "yaw": yaw, "z": z, "rotations": rotations})
            print(f"stop dropped z={z:.4f} yaw={yaw:.4f}", flush=True)
            break
        mean = shift_knots(mean, horizon)
        std = shift_knots(std, horizon).clamp_min(args_cli.std_min)
        snap = capture(base, env_id)
    else:
        records.append({"event": "result", "stop": "step limit", "yaw": yaw, "z": z, "rotations": rotations})
        print(f"result stop=step limit yaw={yaw:.4f} z={z:.4f} rotations={rotations}", flush=True)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("".join(json.dumps(row) + "\n" for row in records))
    print(f"log {LOG_PATH}", flush=True)
    if video is not None:
        video.close()
        print(f"video {VIDEO_PATH} frames={video.count}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
