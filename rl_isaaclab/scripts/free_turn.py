"""FREE MPC on the existing Sharpa grasp in Isaac.

The plant is Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0. The grasp is the same
live search used by the current Sharpa scripts. The planner is Jin,
Complementarity-Free Dexterous Manipulation (arXiv:2408.07855): a 4-step
IPOPT problem whose contact force is a soft-plus of the linearized gap, so a
finger can leave. Five elastomer pads are the contacts. The cylinder radius
is the loaded-pad distance to the axis, so a pad that is already touching
starts at zero gap.
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

parser = argparse.ArgumentParser(description="Complementarity-free MPC on one Sharpa grasp.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=1999979387)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--exec_steps", type=int, default=8)
parser.add_argument("--goal_yaw", type=float, default=math.pi / 2)
parser.add_argument("--goal_tol", type=float, default=0.04)
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
from rl_isaaclab.scripts.free_explicit import N_FINGERS, FreeExplicit
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

LOG_PATH = Path("logs/live_diag/free_turn.jsonl")
VIDEO_PATH = Path("logs/to_delete/free_turn.mp4")
DROP = 0.02
CYLINDER_HALF_HEIGHT = 0.032
MU = 0.5
LOAD_MIN = 0.5


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


def xyzw_to_wxyz(quat):
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)


def wxyz_to_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


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


def pad_jacobian(base, env_id):
    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    columns = [dof0 + int(joint) for joint in base.actuated_dof_indices]
    blocks = []
    for body in base.elastomer_ids:
        jac = base.hand.data.body_link_jacobian_w.torch[env_id, int(body) - body_offset, :3, :]
        blocks.append(jac[:, columns].detach().cpu().numpy())
    return np.stack(blocks, axis=0)


def contact_model(obj_pos, obj_quat_wxyz, pads, pad_jac, loads):
    rotation = wxyz_to_matrix(obj_quat_wxyz)
    axis = rotation[:, 2]
    samples = []
    loaded_radii = []
    for finger in range(N_FINGERS):
        delta = pads[finger] - obj_pos
        axial = float(np.dot(delta, axis))
        radial_vec = delta - axial * axis
        radial = float(np.linalg.norm(radial_vec))
        samples.append((axial, radial, radial_vec, delta))
        if loads[finger] > LOAD_MIN and radial > 1e-4:
            loaded_radii.append(radial)
    surface = float(np.median(loaded_radii)) if loaded_radii else 0.02
    phi = np.ones(N_FINGERS * 4)
    jac = np.zeros((N_FINGERS * 4, 6 + pad_jac.shape[-1]))
    gaps = []
    for finger, (axial, radial, radial_vec, delta) in enumerate(samples):
        if radial < 1e-5 or abs(axial) > CYLINDER_HALF_HEIGHT + 0.01:
            gaps.append(1.0)
            continue
        normal = radial_vec / radial
        tangent = np.cross(axis, normal)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        directions = [tangent, axis, -tangent, -axis]
        point = np.zeros((3, 6))
        point[:, 0:3] = np.eye(3)
        point[:, 3:6] = -skew(delta) @ rotation
        relative = np.zeros((3, jac.shape[1]))
        relative[:, 0:6] = -point
        relative[:, 6:] = pad_jac[finger]
        gap = radial - surface
        gaps.append(gap)
        for row, direction in enumerate(directions):
            cone = normal + MU * direction
            jac[4 * finger + row] = cone @ relative
            phi[4 * finger + row] = gap
    return phi, jac, surface, gaps


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
    print(f"free seed {env_cfg.seed} envs={args_cli.num_envs}", flush=True)

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
    axis = torch.tensor(base.cfg.rot_axis, dtype=torch.float32, device=device)
    lower, upper = joint_limits(base, env_id)
    action_scale = float(base.cfg.action_scale)
    substeps = max(int(math.ceil(0.2 / action_scale)), 1)
    planner = FreeExplicit(int(base.cfg.action_space), lower.detach().cpu().numpy(), upper.detach().cpu().numpy())
    home = base.object_pos[env_id].detach().clone()
    z0 = float(home[2].item())
    goal = quat_mul(yaw_about(axis, args_cli.goal_yaw, device).view(1, 4), base.object_rot[env_id].view(1, 4))[0]
    yaw = 0.0
    rotations = 0
    loads = pad_loads(base, env_id).detach().cpu().tolist()
    records = [{"event": "start", "z": z0, "substeps": substeps, "loads": loads}]
    print(
        f"settled z={z0:.4f} substeps={substeps} loads={[round(v, 3) for v in loads]}",
        flush=True,
    )

    for step_id in range(args_cli.exec_steps):
        if not simulation_app.is_running():
            break
        base._refresh_lab()
        origin = base.scene.env_origins[env_id].detach().cpu().numpy()
        pads = base.hand.data.body_link_state_w[env_id, base.elastomer_ids, :3].detach().cpu().numpy() - origin
        pad_jac = pad_jacobian(base, env_id)
        obj_pos = base.object_pos[env_id].detach().cpu().numpy()
        obj_xyzw = base.object_rot[env_id].detach().cpu().numpy()
        obj_wxyz = xyzw_to_wxyz(obj_xyzw)
        loads_now = pad_loads(base, env_id).detach().cpu().numpy()
        phi, jac, surface, gaps = contact_model(obj_pos, obj_wxyz, pads, pad_jac, loads_now)
        q_hand = actuated_position(base, env_id).detach().cpu().numpy()
        curr_q = np.concatenate((obj_pos, obj_wxyz, q_hand))
        target_q = xyzw_to_wxyz(goal.detach().cpu().numpy())
        if float(np.dot(target_q, target_q)) > 0:
            target_q = target_q / np.linalg.norm(target_q)
        action, status, cost, predicted = planner.plan(
            curr_q,
            home.detach().cpu().numpy(),
            target_q,
            phi,
            jac,
            pads.reshape(-1),
            pad_jac.reshape(N_FINGERS * 3, -1),
            q_hand,
        )
        q_des = torch.as_tensor(q_hand + action, dtype=lower.dtype, device=device)
        q_before = base.object_rot[env_id].detach().clone()
        for _ in range(substeps):
            env.step(targets_to_actions(
                q_des.view(1, -1).expand(args_cli.num_envs, -1),
                base.prev_targets,
                action_scale,
                lower,
                upper,
            ))
        base._refresh_lab()
        turned = axis_angle_from_quat(quat_mul(base.object_rot[env_id], quat_conjugate(q_before)))
        yaw += float(torch.dot(turned, axis).item())
        curr_wxyz = xyzw_to_wxyz(base.object_rot[env_id].detach().cpu().numpy())
        quat_error = 1.0 - float(np.dot(curr_wxyz, target_q) ** 2)
        z = float(base.object_pos[env_id, 2].item())
        if quat_error < args_cli.goal_tol:
            rotations += 1
            goal = quat_mul(
                yaw_about(axis, args_cli.goal_yaw, device).view(1, 4),
                base.object_rot[env_id].view(1, 4),
            )[0]
            print(f"goal advance {rotations} at step {step_id}", flush=True)
        loads = pad_loads(base, env_id).detach().cpu().tolist()
        row = {
            "event": "step",
            "step": step_id,
            "yaw": yaw,
            "quat_error": quat_error,
            "rotations": rotations,
            "z": z,
            "status": status,
            "cost": cost,
            "action_norm": float(np.linalg.norm(action)),
            "predicted_pos": predicted[:3].tolist(),
            "surface": surface,
            "gaps": gaps,
            "loads": loads,
        }
        records.append(row)
        print(
            f"exec {step_id} status={status} yaw={yaw:.4f} quat_error={quat_error:.3f} "
            f"z={z:.4f} loads={[round(v, 3) for v in loads]} "
            f"gaps={[round(v, 4) for v in gaps]} |u|={row['action_norm']:.3f} "
            f"pred_z={predicted[2]:.4f} pred_xy=({predicted[0]:.4f},{predicted[1]:.4f})",
            flush=True,
        )
        record_frame(video, base)
        if z < z0 - DROP:
            records.append({"event": "result", "stop": "dropped", "yaw": yaw, "z": z, "rotations": rotations})
            print(f"stop dropped z={z:.4f} yaw={yaw:.4f}", flush=True)
            break
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
