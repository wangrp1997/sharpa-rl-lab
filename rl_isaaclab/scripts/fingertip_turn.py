"""FREE fingertip MPC on the Sharpa cylinder.

The published example is the in-air fingertip task of Jin, arXiv:2408.07855:
three fingertips start around the object, and each command is a Cartesian
step of at most 5 mm. Their low-level loop is a PD toward that point; the
solid object stops a fingertip that would otherwise enter it. This script
tracks the same point on the Sharpa elastomers, and leaves a command inside
the cylinder on the surface instead of dragging the pad through the object.
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

parser = argparse.ArgumentParser(description="Three-fingertip MPC on one Sharpa grasp.")
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
from rl_isaaclab.scripts.fingertip_explicit import N_TIPS, FingertipExplicit
from rl_isaaclab.scripts.reach_shift_step import closest_on_cylinder, cylinder_size
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

LOG_PATH = Path("logs/live_diag/fingertip_turn.jsonl")
VIDEO_PATH = Path("logs/to_delete/fingertip_turn.mp4")
DROP = 0.02
HALF_HEIGHT = 0.032
MU = 0.5
LOAD_ON = 0.5
FINGER_PREFIXES = (
    "right_thumb_",
    "right_index_",
    "right_middle_",
    "right_ring_",
    "right_pinky_",
)
NAMES = ("thumb", "index", "middle", "ring", "pinky")


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


def pad_loads(base, env_id):
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


def finger_groups(base):
    names = base.hand.joint_names
    return [
        [i for i, joint_id in enumerate(base.actuated_dof_indices) if names[joint_id].startswith(prefix)]
        for prefix in FINGER_PREFIXES
    ]


def pad_jacobian(base, env_id):
    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    columns = [dof0 + int(joint) for joint in base.actuated_dof_indices]
    blocks = []
    for body in base.elastomer_ids:
        jac = base.hand.data.body_link_jacobian_w.torch[env_id, int(body) - body_offset, :3, :]
        blocks.append(jac[:, columns].detach().cpu().numpy())
    return np.stack(blocks, axis=0)


def pads_of(base, env_id):
    origin = base.scene.env_origins[env_id].detach().cpu().numpy()
    state = base.hand.data.body_link_state_w[env_id, base.elastomer_ids, :7].detach().cpu().numpy()
    return state[:, :3] - origin, state[:, 3:7]


def xyzw_matrix(quat):
    x, y, z, w = quat
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def face_offset(pads, quats, obj_pos, axis, loads):
    radius, half_length = cylinder_size(0.5)
    for finger in range(5):
        if loads[finger] <= LOAD_ON:
            continue
        closest = closest_on_cylinder(pads[finger], obj_pos, axis, radius, half_length)
        return xyzw_matrix(quats[finger]).T @ (closest - pads[finger])
    return np.zeros(3)


def object_axis(quat_wxyz):
    return wxyz_to_matrix(quat_wxyz)[:, 2]


def radial(pad, obj_pos, axis):
    delta = pad - obj_pos
    axial = float(np.dot(delta, axis))
    radial_vec = delta - axial * axis
    return axial, float(np.linalg.norm(radial_vec)), radial_vec


def surface_radius(pads, obj_pos, axis, loads):
    radii = []
    for finger in range(5):
        _, dist, _ = radial(pads[finger], obj_pos, axis)
        if loads[finger] > LOAD_ON and dist > 1e-4:
            radii.append(dist)
    return float(np.median(radii)) if radii else 0.02


def choose_tips(loads, dists, surface):
    loaded = [i for i in range(5) if loads[i] > LOAD_ON]
    if len(loaded) >= N_TIPS:
        loaded.sort(key=lambda i: -loads[i])
        return loaded[:N_TIPS]
    # Thumb, index, and middle are the three-fingertip tripod. Ring can sit
    # 1 mm outside the loaded radius and a inward step pokes the cylinder.
    if loads[0] > LOAD_ON and loads[2] > LOAD_ON:
        return [0, 1, 2]
    loaded = [i for i in range(5) if loads[i] > LOAD_ON]
    outside = [i for i in range(5) if i not in loaded and dists[i] >= surface - 0.002]
    outside.sort(key=lambda i: dists[i])
    chosen = loaded[:N_TIPS]
    for finger in outside:
        if len(chosen) >= N_TIPS:
            break
        chosen.append(finger)
    if len(chosen) < N_TIPS:
        rest = [i for i in range(5) if i not in chosen]
        rest.sort(key=lambda i: abs(dists[i] - surface))
        chosen.extend(rest[: N_TIPS - len(chosen)])
    return chosen[:N_TIPS]


def move_finger(env, base, env_id, groups, finger, delta, pad_jac, lower, upper, scale, num_envs):
    cols = groups[finger]
    if len(cols) == 0 or np.linalg.norm(delta) < 1e-6:
        return
    jac = pad_jac[finger][:, cols]
    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), delta)
    dq = np.clip(dq, -0.008, 0.008)
    q_des = base.prev_targets[env_id].detach().cpu().numpy().copy()
    q_des[cols] += dq
    q_des_t = torch.as_tensor(q_des, dtype=lower.dtype, device=lower.device)
    substeps = max(int(math.ceil(float(np.max(np.abs(dq))) / scale)), 1)
    for _ in range(substeps):
        env.step(targets_to_actions(
            q_des_t.view(1, -1).expand(num_envs, -1),
            base.prev_targets,
            scale,
            lower,
            upper,
        ))


def slide_step(current, desired, obj_pos, axis, surface):
    """One short move toward the PD target that does not enter the cylinder.

    Their fingertip is a free ball. Contact stops the inward part of the PD
    pull, and the rest of the pull slides the ball on the surface or off it.
    """
    error = desired - current
    _, dist, radial_vec = radial(current, obj_pos, axis)
    if dist < 1e-6:
        return error
    normal = radial_vec / dist
    _, end_dist, _ = radial(current + error, obj_pos, axis)
    if dist <= surface + 0.001 or end_dist < surface:
        inward = min(float(np.dot(error, normal)), 0.0)
        error = error - inward * normal
        if dist < surface:
            error = error + (surface - dist) * normal
    return error


def track_tips(env, base, env_id, groups, fingers, delta, lower, upper, scale, num_envs, obj_pos, axis, surface):
    """Track current pad position plus the planned step, sliding on the cylinder."""
    command = np.asarray(delta, dtype=float).reshape(len(fingers), 3)
    base._refresh_lab()
    start, _ = pads_of(base, env_id)
    desired = start[list(fingers)] + command
    lower_np = lower.detach().cpu().numpy()
    upper_np = upper.detach().cpu().numpy()
    for _ in range(40):
        base._refresh_lab()
        pads, _ = pads_of(base, env_id)
        current = pads[list(fingers)]
        error = np.stack([
            slide_step(current[row], desired[row], obj_pos, axis, surface)
            for row in range(len(fingers))
        ])
        norms = np.linalg.norm(error, axis=1)
        if float(np.max(norms)) < 0.001:
            break
        pad_jac = pad_jacobian(base, env_id)
        q_des = base.prev_targets[env_id].detach().cpu().numpy().copy()
        moved = False
        for row, finger in enumerate(fingers):
            if norms[row] < 0.001:
                continue
            step = error[row]
            length = float(np.linalg.norm(step))
            if length > 0.0015:
                step = step / length * 0.0015
            cols = groups[finger]
            jac = pad_jac[finger][:, cols]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(3), step)
            room_lo = np.maximum(lower_np[cols] - q_des[cols], -0.03)
            room_hi = np.minimum(upper_np[cols] - q_des[cols], 0.03)
            dq = np.clip(dq, room_lo, room_hi)
            if float(np.max(np.abs(dq))) < 1e-4:
                continue
            q_des[cols] = q_des[cols] + dq
            moved = True
        if not moved:
            break
        gap = float(np.max(np.abs(q_des - base.prev_targets[env_id].detach().cpu().numpy())))
        q_des_t = torch.as_tensor(q_des, dtype=lower.dtype, device=lower.device)
        for _ in range(max(int(math.ceil(gap / scale)), 1)):
            env.step(targets_to_actions(
                q_des_t.view(1, -1).expand(num_envs, -1),
                base.prev_targets,
                scale,
                lower,
                upper,
            ))
    base._refresh_lab()
    pads, _ = pads_of(base, env_id)
    current = pads[list(fingers)]
    achieved = current - start[list(fingers)]
    miss = desired - current
    blocked = []
    for row in range(len(fingers)):
        _, _, radial_vec = radial(current[row], obj_pos, axis)
        dist = float(np.linalg.norm(radial_vec))
        normal = radial_vec / dist if dist > 1e-6 else np.zeros(3)
        blocked.append(max(-float(np.dot(miss[row], normal)), 0.0))
    return (
        np.linalg.norm(miss, axis=1),
        np.linalg.norm(command, axis=1),
        np.asarray(blocked),
        np.linalg.norm(achieved, axis=1),
    )


def tip_contact(obj_pos, obj_quat_wxyz, tips, surface):
    rotation = wxyz_to_matrix(obj_quat_wxyz)
    axis = rotation[:, 2]
    phi = np.ones(N_TIPS * 4)
    jac = np.zeros((N_TIPS * 4, 6 + N_TIPS * 3))
    gaps = []
    for finger in range(N_TIPS):
        axial, dist, radial_vec = radial(tips[finger], obj_pos, axis)
        if dist < 1e-5 or abs(axial) > HALF_HEIGHT + 0.01:
            gaps.append(1.0)
            continue
        normal = radial_vec / dist
        tangent = np.cross(axis, normal)
        tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
        delta = tips[finger] - obj_pos
        point = np.zeros((3, 6))
        point[:, 0:3] = np.eye(3)
        point[:, 3:6] = -skew(delta) @ rotation
        relative = np.zeros((3, jac.shape[1]))
        relative[:, 0:6] = -point
        relative[:, 6 + 3 * finger: 9 + 3 * finger] = np.eye(3)
        gap = dist - surface
        gaps.append(gap)
        for row, direction in enumerate((tangent, axis, -tangent, -axis)):
            jac[4 * finger + row] = (normal + MU * direction) @ relative
            phi[4 * finger + row] = 0.5 * gap
    return phi, jac, gaps


def read_pose(base, env_id):
    base._refresh_lab()
    pads, _quats = pads_of(base, env_id)
    obj_pos = base.object_pos[env_id].detach().cpu().numpy()
    obj_wxyz = xyzw_to_wxyz(base.object_rot[env_id].detach().cpu().numpy())
    loads = pad_loads(base, env_id).detach().cpu().numpy()
    axis = object_axis(obj_wxyz)
    dists = [radial(pads[i], obj_pos, axis)[1] for i in range(5)]
    return pads, obj_pos, obj_wxyz, loads, axis, dists


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree("outputs/", ignore_errors=True)
    env_cfg.hold_pose = False
    env_cfg.freeze_on_success = False
    env_cfg.replay_cache = None
    env_cfg.randomize_mass = False
    env_cfg.gravity_curriculum = False
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    print(f"fingertip seed {env_cfg.seed} envs={args_cli.num_envs}", flush=True)

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
    streak = torch.zeros(args_cli.num_envs, device=base.device)
    env_id = None
    for search_step in range(4000):
        if not simulation_app.is_running():
            return
        env.step(env.zero_actions())
        loads_all = torch.stack([
            torch.linalg.norm(base._contact_sensor[sensor_id].data.normal_force_matrix_w[:, 0, 0, :], dim=-1)
            for sensor_id in range(5)
        ], dim=-1)
        height_ok = (base.object_pos[:, 2] > 0.610) & (base.object_pos[:, 2] < 0.625)
        three_pads = ((loads_all > LOAD_ON).sum(-1) >= 3) & height_ok
        streak = torch.where(three_pads, streak + 1, torch.zeros_like(streak))
        if search_step % 100 == 0:
            print(f"search {search_step} streak={int(streak.max().item())}", flush=True)
        ready = torch.nonzero(streak >= 8).flatten()
        if ready.numel() == 0:
            continue
        env_id = int(ready[0].item())
        base.frozen = True
        base.frozen_env_id = env_id
        base.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
        stable = 0
        accepted = False
        for _ in range(40):
            env.step(env.zero_actions())
            loads_one = pad_loads(base, env_id)
            height = float(base.object_pos[env_id, 2].item())
            if int((loads_one > LOAD_ON).sum().item()) >= 3 and 0.610 < height < 0.625:
                stable += 1
            else:
                stable = 0
            if stable >= 8:
                accepted = True
                break
        if accepted:
            print(f"grasp env {env_id} held three pads under gravity", flush=True)
            break
        print(f"reject env {env_id}: three pads did not hold with gravity down", flush=True)
        base.frozen = False
        env_id = None
        streak.zero_()
    if env_id is None:
        print("no three-fingertip grasp held", flush=True)
        return

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
    lower, upper = joint_limits(base, env_id)
    scale = float(base.cfg.action_scale)
    groups = finger_groups(base)
    pads, obj_pos, obj_wxyz, loads, axis, dists = read_pose(base, env_id)
    surface = surface_radius(pads, obj_pos, axis, loads)
    tips = choose_tips(loads, dists, surface)
    z0 = float(obj_pos[2])
    records = [{
        "event": "settle",
        "z": z0,
        "loads": loads.tolist(),
        "dists": dists,
        "surface": surface,
        "tips": [NAMES[i] for i in tips],
    }]
    print(
        f"settle z={z0:.4f} loads={[round(v, 3) for v in loads]} "
        f"tips={[NAMES[i] for i in tips]} surface={surface:.4f}",
        flush=True,
    )

    radius, half_length = cylinder_size(0.5)
    _, quats = pads_of(base, env_id)
    offset = face_offset(pads, quats, obj_pos, axis, loads)
    print(f"pad face offset {np.round(offset, 4).tolist()}", flush=True)
    stuck = 0
    if all(loads[i] > LOAD_ON for i in tips):
        print("three pads already loaded, skip seating", flush=True)
    for _ in range(0 if all(loads[i] > LOAD_ON for i in tips) else 40):
        if not simulation_app.is_running():
            return
        base._refresh_lab()
        pads, quats = pads_of(base, env_id)
        obj_pos = base.object_pos[env_id].detach().cpu().numpy()
        loads = pad_loads(base, env_id).detach().cpu().numpy()
        axis = object_axis(xyzw_to_wxyz(base.object_rot[env_id].detach().cpu().numpy()))
        if all(loads[i] > LOAD_ON for i in tips):
            break
        if float(obj_pos[2]) < z0 - DROP:
            records.append({"event": "result", "stop": "dropped while seating", "z": float(obj_pos[2])})
            print("stop dropped while seating", flush=True)
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOG_PATH.write_text("".join(json.dumps(row) + "\n" for row in records))
            if video is not None:
                video.close()
            return
        missing = [i for i in tips if loads[i] <= LOAD_ON]
        finger = missing[0]
        face = pads[finger] + xyzw_matrix(quats[finger]) @ offset
        target = closest_on_cylinder(face, obj_pos, axis, radius, half_length)
        error = target - face
        gap = float(np.linalg.norm(error))
        print(
            f"seat {NAMES[finger]} face_gap={gap:.4f} z={float(obj_pos[2]):.4f} "
            f"loads={[round(float(v), 3) for v in loads]}",
            flush=True,
        )
        if gap < 0.002:
            stuck += 1
            if stuck > 2:
                break
        else:
            stuck = 0
        step = error / max(gap, 1e-6) * min(gap, 0.0005)
        pad_jac = pad_jacobian(base, env_id)
        move_finger(
            env, base, env_id, groups, finger, step,
            pad_jac, lower, upper, scale, args_cli.num_envs,
        )
    pads, obj_pos, obj_wxyz, loads, axis, dists = read_pose(base, env_id)
    records.append({"event": "seated", "z": float(obj_pos[2]), "loads": loads.tolist(), "dists": dists})
    print(f"seated loads={[round(float(v), 3) for v in loads]} z={float(obj_pos[2]):.4f}", flush=True)
    if not all(loads[i] > LOAD_ON for i in tips):
        records.append({"event": "result", "stop": "fingertip grasp not formed", "loads": loads.tolist()})
        print("stop fingertip grasp not formed", flush=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("".join(json.dumps(row) + "\n" for row in records))
        if video is not None:
            video.close()
        return

    for _ in range(25):
        if not simulation_app.is_running():
            return
        pads, obj_pos, obj_wxyz, loads, axis, dists = read_pose(base, env_id)
        extras = [i for i in range(5) if i not in tips]
        if all(loads[i] < 0.2 and dists[i] - surface > 0.004 for i in extras):
            break
        if float(obj_pos[2]) < z0 - 0.01 or any(loads[i] < 0.3 for i in tips):
            break
        pad_jac = pad_jacobian(base, env_id)
        for finger in extras:
            if loads[finger] < 0.2 and dists[finger] - surface > 0.004:
                continue
            _, dist, radial_vec = radial(pads[finger], obj_pos, axis)
            if dist < 1e-5:
                continue
            move_finger(
                env, base, env_id, groups, finger, radial_vec / dist * 0.002,
                pad_jac, lower, upper, scale, args_cli.num_envs,
            )
            pads, obj_pos, obj_wxyz, loads, axis, dists = read_pose(base, env_id)
    pads, obj_pos, obj_wxyz, loads, axis, dists = read_pose(base, env_id)
    held = all(loads[i] > LOAD_ON for i in tips)
    records.append({
        "event": "retracted",
        "z": float(obj_pos[2]),
        "loads": loads.tolist(),
        "dists": dists,
        "held": held,
    })
    print(
        f"retracted loads={[round(float(v), 3) for v in loads]} held={held} z={float(obj_pos[2]):.4f}",
        flush=True,
    )
    if not held or float(obj_pos[2]) < z0 - DROP:
        records.append({"event": "result", "stop": "tripod lost while retracting", "loads": loads.tolist()})
        print("stop tripod lost while retracting", flush=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text("".join(json.dumps(row) + "\n" for row in records))
        if video is not None:
            video.close()
        return

    planner = FingertipExplicit()
    home = obj_pos.copy()
    z_hold = float(home[2])
    rot_axis = torch.tensor(base.cfg.rot_axis, dtype=torch.float32, device=device)
    goal = quat_mul(yaw_about(rot_axis, args_cli.goal_yaw, device).view(1, 4), base.object_rot[env_id].view(1, 4))[0]
    yaw = 0.0
    rotations = 0
    surface = surface_radius(pads, obj_pos, axis, loads)

    for step_id in range(args_cli.exec_steps):
        if not simulation_app.is_running():
            break
        pads, obj_pos, obj_wxyz, loads, axis, dists = read_pose(base, env_id)
        tips_pos = pads[list(tips)]
        phi, jac, gaps = tip_contact(obj_pos, obj_wxyz, tips_pos, surface)
        curr_q = np.concatenate((obj_pos, obj_wxyz, tips_pos.reshape(-1)))
        target_q = xyzw_to_wxyz(goal.detach().cpu().numpy())
        target_q = target_q / np.linalg.norm(target_q)
        action, status, cost, predicted = planner.plan(curr_q, home, target_q, phi, jac)
        q_before = base.object_rot[env_id].detach().clone()
        residual, commanded, blocked, achieved = track_tips(
            env, base, env_id, groups, tips, action,
            lower, upper, scale, args_cli.num_envs,
            obj_pos, axis, surface,
        )
        base._refresh_lab()
        turned = axis_angle_from_quat(quat_mul(base.object_rot[env_id], quat_conjugate(q_before)))
        yaw += float(torch.dot(turned, rot_axis).item())
        curr_wxyz = xyzw_to_wxyz(base.object_rot[env_id].detach().cpu().numpy())
        quat_error = 1.0 - float(np.dot(curr_wxyz, target_q) ** 2)
        z = float(base.object_pos[env_id, 2].item())
        if quat_error < args_cli.goal_tol:
            rotations += 1
            goal = quat_mul(
                yaw_about(rot_axis, args_cli.goal_yaw, device).view(1, 4),
                base.object_rot[env_id].view(1, 4),
            )[0]
            print(f"goal advance {rotations} at step {step_id}", flush=True)
        loads = pad_loads(base, env_id).detach().cpu().numpy()
        row = {
            "event": "step",
            "step": step_id,
            "yaw": yaw,
            "quat_error": quat_error,
            "z": z,
            "status": status,
            "cost": cost,
            "action_norm": float(np.linalg.norm(action)),
            "track_residual_mm": (residual * 1000).tolist(),
            "commanded_mm": (commanded * 1000).tolist(),
            "blocked_inward_mm": (blocked * 1000).tolist(),
            "achieved_mm": (achieved * 1000).tolist(),
            "predicted_z": float(predicted[2]),
            "gaps": gaps,
            "loads": loads.tolist(),
        }
        records.append(row)
        print(
            f"exec {step_id} status={status} yaw={yaw:.4f} quat_error={quat_error:.3f} "
            f"z={z:.4f} loads={[round(float(v), 3) for v in loads]} "
            f"|u|={row['action_norm']:.4f} pred_z={predicted[2]:.4f} "
            f"track_mm={[round(float(v), 2) for v in residual * 1000]} "
            f"cmd_mm={[round(float(v), 2) for v in commanded * 1000]} "
            f"blocked_mm={[round(float(v), 2) for v in blocked * 1000]} "
            f"got_mm={[round(float(v), 2) for v in achieved * 1000]}",
            flush=True,
        )
        record_frame(video, base)
        xy = base.object_pos[env_id, :2].detach().cpu().numpy()
        left_xy = float(np.linalg.norm(xy - home[:2])) > 0.03
        pads_clear = bool(np.all(loads < 0.05))
        if z < z_hold - DROP or left_xy or pads_clear:
            reason = "dropped" if z < z_hold - DROP or left_xy else "pads left the cylinder"
            records.append({
                "event": "result", "stop": reason, "yaw": yaw, "z": z,
                "rotations": rotations, "loads": loads.tolist(),
            })
            print(f"stop {reason} z={z:.4f} yaw={yaw:.4f}", flush=True)
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
