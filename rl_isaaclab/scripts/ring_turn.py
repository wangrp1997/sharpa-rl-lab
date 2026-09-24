# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Turn the cylinder about z under the current finger ring.

The object pose is read from the simulator. Each control step moves the
fingertips a few millimetres along the tangential direction of that pose.
At most one finger may leave the settled joint pose; the other four stay
near it. Fingertip force is logged and does not gate the motion.
"""

import argparse
import sys
import shutil
import time

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Turn a cylinder inside the current finger ring.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from isaaclab.envs import DirectRLEnvCfg
import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FINGER_PREFIXES = (
    "right_thumb_",
    "right_index_",
    "right_middle_",
    "right_ring_",
    "right_pinky_",
)
HOLD_BAND = 0.08
LEAVE_BAND = 0.35
TANGENT_STEP = 0.0005
LIFT_STEP = 0.0006
MAX_GAP = 2.0


def yaw_xyzw(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat[0], quat[1], quat[2], quat[3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def finger_groups(joint_names, actuated):
    groups = []
    for prefix in FINGER_PREFIXES:
        groups.append([i for i, joint_id in enumerate(actuated) if joint_names[joint_id].startswith(prefix)])
    return groups


def object_in_ring(tip_xy: torch.Tensor, obj_xy: torch.Tensor) -> tuple[bool, int, float]:
    rel = tip_xy - obj_xy
    ang = torch.atan2(rel[:, 1], rel[:, 0])
    order = torch.argsort(ang)
    ang_s = ang[order]
    gaps = torch.diff(ang_s, append=ang_s[:1] + 2.0 * torch.pi)
    gap_id = int(torch.argmax(gaps).item())
    # The finger on the counter-clockwise side of the largest gap can move into it.
    leaving = int(order[(gap_id + 1) % tip_xy.shape[0]].item())
    return bool(gaps.max().item() < MAX_GAP), leaving, float(gaps.max().item())


def fingertip_delta(tip_xy, obj_xy, leaving, ring_ok, radii, radius0):
    rel = tip_xy - obj_xy
    radius = torch.linalg.norm(rel, dim=-1).clamp_min(1e-4)
    tangent = torch.stack((-rel[:, 1], rel[:, 0]), dim=-1) / radius[:, None]
    radial = rel / radius[:, None]
    delta = tangent * TANGENT_STEP
    if ring_ok:
        outward = radius[leaving] < radius0[leaving] + 0.008
        if outward:
            delta[leaving] = radial[leaving] * LIFT_STEP + tangent[leaving] * TANGENT_STEP
        else:
            delta[leaving] = tangent[leaving] * (2.0 * TANGENT_STEP)
    else:
        # The ring has opened. Pull every fingertip back toward the settled radius.
        delta = -radial * torch.clamp(radii - radius0, min=0.0).unsqueeze(-1)
    return delta


def joint_increment(jac_lin, groups, columns, delta):
    """jac_lin: (5, 3, n_dof) on CPU, rows x/y/z. dq follows the actuated list."""
    dq = torch.zeros(len(columns), dtype=torch.float32)
    eye = torch.eye(delta.shape[-1])
    for finger, cols in enumerate(groups):
        if len(cols) == 0:
            continue
        joint_ids = [columns[c] for c in cols]
        j_tip = jac_lin[finger][:, joint_ids]
        gram = j_tip @ j_tip.T + 1e-2 * eye
        step = torch.clamp(j_tip.T @ torch.linalg.solve(gram, delta[finger]), -0.008, 0.008)
        for local, col in enumerate(cols):
            dq[col] = step[local]
    return dq


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree("outputs/", ignore_errors=True)
    env_cfg.hold_pose = True
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(torch.initial_seed() % (2**31 - 1))
    print(f"grasp seed {env_cfg.seed}", flush=True)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()
    base = env.unwrapped
    groups = finger_groups(base.hand.joint_names, base.actuated_dof_indices)
    print("finger joint groups:", groups, flush=True)

    for _ in range(args_cli.settle_steps):
        env.step(env.zero_actions())
        time.sleep(0.05)
        if not simulation_app.is_running():
            return

    base._refresh_lab()
    anchor = base.hand_dof_pos[0, base.actuated_dof_indices].clone()
    radius0 = torch.linalg.norm(base.fingertip_pos[0, :, :2] - base.object_pos[0, :2], dim=-1).clone()
    yaw0 = float(yaw_xyzw(base.object_rot[0]).item())
    z0 = float(base.object_pos[0, 2].item())
    print(f"settled yaw={yaw0:.3f} z={z0:.3f} radii={radius0.tolist()}", flush=True)

    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    jac_bodies = [b - body_offset for b in base.finger_bodies]
    jac_columns = [dof0 + j for j in base.actuated_dof_indices]
    print(
        f"jacobian map fixed_base={base.hand.is_fixed_base} dof0={dof0} "
        f"bodies={jac_bodies} columns={jac_columns}",
        flush=True,
    )
    tangent_sign = 1.0
    yaw_mark = yaw0

    step_id = 0
    while simulation_app.is_running():
        base._refresh_lab()
        q = base.hand_dof_pos[0, base.actuated_dof_indices]
        obj = base.object_pos[0]
        yaw = float(yaw_xyzw(base.object_rot[0]).item())
        z = float(obj[2].item())
        sinking = z < z0 - 0.006
        tip_xy = base.fingertip_pos[0, :, :2]
        ring_ok, leaving, gap = object_in_ring(tip_xy, obj[:2])
        radii = torch.linalg.norm(tip_xy - obj[:2], dim=-1)
        if sinking or (not ring_ok):
            rel = tip_xy - obj[:2]
            radius = torch.linalg.norm(rel, dim=-1).clamp_min(1e-4)
            delta_xy = -(rel / radius[:, None]) * 0.0008
            leaving = -1
        else:
            delta_xy = tangent_sign * fingertip_delta(tip_xy, obj[:2], leaving, True, radii, radius0)
        delta = torch.zeros(5, 3)
        delta[:, :2] = delta_xy.detach().cpu()
        jac = base.hand.data.body_link_jacobian_w.torch[0].detach().cpu()
        jac_lin = jac[jac_bodies][:, :3, :]
        dq = joint_increment(jac_lin, groups, jac_columns, delta)
        if step_id > 0 and step_id % 40 == 0:
            if (yaw - yaw_mark) < 0.005:
                tangent_sign *= -1.0
                print(f"flip tangent sign to {tangent_sign:+.0f}", flush=True)
            yaw_mark = yaw
        target = q + dq.to(q.device)
        lo = base.hand_dof_lower_limits[0, base.actuated_dof_indices]
        hi = base.hand_dof_upper_limits[0, base.actuated_dof_indices]
        for finger, cols in enumerate(groups):
            band = LEAVE_BAND if finger == leaving else HOLD_BAND
            for col in cols:
                target[col] = torch.clamp(target[col], anchor[col] - band, anchor[col] + band)
        target = torch.maximum(torch.minimum(target, hi), lo)

        prev = base.prev_targets[0, base.actuated_dof_indices]
        command = torch.zeros(base.num_hand_dofs, device=env.device)
        command[base.actuated_dof_indices] = torch.clamp((target - prev) / base.cfg.action_scale, -1.0, 1.0)
        env.step(command.view(1, -1))
        time.sleep(0.08)

        if step_id % 20 == 0:
            print(
                f"step={step_id} yaw={yaw - yaw0:+.3f} z={float(obj[2]):.3f} "
                f"leave={leaving} gap={gap:.2f}",
                flush=True,
            )
        step_id += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
