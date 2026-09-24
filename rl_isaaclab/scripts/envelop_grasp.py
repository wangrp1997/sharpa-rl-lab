# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Lock the cylinder, close the fingers onto it, then test six gravity directions.

The object pose is written back every physics step while the fingers move.
Each finger stops when its elastomer force is in a small band, and backs off
if the force grows past that band. Gravity is changed only after the lock
is released.
"""

import argparse
import sys
import shutil
import time

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Envelop a locked cylinder, then test gravity.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
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
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
CLOSE_STEP = 0.00035
FORCE_HOLD = 0.8
FORCE_BACK = 2.5
GRAVITY = (
    ("down", (0.0, 0.0, -9.81)),
    ("up", (0.0, 0.0, 9.81)),
    ("+y", (0.0, 9.81, 0.0)),
    ("-y", (0.0, -9.81, 0.0)),
    ("+x", (9.81, 0.0, 0.0)),
    ("-x", (-9.81, 0.0, 0.0)),
)


def finger_groups(joint_names, actuated):
    groups = []
    for prefix in FINGER_PREFIXES:
        groups.append([i for i, joint_id in enumerate(actuated) if joint_names[joint_id].startswith(prefix)])
    return groups


def elastomer_force(base) -> torch.Tensor:
    forces = []
    for sensor_id in range(5):
        hist = base._contact_sensor[sensor_id].data.net_forces_w_history
        forces.append(torch.linalg.norm(hist[0, 0, 0]))
    return torch.stack(forces)


def lock_object(base, pose_w: torch.Tensor) -> None:
    base.object.write_root_pose_to_sim(pose_w.view(1, 7), torch.tensor([0], device=base.device))
    base.object.write_root_velocity_to_sim(torch.zeros((1, 6), device=base.device), torch.tensor([0], device=base.device))


def joint_increment(jac_lin, groups, columns, delta):
    dq = torch.zeros(len(columns), dtype=torch.float32)
    eye = torch.eye(3)
    for finger, cols in enumerate(groups):
        if len(cols) == 0:
            continue
        joint_ids = [columns[c] for c in cols]
        j_tip = jac_lin[finger][:, joint_ids]
        gram = j_tip @ j_tip.T + 1e-2 * eye
        step = torch.clamp(j_tip.T @ torch.linalg.solve(gram, delta[finger]), -0.004, 0.004)
        for local, col in enumerate(cols):
            dq[col] = step[local]
    return dq


def apply_increment(env, base, dq, anchor_target):
    q = base.hand_dof_pos[0, base.actuated_dof_indices]
    target = anchor_target.clone()
    move = dq.abs() > 0
    target = torch.where(move.to(q.device), q + dq.to(q.device), target)
    lo = base.hand_dof_lower_limits[0, base.actuated_dof_indices]
    hi = base.hand_dof_upper_limits[0, base.actuated_dof_indices]
    target = torch.maximum(torch.minimum(target, hi), lo)
    prev = base.prev_targets[0, base.actuated_dof_indices]
    command = torch.zeros(base.num_hand_dofs, device=env.device)
    command[base.actuated_dof_indices] = torch.clamp((target - prev) / base.cfg.action_scale, -1.0, 1.0)
    env.step(command.view(1, -1))
    return target


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree("outputs/", ignore_errors=True)
    env_cfg.hold_pose = True
    env_cfg.decimation = 1
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(torch.initial_seed() % (2**31 - 1))
    print(f"grasp seed {env_cfg.seed}", flush=True)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()
    base = env.unwrapped
    groups = finger_groups(base.hand.joint_names, base.actuated_dof_indices)
    body_offset = 1 if base.hand.is_fixed_base else 0
    dof0 = int(base.hand.num_base_dofs)
    jac_bodies = [b - body_offset for b in base.elastomer_ids]
    jac_columns = [dof0 + j for j in base.actuated_dof_indices]
    print(f"elastomer jacobian bodies={jac_bodies}", flush=True)

    # One second of simulation lets the palm catch the cylinder before the fingers close.
    for settle_id in range(480):
        env.step(env.zero_actions())
        if settle_id % 20 == 0:
            time.sleep(0.01)
        if not simulation_app.is_running():
            return
    base._refresh_lab()
    z0 = float(base.object_pos[0, 2].item())
    print(f"settled z={z0:.3f}", flush=True)
    if z0 < 0.55:
        print("object is not in the hand, stop", flush=True)
        return

    # Heavy object and no gravity: fingers can press on it without pushing it away.
    # Lift it 2 cm into the fingers. At the cradled height the pads sit above the cylinder.
    base.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
    base.set_mass(base.object, torch.tensor([20.0], device=base.device), base.num_envs)
    base._refresh_lab()
    pose_w = torch.cat([base.object.data.root_pos_w.torch[0], base.object.data.root_quat_w.torch[0]]).clone()
    pose_w[2] += 0.02
    lock_object(base, pose_w)
    held = base.prev_targets[0, base.actuated_dof_indices].clone()
    flex_cols = []
    for col, joint_id in enumerate(base.actuated_dof_indices):
        name = base.hand.joint_names[joint_id]
        if name.endswith(("_FE", "_IP", "_PIP", "_DIP")):
            finger = next(i for i, prefix in enumerate(FINGER_PREFIXES) if name.startswith(prefix))
            flex_cols.append((col, finger))
    print("object held fixed, fingers closing", flush=True)

    for step_id in range(220):
        if not simulation_app.is_running():
            return
        base._refresh_lab()
        force = elastomer_force(base)
        if int((force > FORCE_HOLD).sum().item()) >= 3 and float(force.max()) < FORCE_BACK:
            print(f"contact at step {step_id} force={[round(float(v), 3) for v in force]}", flush=True)
            break
        dq = torch.zeros(len(base.actuated_dof_indices))
        for col, finger in flex_cols:
            if float(force[finger]) > FORCE_BACK:
                dq[col] = -0.002
            elif float(force[finger]) < FORCE_HOLD:
                dq[col] = 0.0012
        held = apply_increment(env, base, dq, held)
        if step_id % 40 == 0:
            print(f"close step={step_id} force={[round(float(v), 3) for v in force]}", flush=True)
        time.sleep(0.02)
    else:
        print(f"close finished without 3 contacts, force={[round(float(v), 3) for v in elastomer_force(base)]}", flush=True)

    for _ in range(60):
        env.step(env.zero_actions())
        time.sleep(0.01)
        if not simulation_app.is_running():
            return
    base._refresh_lab()
    z0 = float(base.object_pos[0, 2].item())
    base.set_mass(base.object, torch.tensor([0.05], device=base.device), base.num_envs)
    print(f"release lock z={z0:.3f} force={[round(float(v), 3) for v in elastomer_force(base)]}", flush=True)

    for name, gravity in GRAVITY:
        if not simulation_app.is_running():
            return
        base.physics_sim_view.set_gravity(carb.Float3(*gravity))
        # A settled body stays asleep across a gravity change, so give it a small shove.
        wake = torch.zeros((1, 6), device=base.device)
        wake[0, 2] = 0.05
        base.object.write_root_velocity_to_sim(wake, torch.tensor([0], device=base.device))
        z_min = z0
        for _ in range(480):
            env.step(env.zero_actions())
            base._refresh_lab()
            z_min = min(z_min, float(base.object_pos[0, 2].item()))
            time.sleep(0.004)
        held_up = z_min > z0 - 0.02
        print(
            f"gravity {name}: {'hold' if held_up else 'dropped'} z_min={z_min:.3f} "
            f"force={[round(float(v), 2) for v in elastomer_force(base)]}",
            flush=True,
        )

    print("gravity test done, window stays open", flush=True)
    while simulation_app.is_running():
        env.step(env.zero_actions())
        time.sleep(0.02)


if __name__ == "__main__":
    main()
    simulation_app.close()
