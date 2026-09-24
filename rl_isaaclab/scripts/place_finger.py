"""Place unloaded fingers on the known cylinder. Do not spin it.

Contact is each elastomer's normal force against its own reading at the start.
Fingers that are already loaded keep a zero command. A free finger is moved
toward the closest point on the cylinder when that increment lies inside the
joint limits. Tangential rotation is not commanded.
"""

import argparse
import json
import sys
import shutil
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Place free fingers on the cylinder.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=15)
parser.add_argument("--place_steps", type=int, default=25)
parser.add_argument("--scale", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import numpy as np
import torch
import importlib.util

from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from isaaclab.envs import DirectRLEnvCfg
import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

_step_path = Path(__file__).with_name("reach_shift_step.py")
_step = importlib.util.spec_from_file_location("reach_shift_step", _step_path)
reach_shift_step = importlib.util.module_from_spec(_step)
_step.loader.exec_module(reach_shift_step)

cylinder_axis = reach_shift_step.cylinder_axis
cylinder_size = reach_shift_step.cylinder_size
closest_on_cylinder = reach_shift_step.closest_on_cylinder
joint_increment = reach_shift_step.joint_increment
MAX_DQ = reach_shift_step.MAX_DQ
LAND_TOL = reach_shift_step.LAND_TOL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FINGER_PREFIXES = (
    "right_thumb_",
    "right_index_",
    "right_middle_",
    "right_ring_",
    "right_pinky_",
)
NAMES = ("thumb", "index", "middle", "ring", "pinky")
LOG_PATH = Path("logs/live_diag/place_finger.jsonl")


def finger_groups(joint_names, actuated):
    return [
        [i for i, joint_id in enumerate(actuated) if joint_names[joint_id].startswith(prefix)]
        for prefix in FINGER_PREFIXES
    ]


def normal_loads(base) -> np.ndarray:
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.normal_force_matrix_w[:, 0, 0, :]
        columns.append(torch.linalg.norm(force, dim=-1))
    return torch.stack(columns, dim=-1).detach().cpu().numpy()


def write_action(actions, env_id, dq, actuated, scale):
    for column, joint_id in enumerate(actuated):
        actions[env_id, joint_id] = float(np.clip(dq[column] / scale, -1.0, 1.0))


def pads_of(base, env_id):
    origin = base.scene.env_origins[env_id].detach().cpu().numpy()
    return base.hand.data.body_link_state_w[env_id, base.elastomer_ids, :3].detach().cpu().numpy() - origin


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree("outputs/", ignore_errors=True)
    env_cfg.hold_pose = False
    env_cfg.freeze_on_success = True
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(torch.initial_seed() % (2**31 - 1))
    print(f"grasp seed {env_cfg.seed}", flush=True)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

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
    for _ in range(args_cli.settle_steps):
        if not simulation_app.is_running():
            return
        env.step(env.zero_actions())

    env_id = int(base.frozen_env_id)
    base._focus_env(env_id)
    print(f"PLACE grasp env {env_id}", flush=True)

    groups = finger_groups(base.hand.joint_names, base.actuated_dof_indices)
    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    elastomer_jac = [int(body) - body_offset for body in base.elastomer_ids]
    jac_columns = [dof0 + int(joint) for joint in base.actuated_dof_indices]
    action_scale = float(base.cfg.action_scale)
    noise = float(base.cfg.contact_sensor_noise)
    radius, half_length = cylinder_size(args_cli.scale)
    records = []

    base._refresh_lab()
    loads0 = normal_loads(base)[env_id]
    empty = float(np.min(loads0))
    loaded = [i for i in range(5) if loads0[i] > empty + noise]
    free = [i for i in range(5) if i not in loaded]
    pads = pads_of(base, env_id)
    center0 = base.object_pos[env_id].detach().cpu().numpy()
    axis0 = cylinder_axis(base.object_rot[env_id].detach().cpu().numpy())
    z0 = float(center0[2])
    gaps = []
    for finger in free:
        surface = closest_on_cylinder(pads[finger], center0, axis0, radius, half_length)
        gaps.append((float(np.linalg.norm(surface - pads[finger])), finger))
    gaps.sort()
    contact_count = len(loaded)
    records.append({
        "event": "contacts",
        "env": env_id,
        "seed": int(env_cfg.seed),
        "count": contact_count,
        "loads": loads0.tolist(),
        "loaded": [NAMES[i] for i in loaded],
        "free": [NAMES[i] for _, i in gaps],
        "gaps": [gap for gap, _ in gaps],
    })
    print(
        f"count={contact_count} loaded={[NAMES[i] for i in loaded]} "
        f"free={[NAMES[i] for _, i in gaps]} loads={[round(float(v), 3) for v in loads0]}",
        flush=True,
    )

    held_ok = True
    touched = []
    if contact_count != 3:
        records.append({"event": "skip", "reason": "adjust contacts only for a three-finger grasp", "count": contact_count})
        print(f"count={contact_count}, do not adjust contacts", flush=True)
        gaps = []
    for gap0, finger in gaps:
        if not held_ok or not simulation_app.is_running():
            break
        for step_id in range(args_cli.place_steps):
            if not simulation_app.is_running():
                return
            base._refresh_lab()
            pads = pads_of(base, env_id)
            center = base.object_pos[env_id].detach().cpu().numpy()
            axis = cylinder_axis(base.object_rot[env_id].detach().cpu().numpy())
            surface = closest_on_cylinder(pads[finger], center, axis, radius, half_length)
            q_all = base.hand_dof_pos[env_id].detach().cpu().numpy()
            lower_all = base.hand_dof_lower_limits[env_id].detach().cpu().numpy()
            upper_all = base.hand_dof_upper_limits[env_id].detach().cpu().numpy()
            jac = base.hand.data.body_link_jacobian_w.torch[env_id][:, :3, :].detach().cpu().numpy()
            cols = groups[finger]
            columns = [jac_columns[col] for col in cols]
            block = jac[elastomer_jac[finger]][:, columns]
            joint_ids = [int(base.actuated_dof_indices[col]) for col in cols]
            step, residual = joint_increment(
                block,
                surface - pads[finger],
                q_all[joint_ids],
                lower_all[joint_ids],
                upper_all[joint_ids],
                MAX_DQ,
            )
            if float(np.linalg.norm(step)) < 1e-5:
                records.append({
                    "event": "no_increment",
                    "finger": NAMES[finger],
                    "step": step_id,
                    "residual": residual,
                    "gap0": gap0,
                })
                print(f"{NAMES[finger]} no increment at {step_id}, residual={residual:.4f}", flush=True)
                break
            actions = torch.zeros_like(base.prev_targets)
            dq = np.zeros(len(base.actuated_dof_indices), dtype=np.float64)
            for local, col in enumerate(cols):
                dq[col] = step[local]
            write_action(actions, env_id, dq, base.actuated_dof_indices, action_scale)
            env.step(actions)
            base._refresh_lab()
            loads = normal_loads(base)[env_id]
            pos = base.object_pos[env_id].detach().cpu().numpy()
            on_finger = float(loads[finger]) > float(loads0[finger]) + noise
            held_ok = all(float(loads[i]) > empty + noise for i in loaded)
            records.append({
                "event": "place",
                "finger": NAMES[finger],
                "step": step_id,
                "residual": residual,
                "loads": loads.tolist(),
                "z": float(pos[2]),
                "touched": on_finger,
                "held": held_ok,
            })
            print(
                f"place {NAMES[finger]} {step_id} residual={residual:.4f} "
                f"loads={[round(float(v), 3) for v in loads]} held={held_ok}",
                flush=True,
            )
            if float(pos[2]) < z0 - 0.02:
                records.append({"event": "dropped", "finger": NAMES[finger]})
                print("object dropped", flush=True)
                held_ok = False
                break
            if on_finger:
                touched.append(NAMES[finger])
                break
            if not held_ok:
                print("a loaded finger returned to its empty reading", flush=True)
                break
            if residual <= LAND_TOL:
                break

    loads1 = normal_loads(base)[env_id]
    records.append({
        "event": "result",
        "loads0": loads0.tolist(),
        "loads1": loads1.tolist(),
        "touched": touched,
        "held": held_ok,
        "ok": bool(held_ok and len(free) > 0 and len(touched) == len(free)),
    })
    print(
        f"result touched={touched} held={held_ok} loads={[round(float(v), 3) for v in loads1]}",
        flush=True,
    )
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {LOG_PATH}", flush=True)
    while simulation_app.is_running():
        env.step(env.zero_actions())


if __name__ == "__main__":
    main()
    simulation_app.close()
