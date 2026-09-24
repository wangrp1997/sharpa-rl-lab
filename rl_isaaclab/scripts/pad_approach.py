"""After a live grasp freezes, move an unloaded fingertip pad toward the surface.

The surface is the same Gaussian-process implicit surface as contact_gpis.py.
Only joints of the chosen finger are used. A finger is left still when no
joint increment inside the limits moves its elastomer 0.5 mm inward. A step
stops if thumb, middle, or ring force falls below 1 N. Poses are not written
back.
"""

import argparse
import json
import sys
import shutil
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Approach with an unloaded fingertip pad.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=15)
parser.add_argument("--max_steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import numpy as np
import torch

from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from isaaclab.envs import DirectRLEnvCfg
import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config
import importlib.util

_gpis_path = Path(__file__).with_name("contact_gpis.py")
_gpis = importlib.util.spec_from_file_location("contact_gpis", _gpis_path)
contact_gpis = importlib.util.module_from_spec(_gpis)
_gpis.loader.exec_module(contact_gpis)
NAMES = contact_gpis.NAMES
LOAD_MIN = contact_gpis.LOAD_MIN
fit = contact_gpis.fit
predict = contact_gpis.predict

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FINGER_PREFIXES = (
    "right_thumb_",
    "right_index_",
    "right_middle_",
    "right_ring_",
    "right_pinky_",
)
HOLD_MIN = 1.0
PAD_STEP = 5e-4
MAX_DQ = 0.008
LOG_PATH = Path("logs/live_diag/pad_approach.jsonl")


def finger_groups(joint_names, actuated):
    return [
        [i for i, joint_id in enumerate(actuated) if joint_names[joint_id].startswith(prefix)]
        for prefix in FINGER_PREFIXES
    ]


def fingertip_loads(base) -> np.ndarray:
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.force_matrix_w[:, 0, 0, :]
        columns.append(torch.linalg.norm(force, dim=-1))
    return torch.stack(columns, dim=-1).detach().cpu().numpy()


def surface_at(point, points, length, gram, weight, center, scale):
    local = (point - center) / scale
    mean, variance = predict(local, points, length, gram, weight)
    grad = np.zeros(3)
    eps = 1e-4
    for axis in range(3):
        nudged = local.copy()
        nudged[axis] += eps / scale
        mean_hi, _ = predict(nudged, points, length, gram, weight)
        grad[axis] = (mean_hi - mean) / eps
    return mean, variance, grad


def fit_loaded(pads, forces, loads):
    active = [i for i in range(5) if loads[i] > LOAD_MIN and np.linalg.norm(forces[i]) > 1e-6]
    points = pads[active]
    normals = forces[active]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    center = points.mean(axis=0)
    scale = max(float(np.max(np.abs(points - center))), 1e-4)
    local = (points - center) / scale
    dists = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1)
    length = 1.1 * float(np.max(dists))
    gram, weight = fit(local, normals, length)
    return active, local, length, gram, weight, center, scale


def pad_step(jac_row, groups, finger, direction):
    cols = groups[finger]
    if not cols:
        return None, "no joints"
    block = jac_row[:, cols]
    gram = block @ block.T + 1e-4 * np.eye(3)
    step = block.T @ np.linalg.solve(gram, direction)
    step = np.clip(step, -MAX_DQ, MAX_DQ)
    achieved = block @ step
    if np.linalg.norm(achieved - direction) > 0.5 * np.linalg.norm(direction):
        return None, "jacobian cannot make the pad step"
    if np.linalg.norm(step) < 1e-5:
        return None, "joint increment is zero"
    return (cols, step), "ok"


def write_finger(actions, env_id, cols, step, actuated, scale, q, lower, upper):
    dq = np.zeros(len(actuated))
    for local, col in enumerate(cols):
        joint_id = int(actuated[col])
        moved = float(q[joint_id] + step[local])
        if moved < lower[joint_id] or moved > upper[joint_id]:
            return False
        dq[col] = step[local]
    for col, joint_id in enumerate(actuated):
        actions[env_id, joint_id] = float(np.clip(dq[col] / scale, -1.0, 1.0))
    return True


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

    live = torch.nonzero(base.live_success.reshape(-1)).flatten().tolist()
    if not live:
        print("no live grasp", flush=True)
        return
    env_id = int(live[0])
    base._focus_env(env_id)
    print(f"PAD grasp env {env_id}", flush=True)

    groups = finger_groups(base.hand.joint_names, base.actuated_dof_indices)
    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    elastomer_jac = [int(body) - body_offset for body in base.elastomer_ids]
    jac_columns = [dof0 + int(joint) for joint in base.actuated_dof_indices]
    scale = float(base.cfg.action_scale)
    lower = base.hand_dof_lower_limits[env_id].detach().cpu().numpy()
    upper = base.hand_dof_upper_limits[env_id].detach().cpu().numpy()
    records = []

    for _attempt in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        base._refresh_lab()
        loads = fingertip_loads(base)[env_id]
        origin = base.scene.env_origins[env_id].detach().cpu().numpy()
        pads = base.hand.data.body_link_state_w[env_id, base.elastomer_ids, :3].detach().cpu().numpy() - origin
        forces = []
        for sensor_id in range(5):
            forces.append(base._contact_sensor[sensor_id].data.force_matrix_w[env_id, 0, 0].detach().cpu().numpy())
        forces = np.stack(forces)
        if int(np.sum(loads > LOAD_MIN)) < 3:
            records.append({"event": "too_few_contacts", "loads": loads.tolist()})
            print(f"stop: loaded contacts {[round(v, 3) for v in loads]}", flush=True)
            break
        active, local, length, gram, weight, center, pos_scale = fit_loaded(pads, forces, loads)
        idle = [i for i in range(5) if i not in active]
        if not idle:
            records.append({"event": "all_pads_loaded", "loads": loads.tolist()})
            break
        ranked = []
        for finger in idle:
            _, variance, _ = surface_at(pads[finger], local, length, gram, weight, center, pos_scale)
            ranked.append((variance, finger))
        ranked.sort(reverse=True)
        moved = False
        for _, finger in ranked:
            mean, _, grad = surface_at(pads[finger], local, length, gram, weight, center, pos_scale)
            grad_norm = np.linalg.norm(grad)
            row = {
                "event": "check",
                "finger": NAMES[finger],
                "mean": mean,
                "loads": loads.tolist(),
            }
            if grad_norm < 1e-6:
                row.update({"moved": False, "reason": "no gradient"})
                records.append(row)
                print(f"skip {NAMES[finger]}: no gradient", flush=True)
                continue
            inward = -grad / grad_norm * PAD_STEP
            jac = base.hand.data.body_link_jacobian_w.torch[env_id, elastomer_jac[finger], :3, :]
            jac = jac[:, jac_columns].detach().cpu().numpy()
            solved = pad_step(jac, groups, finger, inward)
            if solved[0] is None:
                row.update({"moved": False, "reason": solved[1]})
                records.append(row)
                print(f"skip {NAMES[finger]}: {solved[1]}", flush=True)
                continue
            cols, step = solved[0]
            actions = torch.zeros_like(base.prev_targets)
            q = base.hand_dof_pos[env_id].detach().cpu().numpy()
            allowed = write_finger(
                actions, env_id, cols, step, base.actuated_dof_indices, scale, q, lower, upper
            )
            if not allowed:
                row.update({"moved": False, "reason": "joint limit"})
                records.append(row)
                print(f"skip {NAMES[finger]}: joint limit", flush=True)
                continue
            env.step(actions)
            base._refresh_lab()
            loads_after = fingertip_loads(base)[env_id]
            held = all(loads_after[i] >= HOLD_MIN for i in (0, 2, 3))
            touched = bool(loads_after[finger] > LOAD_MIN)
            row.update({
                "moved": True,
                "reason": "ok",
                "loads_after": loads_after.tolist(),
                "held": held,
                "touched": touched,
            })
            records.append(row)
            print(
                f"step {NAMES[finger]} loads={[round(float(v), 3) for v in loads_after]} "
                f"held={held} touched={touched}",
                flush=True,
            )
            moved = True
            if touched or not held:
                moved = False
            break
        if not moved:
            break

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
