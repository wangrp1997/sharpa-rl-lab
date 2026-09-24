"""Turn a frozen grasp one small step at a time from the contact surface.

The step is surface_turn_step.plan_once. Only fingers that already carry
more than 0.5 N are commanded. Poses are not written back.
"""

import argparse
import json
import sys
import shutil
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Turn from the contact surface.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=15)
parser.add_argument("--max_steps", type=int, default=24)
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
_step_path = Path(__file__).with_name("surface_turn_step.py")
_step = importlib.util.spec_from_file_location("surface_turn_step", _step_path)
surface_turn_step = importlib.util.module_from_spec(_step)
_step.loader.exec_module(surface_turn_step)

NAMES = contact_gpis.NAMES
LOAD_MIN = contact_gpis.LOAD_MIN
fit = contact_gpis.fit
predict = contact_gpis.predict
plan_once = surface_turn_step.plan_once
axis_from_normals = surface_turn_step.axis_from_normals
yaw_about = surface_turn_step.yaw_about
explicit_step = surface_turn_step.explicit_step

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

FINGER_PREFIXES = (
    "right_thumb_",
    "right_index_",
    "right_middle_",
    "right_ring_",
    "right_pinky_",
)
LOG_PATH = Path("logs/live_diag/surface_turn.jsonl")
DROP = 0.02


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
    mean, _ = predict(local, points, length, gram, weight)
    grad = np.zeros(3)
    eps = 1e-4
    for axis in range(3):
        nudged = local.copy()
        nudged[axis] += eps / scale
        mean_hi, _ = predict(nudged, points, length, gram, weight)
        grad[axis] = (mean_hi - mean) / eps
    return mean, grad


def fit_loaded(pads, forces, active):
    points = pads[active]
    normals = forces[active]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    center = points.mean(axis=0)
    scale = max(float(np.max(np.abs(points - center))), 1e-4)
    local = (points - center) / scale
    dists = np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1)
    length = 1.1 * float(np.max(dists))
    gram, weight = fit(local, normals, length)
    return local, length, gram, weight, center, scale, normals


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
    print(f"TURN grasp env {env_id}", flush=True)

    groups = finger_groups(base.hand.joint_names, base.actuated_dof_indices)
    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    elastomer_jac = [int(body) - body_offset for body in base.elastomer_ids]
    jac_columns = [dof0 + int(joint) for joint in base.actuated_dof_indices]
    scale = float(base.cfg.action_scale)
    q_lower = base.hand_dof_lower_limits[env_id].detach().cpu().numpy()
    q_upper = base.hand_dof_upper_limits[env_id].detach().cpu().numpy()
    actuated = [int(j) for j in base.actuated_dof_indices]
    n_act = len(actuated)
    records = []

    base._refresh_lab()
    z0 = float(base.object_pos[env_id, 2].item())
    q0 = base.object_rot[env_id].detach().cpu().numpy().astype(np.float64)

    for step_id in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break
        base._refresh_lab()
        loads = fingertip_loads(base)[env_id]
        active = [i for i in range(5) if loads[i] > LOAD_MIN]
        z = float(base.object_pos[env_id, 2].item())
        if z < z0 - DROP or len(active) < 3:
            records.append({
                "event": "stop",
                "step": step_id,
                "loads": loads.tolist(),
                "z": z,
                "reason": "dropped" if z < z0 - DROP else "too_few_contacts",
            })
            print(f"stop step {step_id} z={z:.4f} loads={[round(float(v), 3) for v in loads]}", flush=True)
            break
        origin = base.scene.env_origins[env_id].detach().cpu().numpy()
        pads = base.hand.data.body_link_state_w[env_id, base.elastomer_ids, :3].detach().cpu().numpy() - origin
        obj_pos = base.object_pos[env_id].detach().cpu().numpy().astype(np.float64)
        quat = base.object_rot[env_id].detach().cpu().numpy().astype(np.float64)
        forces = np.stack([
            base._contact_sensor[sensor_id].data.force_matrix_w[env_id, 0, 0].detach().cpu().numpy()
            for sensor_id in range(5)
        ])
        local, length, gram, weight, center, pos_scale, normals = fit_loaded(pads, forces, active)
        axis, axis_value = axis_from_normals(normals)
        phi = []
        rows = []
        jac_all = base.hand.data.body_link_jacobian_w.torch
        for finger in active:
            mean, grad = surface_at(pads[finger], local, length, gram, weight, center, pos_scale)
            pad_jac = jac_all[env_id, elastomer_jac[finger], :3, :][:, jac_columns].detach().cpu().numpy()
            radius = pads[finger] - obj_pos
            row = np.concatenate([
                -grad,
                -np.cross(radius, grad),
                pad_jac.T @ grad,
            ])
            phi.append(mean)
            rows.append(row)
        phi = np.asarray(phi, dtype=np.float64)
        jac = np.stack(rows)
        q = base.hand_dof_pos[env_id].detach().cpu().numpy()
        joints = np.array([q[joint_id] for joint_id in actuated], dtype=np.float64)
        lower = np.zeros(n_act)
        upper = np.zeros(n_act)
        allowed = []
        for finger in active:
            allowed.extend(groups[finger])
        for col in allowed:
            joint_id = actuated[col]
            lower[col] = max(-scale, float(q_lower[joint_id] - q[joint_id]))
            upper[col] = min(scale, float(q_upper[joint_id] - q[joint_id]))
            if lower[col] > upper[col]:
                lower[col] = 0.0
                upper[col] = 0.0
        u, cost, target_q = plan_once(obj_pos, quat, joints, phi, jac, axis, lower, upper)
        pred_pos, pred_quat, _ = explicit_step(obj_pos, quat, joints, u, phi, jac)
        actions = torch.zeros_like(base.prev_targets)
        for col, joint_id in enumerate(actuated):
            actions[env_id, joint_id] = float(np.clip(u[col] / scale, -1.0, 1.0))
        env.step(actions)
        base._refresh_lab()
        quat_after = base.object_rot[env_id].detach().cpu().numpy().astype(np.float64)
        loads_after = fingertip_loads(base)[env_id]
        yaw = yaw_about(quat_after, q0, axis)
        pred_yaw = yaw_about(pred_quat, quat, axis)
        row = {
            "event": "step",
            "step": step_id,
            "fingers": [NAMES[i] for i in active],
            "phi": phi.tolist(),
            "axis": axis.tolist(),
            "axis_eigenvalue": axis_value,
            "u_norm": float(np.linalg.norm(u)),
            "cost": cost,
            "pred_yaw": pred_yaw,
            "yaw": yaw,
            "z": float(base.object_pos[env_id, 2].item()),
            "loads": loads_after.tolist(),
            "target_quat_xyzw": target_q.tolist(),
        }
        records.append(row)
        print(
            f"step {step_id} yaw={yaw:+.4f} pred={pred_yaw:+.4f} "
            f"u={row['u_norm']:.4f} loads={[round(float(v), 3) for v in loads_after]}",
            flush=True,
        )

    if records:
        last = records[-1]
        yaw_end = last.get("yaw", 0.0)
        print(f"TURN done steps={len([r for r in records if r['event']=='step'])} yaw={yaw_end:+.4f}", flush=True)
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
