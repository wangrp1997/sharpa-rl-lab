"""Tangential step with link-sphere gaps as inequalities.

Loaded pads move along axis × normal. A pair of collision links contributes
a slack when two surface spheres overlap. Adjacent
links are omitted. The author MATLAB HQP is not called; osqp solves the
same inequalities.
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Belief tangent step with link spheres.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=15)
parser.add_argument("--turn_steps", type=int, default=8)
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


def _load(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


turn_step = _load("tangent_turn_step.py")
belief_step = _load("belief_turn_step.py")
spheres = _load("link_spheres.py")
hold_step = _load("stable_hold_step.py")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

NAMES = ("thumb", "index", "middle", "ring", "pinky")


def yaw_about(quat, q0, axis):
    """Rotation of quat relative to q0 about axis. Both quaternions are xyzw."""
    x, y, z, w = (float(v) for v in q0)
    ax, ay, az, aw = (float(v) for v in quat)
    bx, by, bz, bw = -x, -y, -z, w
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    rw = aw * bw - ax * bx - ay * by - az * bz
    xyz = np.array([rx, ry, rz], dtype=np.float64)
    norm_v = float(np.linalg.norm(xyz))
    if norm_v < 1e-8:
        return 0.0
    angle = 2.0 * np.arctan2(norm_v, rw)
    return float(angle * np.dot(xyz, axis) / norm_v)


def object_shifts(pads, center, quat, pads0, center0, quat0, fingers):
    rows = []
    for finger in fingers:
        point = turn_step.point_in_object(pads[finger], center, quat)
        start = turn_step.point_in_object(pads0[finger], center0, quat0)
        rows.append({"finger": NAMES[finger], "shift_m": float(np.linalg.norm(point - start))})
    return rows
LOG_PATH = Path("logs/live_diag/belief_turn.jsonl")
DROP = 0.02
NEW_POINT = 0.002
AXIS_STD0 = 0.05


def pad_forces(base) -> np.ndarray:
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.normal_force_matrix_w[:, 0, 0, :]
        columns.append(force)
    return torch.stack(columns, dim=1).detach().cpu().numpy()


def write_action(actions, env_id, dq, actuated, scale):
    for column, joint_id in enumerate(actuated):
        actions[env_id, joint_id] = float(np.clip(dq[column] / scale, -1.0, 1.0))


def pads_of(base, env_id):
    origin = base.scene.env_origins[env_id].detach().cpu().numpy()
    return base.hand.data.body_link_state_w[env_id, base.elastomer_ids, :3].detach().cpu().numpy() - origin


def object_pose(base, env_id):
    center = base.object_pos[env_id].detach().cpu().numpy().astype(np.float64)
    quat = base.object_rot[env_id].detach().cpu().numpy().astype(np.float64)
    return center, quat


def link_pack(base, env_id, jac_columns, body_offset):
    origin = base.scene.env_origins[env_id].detach().cpu().numpy()
    state = base.hand.data.body_link_state_w[env_id].detach().cpu().numpy()
    jac = base.hand.data.body_link_jacobian_w.torch[env_id].detach().cpu().numpy()
    names, origins, quats, lin, ang = [], [], [], [], []
    for body_id, name in enumerate(base.hand.body_names):
        if name not in spheres.SPHERES:
            continue
        jac_index = int(body_id) - body_offset
        names.append(name)
        origins.append(state[body_id, :3] - origin)
        quats.append(state[body_id, 3:7])
        lin.append(jac[jac_index, :3, :][:, jac_columns])
        ang.append(jac[jac_index, 3:6, :][:, jac_columns])
    return names, np.stack(origins), np.stack(quats), np.stack(lin), np.stack(ang)


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
    print(f"BELIEF grasp env {env_id}", flush=True)

    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    elastomer_jac = [int(body) - body_offset for body in base.elastomer_ids]
    jac_columns = [dof0 + int(joint) for joint in base.actuated_dof_indices]
    actuated = [int(joint) for joint in base.actuated_dof_indices]
    action_scale = float(base.cfg.action_scale)
    noise = float(base.cfg.contact_sensor_noise)
    records = []
    mean = np.zeros(3)
    covariance = np.eye(3) * AXIS_STD0 ** 2

    base._refresh_lab()
    forces0 = pad_forces(base)[env_id]
    loads0 = np.linalg.norm(forces0, axis=-1)
    empty = float(np.min(loads0))
    loaded = [i for i in range(5) if loads0[i] > empty + noise]
    center0, quat0 = object_pose(base, env_id)
    z0 = float(center0[2])
    pads0 = pads_of(base, env_id)
    print(
        f"loaded={[NAMES[i] for i in loaded]} loads={[round(float(v), 3) for v in loads0]}",
        flush=True,
    )
    records.append({
        "event": "contacts",
        "env": env_id,
        "seed": int(env_cfg.seed),
        "loads": loads0.tolist(),
        "loaded": [NAMES[i] for i in loaded],
        "z": z0,
    })

    held_ok = len(loaded) >= 2
    new_points = []
    surface = []
    axis = None
    if not held_ok:
        records.append({"event": "skip", "reason": "fewer than two pad normals"})
        print("fewer than two pad normals, do not move", flush=True)
    else:
        normals0 = forces0[loaded] / np.linalg.norm(forces0[loaded], axis=1, keepdims=True)
        for finger, normal in zip(loaded, normals0):
            point = turn_step.point_in_object(pads0[finger], center0, quat0)
            normal_obj = turn_step.quat_xyzw_to_matrix(quat0).T @ normal
            surface.append((point, normal_obj, finger))
        for step_id in range(args_cli.turn_steps):
            if not simulation_app.is_running():
                return
            base._refresh_lab()
            forces = pad_forces(base)[env_id]
            loads = np.linalg.norm(forces, axis=-1)
            center, quat = object_pose(base, env_id)
            active_now = [i for i in range(5) if float(loads[i]) > empty + noise]
            held_ok = len(active_now) >= 2 and float(center[2]) >= z0 - DROP
            if not held_ok:
                pads_stop = pads_of(base, env_id)
                records.append({
                    "event": "stop",
                    "step": step_id,
                    "loads": loads.tolist(),
                    "z": float(center[2]),
                    "held": held_ok,
                    "yaw": yaw_about(quat, quat0, axis) if axis is not None else None,
                    "pad_shift": object_shifts(pads_stop, center, quat, pads0, center0, quat0, loaded),
                })
                print(f"stop step {step_id} held={held_ok} z={float(center[2]):.4f}", flush=True)
                held_ok = False
                break
            active = active_now
            normals = [forces[i] / np.linalg.norm(forces[i]) for i in active]
            jac = base.hand.data.body_link_jacobian_w.torch[env_id].detach().cpu().numpy()
            contact_jacs = [jac[elastomer_jac[i], :3, :][:, jac_columns] for i in active]
            q_all = base.hand_dof_pos[env_id].detach().cpu().numpy()[actuated]
            lower_all = base.hand_dof_lower_limits[env_id].detach().cpu().numpy()[actuated]
            upper_all = base.hand_dof_upper_limits[env_id].detach().cpu().numpy()[actuated]
            names, origins, quats, lin, ang = link_pack(base, env_id, jac_columns, body_offset)
            close = spheres.watch_pairs(
                names, origins, quats, lin, ang, belief_step.WATCH_DISTANCE
            )
            axis = turn_step.axis_from_normals(normals)
            axis_std = AXIS_STD0 if axis is None else float(np.sqrt(max(axis @ covariance @ axis, 0.0)))
            pads_before = pads_of(base, env_id)
            centroid = np.mean(pads_before[active], axis=0)
            radius = float(np.mean([np.linalg.norm(pads_before[i] - centroid) for i in active]))
            shells = []
            for finger in range(5):
                if finger in active:
                    continue
                point = pads_before[finger]
                delta = point - centroid
                dist = float(np.linalg.norm(delta))
                if dist < 1e-5:
                    continue
                shells.append((
                    delta / dist,
                    jac[elastomer_jac[finger], :3, :][:, jac_columns],
                    point,
                    max(dist - radius, 0.0),
                ))
            plan = belief_step.plan_belief_step(
                normals,
                contact_jacs,
                [pads_before[i] for i in active],
                [q_all],
                [lower_all],
                [upper_all],
                [item[:4] for item in close],
                axis_std,
                shells=shells,
            )
            pair_gaps = [
                {"pair": f"{item[4]}|{item[5]}", "gap_m": float(item[3])} for item in close
            ]
            records.append({
                "event": "read",
                "step": step_id,
                "loads": loads.tolist(),
                "z": float(center[2]),
                "pairs": len(close),
                "pair_gaps": pair_gaps,
                "dq": plan["dq"].tolist(),
                "linear": plan["linear"].tolist(),
                "angular": plan["angular"].tolist(),
                "enclosed": plan["enclosed"],
                "length": plan["length"],
                "axis_std": axis_std,
                "moved": [NAMES[active[i]] for i in plan["moved"]],
            })
            print(
                f"belief {step_id} pairs={len(close)} length={plan['length']:.5f} "
                f"std={axis_std:.4f} moved={[NAMES[active[i]] for i in plan['moved']]} "
                f"loads={[round(float(v), 3) for v in loads]}",
                flush=True,
            )
            if float(np.linalg.norm(plan["dq"])) < 1e-8:
                print("zero increment, keep reading", flush=True)
                env.step(env.zero_actions())
                continue
            actions = torch.zeros_like(base.prev_targets)
            write_action(actions, env_id, plan["dq"], base.actuated_dof_indices, action_scale)
            env.step(actions)
            if axis is not None:
                mean, covariance = belief_step.update_belief(mean, covariance, axis, 0.0, 1e-4)
            base._refresh_lab()
            pads = pads_of(base, env_id)
            forces_after = pad_forces(base)[env_id]
            loads_after = np.linalg.norm(forces_after, axis=-1)
            center_after, quat_after = object_pose(base, env_id)
            tangent = []
            for local, finger in enumerate(active):
                direction = plan["directions"][local]
                if direction is None:
                    continue
                delta = pads[finger] - pads_before[finger]
                tangent.append({
                    "finger": NAMES[finger],
                    "along_m": float(np.dot(delta, direction)),
                    "command_m": float(plan["length"]),
                })
            records.append({
                "event": "after",
                "step": step_id,
                "yaw": yaw_about(quat_after, quat0, axis) if axis is not None else None,
                "tangent": tangent,
                "pad_shift": object_shifts(pads, center_after, quat_after, pads0, center0, quat0, loaded),
                "loads": loads_after.tolist(),
                "z": float(center_after[2]),
            })
            for finger in loaded:
                if float(loads_after[finger]) <= empty + noise:
                    continue
                point = turn_step.point_in_object(pads[finger], center_after, quat_after)
                start = turn_step.point_in_object(pads0[finger], center0, quat0)
                shift = float(np.linalg.norm(point - start))
                previous = max((item[1] for item in new_points if item[0] == NAMES[finger]), default=0.0)
                if shift < max(NEW_POINT, previous + NEW_POINT):
                    continue
                normal = forces_after[finger] / np.linalg.norm(forces_after[finger])
                normal_obj = turn_step.quat_xyzw_to_matrix(quat_after).T @ normal
                surface.append((point, normal_obj, finger))
                new_points.append((NAMES[finger], shift))
                print(f"new point {NAMES[finger]} shift={shift:.4f}", flush=True)

    if surface:
        shape = hold_step.ShapeRecord(
            np.stack([item[0] for item in surface]),
            np.stack([item[1] for item in surface]),
        )
        records.append({
            "event": "shape",
            "points": len(surface),
            "new_points": len(new_points),
            "variance_at_first": shape.query(surface[0][0])[1],
        })
    loads1 = np.linalg.norm(pad_forces(base)[env_id], axis=-1)
    center1, quat1 = object_pose(base, env_id)
    records.append({
        "event": "result",
        "loads0": loads0.tolist(),
        "loads1": loads1.tolist(),
        "held": held_ok,
        "new_points": [{"finger": name, "shift_m": shift} for name, shift in new_points],
        "z0": z0,
        "z1": float(center1[2]),
        "seed": int(env_cfg.seed),
        "env": env_id,
        "quat0": quat0.tolist(),
        "quat1": quat1.tolist(),
        "axis_variance": np.diag(covariance).tolist(),
    })
    print(
        f"result held={held_ok} new={len(new_points)} "
        f"loads={[round(float(v), 3) for v in loads1]} z={float(center1[2]):.4f}",
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
