"""Move pads that already read force along axis × normal.

The axis is the pairwise cross product of those force normals. The cylinder
mesh is not used to choose the step. Object pose is read only afterwards, to
see whether the contact moved to a new place on the object.
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tangential step of loaded pads.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=15)
parser.add_argument("--turn_steps", type=int, default=15)
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

_step_path = Path(__file__).with_name("tangent_turn_step.py")
_step = importlib.util.spec_from_file_location("tangent_turn_step", _step_path)
turn_step = importlib.util.module_from_spec(_step)
_step.loader.exec_module(turn_step)

_hold_path = Path(__file__).with_name("stable_hold_step.py")
_hold = importlib.util.spec_from_file_location("stable_hold_step", _hold_path)
hold_step = importlib.util.module_from_spec(_hold)
_hold.loader.exec_module(hold_step)

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
LOG_PATH = Path("logs/live_diag/tangent_turn.jsonl")
DROP = 0.02
NEW_POINT = 0.002


def finger_groups(joint_names, actuated):
    return [
        [i for i, joint_id in enumerate(actuated) if joint_names[joint_id].startswith(prefix)]
        for prefix in FINGER_PREFIXES
    ]


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
    print(f"TURN grasp env {env_id}", flush=True)

    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    elastomer_jac = [int(body) - body_offset for body in base.elastomer_ids]
    jac_columns = [dof0 + int(joint) for joint in base.actuated_dof_indices]
    groups = finger_groups(base.hand.joint_names, base.actuated_dof_indices)
    actuated = [int(joint) for joint in base.actuated_dof_indices]
    action_scale = float(base.cfg.action_scale)
    noise = float(base.cfg.contact_sensor_noise)
    records = []

    base._refresh_lab()
    forces0 = pad_forces(base)[env_id]
    loads0 = np.linalg.norm(forces0, axis=-1)
    empty = float(np.min(loads0))
    loaded = [i for i in range(5) if loads0[i] > empty + noise]
    center0, quat0 = object_pose(base, env_id)
    z0 = float(center0[2])
    pads0 = pads_of(base, env_id)
    records.append({
        "event": "contacts",
        "env": env_id,
        "seed": int(env_cfg.seed),
        "loads": loads0.tolist(),
        "loaded": [NAMES[i] for i in loaded],
        "z": z0,
    })
    print(
        f"loaded={[NAMES[i] for i in loaded]} loads={[round(float(v), 3) for v in loads0]}",
        flush=True,
    )

    held_ok = bool(loaded)
    new_points = []
    surface = []
    if len(loaded) < 2:
        records.append({"event": "skip", "reason": "fewer than two pad normals"})
        print("fewer than two pad normals, do not move", flush=True)
        held_ok = False
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
            held_ok = all(float(loads[i]) > empty + noise for i in loaded)
            records.append({
                "event": "read",
                "step": step_id,
                "loads": loads.tolist(),
                "z": float(center[2]),
                "held": held_ok,
                "new_points": len(new_points),
            })
            print(
                f"turn {step_id} loads={[round(float(v), 3) for v in loads]} "
                f"held={held_ok} new={len(new_points)}",
                flush=True,
            )
            if float(center[2]) < z0 - DROP:
                records.append({"event": "dropped", "z": float(center[2])})
                print("object dropped", flush=True)
                held_ok = False
                break
            if not held_ok:
                print("a loaded finger returned to its empty reading", flush=True)
                break
            active = [i for i in loaded if float(loads[i]) > empty + noise]
            normals = []
            blocks = []
            joints = []
            lowers = []
            uppers = []
            q_all = base.hand_dof_pos[env_id].detach().cpu().numpy()
            lower_all = base.hand_dof_lower_limits[env_id].detach().cpu().numpy()
            upper_all = base.hand_dof_upper_limits[env_id].detach().cpu().numpy()
            jac = base.hand.data.body_link_jacobian_w.torch[env_id][:, :3, :].detach().cpu().numpy()
            for finger in active:
                force = forces[finger]
                normals.append(force / np.linalg.norm(force))
                cols = groups[finger]
                columns = [jac_columns[col] for col in cols]
                joint_ids = [actuated[col] for col in cols]
                blocks.append(jac[elastomer_jac[finger]][:, columns])
                joints.append(q_all[joint_ids])
                lowers.append(lower_all[joint_ids])
                uppers.append(upper_all[joint_ids])
            plan = turn_step.plan_tangent_steps(normals, blocks, joints, lowers, uppers)
            if plan["axis"] is None or not plan["moved"]:
                records.append({"event": "no_increment", "step": step_id})
                print("no tangential increment", flush=True)
                break
            dq = np.zeros(len(actuated), dtype=np.float64)
            for local, finger in enumerate(active):
                if local not in plan["moved"]:
                    continue
                for offset, col in enumerate(groups[finger]):
                    dq[col] = plan["steps"][local][offset]
            actions = torch.zeros_like(base.prev_targets)
            write_action(actions, env_id, dq, base.actuated_dof_indices, action_scale)
            env.step(actions)
            base._refresh_lab()
            pads = pads_of(base, env_id)
            forces_after = pad_forces(base)[env_id]
            loads_after = np.linalg.norm(forces_after, axis=-1)
            center_after, quat_after = object_pose(base, env_id)
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
                records.append({
                    "event": "new_point",
                    "step": step_id,
                    "finger": NAMES[finger],
                    "shift_m": shift,
                })
                print(f"new point {NAMES[finger]} shift={shift:.4f}", flush=True)

    if len(surface) >= 1:
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
