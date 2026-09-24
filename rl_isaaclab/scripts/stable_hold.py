"""Live check of a grasp-holding increment on one frozen Sharpa hand.

Pads that already read force are recorded as a Gaussian implicit surface and
are kept from translating. Fingers without a reading take a small flexion.
The surface is not used to choose a finger. The cylinder mesh is not read.
"""

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hold loaded pads and flex the others.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--settle_steps", type=int, default=15)
parser.add_argument("--hold_steps", type=int, default=25)
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

_step_path = Path(__file__).with_name("stable_hold_step.py")
_step = importlib.util.spec_from_file_location("stable_hold_step", _step_path)
hold_step = importlib.util.module_from_spec(_step)
_step.loader.exec_module(hold_step)

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
LOG_PATH = Path("logs/live_diag/stable_hold.jsonl")
DROP = 0.02


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


def flexion_columns(joint_names, actuated):
    columns = []
    for col, joint_id in enumerate(actuated):
        name = joint_names[joint_id]
        if not name.endswith(("_FE", "_IP", "_PIP", "_DIP")):
            continue
        finger = next(i for i, prefix in enumerate(FINGER_PREFIXES) if name.startswith(prefix))
        columns.append((col, finger))
    return columns


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
    print(f"HOLD grasp env {env_id}", flush=True)

    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    elastomer_jac = [int(body) - body_offset for body in base.elastomer_ids]
    jac_columns = [dof0 + int(joint) for joint in base.actuated_dof_indices]
    action_scale = float(base.cfg.action_scale)
    noise = float(base.cfg.contact_sensor_noise)
    flex = flexion_columns(base.hand.joint_names, base.actuated_dof_indices)
    records = []

    base._refresh_lab()
    forces0 = pad_forces(base)[env_id]
    loads0 = np.linalg.norm(forces0, axis=-1)
    empty = float(np.min(loads0))
    loaded = [i for i in range(5) if loads0[i] > empty + noise]
    z0 = float(base.object_pos[env_id, 2].detach().cpu().numpy())
    records.append({
        "event": "contacts",
        "env": env_id,
        "seed": int(env_cfg.seed),
        "loads": loads0.tolist(),
        "loaded": [NAMES[i] for i in loaded],
        "empty": empty,
        "z": z0,
    })
    print(
        f"loaded={[NAMES[i] for i in loaded]} loads={[round(float(v), 3) for v in loads0]}",
        flush=True,
    )

    held_ok = True
    explored = []
    if not loaded:
        records.append({"event": "skip", "reason": "no pad reading to hold"})
        print("no pad reading, do not move", flush=True)
    else:
        pads = pads_of(base, env_id)
        normals = forces0[loaded] / np.linalg.norm(forces0[loaded], axis=1, keepdims=True)
        shape = hold_step.ShapeRecord(pads[loaded], normals)
        for finger in range(5):
            _mean, variance = shape.query(pads[finger])
            records.append({"event": "shape", "finger": NAMES[finger], "variance": variance})
        for step_id in range(args_cli.hold_steps):
            if not simulation_app.is_running():
                return
            base._refresh_lab()
            forces = pad_forces(base)[env_id]
            loads = np.linalg.norm(forces, axis=-1)
            pos_z = float(base.object_pos[env_id, 2].detach().cpu().numpy())
            held_ok = all(float(loads[i]) > empty + noise for i in loaded)
            for finger in range(5):
                if finger in loaded or NAMES[finger] in explored:
                    continue
                if float(loads[finger]) > float(loads0[finger]) + noise:
                    explored.append(NAMES[finger])
            records.append({
                "event": "read",
                "step": step_id,
                "loads": loads.tolist(),
                "z": pos_z,
                "held": held_ok,
                "explored": list(explored),
            })
            print(
                f"hold {step_id} loads={[round(float(v), 3) for v in loads]} "
                f"held={held_ok} explored={explored}",
                flush=True,
            )
            if pos_z < z0 - DROP:
                records.append({"event": "dropped", "z": pos_z})
                print("object dropped", flush=True)
                held_ok = False
                break
            if not held_ok:
                print("a loaded finger returned to its empty reading", flush=True)
                break
            protected = [i for i in range(5) if float(loads[i]) > empty + noise]
            q_all = base.hand_dof_pos[env_id].detach().cpu().numpy()
            upper_all = base.hand_dof_upper_limits[env_id].detach().cpu().numpy()
            actuated = [int(joint) for joint in base.actuated_dof_indices]
            free_columns = []
            for col, finger in flex:
                if finger in protected:
                    continue
                joint_id = actuated[col]
                if float(q_all[joint_id]) + hold_step.CLOSE_STEP > float(upper_all[joint_id]):
                    continue
                free_columns.append(col)
            proposal = hold_step.closing_proposal(len(actuated), free_columns)
            if float(np.linalg.norm(proposal)) < 1e-8:
                records.append({"event": "no_increment", "step": step_id})
                print("no flexion left inside the limits", flush=True)
                break
            jac = base.hand.data.body_link_jacobian_w.torch[env_id][:, :3, :].detach().cpu().numpy()
            blocks = [jac[elastomer_jac[finger]][:, jac_columns] for finger in protected]
            columns = [list(range(len(actuated))) for _ in protected]
            dq = hold_step.hold_increment(proposal, blocks, columns)
            lower_all = base.hand_dof_lower_limits[env_id].detach().cpu().numpy()
            room_lo = np.array([float(lower_all[joint] - q_all[joint]) for joint in actuated])
            room_hi = np.array([float(upper_all[joint] - q_all[joint]) for joint in actuated])
            dq = np.clip(dq, room_lo, room_hi)
            if float(np.linalg.norm(dq)) < 1e-8:
                records.append({"event": "no_increment", "step": step_id, "after": "projection"})
                print("projection removed the increment", flush=True)
                break
            actions = torch.zeros_like(base.prev_targets)
            write_action(actions, env_id, dq, base.actuated_dof_indices, action_scale)
            env.step(actions)

    loads1 = np.linalg.norm(pad_forces(base)[env_id], axis=-1)
    records.append({
        "event": "result",
        "loads0": loads0.tolist(),
        "loads1": loads1.tolist(),
        "held": held_ok and bool(loaded),
        "explored": explored,
        "seed": int(env_cfg.seed),
        "env": env_id,
    })
    print(
        f"result held={held_ok and bool(loaded)} explored={explored} "
        f"loads={[round(float(v), 3) for v in loads1]}",
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
