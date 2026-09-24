"""Screen 128 grasps, then compare the exploration switch on the live hands.

Gravity is set downward after the screen. Poses are not written back.
One arm may open the exploration step only when two contact hypotheses
disagree on the next joint increment. That step keeps the yaw task and adds
a tangential nudge on the disputed finger. The other arm keeps the switch shut.
Both arms take the same number of control steps.
"""

import argparse
import json
import sys
import shutil
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the action-disagreement switch on a live grasp.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--trial_steps", type=int, default=24)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--arm", choices=("split", "gated", "off"), default="split")
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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_PREFIXES = (
    "right_thumb_",
    "right_index_",
    "right_middle_",
    "right_ring_",
    "right_pinky_",
)
LOAD_MIN = 0.5
DISAGREE = 5e-4
YAW_HIT = 0.002
TANGENT_STEP = 4e-4
PSI_DES = 0.01
LOG_PATH = {
    "split": Path("logs/live_diag/trace.jsonl"),
    "gated": Path("logs/live_diag/trace_gated.jsonl"),
    "off": Path("logs/live_diag/trace_off.jsonl"),
}[args_cli.arm]


def finger_groups(joint_names, actuated):
    return [
        [i for i, joint_id in enumerate(actuated) if joint_names[joint_id].startswith(prefix)]
        for prefix in FINGER_PREFIXES
    ]


def cylinder_axis(quat: torch.Tensor) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat.tolist())
    return np.array(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        dtype=np.float64,
    )


def yaw_about_start(quat: torch.Tensor, q0: torch.Tensor) -> float:
    """Yaw of quat relative to q0 about the cylinder axis, both xyzw."""
    ax, ay, az, aw = (-float(q0[0]), -float(q0[1]), -float(q0[2]), float(q0[3]))
    bx, by, bz, bw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    rel = np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    w = aw * bw - ax * bx - ay * by - az * bz
    norm_v = float(np.linalg.norm(rel))
    if norm_v < 1e-8:
        return 0.0
    angle = 2.0 * np.arctan2(norm_v, w)
    return float(angle * rel[2] / norm_v)


def fingertip_loads(base) -> np.ndarray:
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.force_matrix_w[:, 0, 0, :]
        columns.append(torch.linalg.norm(force, dim=-1))
    return torch.stack(columns, dim=-1).detach().cpu().numpy()


def coupled_increment(jac, tips, center, axis, fingers, groups, jac_bodies, jac_columns):
    """Least-squares joint increment whose transmitting fingers follow one yaw step."""
    cols = sorted({col for finger in fingers for col in groups[finger]})
    dq = np.zeros(len(jac_columns), dtype=np.float64)
    if not cols:
        return dq, 0.0
    jac_np = jac.detach().cpu().numpy()
    tips_np = tips.detach().cpu().numpy()
    center_np = center.detach().cpu().numpy()
    rows = []
    targets = []
    index = {col: local for local, col in enumerate(cols)}
    for finger in fingers:
        lever = tips_np[finger] - center_np
        tang = np.cross(axis, lever)
        block = jac_np[jac_bodies[finger], :3, :]
        for axis_id in range(3):
            row = np.zeros(len(cols) + 1, dtype=np.float64)
            for col in groups[finger]:
                row[index[col]] = block[axis_id, jac_columns[col]]
            row[-1] = -tang[axis_id]
            rows.append(row)
            targets.append(0.0)
    task = np.zeros(len(cols) + 1, dtype=np.float64)
    task[-1] = 8.0
    rows.append(task)
    targets.append(8.0 * PSI_DES)
    for local in range(len(cols)):
        damp = np.zeros(len(cols) + 1, dtype=np.float64)
        damp[local] = 0.05
        rows.append(damp)
        targets.append(0.0)
    solution, *_ = np.linalg.lstsq(np.stack(rows), np.asarray(targets), rcond=None)
    dq[cols] = np.clip(solution[:-1], -0.008, 0.008)
    return dq, float(solution[-1])


def probe_increment(jac, tips, center, axis, finger, groups, jac_bodies, jac_columns):
    jac_np = jac.detach().cpu().numpy()
    tips_np = tips.detach().cpu().numpy()
    center_np = center.detach().cpu().numpy()
    lever = tips_np[finger] - center_np
    tang = np.cross(axis, lever)
    norm = np.linalg.norm(tang)
    direction = np.zeros(3) if norm < 1e-8 else tang / norm * TANGENT_STEP
    cols = groups[finger]
    dq = np.zeros(len(jac_columns), dtype=np.float64)
    if not cols:
        return dq
    joint_ids = [jac_columns[col] for col in cols]
    block = jac_np[jac_bodies[finger], :3][:, joint_ids]
    gram = block @ block.T + 1e-3 * np.eye(3)
    step = np.clip(block.T @ np.linalg.solve(gram, direction), -0.008, 0.008)
    for local, col in enumerate(cols):
        dq[col] = step[local]
    return dq


def as_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy().reshape(-1).tolist()
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def matrix_force(base, env_id, name):
    columns = []
    for sensor_id in range(5):
        block = getattr(base._contact_sensor[sensor_id].data, name, None)
        if block is None:
            return None
        columns.append(block[env_id, 0, 0].detach().float().cpu().numpy().tolist())
    return columns


def privileged(base, env_id):
    record = {}
    try:
        record["object_mass"] = float(base.object.root_physx_view.get_masses().reshape(-1)[env_id].item())
    except Exception:
        record["object_mass"] = None
    try:
        record["object_material"] = as_list(base.object.root_physx_view.get_material_properties()[env_id])
    except Exception:
        record["object_material"] = None
    gravity = base.physics_sim_view.get_gravity()
    record["gravity"] = [float(gravity[0]), float(gravity[1]), float(gravity[2])]
    return record


def write_action(actions, env_id, dq, actuated, scale):
    for column, joint_id in enumerate(actuated):
        actions[env_id, joint_id] = float(np.clip(dq[column] / scale, -1.0, 1.0))


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
    if args_cli.arm == "off":
        gated_ids, off_ids = [], live
    elif args_cli.arm == "gated":
        gated_ids, off_ids = live, []
    else:
        gated_ids, off_ids = live[0::2], live[1::2]
    print(f"live={live} gated={gated_ids} switch_off={off_ids}", flush=True)
    if live:
        base._focus_env(live[0])

    groups = finger_groups(base.hand.joint_names, base.actuated_dof_indices)
    dof0 = int(base.hand.num_base_dofs)
    body_offset = 1 if base.hand.is_fixed_base else 0
    jac_bodies = [body - body_offset for body in base.finger_bodies]
    jac_columns = [dof0 + joint for joint in base.actuated_dof_indices]
    scale = float(base.cfg.action_scale)

    base._refresh_lab()
    q0 = base.object_rot.clone()
    z0 = base.object_pos[:, 2].clone()
    # None until a probe labels the finger. True means it transmitted yaw.
    label = {env_id: [None] * 5 for env_id in gated_ids}
    stats = {
        env_id: {"arm": "gated" if env_id in gated_ids else "off", "open": 0, "shut": 0, "dropped": False}
        for env_id in live
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for env_id in live:
        records.append({
            "event": "frozen",
            "env": env_id,
            "arm": stats[env_id]["arm"],
            "joint_names": list(base.hand.joint_names),
            "object_pos": as_list(base.object_pos[env_id]),
            "object_quat_xyzw": as_list(base.object_rot[env_id]),
            "q": as_list(base.hand_dof_pos[env_id]),
            "fingertip_pos": base.fingertip_pos[env_id].detach().float().cpu().numpy().tolist(),
            "loads": fingertip_loads(base)[env_id].tolist(),
            "note": "diagnostic snapshot only; writing this pose back breaks contact",
            **privileged(base, env_id),
        })

    for step_id in range(args_cli.trial_steps):
        if not simulation_app.is_running():
            break
        base._refresh_lab()
        loads = fingertip_loads(base)
        jac_all = base.hand.data.body_link_jacobian_w.torch
        actions = torch.zeros_like(base.prev_targets)
        yaw_before = {
            env_id: yaw_about_start(base.object_rot[env_id], q0[env_id]) for env_id in live
        }
        pending = {}
        planned = {}
        for env_id in live:
            if stats[env_id]["dropped"]:
                continue
            loaded = [i for i in range(5) if loads[env_id, i] > LOAD_MIN]
            usable = loaded
            if env_id in label:
                usable = [i for i in loaded if label[env_id][i] is not False]
            axis = cylinder_axis(base.object_rot[env_id])
            jac = jac_all[env_id]
            tips = base.fingertip_pos[env_id]
            center = base.object_pos[env_id]
            if env_id in label:
                unknown = [i for i in usable if label[env_id][i] is None]
            else:
                unknown = list(usable)
            disputed = None
            if unknown:
                disputed = min(unknown, key=lambda i: loads[env_id, i])
            u_with, _ = coupled_increment(
                jac, tips, center, axis, usable, groups, jac_bodies, jac_columns
            )
            others = [i for i in usable if i != disputed]
            u_without, _ = coupled_increment(
                jac, tips, center, axis, others, groups, jac_bodies, jac_columns
            )
            gap = float(np.max(np.abs(u_with - u_without))) if disputed is not None else 0.0
            opened = env_id in label and disputed is not None and gap > DISAGREE
            if opened:
                nudge = probe_increment(
                    jac, tips, center, axis, disputed, groups, jac_bodies, jac_columns
                )
                dq = np.clip(u_with + nudge, -0.008, 0.008)
                stats[env_id]["open"] += 1
                pending[env_id] = disputed
            else:
                nudge = np.zeros_like(u_with)
                dq = u_with
                if env_id in stats:
                    stats[env_id]["shut"] += 1
            planned[env_id] = {
                "switch_open": opened,
                "probe": "u_with_plus_tangent",
                "gap": gap,
                "disputed": None if disputed is None else NAMES[disputed],
                "loaded": [NAMES[i] for i in loaded],
                "u_with": u_with.tolist(),
                "u_without": u_without.tolist(),
                "u_nudge": nudge.tolist(),
                "u_exec": dq.tolist(),
            }
            write_action(actions, env_id, dq, base.actuated_dof_indices, scale)
        env.step(actions)
        base._refresh_lab()
        loads_after = fingertip_loads(base)
        for env_id, finger in pending.items():
            dyaw = yaw_about_start(base.object_rot[env_id], q0[env_id]) - yaw_before[env_id]
            label[env_id][finger] = abs(dyaw) > YAW_HIT
        for env_id in live:
            z = float(base.object_pos[env_id, 2].item())
            if z < float(z0[env_id].item()) - 0.02:
                stats[env_id]["dropped"] = True
            yaw = yaw_about_start(base.object_rot[env_id], q0[env_id])
            plan = planned.get(env_id, {})
            row = {
                "event": "step",
                "step": step_id,
                "env": env_id,
                "arm": stats[env_id]["arm"],
                "yaw": yaw,
                "dyaw": yaw - yaw_before[env_id],
                "z": z,
                "dropped": stats[env_id]["dropped"],
                "object_pos": as_list(base.object_pos[env_id]),
                "object_quat_xyzw": as_list(base.object_rot[env_id]),
                "object_linvel": as_list(base.object_linvel[env_id]),
                "object_angvel": as_list(base.object_angvel[env_id]),
                "q": as_list(base.hand_dof_pos[env_id]),
                "qd": as_list(base.hand_dof_vel[env_id]),
                "torque": as_list(base.hand_dof_torque[env_id]),
                "fingertip_pos": base.fingertip_pos[env_id].detach().float().cpu().numpy().tolist(),
                "loads": loads_after[env_id].tolist(),
                "normal_force": matrix_force(base, env_id, "normal_force_matrix_w"),
                "friction_force": matrix_force(base, env_id, "friction_force_matrix_w"),
                **plan,
                **privileged(base, env_id),
            }
            records.append(row)
        if step_id % 4 == 0 or step_id + 1 == args_cli.trial_steps:
            brief = " ".join(
                f"{item['env']}:{item['arm'][0]} yaw={item['yaw']:+.4f} z={item['z']:.3f}"
                for item in records[-len(live):]
            )
            print(f"step {step_id} {brief}", flush=True)

    summary = []
    for env_id in live:
        yaw = yaw_about_start(base.object_rot[env_id], q0[env_id])
        item = {
            "env": env_id,
            "arm": stats[env_id]["arm"],
            "yaw": yaw,
            "open": stats[env_id]["open"],
            "shut": stats[env_id]["shut"],
            "dropped": stats[env_id]["dropped"],
            "z": float(base.object_pos[env_id, 2].item()),
        }
        summary.append(item)
        print(
            f"RESULT env {env_id} {item['arm']} yaw={yaw:+.4f} "
            f"open={item['open']} shut={item['shut']} dropped={item['dropped']}",
            flush=True,
        )
    with LOG_PATH.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
        handle.write(json.dumps({"summary": summary}) + "\n")
    print(f"wrote {LOG_PATH}", flush=True)
    while simulation_app.is_running():
        env.step(env.zero_actions())


if __name__ == "__main__":
    main()
    simulation_app.close()
