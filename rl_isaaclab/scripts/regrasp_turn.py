"""Closed-loop OPT1/OPT2 on one frozen grasp. Records an mp4 when --record is set.

belief_turn.py is left in place. The simulator executes only the first joint increment.
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

RECORD = "--record" in sys.argv
sys.argv = [arg for arg in sys.argv if arg not in ("--headless", "--record")]
if RECORD:
    sys.argv += ["--viz", "kit"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Closed-loop kinematic regrasp on one frozen grasp.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=1999979387)
parser.add_argument("--settle_steps", type=int, default=10)
parser.add_argument("--turn_steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg
import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper


def _load(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shift_step = _load("reach_shift_step.py")
regrasp_step = _load("regrasp_step.py")

NAMES = ("thumb", "index", "middle", "ring", "pinky")
PALM = "right_hand_C_MC"
LOG_PATH = Path("logs/live_diag/regrasp_turn.jsonl")
VIDEO_PATH = Path("logs/to_delete/regrasp_turn.mp4")
MESH_RADIUS, MESH_HALF = shift_step.cylinder_size(0.5)
DROP = 0.02


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


def pad_forces(base):
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.normal_force_matrix_w[:, 0, 0, :]
        columns.append(force)
    return torch.stack(columns, dim=1).detach().cpu().numpy()


def write_action(actions, env_id, dq, actuated, scale):
    for column, joint_id in enumerate(actuated):
        actions[env_id, joint_id] = float(np.clip(dq[column] / scale, -1.0, 1.0))


def quat_xyzw_matrix(quat):
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def palm_pose(base, env_id):
    origin = base.scene.env_origins[env_id].detach().cpu().numpy()
    bodies = list(base.hand.body_names)
    index = bodies.index(PALM)
    state = base.hand.data.body_link_state_w[env_id, index].detach().cpu().numpy()
    return state[:3] - origin, state[3:7]


def to_palm(point, palm_pos, palm_quat):
    rotation = quat_xyzw_matrix(palm_quat)
    return (np.asarray(point, dtype=np.float64) - palm_pos) @ rotation


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree("outputs/", ignore_errors=True)
    env_cfg.hold_pose = False
    env_cfg.freeze_on_success = True
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    print(f"regrasp seed {env_cfg.seed}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()
    base = env.unwrapped
    while simulation_app.is_running() and not base.frozen:
        env.step(env.zero_actions())
    if not base.frozen:
        print("screen stopped before a grasp held", flush=True)
        return

    base.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
    env_id = int(base.frozen_env_id)
    base._focus_env(env_id)
    print(f"REGRASP grasp env {env_id}", flush=True)
    video = FrameWriter(VIDEO_PATH) if RECORD else None
    for _ in range(args_cli.settle_steps):
        if not simulation_app.is_running():
            return
        env.step(env.zero_actions())
        if video is not None:
            frame = grab_frame(base)
            if frame is not None:
                video.add(frame)

    actuated = [int(joint) for joint in base.actuated_dof_indices]
    action_scale = float(base.cfg.action_scale)
    noise = float(base.cfg.contact_sensor_noise)
    joint_names = list(base.hand.joint_names)
    fk = regrasp_step.HandFK()
    limits = {}
    lower = base.hand_dof_lower_limits[env_id].detach().cpu().numpy()
    upper = base.hand_dof_upper_limits[env_id].detach().cpu().numpy()
    for index, name in enumerate(joint_names):
        limits[name] = (float(lower[index]), float(upper[index]))

    base._refresh_lab()
    loads0 = np.linalg.norm(pad_forces(base)[env_id], axis=-1)
    empty = float(np.min(loads0))
    print(f"loaded loads={[round(float(v), 3) for v in loads0]}", flush=True)
    z0 = float(base.object_pos[env_id, 2].item())
    records = [{"event": "contacts", "env": env_id, "loads": loads0.tolist(), "z": z0}]
    name_index = {name: index for index, name in enumerate(fk.revolute)}
    stop = "step limit"

    for step_id in range(args_cli.turn_steps):
        if not simulation_app.is_running():
            break
        base._refresh_lab()
        loads = np.linalg.norm(pad_forces(base)[env_id], axis=-1)
        loaded = [finger for finger in range(5) if float(loads[finger]) > empty + noise]
        center, quat = base.object_pos[env_id].detach().cpu().numpy(), base.object_rot[env_id].detach().cpu().numpy()
        axis = shift_step.cylinder_axis(quat)
        palm_pos, palm_quat = palm_pose(base, env_id)
        q = base.hand.data.joint_pos[env_id].detach().cpu().numpy()
        positions = {name: float(q[joint_names.index(name)]) for name in fk.revolute}
        planned = regrasp_step.plan_cycle_step(
            fk,
            positions,
            limits,
            loaded,
            to_palm(center, palm_pos, palm_quat),
            to_palm(palm_pos + axis, palm_pos, palm_quat),
            MESH_RADIUS,
            MESH_HALF,
        )
        mode = None if planned is None else planned["mode"]
        finger = None if planned is None else NAMES[planned["finger"]]
        gap = None if planned is None or "gap" not in planned else float(planned["gap"])
        yaw = None if planned is None or "yaw" not in planned else float(planned["yaw"])
        print(
            f"regrasp {step_id} mode={mode} finger={finger} gap={gap} yaw={yaw} loads={[round(float(v), 3) for v in loads]}",
            flush=True,
        )
        records.append({
            "event": "step",
            "step": step_id,
            "mode": mode,
            "finger": finger,
            "gap": gap,
            "yaw": yaw,
            "loads": loads.tolist(),
            "z": float(center[2]),
        })
        dq_norm = 0.0
        actions = torch.zeros_like(base.prev_targets)
        if planned is not None:
            dq = np.zeros(len(actuated))
            for column, joint_id in enumerate(actuated):
                dq[column] = planned["dq"][name_index[joint_names[joint_id]]]
            dq_norm = float(np.max(np.abs(dq)))
            write_action(actions, env_id, dq, actuated, action_scale)
        if planned is None or dq_norm < 1e-8:
            stop = "no increment"
            print(f"stop step {step_id} {stop}", flush=True)
            break
        if float(center[2]) < z0 - DROP:
            stop = "dropped"
            print(f"stop step {step_id} {stop} z={float(center[2]):.4f}", flush=True)
            break
        env.step(actions)
        if video is not None:
            frame = grab_frame(base)
            if frame is not None:
                video.add(frame)

    loads1 = np.linalg.norm(pad_forces(base)[env_id], axis=-1)
    z1 = float(base.object_pos[env_id, 2].item())
    if stop == "step limit" and z1 < z0 - DROP:
        stop = "dropped"
    records.append({"event": "result", "stop": stop, "loads": loads1.tolist(), "z": z1})
    print(f"result stop={stop} z={z1:.4f} loads={[round(float(v), 3) for v in loads1]}", flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("".join(json.dumps(row) + "\n" for row in records))
    if video is not None:
        video.close()
        print(f"video {VIDEO_PATH} frames={video.count}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
