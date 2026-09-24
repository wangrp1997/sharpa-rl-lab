"""Load one cached Sharpa grasp at reset and let contact form under downward gravity.

The pose is written only during reset, before the episode steps. Gravity stays down.
"""

import argparse
import sys
import shutil

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Settle one grasp from the Sharpa cache.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cache", type=str, default="cache/sharpa_grasp_linspace_0.4-0.6-8.npy")
parser.add_argument("--bucket", type=int, default=4, help="Scale bucket in the 8-bin cache. 4 is scale 0.514.")
parser.add_argument("--pose", type=int, default=0, help="First index inside the chosen bucket.")
parser.add_argument("--scan", type=int, default=40, help="How many poses in the bucket to try.")
parser.add_argument("--settle_steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym

from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from isaaclab.envs import DirectRLEnvCfg
import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

NAMES = ("thumb", "index", "middle", "ring", "pinky")


def bucket_scale(bucket: int) -> float:
    return float(np.linspace(0.4, 0.6, 8)[bucket])


def bucket_rows(path: str, bucket: int) -> np.ndarray:
    data = np.load(path)
    per_bucket = data.shape[0] // 8
    return data[bucket * per_bucket : (bucket + 1) * per_bucket]


def loads(base) -> np.ndarray:
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.force_matrix_w[:, 0, 0, :]
        columns.append(torch.linalg.norm(force, dim=-1))
    return torch.stack(columns, dim=-1)[0].detach().cpu().numpy()


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree("outputs/", ignore_errors=True)
    rows = bucket_rows(args_cli.cache, args_cli.bucket)
    pose = rows[args_cli.pose]
    pose_path = "/tmp/sharpa_one_grasp.npy"
    np.save(pose_path, pose.reshape(1, -1))
    scale = bucket_scale(args_cli.bucket)
    env_cfg.hold_pose = True
    env_cfg.freeze_on_success = False
    env_cfg.replay_cache = pose_path
    env_cfg.scale_range = [scale, scale, 1]
    env_cfg.events.randomize_scale.params["scale_range"] = [scale, scale, 1]
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    look = pose[22:25]
    env_cfg.viewer.lookat = tuple(float(v) for v in look)
    env_cfg.viewer.eye = tuple(float(v) for v in (look + np.array([0.126, 0.125, 0.361])))
    print(
        f"bucket={args_cli.bucket} scale={scale:.4f} pose={args_cli.pose} "
        f"object_xyz={look.tolist()}",
        flush=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()
    base = env.unwrapped
    chosen = None
    for offset in range(args_cli.scan):
        pose_id = args_cli.pose + offset
        if pose_id >= len(rows):
            break
        base.replay_pose = torch.tensor(rows[pose_id], dtype=torch.float32, device=base.device)
        env.reset()
        for _ in range(args_cli.settle_steps):
            if not simulation_app.is_running():
                return
            env.step(env.zero_actions())
        base._refresh_lab()
        force = loads(base)
        z = float(base.object_pos[0, 2].item())
        active = [NAMES[i] for i in range(5) if force[i] > 0.5]
        print(
            f"pose {pose_id} z={z:.4f} loads={np.round(force, 3).tolist()} above_0.5={active}",
            flush=True,
        )
        if z > 0.55 and len(active) >= 3:
            chosen = pose_id
            break
    if chosen is None:
        print("no cached pose in this scan kept 3 fingertip loads", flush=True)
        return
    print(f"holding pose {chosen} with zero actions", flush=True)
    while simulation_app.is_running():
        env.step(env.zero_actions())


if __name__ == "__main__":
    main()
    simulation_app.close()
