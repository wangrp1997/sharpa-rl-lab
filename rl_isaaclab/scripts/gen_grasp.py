# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import sys
import shutil

# Isaac Lab 3 dropped --headless. With no visualizer selected, the app is already headless.
sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent.")
parser.add_argument("--num_envs", type=int, default=16384, help="Number of environments to simulate.")
parser.add_argument("--hold", action="store_true", help="Hold the nominal grasp pose instead of resampling.")
parser.add_argument("--max_grasps", type=int, default=None, help="Stop after this many stable grasps.")
parser.add_argument("--freeze", action="store_true", help="Keep the live grasp on screen once it holds.")
parser.add_argument("--cache", type=str, default=None, help="Replay this grasp cache in the viewer.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment")
parser.add_argument("--max_agent_steps", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default=None, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

from isaaclab.envs import DirectRLEnvCfg

import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    shutil.rmtree('outputs/', ignore_errors=True)
    env_cfg.hold_pose = args_cli.hold or args_cli.cache is not None
    env_cfg.freeze_on_success = args_cli.freeze
    if args_cli.max_grasps is not None:
        env_cfg.max_grasps = args_cli.max_grasps
    if args_cli.cache is not None:
        import numpy as np
        env_cfg.replay_cache = args_cli.cache
        pose = np.load(args_cli.cache)[0]
        look = pose[22:25]
        env_cfg.viewer.lookat = tuple(float(v) for v in look)
        env_cfg.viewer.eye = tuple(float(v) for v in (look + np.array([0.126, 0.125, 0.361])))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg['seed']
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg["algorithm"]['minibatch_size'] = min([args_cli.num_envs * 8, 32768])

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)

    env.reset()
    if args_cli.freeze and env.unwrapped.num_envs > 1:
        base = env.unwrapped
        center = base.scene.env_origins.detach().float().mean(dim=0).cpu()
        eye = center + torch.tensor([0.0, -7.0, 6.0])
        base.sim.set_camera_view(
            tuple(float(v) for v in eye.tolist()),
            tuple(float(v) for v in center.tolist()),
        )
    while True:
        actions = env.zero_actions()
        _ = env.step(actions)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
