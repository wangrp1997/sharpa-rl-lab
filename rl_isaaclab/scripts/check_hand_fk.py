"""Compare URDF elastomer positions with the simulator at the same joint angles.

Headless. One environment. The palm frame cancels the robot base pose.
A second joint vector, written into the articulation and refreshed by
forward kinematics, checks the joint axes away from the reset pose.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

sys.argv = [arg for arg in sys.argv if arg != "--headless"]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check Sharpa URDF forward kinematics against Isaac.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg
import rl_isaaclab.tasks.inhand_rotate
from isaaclab_tasks.utils.hydra import hydra_task_config
from rl_isaaclab.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper

_fk_path = Path(__file__).with_name("hand_fk.py")
_spec = importlib.util.spec_from_file_location("hand_fk", _fk_path)
hand_fk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hand_fk)

NAMES = ("thumb", "index", "middle", "ring", "pinky")
PALM = "right_hand_C_MC"
MATCH_M = 0.002


def quat_xyzw_matrix(quat):
    """This articulation stores body quaternions as xyzw."""
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def palm_frame(points, palm_pos, palm_quat):
    rotation = quat_xyzw_matrix(palm_quat)
    return (points - palm_pos) @ rotation


def read_pads(base, env_id, joint_names, revolute):
    pos = base.hand.data.body_link_pos_w[env_id].detach().cpu().numpy()
    quat = base.hand.data.body_link_quat_w[env_id].detach().cpu().numpy()
    bodies = list(base.hand.body_names)
    palm = bodies.index(PALM)
    pads = np.stack([pos[bodies.index(name)] for name in hand_fk.PAD_LINKS])
    q = base.hand.data.joint_pos[env_id].detach().cpu().numpy()
    positions = {name: float(q[joint_names.index(name)]) for name in revolute}
    return palm_frame(pads, pos[palm], quat[palm]), positions


def report(label, sim_pads, urdf_pads):
    error = np.linalg.norm(sim_pads - urdf_pads, axis=1)
    for name, value in zip(NAMES, error):
        print(f"{label} {name} {value * 1000:.3f} mm", flush=True)
    worst = float(np.max(error))
    print(f"{label} max {worst * 1000:.3f} mm", flush=True)
    return worst


@hydra_task_config(args_cli.task, "agent_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = GymStyleEnvWrapper(env, clip_actions=env_cfg.clip_actions)
    env.reset()
    env.step(env.zero_actions())
    base = env.unwrapped
    fk = hand_fk.HandFK()
    joint_names = list(base.hand.joint_names)
    missing = [name for name in fk.revolute if name not in joint_names]
    if missing:
        print(f"sim is missing joints {missing}", flush=True)
        simulation_app.close()
        sys.exit(1)

    sim_pads, positions = read_pads(base, 0, joint_names, fk.revolute)
    worst = report("reset", sim_pads, fk.pads(positions))

    q_now = base.hand.data.joint_pos[0].detach().clone()
    lower = base.hand_dof_lower_limits[0]
    upper = base.hand_dof_upper_limits[0]
    for name in fk.revolute:
        index = joint_names.index(name)
        q_now[index] = torch.clamp(q_now[index] + 0.25, lower[index], upper[index])
    base.hand.write_joint_position_to_sim_index(position=q_now.unsqueeze(0))
    base.sim.forward()
    base.hand.update(0.0)
    sim_pads, positions = read_pads(base, 0, joint_names, fk.revolute)
    worst = max(worst, report("shifted", sim_pads, fk.pads(positions)))
    print("hand_fk match" if worst < MATCH_M else "hand_fk mismatch", flush=True)
    simulation_app.close()
    sys.exit(0 if worst < MATCH_M else 1)


if __name__ == "__main__":
    main()
    simulation_app.close()
