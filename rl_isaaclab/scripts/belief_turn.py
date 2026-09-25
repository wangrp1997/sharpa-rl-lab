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
import subprocess
import sys
from pathlib import Path

RECORD = "--record" in sys.argv
sys.argv = [arg for arg in sys.argv if arg not in ("--headless", "--record")]
if RECORD:
    sys.argv += ["--viz", "kit"]

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
shift_step = _load("reach_shift_step.py")

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
VIDEO_PATH = Path("logs/to_delete/belief_turn.mp4")


class FrameWriter:
    """RGB frames to one mp4. The directory name marks the file for deletion."""

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
    """One viewport frame. This Kit build has no replicator, so capture goes through a png."""
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


def open_capture(eye, target):
    import omni.replicator.core as rep

    camera = rep.create.camera(position=tuple(eye), look_at=tuple(target))
    product = rep.create.render_product(camera, (960, 540))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach([product])
    grab_frame.annotator = annotator
DROP = 0.02
NEW_POINT = 0.002
AXIS_STD0 = 0.05
MESH_RADIUS, MESH_HALF = shift_step.cylinder_size(0.5)
AHEAD = 0.15


def correct_center(center, axis, pads, loaded, radius):
    """Move the cylinder axis so loaded pads sit on the surface."""
    center = np.asarray(center, dtype=np.float64).copy()
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    for _ in range(4):
        rows = []
        residual = []
        for pad, holding in zip(pads, loaded):
            if not holding:
                continue
            radial = pad - center
            radial = radial - float(np.dot(radial, axis)) * axis
            norm = float(np.linalg.norm(radial))
            if norm < 1e-6:
                continue
            rows.append(-radial / norm)
            residual.append(norm - radius)
        if not rows:
            break
        delta, *_ = np.linalg.lstsq(np.vstack(rows), -np.asarray(residual), rcond=None)
        center = center + delta
    return center


def pad_forces(base) -> np.ndarray:
    columns = []
    for sensor_id in range(5):
        force = base._contact_sensor[sensor_id].data.normal_force_matrix_w[:, 0, 0, :]
        columns.append(force)
    return torch.stack(columns, dim=1).detach().cpu().numpy()


def write_action(actions, env_id, dq, actuated, scale):
    for column, joint_id in enumerate(actuated):
        actions[env_id, joint_id] = float(np.clip(dq[column] / scale, -1.0, 1.0))


KEEP_GAP = 0.008


def nearest_shell(finger, names, origins, quats, lin, ang, cage, radius):
    """Sphere on this finger nearest the coarse surface, not the deepest one."""
    best = None
    prefix = f"right_{finger}_"
    for index, name in enumerate(names):
        if not str(name).startswith(prefix):
            continue
        patches = spheres.SPHERES.get(name)
        if not patches:
            continue
        rotation = spheres._rotation(quats[index])
        for local, rad in patches:
            offset = rotation @ np.asarray(local, dtype=np.float64)
            center = np.asarray(origins[index], dtype=np.float64) + offset
            delta = center - cage
            dist = float(np.linalg.norm(delta))
            gap = dist - radius - float(rad)
            if gap > KEEP_GAP:
                continue
            distal = any(token in str(name) for token in ("elastomer", "_DP", "fingertip", "_PP", "_MP"))
            jac = lin[index] - spheres._skew(offset) @ ang[index]
            normal = delta / dist if dist > 1e-8 else np.array([0.0, 0.0, 1.0])
            rank = (1 if distal else 0, gap)
            if best is None or rank > best[0]:
                best = (rank, gap, center, normal, jac)
    if best is None:
        return None
    return best[1:]


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
    env_id = int(base.frozen_env_id)
    base._focus_env(env_id)
    print(f"BELIEF grasp env {env_id}", flush=True)
    video = FrameWriter(VIDEO_PATH) if RECORD else None
    if video is not None and not any(hasattr(viz, "render_rgb_array") for viz in getattr(base.sim, "visualizers", [])):
        origin = base.scene.env_origins[env_id]
        look = (origin + base.object_pos[env_id]).detach().cpu()
        eye = look + torch.tensor([0.126, 0.125, 0.361])
        open_capture([float(v) for v in eye.tolist()], [float(v) for v in look.tolist()])
    for _ in range(args_cli.settle_steps):
        if not simulation_app.is_running():
            return
        env.step(env.zero_actions())
        if video is not None:
            frame = grab_frame(base)
            if frame is not None:
                video.add(frame)

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
        last_axis = shift_step.cylinder_axis(quat0)
        pose_hat = (center0.copy(), last_axis.copy())
        pose_rng = np.random.default_rng(0)
        for step_id in range(args_cli.turn_steps):
            if not simulation_app.is_running():
                return
            base._refresh_lab()
            forces = pad_forces(base)[env_id]
            loads = np.linalg.norm(forces, axis=-1)
            center, quat = object_pose(base, env_id)
            jac = base.hand.data.body_link_jacobian_w.torch[env_id].detach().cpu().numpy()
            names, origins, quats, lin, ang = link_pack(base, env_id, jac_columns, body_offset)
            pads_before = pads_of(base, env_id)
            loaded_now = [float(loads[finger]) > empty + noise for finger in range(5)]
            if step_id % 3 == 2:
                center_hat, axis_hat = pose_hat
                vision = False
            else:
                axis_hat = shift_step.cylinder_axis(quat)
                axis_hat = shift_step._rodrigues(pose_rng.normal(0.0, 0.05, 3)) @ axis_hat
                center_hat = center + pose_rng.normal(0.0, 0.002, 3)
                vision = True
            center_hat = correct_center(center_hat, axis_hat, pads_before, loaded_now, MESH_RADIUS)
            pose_hat = (center_hat, axis_hat)
            active = list(range(5))
            normals = []
            contact_jacs = []
            contact_points = []
            directions = []
            stay = []
            radii = []
            for finger in range(5):
                pad = pads_before[finger]
                pad_jac = jac[elastomer_jac[finger], :3, :][:, jac_columns]
                mesh_point = shift_step.closest_on_cylinder(pad, center_hat, axis_hat, MESH_RADIUS, MESH_HALF)
                radial = mesh_point - center_hat
                radial = radial - float(np.dot(radial, axis_hat)) * axis_hat
                radial_norm = float(np.linalg.norm(radial))
                normal = radial / radial_norm if radial_norm > 1e-8 else axis_hat
                holding = bool(loaded_now[finger])
                stay.append(holding)
                radii.append(pad - center_hat)
                if holding:
                    directions.append(None)
                else:
                    delta = mesh_point - pad
                    delta_norm = float(np.linalg.norm(delta))
                    tangent = np.cross(axis_hat, normal)
                    tangent_norm = float(np.linalg.norm(tangent))
                    tangent = tangent / tangent_norm if tangent_norm > 1e-8 else normal
                    directions.append(delta / delta_norm if delta_norm > 1e-8 else tangent)
                normals.append(normal)
                contact_jacs.append(pad_jac)
                contact_points.append(mesh_point)
            release = None
            object_shift = None
            open_fingers = [finger for finger in range(5) if not stay[finger]]
            if open_fingers:
                nearest_gap = None
                nearest_norm = None
                for finger in open_fingers:
                    gap = pads_before[finger] - contact_points[finger]
                    gap = gap - float(np.dot(gap, axis_hat)) * axis_hat
                    gap_norm = float(np.linalg.norm(gap))
                    if nearest_norm is None or gap_norm < nearest_norm:
                        nearest_gap = gap
                        nearest_norm = gap_norm
                if nearest_gap is not None and nearest_norm > 0.004:
                    object_shift = nearest_gap / nearest_norm * min(0.0015, nearest_norm)
            candidates = []
            for finger in range(5):
                if open_fingers or not stay[finger]:
                    continue
                others = [pads_before[other] for other in range(5) if stay[other] and other != finger]
                if len(others) < 3 or not belief_step.support_holds(others, center_hat, MESH_RADIUS):
                    continue
                tangent = np.cross(axis_hat, normals[finger])
                tangent_norm = float(np.linalg.norm(tangent))
                if tangent_norm < 1e-8:
                    continue
                gain = (tangent / tangent_norm) @ contact_jacs[finger]
                reachable = 0.0
                for component in gain:
                    reachable += max(component * -belief_step.MAX_DQ, component * belief_step.MAX_DQ)
                candidates.append((reachable, finger))
            if candidates:
                release = min(candidates)[1]
                normal = normals[release]
                landing = pads_before[release] + normal * 0.003
                delta = landing - pads_before[release]
                delta_norm = float(np.linalg.norm(delta))
                tangent = np.cross(axis_hat, normal)
                tangent_norm = float(np.linalg.norm(tangent))
                tangent = tangent / tangent_norm if tangent_norm > 1e-8 else normal
                directions[release] = delta / delta_norm if delta_norm > 1e-8 else tangent
                stay[release] = False
            held_ok = sum(loaded_now) >= 2 and float(center[2]) >= z0 - DROP
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
            q_all = base.hand_dof_pos[env_id].detach().cpu().numpy()[actuated]
            lower_all = base.hand_dof_lower_limits[env_id].detach().cpu().numpy()[actuated]
            upper_all = base.hand_dof_upper_limits[env_id].detach().cpu().numpy()[actuated]
            close = spheres.watch_pairs(
                names, origins, quats, lin, ang, belief_step.WATCH_DISTANCE
            )
            axis = axis_hat
            if last_axis is not None and float(np.dot(axis, last_axis)) < 0.0:
                axis = -axis
            last_axis = axis
            plan = belief_step.plan_rigid_step(
                contact_jacs,
                radii,
                normals,
                stay,
                directions,
                axis,
                [q_all],
                [lower_all],
                [upper_all],
                [item[:4] for item in close],
                object_shift=object_shift,
            )
            plan["forces"] = [1.0 if holding else 0.0 for holding in stay]
            plan["length"] = float(np.linalg.norm(np.cross(plan["angular"], radii[0])))
            plan["enclosed"] = bool(sum(stay) >= 2)
            axis_std = float(np.sqrt(max(axis @ covariance @ axis, 0.0)))
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
                f"force={[round(plan['forces'][i], 2) for i in range(len(active))]} "
                f"vision={int(vision)} release={NAMES[release] if release is not None else '-'} "
                f"shift={0.0 if object_shift is None else float(np.linalg.norm(object_shift)):.4f} "
                f"loads={[round(float(v), 3) for v in loads]}",
                flush=True,
            )
            if float(np.linalg.norm(plan["dq"])) < 1e-8:
                print("zero increment, keep reading", flush=True)
                env.step(env.zero_actions())
                if video is not None:
                    frame = grab_frame(base)
                    if frame is not None:
                        video.add(frame)
                continue
            actions = torch.zeros_like(base.prev_targets)
            write_action(actions, env_id, plan["dq"], base.actuated_dof_indices, action_scale)
            env.step(actions)
            if video is not None:
                frame = grab_frame(base)
                if frame is not None:
                    video.add(frame)
            gaps = []
            for finger, holding in enumerate(loaded_now):
                if not holding:
                    continue
                rel = pads_before[finger] - center_hat
                radial = rel - float(np.dot(rel, axis)) * axis
                gaps.append(float(np.linalg.norm(radial)) - MESH_RADIUS)
            measured = float(np.mean(gaps)) if gaps else 0.0
            mean, covariance = belief_step.update_belief(mean, covariance, axis, measured, 1e-4)
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
    if video is not None:
        video.close()
        print(f"wrote {VIDEO_PATH} frames={video.count}", flush=True)
    return


if __name__ == "__main__":
    main()
    simulation_app.close()
