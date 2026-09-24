"""Offline check of the contact program on the screened grasp.

Contact indicators are enumerated. For each subset, two convex programs are solved
with HiGHS: whether the origin lies in the convex hull of the cylinder normals,
and whether polyhedral friction cones can balance gravity. The friction coefficient
is a declared number, not the simulator value.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

NAMES = ("thumb", "index", "middle", "ring", "pinky")
LOG = Path("logs/safe_explore/diag.jsonl")
DECLARED_MU = (0.2, 0.3, 0.5, 0.8)
F_MAX = 5.0
CONE_SIDES = 8


def load_settled(path: Path) -> dict:
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "settled":
                return row
    raise RuntimeError(f"no settled record in {path}")


def cylinder_frames(tips: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    radial = tips[:, :2] - center[:2]
    length = np.linalg.norm(radial, axis=1)
    normal_xy = radial / length[:, None]
    tangent_xy = np.stack((-normal_xy[:, 1], normal_xy[:, 0]), axis=1)
    normal = np.column_stack((normal_xy, np.zeros(5)))
    tangent = np.column_stack((tangent_xy, np.zeros(5)))
    bitangent = np.tile(np.array([0.0, 0.0, 1.0]), (5, 1))
    lever = tips - center
    return normal, tangent, bitangent, lever


def origin_in_hull(normals: np.ndarray) -> tuple[bool, float]:
    """True when the origin is a convex combination of the planar normals."""
    count = normals.shape[0]
    A_eq = np.vstack((normals.T, np.ones(count)))
    b_eq = np.array([0.0, 0.0, 1.0])
    result = linprog(
        np.zeros(count),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * count,
        method="highs",
    )
    if not result.success:
        return False, np.inf
    residual = float(np.linalg.norm(normals.T @ result.x))
    return residual <= 1e-6, residual


def balance_gravity(
    normal: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    lever: np.ndarray,
    active: np.ndarray,
    mass: float,
    mu: float,
) -> tuple[bool, float]:
    """Inner polyhedral cone. Feasible here implies the second-order cone is feasible."""
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return False, np.inf
    angles = np.linspace(0.0, 2.0 * np.pi, CONE_SIDES, endpoint=False)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    # Per active finger: lambda[CONE_SIDES], fn is sum(lambda).
    width = idx.size * CONE_SIDES
    # Force on the object: -fn n + mu fn (c t + s b), fn = sum lambda, (c,s) = direction.
    columns = []
    for finger in idx:
        for cosine, sine in directions:
            push = (
                -normal[finger]
                + mu * cosine * tangent[finger]
                + mu * sine * bitangent[finger]
            )
            columns.append(push)
    force_map = np.column_stack(columns)
    torque_map = np.column_stack(
        [
            np.cross(
                lever[finger],
                -normal[finger] + mu * cosine * tangent[finger] + mu * sine * bitangent[finger],
            )
            for finger in idx
            for cosine, sine in directions
        ]
    )
    A_eq = np.vstack((force_map, torque_map))
    weight = np.zeros(6)
    weight[2] = mass * 9.81
    cost = np.ones(width)
    bounds = [(0.0, F_MAX)] * width
    result = linprog(cost, A_eq=A_eq, b_eq=weight, bounds=bounds, method="highs")
    if not result.success:
        return False, np.inf
    return True, float(result.fun)


def main() -> None:
    settled = load_settled(LOG)
    tips = np.asarray(settled["fingertip_pos"], dtype=float)
    center = np.asarray(settled["object_pos"], dtype=float)
    mass = float(settled["object_mass"])
    measured = np.asarray(settled["in_contact"], dtype=bool)
    normal, tangent, bitangent, lever = cylinder_frames(tips, center)
    normal_xy = normal[:, :2]

    print(f"mass_used_for_wrench={mass:.4f} kg  declared_mu={DECLARED_MU}  f_max={F_MAX}")
    print("finger  radius_mm  normal_deg  measured_contact")
    for i, name in enumerate(NAMES):
        radius = np.linalg.norm(tips[i, :2] - center[:2])
        angle = np.degrees(np.arctan2(normal_xy[i, 1], normal_xy[i, 0]))
        print(f"{name:8} {1e3 * radius:8.1f} {angle:11.1f}  {bool(measured[i])}")

    rows = []
    for size in (3, 4, 5):
        for choice in itertools.combinations(range(5), size):
            mask = np.zeros(5, dtype=bool)
            mask[list(choice)] = True
            inside, residual = origin_in_hull(normal_xy[mask])
            entry = {
                "fingers": [NAMES[i] for i in choice],
                "hull": inside,
                "residual": residual,
                "balance": {},
            }
            for mu in DECLARED_MU:
                ok, effort = balance_gravity(normal, tangent, bitangent, lever, mask, mass, mu)
                entry["balance"][str(mu)] = None if not ok else effort
            rows.append(entry)

    current = [NAMES[i] for i, flag in enumerate(measured) if flag]
    print(f"\nmeasured set: {current}")
    for row in rows:
        if row["fingers"] == current:
            print(
                f"current hull={row['hull']} residual={row['residual']:.2e} "
                f"balance_effort={row['balance']}"
            )

    print("\nsubsets whose normals contain the origin and that balance gravity")
    print(f"{'fingers':42} {'hull':5} " + " ".join(f"mu={mu:<4}" for mu in DECLARED_MU))
    feasible_count = 0
    for row in rows:
        if not row["hull"]:
            continue
        efforts = []
        any_ok = False
        for mu in DECLARED_MU:
            value = row["balance"][str(mu)]
            any_ok = any_ok or value is not None
            efforts.append("  no " if value is None else f"{value:5.2f}")
        if not any_ok:
            continue
        feasible_count += 1
        name = ",".join(row["fingers"])
        print(f"{name:42} {str(row['hull']):5} " + " ".join(efforts))
    print(f"feasible_subsets={feasible_count}")


if __name__ == "__main__":
    main()
