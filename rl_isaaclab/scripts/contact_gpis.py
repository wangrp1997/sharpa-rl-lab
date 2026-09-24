"""Fit a 3D implicit surface to one frozen Sharpa grasp.

Observations follow Tactile SLAM: the surface value is 0 at a contact, and
its gradient is the outward unit normal. The released kernel in
gpis-touch-public is the 2D cubic kernel, so this script uses a squared
exponential with the same length-scale rule, R = 1.1 times the maximum
distance between contacts. No joint command is sent.
"""

import json
from pathlib import Path

import numpy as np

NAMES = ("thumb", "index", "middle", "ring", "pinky")
LOAD_MIN = 0.5
NOISE = 1e-4
LOG = Path("logs/live_diag/trace_off.jsonl")
OUT = Path("logs/live_diag/gpis_env93.json")


def load_contacts(path: Path):
    lines = path.read_text().splitlines()
    frozen = json.loads(lines[0])
    step0 = json.loads(lines[1])
    tips = np.asarray(frozen["fingertip_pos"], dtype=np.float64)
    forces = np.asarray(step0["normal_force"], dtype=np.float64)
    loads = np.asarray(frozen["loads"], dtype=np.float64)
    active = [i for i in range(5) if loads[i] > LOAD_MIN]
    points = tips[active]
    normals = forces[active]
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    return frozen, tips, loads, active, points, normals


def kernel_block(a, b, length):
    """4x4 covariance of (value, gradient) for squared-exponential kernel."""
    d = a - b
    r2 = float(d @ d)
    l2 = length * length
    k = np.exp(-0.5 * r2 / l2)
    # dk/da = -d/l2 * k, dk/db = d/l2 * k
    block = np.zeros((4, 4))
    block[0, 0] = k
    block[0, 1:] = k * d / l2
    block[1:, 0] = -k * d / l2
    outer = np.outer(d, d) / (l2 * l2)
    block[1:, 1:] = k * (np.eye(3) / l2 - outer)
    return block


def fit(points, normals, length):
    n = len(points)
    gram = np.zeros((4 * n, 4 * n))
    target = np.zeros(4 * n)
    for i in range(n):
        target[4 * i + 1: 4 * i + 4] = normals[i]
        for j in range(n):
            gram[4 * i: 4 * i + 4, 4 * j: 4 * j + 4] = kernel_block(points[i], points[j], length)
    gram = gram + NOISE * np.eye(4 * n)
    weight = np.linalg.solve(gram, target)
    return gram, weight


def predict(query, points, length, gram, weight):
    cross = np.zeros(4 * len(points))
    for j, point in enumerate(points):
        cross[4 * j: 4 * j + 4] = kernel_block(query, point, length)[0]
    mean = float(cross @ weight)
    solved = np.linalg.solve(gram, cross)
    variance = float(kernel_block(query, query, length)[0, 0] - cross @ solved)
    return mean, max(variance, 0.0)


def main():
    frozen, tips, loads, active, points, normals = load_contacts(LOG)
    center = points.mean(axis=0)
    scale = np.max(np.abs(points - center))
    local = (points - center) / scale
    length = 1.1 * np.max(np.linalg.norm(local[:, None, :] - local[None, :, :], axis=-1))
    gram, weight = fit(local, normals, length)

    queries = {
        NAMES[i]: tips[i] for i in range(5)
    }
    queries["object_center"] = np.asarray(frozen["object_pos"], dtype=np.float64)
    report = []
    for name, point in queries.items():
        mean, variance = predict((point - center) / scale, local, length, gram, weight)
        report.append({
            "name": name,
            "loaded": bool(name in NAMES and loads[NAMES.index(name)] > LOAD_MIN) if name in NAMES else False,
            "position_m": point.tolist(),
            "mean": mean,
            "variance": variance,
        })
    result = {
        "source": str(LOG),
        "env": frozen["env"],
        "contacts": [NAMES[i] for i in active],
        "loads_n": loads.tolist(),
        "normals_outward": {NAMES[i]: normals[k].tolist() for k, i in enumerate(active)},
        "length_scale_normalized": float(length),
        "position_center_m": center.tolist(),
        "position_scale_m": float(scale),
        "queries": report,
        "note": "freeze snapshot positions; normals from the first logged step because the freeze line has force magnitudes only",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"contacts={result['contacts']} length={length:.3f}")
    for item in report:
        print(f"{item['name']:16} mean={item['mean']:+.4f} var={item['variance']:.4f} loaded={item['loaded']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
