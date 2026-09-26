"""Fingertip in-air MPC from Jin, arXiv:2408.07855.

This is examples/mpc/fingertips, not the Allegro palm task. The robot state is
three fingertip positions. A command is a Cartesian step of at most 5 mm.
The path cost keeps each fingertip near the object center and pays for the
sum of the three inward unit vectors, which is their grasp-closure term.
"""

import numpy as np
import casadi as cs


N_TIPS = 3
BETA = 100.0


def _softplus(x):
    z = BETA * x
    return (cs.fmax(z, 0) + cs.log(1 + cs.exp(-cs.fabs(z)))) / BETA


def _rotation(quat):
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    return cs.vertcat(
        cs.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        cs.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        cs.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


class FingertipExplicit:
    def __init__(self, max_iter=80, n_cone=N_TIPS * 4):
        self.n_cmd = N_TIPS * 3
        self.n_qpos = 7 + self.n_cmd
        self.n_qvel = 6 + self.n_cmd
        self.horizon = 4
        self.n_cone = int(n_cone)
        self.sigma = 1.0
        self.h = 0.1
        self.u_bound = 0.005
        self.stiff = 100.0
        self.sol_guess = None

        inertia = np.eye(6)
        inertia[0:3, 0:3] *= 50.0
        inertia[3:6, 3:6] *= 0.05
        self.Q_inv = np.zeros((self.n_qvel, self.n_qvel))
        self.Q_inv[:6, :6] = np.linalg.inv(inertia)
        self.Q_inv[6:, 6:] = np.eye(self.n_cmd) / self.stiff
        self.gravity_wrench = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0])
        self.obj_mass = 0.01
        self.q_lower = np.concatenate(([-1.0, -1.0, 0.0], -1e7 * np.ones(4), -1.0 * np.ones(self.n_cmd)))
        self.q_upper = np.concatenate(([1.0, 1.0, 1.5], 1e7 * np.ones(4), 2.0 * np.ones(self.n_cmd)))
        self._build(max_iter)

    def _build(self, max_iter):
        nq = self.n_qpos
        nv = self.n_qvel
        nu = self.n_cmd
        sigma = cs.SX.sym("sigma", 1)
        phi = cs.SX.sym("phi", self.n_cone)
        jac = cs.SX.sym("jac", self.n_cone, nv)
        q = cs.SX.sym("q", nq)
        cmd = cs.SX.sym("cmd", nu)

        b = cs.vertcat(self.obj_mass * self.gravity_wrench, self.stiff * cmd)
        Q_inv = cs.DM(self.Q_inv)
        v_non = Q_inv @ b / self.h
        bias = jac @ Q_inv @ b
        raw = -sigma * (bias + phi) - 0.1 * sigma * bias / self.h
        vel = v_non + Q_inv @ jac.T @ _softplus(raw) / self.h

        quat = q[3:7]
        quat_map = cs.vertcat(
            cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
            cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
            cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]),
        ).T
        next_q = cs.vertcat(
            q[0:3] + self.h * vel[0:3],
            quat + 0.5 * self.h * quat_map @ vel[3:6],
            q[7:] + self.h * vel[6:],
        )
        step = cs.Function("step_once", [q, cmd, phi, jac, sigma], [next_q])

        target_p = cs.SX.sym("target_p", 3)
        target_q = cs.SX.sym("target_q", 4)
        x = cs.SX.sym("x", nq)
        u = cs.SX.sym("u", nu)
        contact_cost = 0
        closure = 0
        rotation = _rotation(x[3:7])
        for finger in range(N_TIPS):
            tip = x[7 + 3 * finger: 10 + 3 * finger]
            contact_cost += cs.sumsqr(x[0:3] - tip)
            offset = rotation.T @ (tip - x[0:3])
            closure += offset / (cs.norm_2(offset) + 1e-6)
        path_cost = cs.Function(
            "path_cost",
            [x, u, target_p, target_q],
            [contact_cost + 0.05 * cs.sumsqr(closure) + 50 * cs.sumsqr(u)],
        )
        final_cost = cs.Function(
            "final_cost",
            [x, target_p, target_q],
            [10 * (500 * cs.sumsqr(x[0:3] - target_p) + 5.0 * (1 - cs.dot(x[3:7], target_q) ** 2))],
        )

        w, w0, lbw, ubw, g = [], [], [], [], []
        cost = 0
        q0 = cs.SX.sym("q0", nq)
        lbu = cs.SX.sym("lbu", nu)
        ubu = cs.SX.sym("ubu", nu)
        lbq = cs.SX.sym("lbq", nq)
        ubq = cs.SX.sym("ubq", nq)
        qk = q0
        for k in range(self.horizon):
            uk = cs.SX.sym("u" + str(k), nu)
            w.append(uk)
            lbw.append(lbu)
            ubw.append(ubu)
            w0.append(cs.DM.zeros(nu))
            pred = step(qk, uk, phi, jac, sigma)
            cost += path_cost(qk, uk, target_p, target_q)
            qk = cs.SX.sym("q" + str(k + 1), nq)
            w.append(qk)
            w0.append(cs.DM.zeros(nq))
            lbw.append(lbq)
            ubw.append(ubq)
            g.append(pred - qk)
        cost += final_cost(qk, target_p, target_q)

        packed = cs.vvcat([q0, phi, jac, target_p, target_q, sigma])
        program = {"f": cost, "x": cs.vcat(w), "g": cs.vcat(g), "p": packed}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": int(max_iter),
            "ipopt.max_cpu_time": 20,
        }
        self.solver = cs.nlpsol("fingertip_mpc", "ipopt", program, opts)
        self.pack = cs.Function(
            "pack",
            [q0, phi, jac, target_p, target_q, sigma],
            [packed],
        )
        self.bound = cs.Function("bound", [lbu, ubu, lbq, ubq], [cs.vcat(lbw), cs.vcat(ubw)])
        self.w0 = np.array(cs.vcat(w0)).reshape(-1)
        self.n_g = int(cs.vcat(g).shape[0])

    def _guess(self, curr_q):
        if self.sol_guess is not None:
            return self.sol_guess
        pieces = []
        for _ in range(self.horizon):
            pieces.append(np.zeros(self.n_cmd))
            pieces.append(np.asarray(curr_q, dtype=float))
        return {
            "x0": np.concatenate(pieces),
            "lam_x0": np.zeros(self.w0.shape[0]),
            "lam_g0": np.zeros(self.n_g),
        }

    def plan(self, curr_q, target_p, target_q, phi, jac):
        guess = self._guess(curr_q)
        packed = self.pack(curr_q, phi, jac, target_p, target_q, self.sigma)
        lower, upper = self.bound(
            -self.u_bound * np.ones(self.n_cmd),
            self.u_bound * np.ones(self.n_cmd),
            self.q_lower,
            self.q_upper,
        )
        sol = self.solver(
            x0=guess["x0"],
            lam_x0=guess["lam_x0"],
            lam_g0=guess["lam_g0"],
            lbx=lower,
            ubx=upper,
            lbg=0.0,
            ubg=0.0,
            p=packed,
        )
        status = self.solver.stats()["return_status"]
        w_opt = np.array(sol["x"]).reshape(-1)
        self.sol_guess = {
            "x0": w_opt,
            "lam_x0": np.array(sol["lam_x"]).reshape(-1),
            "lam_g0": np.array(sol["lam_g"]).reshape(-1),
        }
        predicted = w_opt[self.n_cmd: self.n_cmd + self.n_qpos].copy()
        return w_opt[: self.n_cmd].copy(), status, float(np.array(sol["f"]).reshape(-1)[0]), predicted
