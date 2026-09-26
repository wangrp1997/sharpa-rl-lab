"""Complementarity-free MPC from Jin, arXiv:2408.07855, Algorithm 1.

The dynamics, cone rows, horizon, and costs match examples/mpc/allegro/cube
in Complementarity-Free-Dexterous-Manipulation. Fingertip positions are a
linear map of the current pad Jacobian, because this hand has no Allegro FK.
Contact gap and Jacobian are parameters, frozen over the horizon, as in
plan_once.
"""

import numpy as np
import casadi as cs


N_FINGERS = 5
BETA = 100.0


def _softplus(x):
    z = BETA * x
    return (cs.fmax(z, 0) + cs.log(1 + cs.exp(-cs.fabs(z)))) / BETA


class FreeExplicit:
    def __init__(self, n_cmd, joint_lower, joint_upper, horizon=4, max_iter=50):
        self.n_cmd = int(n_cmd)
        self.n_qpos = 7 + self.n_cmd
        self.n_qvel = 6 + self.n_cmd
        self.horizon = int(horizon)
        self.max_ncon = N_FINGERS
        self.n_cone = self.max_ncon * 4
        self.sigma = 0.5
        self.h = 0.1
        self.u_bound = 0.2
        self.sol_guess = None

        stiffness = np.eye(self.n_cmd)
        inertia = np.eye(6)
        inertia[0:3, 0:3] *= 50.0
        inertia[3:6, 3:6] *= 0.1
        self.Q_inv = np.zeros((self.n_qvel, self.n_qvel))
        self.Q_inv[:6, :6] = np.linalg.inv(inertia)
        self.Q_inv[6:, 6:] = np.linalg.inv(stiffness)
        self.gravity_wrench = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0])
        self.obj_mass = 0.01

        self.q_lower = np.concatenate(([-0.99, -0.99, 0.0], -1e7 * np.ones(4), np.asarray(joint_lower, dtype=float)))
        self.q_upper = np.concatenate(([0.99, 0.99, 1.5], 1e7 * np.ones(4), np.asarray(joint_upper, dtype=float)))
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

        b = cs.vertcat(self.obj_mass * self.gravity_wrench, cmd)
        Q_inv = cs.DM(self.Q_inv)
        v_non = Q_inv @ b / self.h
        bias = jac @ Q_inv @ b
        raw = -sigma * (bias + phi) - 0.1 * sigma * bias / self.h
        force = _softplus(raw)
        vel = v_non + Q_inv @ jac.T @ force / self.h

        quat = q[3:7]
        quat_map = cs.vertcat(
            cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
            cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
            cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]),
        ).T
        next_pos = q[0:3] + self.h * vel[0:3]
        next_quat = quat + 0.5 * self.h * quat_map @ vel[3:6]
        next_hand = q[7:] + self.h * vel[6:]
        next_q = cs.vertcat(next_pos, next_quat, next_hand)
        step = cs.Function("step_once", [q, cmd, phi, jac, sigma], [next_q])

        target_p = cs.SX.sym("target_p", 3)
        target_q = cs.SX.sym("target_q", 4)
        ftp0 = cs.SX.sym("ftp0", N_FINGERS * 3)
        ftp_jac = cs.SX.sym("ftp_jac", N_FINGERS * 3, nu)
        q_ref = cs.SX.sym("q_ref", nu)

        x = cs.SX.sym("x", nq)
        u = cs.SX.sym("u", nu)
        dq = x[7:] - q_ref
        contact_cost = 0
        for finger in range(N_FINGERS):
            ftp = ftp0[3 * finger: 3 * finger + 3] + ftp_jac[3 * finger: 3 * finger + 3, :] @ dq
            contact_cost += cs.sumsqr(x[0:3] - ftp)
        position_cost = cs.sumsqr(x[0:3] - target_p)
        quaternion_cost = 1 - cs.dot(x[3:7], target_q) ** 2
        path_cost = cs.Function(
            "path_cost",
            [x, u, target_p, target_q, ftp0, ftp_jac, q_ref],
            [contact_cost + 0.1 * cs.sumsqr(u)],
        )
        final_cost = cs.Function(
            "final_cost",
            [x, target_p, target_q, ftp0, ftp_jac, q_ref],
            [10 * (100 * position_cost + 5.0 * quaternion_cost)],
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
            cost += path_cost(qk, uk, target_p, target_q, ftp0, ftp_jac, q_ref)
            qk = cs.SX.sym("q" + str(k + 1), nq)
            w.append(qk)
            w0.append(cs.DM.zeros(nq))
            lbw.append(lbq)
            ubw.append(ubq)
            g.append(pred - qk)
        cost += final_cost(qk, target_p, target_q, ftp0, ftp_jac, q_ref)

        packed = cs.vvcat([q0, phi, jac, target_p, target_q, ftp0, ftp_jac, q_ref, sigma])
        program = {"f": cost, "x": cs.vcat(w), "g": cs.vcat(g), "p": packed}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": int(max_iter),
            "ipopt.max_cpu_time": 20,
        }
        self.solver = cs.nlpsol("free_mpc", "ipopt", program, opts)
        self.pack = cs.Function(
            "pack",
            [q0, phi, jac, target_p, target_q, ftp0, ftp_jac, q_ref, sigma],
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

    def plan(self, curr_q, target_p, target_q, phi, jac, ftp0, ftp_jac, q_ref):
        guess = self._guess(curr_q)
        packed = self.pack(curr_q, phi, jac, target_p, target_q, ftp0, ftp_jac, q_ref, self.sigma)
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
        action = w_opt[: self.n_cmd].copy()
        predicted = w_opt[self.n_cmd: self.n_cmd + self.n_qpos].copy()
        return action, status, float(np.array(sol["f"]).reshape(-1)[0]), predicted
