"""PRUNE-FW: PRUNE whose minimisation over the witness policy q uses Frank-Wolfe instead of a grid over the
(d+1)-sparse faces of the simplex, with a certified lower bound for the stopping rule.

Reward branch of F (see lower_bound.py):  F_1(omega, p) = min_{q in simplex} g(q),
    g(q) = max_{eta >= 0} eta^T b(q) - 1/2 eta^T M(q) eta     (exact (d+1)-dimensional dual for fixed q).

* Sampling (allocation) uses F_hat_1 = min over local minimisers of g found by multi-start Frank-Wolfe over the
  simplex (starts: every pure arm and the empirical LP optimum; linear minimisation oracle = best vertex e_a for
  the gradient of g, step size by a geometric line search). Its cost per evaluation is
  O((K + 1) * iters * n_gamma * (K d + 2^(d+1) (d+1)^3)), polynomial in K, instead of C(K, d+1) * grid.
  g is not convex in q (Remark 1), so F_hat_1 >= F_1: an approximation, which only affects the sampling rule.

* Stopping uses a certified lower bound. For any y >= 0 and any alternative (mu', B') whose LP optimum beats p by
  eps, weak LP duality gives  mu'^T v <= max_a (mu'_a - B'_a^T y)  for every v with B'v <= 0, hence
      Alt_r(nu, p, eps)  subset of  A_y = union_a {lam : mu'_a - B'_a^T y - mu'^T p >= eps},
  a union of K half-spaces in (mu', B'). The divergence to a half-space is in closed form, so
      F_1(omega, p) >= LB(y) = min_a (eps - mu_a + B_a^T y + mu^T p)_+^2 / (2 V_a(y)),
      V_a(y) = sum_{b != a} sigma_r^2 p_b^2 / omega_b + c_a^T Sigma c_a / omega_a,   c_a = (1 - p_a, -y).
  The stopping statistic t * min(max_y LB(y), F_2) (F_2 = cost branch, exact) never exceeds the exact GLR of
  Lemma 2, so stopping on it keeps delta-correctness; it can only stop later than the exact rule.
"""

import itertools

import numpy as np
from scipy.optimize import linprog, minimize, nnls

from .algorithms import PRUNE
from .lower_bound import AltSolver, Evaluation


class FWAltSolver(AltSolver):
    """Same interface as AltSolver (evaluate -> Evaluation), reward branch by multi-start Frank-Wolfe over q."""

    def __init__(self, K, Sigma, eps, iters=40, n_gamma=10, tol=1e-9, qp="auto"):
        """qp: solver of the (d+1)-dimensional non-negative dual QP. "enum" enumerates its 2^(d+1) active sets
        (exact, fastest for d <= 3); "nnls" uses a Cholesky factorisation and Lawson-Hanson NNLS (polynomial in d);
        "auto" picks enum for d <= 3 and nnls otherwise."""
        self.qp = qp
        self.K = K
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.m = self.Sigma.shape[0]
        self.d = self.m - 1
        self.eps = float(eps)
        self.sr2 = self.Sigma[0, 0]
        self.Src = self.Sigma[1:, 0]
        self.Sc = self.Sigma[1:, 1:]
        self.Sc_diag = np.diag(self.Sc).copy()
        self.subsets = [np.array(S) for r in range(1, self.m + 1)
                        for S in itertools.combinations(range(self.m), r)]
        self.iters, self.tol = iters, tol
        self.gammas = 0.5 ** np.arange(n_gamma)          # step sizes tried by the line search
        self._lp_cache = (None, None)

    # ------------------------------------------------------------------ dual QP for fixed q
    def _dual_qp(self, mu, B, omega, p, Q):
        use_nnls = self.qp == "nnls" or (self.qp == "auto" and self.d > 3)
        if not use_nnls:
            return super()._dual_qp(mu, B, omega, p, Q)
        v = Q - p
        W = 1.0 / omega
        b = np.column_stack([self.eps - v @ mu, Q @ B.T])               # (n, m)
        brr, brc, bcc = (v * v) @ W, (Q * v) @ W, (Q * Q) @ W
        n, m = Q.shape[0], self.m
        M = np.empty((n, m, m))
        M[:, 0, 0] = self.sr2 * brr
        M[:, 0, 1:] = -brc[:, None] * self.Src[None]
        M[:, 1:, 0] = M[:, 0, 1:]
        M[:, 1:, 1:] = bcc[:, None, None] * self.Sc[None]
        # max_{eta >= 0} eta^T b - 1/2 eta^T M eta  =  min_{eta >= 0} 1/2 ||L^T eta - L^-1 b||^2 - 1/2 ||L^-1 b||^2
        M = M + 1e-12 * (1.0 + np.trace(M, axis1=1, axis2=2))[:, None, None] * np.eye(m)[None]
        L = np.linalg.cholesky(M)                                       # batched, O(n m^3)
        y = np.linalg.solve(L, b[..., None])[..., 0]
        lam = np.zeros((n, m))
        for i in range(n):
            lam[i] = nnls(L[i].T, y[i])[0]
        val = np.einsum("ij,ij->i", lam, b) - 0.5 * np.einsum("ij,ijk,ik->i", lam, M, lam)
        degenerate = (brr <= 1e-300) & (b[:, 0] > 0)                    # q = p: no alternative in this branch
        val = np.where(degenerate, np.inf, val)
        return val, lam

    # ------------------------------------------------------------------ Frank-Wolfe over q
    def _grad_q(self, mu, B, omega, p, Q, lam):
        """Gradient of g in q (envelope theorem on the dual), rows of Q with optimal multipliers lam."""
        l0, lc = lam[:, :1], lam[:, 1:]
        rc = (lc @ self.Src)[:, None]                                   # Sigma_rc^T eta_c
        cc = np.einsum("ij,jk,ik->i", lc, self.Sc, lc)[:, None]          # eta_c^T Sigma_c eta_c
        W = 1.0 / omega
        return (-l0 * mu[None] + lc @ B
                - (self.sr2 * l0 * l0 * (Q - p) - l0 * rc * (2 * Q - p) + cc * Q) * W[None])

    def _lp(self, mu, B):
        key = (mu.tobytes(), B.tobytes())
        if self._lp_cache[0] != key:
            res = linprog(-mu, A_ub=B, b_ub=np.zeros(len(B)), A_eq=np.ones((1, self.K)), b_eq=[1.0],
                          bounds=[(0, None)] * self.K, method="highs")
            if res.status == 0:
                x = np.clip(res.x, 0, None)
                self._lp_cache = (key, (x / x.sum(), np.maximum(-res.ineqlin.marginals, 0.0)))
            else:
                self._lp_cache = (key, (None, np.zeros(len(B))))
        return self._lp_cache[1]

    def _starts(self, mu, B, p):
        S = [np.eye(self.K)]
        p_lp, _ = self._lp(mu, B)
        if p_lp is not None:
            S.append(p_lp[None])
        Q = np.vstack(S)
        return Q[np.abs(Q - p).sum(1) > 1e-9]                        # q = p has no alternative

    def _fw_search(self, mu, B, omega, p):
        Q = self._starts(mu, B, p)
        val, lam = self._dual_qp(mu, B, omega, p, Q)
        K, n = self.K, len(Q)
        active = np.ones(n, dtype=bool)
        for _ in range(self.iters):
            if not active.any():
                break
            G = self._grad_q(mu, B, omega, p, Q, lam)
            a = G.argmin(1)
            gap = np.einsum("ij,ij->i", G, Q) - G[np.arange(n), a]       # FW gap <grad, q - e_a>
            active &= np.isfinite(val) & (gap > self.tol * np.maximum(np.abs(val), 1e-12))
            idx = np.flatnonzero(active)
            if len(idx) == 0:
                break
            Dv = -Q[idx]
            Dv[np.arange(len(idx)), a[idx]] += 1.0                        # e_a - q
            C = Q[idx, None, :] + self.gammas[None, :, None] * Dv[:, None, :]
            v2, l2 = self._dual_qp(mu, B, omega, p, C.reshape(-1, K))
            v2, l2 = v2.reshape(len(idx), -1), l2.reshape(len(idx), len(self.gammas), -1)
            j = v2.argmin(1)
            better = v2[np.arange(len(idx)), j] < val[idx] - 1e-15
            upd = idx[better]
            Q[upd] = C[better, j[better]]
            val[upd] = v2[better, j[better]]
            lam[upd] = l2[better, j[better]]
            active[idx[~better]] = False
        # one piece per distinct local minimiser
        keep = np.isfinite(val)
        Q, val, lam = Q[keep], val[keep], lam[keep]
        _, first = np.unique(np.round(Q, 6), axis=0, return_index=True)
        return val[first], Q[first], lam[first]

    def _face_search(self, mu, B, omega, p):          # used by AltSolver.evaluate
        if len(self._starts(mu, B, p)) == 0:
            return np.array([np.inf]), np.zeros((1, self.K)), np.zeros((1, self.m))
        val, q, lam = self._fw_search(mu, B, omega, p)
        if len(val) == 0:
            return np.array([np.inf]), np.zeros((1, self.K)), np.zeros((1, self.m))
        return val, q, lam

    # ------------------------------------------------------------------ certified lower bound
    def reward_lower_bound(self, mu, B, omega, p, Y):
        """LB(y) <= F_1(omega, p) for every row y >= 0 of Y (n, d). Returns (n,)."""
        omega = np.maximum(omega, 1e-12)
        W = 1.0 / omega
        S = self.sr2 * np.sum(p * p * W)
        ca0 = 1.0 - p                                                   # (K,)
        # c_a^T Sigma c_a = sr2 (1-p_a)^2 - 2 (1-p_a) Src^T y + y^T Sc y
        rc = Y @ self.Src                                               # (n,)
        yy = np.einsum("ij,jk,ik->i", Y, self.Sc, Y)                    # (n,)
        caSca = self.sr2 * ca0[None] ** 2 - 2 * ca0[None] * rc[:, None] + yy[:, None]
        V = S - self.sr2 * p * p * W + caSca * W[None]                  # (n, K)
        gap = self.eps - mu[None] + Y @ B + mu @ p                      # (n, K): eps - (mu_a - B_a^T y - mu^T p)
        with np.errstate(divide="ignore", invalid="ignore"):
            # V_a = 0 only for a = p pure, whose half-space cannot be reached (gap = eps > 0): +inf
            vals = np.where(V > 0, np.maximum(gap, 0.0) ** 2 / (2 * V), np.inf)
        return np.min(vals, axis=1)

    def certified_reward(self, mu, B, omega, p):
        """max_{y >= 0} LB(y): start from y = 0 and the LP dual of the plug-in problem, refine by Nelder-Mead."""
        _, y_lp = self._lp(mu, B)
        Y0 = np.vstack([np.zeros(self.d), y_lp, 2 * y_lp, 0.5 * y_lp])
        lb0 = self.reward_lower_bound(mu, B, omega, p, Y0)
        best_y, best = Y0[lb0.argmax()], lb0.max()
        f = lambda z: -self.reward_lower_bound(mu, B, omega, p, (z * z)[None])[0]   # y = z^2 >= 0
        res = minimize(f, np.sqrt(best_y), method="Nelder-Mead",
                       options=dict(xatol=1e-8, fatol=1e-12 * max(best, 1e-12), maxiter=200 * self.d))
        return max(best, -res.fun)

    def certified_F(self, mu, B, omega, p):
        """Lower bound on F(omega, p): min(certified reward branch, exact cost branch)."""
        omega = np.maximum(omega, 1e-12)
        slack = -(B @ p)
        if np.any(slack <= 0):
            return 0.0
        P = np.sum(p * p / omega)
        cost = np.min(slack ** 2 / (2 * self.Sc_diag * P))
        return min(self.certified_reward(mu, B, omega, p), cost)


class PRUNEFW(PRUNE):
    """PRUNE with Frank-Wolfe over the witness q.

    stopping : "fw" tests t * F_hat(N_t / t, p_t) with the Frank-Wolfe value, exactly like PRUNE tests the grid
        value (neither is certified, both upper-bound the exact statistic); "certified" tests the certified lower
        bound, which keeps the delta-correctness of Lemma 2 but is loose for d >= 2.
    """

    name = "PRUNE-FW"

    def __init__(self, K, Sigma, delta, eps, fw_iters=40, stopping="fw", **kwargs):
        self.stopping = stopping
        super().__init__(K, Sigma, delta, eps, solver=FWAltSolver(K, Sigma, eps, iters=fw_iters), **kwargs)

    def should_stop(self):
        if self.stopping == "fw":
            return super().should_stop()
        if self.lazy > 1 and self.t % self.lazy:
            return False
        self._update_answer()
        if self.ev is None or self.ev.F <= 0.0:
            return False
        beta = self.beta()
        # ev.F >= F >= certified F: skipping the certificate when t * ev.F is small never misses a stop
        if self.t * self.ev.F < self.verify_factor * beta:
            return False
        self.Z = self.t * self.solver.certified_F(self.mu_hat, self.B_hat, self.N / self.t, self.p)
        return self.Z > beta
