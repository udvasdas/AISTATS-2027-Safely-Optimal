"""PRUNE (Algorithm 1) for unstructured Gaussian bandits with unknown linear constraints.

Every pure-exploration algorithm exposes the same interface, driven by `experiments.run_episode`:

    algo.reset(); pull algo.initial_arms()
    while not algo.should_stop(): a = algo.next_arm(); algo.observe(a, y)
    p_hat = algo.recommend()
"""

import functools
import itertools
import math

import numpy as np
from scipy.optimize import linprog

from .lower_bound import AltSolver, ascend_answer, init_answer, solve_sop
from .thresholds import beta_stylized, beta_theory


class PureExplorationAlgorithm:
    """Book-keeping shared by all algorithms: counts, running means and the stopping threshold."""

    name = "base"

    def __init__(self, K, Sigma, delta, eps, threshold="stylized"):
        self.K = K
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.d = self.Sigma.shape[0] - 1
        self.delta = delta
        self.eps = eps
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.t = 0
        self.N = np.zeros(self.K)
        self.S = np.zeros((self.K, self.d + 1))

    def initial_arms(self):
        return list(range(self.K))

    def observe(self, a, y):
        self.t += 1
        self.N[a] += 1
        self.S[a] += y

    @property
    def means(self):
        return self.S / np.maximum(self.N, 1)[:, None]

    @property
    def mu_hat(self):
        return self.means[:, 0]

    @property
    def B_hat(self):
        return self.means[:, 1:].T

    def beta(self):
        if self.threshold == "theory":
            return beta_theory(self.N, self.delta, self.d)
        return beta_stylized(self.t, self.delta)

    def should_stop(self):
        raise NotImplementedError

    def next_arm(self):
        raise NotImplementedError

    def recommend(self):
        raise NotImplementedError


@functools.lru_cache(maxsize=None)
def _support_pairs(n, K, s):
    """All (rows, cols) index pairs of size s for an n x K game, as two (m, s) arrays."""
    rows = list(itertools.combinations(range(n), s))
    cols = list(itertools.combinations(range(K), s))
    R = np.array([r for r in rows for _ in cols], dtype=int).reshape(-1, s)
    C = np.array([c for _ in rows for c in cols], dtype=int).reshape(-1, s)
    return R, C


def solve_game(P, tol=1e-10):
    """Exact max_{x in simplex} min_i (P x)_i for a small n x K matrix P by support enumeration.

    For every pair of equal-size supports the equaliser strategies of both players are solved in one
    batched linear solve; a pair is optimal when both are non-negative and neither player can improve.
    Returns x, or None if no pair qualifies (degenerate game).
    """
    n, K = P.shape
    scale = max(np.abs(P).max(), 1e-300)
    for s in range(1, min(n, K) + 1):
        R, C = _support_pairs(n, K, s)
        m = len(R)
        Psub = P[R[:, :, None], C[:, None, :]]                  # (m, s, s)
        Ax = np.zeros((m, s + 1, s + 1))
        Ay = np.zeros((m, s + 1, s + 1))
        Ax[:, :s, :s], Ax[:, :s, s], Ax[:, s, :s] = Psub, -1.0, 1.0
        Ay[:, :s, :s], Ay[:, :s, s], Ay[:, s, :s] = Psub.transpose(0, 2, 1), -1.0, 1.0
        rhs = np.zeros((m, s + 1, 1))
        rhs[:, s] = 1.0
        try:
            zx = np.linalg.solve(Ax, rhs)[..., 0]
            zy = np.linalg.solve(Ay, rhs)[..., 0]
        except np.linalg.LinAlgError:
            return None
        x = np.zeros((m, K))
        y = np.zeros((m, n))
        np.put_along_axis(x, C, zx[:, :s], axis=1)
        np.put_along_axis(y, R, zy[:, :s], axis=1)
        v = zx[:, s]
        t = tol * scale
        ok = ((zx[:, :s] >= -t).all(1) & (zy[:, :s] >= -t).all(1)
              & ((x @ P.T).min(1) >= v - t) & ((y @ P).max(1) <= v + t))
        if ok.any():
            xs = np.clip(x[np.argmax(ok)], 0.0, None)
            return xs / xs.sum()
    return None


def fw_direction(H, omega):
    """x = argmax_{x in simplex} min_{h in conv(H)} <x - omega, h>   (Eq. 9)."""
    if H.shape[0] == 1:
        x = np.zeros_like(omega)
        x[np.argmax(H[0])] = 1.0
        return x
    if H.shape[0] <= 6:
        x = solve_game(H - (H @ omega)[:, None])
        if x is not None:
            return x
    K = omega.shape[0]
    # variables (x, z): max z  s.t.  z - h_i . x <= -h_i . omega
    res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.c_[-H, np.ones(len(H))], b_ub=-H @ omega,
                  A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1.0],
                  bounds=[(0, None)] * K + [(None, None)], method="highs")
    x = np.clip(res.x[:K], 0.0, None)
    return x / x.sum()


class PRUNE(PureExplorationAlgorithm):
    """Pure exploRation Under uNknown linEar constraints (Algorithm 1), known covariance Sigma.

    Parameters
    ----------
    delta, eps : confidence level and optimality slack of the SeOP set Pi(nu, eps).
    threshold : "stylized" (log((1 + log t)/delta)) or "theory" (Lemma 2).
    recommend : "certified" returns the answer p_t whose GLR crossed the threshold; "lp" returns the
        plug-in LP solution of line 15 (which lies on the boundary of the estimated constraints).
    r_scale : r_t = r_scale * t^-0.9 / K for the r-subdifferential (Eq. 8).
    p_every : rounds between two trust-region ascent steps on the candidate answer p.
    verify_factor : the (exact) stopping statistic at N_t is only computed once
        t * F(omega_t, p_t) >= verify_factor * beta; skipping it can only delay stopping.
    lazy : update the answer p_t and the gradient of F, and test the stopping rule, only every `lazy`
        rounds (C-tracking of the last FW directions in between). Stopping on a subset of the rounds keeps
        the delta-correctness; used on large instances (IMDB) where one evaluation of F takes seconds.
    """

    name = "PRUNE"

    def __init__(self, K, Sigma, delta, eps, threshold="stylized", recommend="certified",
                 r_scale=1.0, p_every=1, verify_factor=0.5, solver=None, lazy=1):
        self.lazy = int(lazy)
        self.solver = solver or AltSolver(K, Sigma, eps)
        self.recommend_mode = recommend
        self.r_scale = r_scale
        self.p_every = p_every
        self.verify_factor = verify_factor
        super().__init__(K, Sigma, delta, eps, threshold)

    def reset(self):
        super().reset()
        self.omega = np.full(self.K, 1.0 / self.K)     # FW iterate, omega_K = uniform after warm-up
        self.p = None                                  # candidate answer in Pi(nu_hat, eps)
        self.ev = None
        self.rho = 0.1                                 # trust-region radius for p
        self.Z = 0.0                                   # last computed GLR statistic
        self.n_reinit = 0

    # ------------------------------------------------------------------ answer p_t
    def _update_answer(self):
        mu, B = self.mu_hat, self.B_hat
        ev = None
        if self.p is not None:
            ev = self.solver.evaluate(mu, B, self.omega, self.p)
        if ev is None or ev.F <= 0.0:
            # p_t left the relative interior of Pi(nu_hat, eps): restart from a central answer
            self.n_reinit += 1
            self.p, ev = init_answer(self.solver, mu, B, self.omega)
            if self.p is None:
                self.ev = None
                return
            self.rho = 0.1
            self.p, ev, self.rho = ascend_answer(self.solver, mu, B, self.omega, self.p, ev, self.rho,
                                                 iters=5)
        elif self.t % self.p_every == 0:
            self.p, ev, self.rho = ascend_answer(self.solver, mu, B, self.omega, self.p, ev,
                                                 max(self.rho, 1e-3), iters=1)
        self.ev = ev

    # ------------------------------------------------------------------ interface
    def should_stop(self):
        """Chernoff stopping rule of Lemma 2(a) for the candidate answer p_t."""
        if self.lazy > 1 and self.t % self.lazy:
            return False
        self._update_answer()
        if self.ev is None or self.ev.F <= 0.0:
            return False
        beta = self.beta()
        if self.t * self.ev.F < self.verify_factor * beta:
            return False
        # GLR: min over the alternative set of sum_a N_a KL(nu_hat_a || lambda_a) = t F(N_t / t, p_t)
        ev_n = self.solver.evaluate(self.mu_hat, self.B_hat, self.N / self.t, self.p)
        self.Z = self.t * ev_n.F
        return self.Z > beta

    def next_arm(self):
        t, K = self.t, self.K
        forced = (t % K == 0 and math.isqrt(t // K) ** 2 == t // K)
        if forced or self.ev is None or self.ev.F <= 0.0:
            x = np.full(K, 1.0 / K)                     # forced exploration
        else:
            ev = self.ev
            r = self.r_scale * t ** -0.9 / K
            active = np.isfinite(ev.vals) & (ev.vals < ev.F + r)
            x = fw_direction(ev.grad_w[active], self.omega)
        self.omega = (t * self.omega + x) / (t + 1)
        # C-tracking of the cumulative FW directions (Lemma 5): sum_{s <= t+1} x_s = (t+1) omega_{t+1}
        return int(np.argmin(self.N - (t + 1) * self.omega))

    def recommend(self):
        if self.recommend_mode == "lp" or self.p is None:
            p, _ = solve_sop(self.mu_hat, self.B_hat)
            return p if p is not None else np.full(self.K, 1.0 / self.K)
        return self.p.copy()


class PRUNEUniform(PRUNE):
    """Uniform baseline for unknown constraints: arms drawn i.i.d. uniformly, with PRUNE's stopping rule and
    certified recommendation (the answer p_t is ascended for the uniform allocation). Only the sampling rule
    differs from PRUNE."""

    name = "Uniform"

    def reset(self):
        super().reset()
        self.rng = np.random.default_rng()

    def next_arm(self):
        return int(self.rng.integers(self.K))
