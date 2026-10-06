"""Known-constraint baselines of Carlsson, Basu, Johansson & Dubhashi (2024), "Pure exploration in bandits
with linear constraints": CTnS (Algorithm 1), CGE (Algorithm 2) and the uniform sampler of their Section 5.

Setting: Gaussian rewards N(mu_a, sigma^2), constraints B pi <= 0 KNOWN to the learner (the cost
observations of the environment are ignored), target = the exact optimal policy pi* = argmax_{pi in F} mu^T pi,
F = {pi in simplex : B pi <= 0}. All three share
  * recommendation: pi_hat_t = argmax_{pi in F} mu_hat_t^T pi (a vertex of F, enumerated once);
  * stopping (Eq. 10): min_{pi' in nbrs(pi_hat_t)} (mu_hat^T (pi_hat - pi'))_+^2 / (2 sigma^2 sum_a (pi_hat - pi')_a^2 / N_a)
    > c(t, delta) = log((1 + log t) / delta), the threshold of their experiments;
  * exploration set Pi: F itself for anytime constraints (their Fig. 3c/3d), the simplex for end-of-time.
Sampling rules:
  * CTnS: w*_t in argmax_{w in Pi} D(w, mu_hat_t, F) (cutting planes), mixed with the projected uniform
    allocation at rate eps_t = 1 / (2 sqrt(K^2 + t)), C-tracking of the cumulative mixed allocations.
  * CGE: AdaGrad over Pi (allocation player), best response over the neighbours (instance player), optimistic
    gains U_a = max(f(t) / N_a, max_{xi in {alpha_a, beta_a}} d(xi, lambda_a)), [alpha_a, beta_a] =
    {xi : N_a d(mu_hat_a, xi) <= f(t)}, f(t) = log t; same tracking as CTnS.
  * Uniform: arms drawn i.i.d. from the uniform allocation, projected onto Pi when it is not in Pi.

`relaxed=True`: eps-optimal identification (their Appendix D). The answers are the vertices pi of F and
the alternative set of pi is {lambda : lambda^T (v - pi) > eps for some vertex v}; with eps > 0 the
neighbours no longer suffice, so every vertex v != pi enters, with margin (mu^T (pi - v) + eps)_+.
  * stopping: max over vertices pi of GLR(pi) > c(t, delta), recommend the maximiser (only vertices that
    are eps-good under mu_hat have GLR > 0), the standard GLR rule for multiple correct answers;
  * CTnS: w*_t from the eps-good vertex (under mu_hat) with the largest oracle value max_w F_pi(w);
  * CGE: the instance player best-responds for the eps-good vertex that the current w certifies best.
Answers are not sticky; Appendix D warns that without stickiness the target may switch between
eps-good vertices.
"""

import numpy as np
from scipy.optimize import minimize

from .algorithms import PureExplorationAlgorithm
from .hardness import _kelley, _vertex_pieces
from .lagex import PolytopeVertices, gaussian_projections, neighbours, project_simplex_weighted


def project_onto(y, B, Q=None, x0=None):
    """argmin_{x in simplex, B x <= 0} (x - y)^T Q (x - y), Q = I by default (B may have zero rows)."""
    if Q is None:
        x = project_simplex_weighted(y, np.ones_like(y))
        if len(B) == 0 or np.all(B @ x <= 1e-12):
            return x
        Q = np.eye(len(y))
    res = minimize(lambda x: (x - y) @ Q @ (x - y), np.full(len(y), 1.0 / len(y)) if x0 is None else x0,
                   jac=lambda x: 2 * Q @ (x - y), method="SLSQP", bounds=[(0, 1)] * len(y),
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1, "jac": lambda x: np.ones_like(x)},
                                {"type": "ineq", "fun": lambda x: -B @ x, "jac": lambda x: -B}],
                   options=dict(ftol=1e-12, maxiter=200))
    x = np.maximum(res.x, 0.0)
    return x / x.sum()


class _KnownConstraintExplorer(PureExplorationAlgorithm):
    """Recommendation, stopping and tracking shared by CTnS, CGE and Uniform."""

    def __init__(self, K, Sigma, delta, eps, B, exploration="anytime", threshold="stylized", relaxed=False,
                 reward_var=None):
        self.B = np.atleast_2d(np.asarray(B, float))
        self.exploration, self.relaxed = exploration, relaxed
        self.B_pi = self.B if exploration == "anytime" else np.zeros((0, K))      # constraints of Pi
        # reward variance per arm (known); default: the reward variance of Sigma for every arm
        self.var = np.full(K, float(np.asarray(Sigma, float)[0, 0])) if reward_var is None \
            else np.asarray(reward_var, float)
        self.V, tight, Ineq = PolytopeVertices(K, self.B.shape[0])(self.B)
        if relaxed:                                          # every other vertex is an alternative
            self.nbrs = [np.delete(self.V, k, axis=0) for k in range(len(self.V))]
        else:
            self.nbrs = [neighbours(k, self.V, tight, Ineq) for k in range(len(self.V))]
        self.r = float(eps) if relaxed else 0.0
        self.uniform = project_onto(np.full(K, 1.0 / K), self.B_pi)
        super().__init__(K, Sigma, delta, eps, threshold)

    def reset(self):
        super().reset()
        self.cum_w = np.zeros(self.K)
        self.rng = np.random.default_rng()

    def _vertex(self):
        return int(np.argmax(self.V @ self.mu_hat))

    def _candidates(self):
        """Vertices that may be recommended: pi_hat* (exact), or the eps-good vertices under mu_hat (relaxed)."""
        if not self.relaxed:
            return [self._vertex()]
        vals = self.V @ self.mu_hat
        return list(np.flatnonzero(vals > vals.max() - self.r))

    def _values(self, k, w):
        # per-arm variances enter as effective weights w_a / sigma_a^2
        return gaussian_projections(w / self.var, self.mu_hat, self.V[k], self.nbrs[k], self.r, 1.0)

    def glr(self):
        """max over candidate answers of the GLR statistic; sets the answer to recommend."""
        best = -1.0
        for k in self._candidates():
            z = self._values(k, self.N)[0].min()
            if z > best:
                best, self.answer = z, k
        return best

    def should_stop(self):
        return self.glr() > self.beta()

    def recommend(self):
        self.glr()
        return self.V[self.answer].copy()

    def _track(self, w):
        """C-tracking of the allocations mixed with the (projected) uniform one at rate eps_t."""
        e = 1.0 / (2.0 * np.sqrt(self.K ** 2 + self.t))
        self.cum_w += (1.0 - self.K * e) * w + self.K * e * self.uniform
        return int(np.argmin(self.N - self.cum_w))


class CTnS(_KnownConstraintExplorer):
    """Constrained Track-and-Stop (Algorithm 1 of Carlsson et al., 2024).

    lazy : re-solve the oracle only every `lazy` rounds and track the last solution in between (lazy TaS);
        the stopping rule is still tested every round.
    """

    name = "CTnS"

    def __init__(self, *args, lazy=1, **kwargs):
        self.lazy = int(lazy)
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.w_cache = None

    def next_arm(self):
        if self.lazy > 1 and self.w_cache is not None and self.t % self.lazy:
            return self._track(self.w_cache)
        w, val = self.uniform, 0.0
        for k in self._candidates():
            wk, vk = _kelley(_vertex_pieces(self.mu_hat, self.V[k], self.nbrs[k], self.r, self.var), self.K,
                             A_ub=self.B_pi if len(self.B_pi) else None, tol=1e-3)
            if vk > val:
                w, val = wk, vk
        self.w_cache = w
        return self._track(w)


class CGE(_KnownConstraintExplorer):
    """Constrained Game Explorer (Algorithm 2 of Carlsson et al., 2024)."""

    name = "CGE"

    def __init__(self, K, Sigma, delta, eps, B, exploration="anytime", threshold="stylized", relaxed=False,
                 reward_var=None, eta=1 / np.sqrt(2), mu_range=(-1.0, 10.0)):
        self.eta, self.mu_range = eta, mu_range
        super().__init__(K, Sigma, delta, eps, B, exploration, threshold, relaxed, reward_var)

    def reset(self):
        super().reset()
        self.w = self.uniform.copy()
        self.H = 0.1 * np.eye(self.K)                     # delta I + sum_s U_s U_s^T

    def next_arm(self):
        w, mu, N = self.w, self.mu_hat, self.N
        # best response of the instance player (relaxed: for the eps-good answer that w certifies best)
        best = -1.0
        for k in self._candidates():
            vals, lams = self._values(k, np.maximum(w, 1e-12))
            if vals.min() > best:
                best, lam = vals.min(), lams[np.argmin(vals)]
        f = np.log(self.t)
        rad = np.sqrt(2 * self.var * f / N)
        lo, hi = np.maximum(mu - rad, self.mu_range[0]), np.minimum(mu + rad, self.mu_range[1])
        kl = lambda x: (x - lam) ** 2 / (2 * self.var)
        U = np.maximum(f / N, np.maximum(kl(lo), kl(hi)))             # optimistic gains
        arm = self._track(w)
        # full-matrix AdaGrad ascent step on <w, U> over Pi: w <- Proj^{H^1/2}_Pi(w + eta H^-1/2 U)
        self.H += np.outer(U, U)
        e, P = np.linalg.eigh(self.H)
        root = np.sqrt(np.maximum(e, 1e-300))
        self.w = project_onto(w + self.eta * (P @ ((P.T @ U) / root)), self.B_pi, (P * root) @ P.T, x0=w)
        return arm


class Uniform(_KnownConstraintExplorer):
    """Arms drawn from the uniform allocation (projected onto Pi when needed)."""

    name = "Uniform"

    def next_arm(self):
        return int(self.rng.choice(self.K, p=self.uniform))
