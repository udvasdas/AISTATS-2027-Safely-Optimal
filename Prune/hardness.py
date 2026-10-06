"""Characteristic times T(mu) of the different problem formulations, for hardness maps.

All for Gaussian arms with reward variance sigma^2 and constraints B pi <= 0 (B = A - b 1^T).

* `t_bai`             unconstrained best-arm identification (Garivier & Kaufmann, 2016).
* `t_known_exact`     known constraints, exact optimal policy (Carlsson et al., 2024, Eq. 5/8);
                      `anytime=True` restricts the allocation to omega in F (their Fig. 3b).
* `t_lagrangian`      unknown constraints, Lagrangian relaxation (Das & Basu, 2026, Eq. 7 / Thm 4) at
                      the population level (A_tilde = A), with r-optimality margin (gap + r)^2 and
                      L = {l >= 0, ||l||_1 <= D / gamma}, gamma = min_i (-A_i pi*). When a constraint is
                      tight at pi* (gamma = 0) the minimisation over l forces omega in F.
* `t_known_eps`       known constraints, eps-optimal safe policy with unrestricted allocation: PRUNE's
                      Theorem 1 when the costs are known exactly (no cost alternatives).
* `t_prune`           unknown constraints, eps-optimal safe policy: PRUNE's Theorem 1 / Corollary 1 with
                      the joint (reward, cost) covariance, e.g. reward-cost correlation rho.
"""

import numpy as np
from scipy.optimize import linprog, minimize

from .lagex import PolytopeVertices, neighbours
from .lower_bound import Evaluation, characteristic_time

INF = np.inf


def _kelley(pieces, K, A_ub=None, iters=500, tol=1e-6, w_min=1e-9):
    """max_{omega in simplex, A_ub omega <= 0} min_i f_i(omega) for concave pieces (cutting planes).

    pieces(omega) -> (values (n,), gradients (n, K)). Returns (omega*, value).
    """
    omega = np.full(K, 1.0 / K)
    if A_ub is not None and np.any(A_ub @ omega > 0):
        # interior feasible start: max s s.t. A_ub omega + s <= 0, omega >= 1e-3, s <= 1
        m = len(A_ub)
        res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.c_[A_ub, np.ones(m)], b_ub=np.zeros(m),
                      A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1.0],
                      bounds=[(1e-3, None)] * K + [(None, 1.0)], method="highs")
        omega = res.x[:K]
    cuts_A, cuts_b = [], []
    best_w, best, ub = omega, -1.0, INF
    extra_A = np.zeros((0, K + 1)) if A_ub is None else np.c_[A_ub, np.zeros(len(A_ub))]
    for _ in range(iters):
        vals, grads = pieces(np.maximum(omega, w_min))
        if vals.min() > best:
            best_w, best = omega.copy(), vals.min()
        cuts_A.append(np.c_[-grads, np.ones(len(vals))])
        cuts_b.append(vals - grads @ omega)
        res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.vstack(cuts_A + [extra_A]),
                      b_ub=np.concatenate(cuts_b + [np.zeros(len(extra_A))]),
                      A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1.0],
                      bounds=[(w_min, None)] * K + [(None, None)], method="highs")
        if res.status != 0:
            break
        ub = min(ub, -res.fun)
        omega = res.x[:K]
        if ub - best <= tol * max(best, 1e-300):
            break
    return best_w, best


def _vertex_pieces(mu, pi, alts, r, sigma2):
    """Pieces (gap + r)_+^2 / (2 ||pi - v||^2_{diag(sigma^2/omega)}) over alternative vertices v.

    sigma2: reward variance, a scalar or per arm (K,)."""
    V = pi - alts
    x2 = np.maximum(V @ mu + r, 0.0) ** 2 / 2
    V2 = V * V * sigma2

    def pieces(w):
        S = (V2 / w).sum(1)
        return x2 / S, (x2 / S ** 2)[:, None] * V2 / (w * w)
    return pieces


def _optimum_and_neighbours(mu, B, tie_tol=1e-9, allow_ties=False):
    """LP optimum pi* and its neighbouring vertices. With exact identification (r = 0) a non-unique
    optimum makes T infinite (returns None); with r > 0 a tied vertex only costs r^2, so any optimum is used."""
    K, d = len(mu), B.shape[0]
    V, tight, Ineq = PolytopeVertices(K, d)(B)
    vals = V @ mu
    order = np.argsort(-vals)
    if not allow_ties and len(V) > 1 and vals[order[0]] - vals[order[1]] <= tie_tol:
        return None, None                           # optimal policy not unique: T = infinity
    k = int(order[0])
    return V[k], neighbours(k, V, tight, Ineq)


def t_bai(mu, sigma2=1.0):
    mu = np.asarray(mu, float)
    return t_known_exact(mu, np.zeros((0, len(mu))), sigma2, anytime=False)


def t_known_exact(mu, B, sigma2=1.0, anytime=True, r=0.0):
    mu = np.asarray(mu, float)
    pi, nbrs = _optimum_and_neighbours(mu, B, allow_ties=r > 0)
    if pi is None:
        return INF
    _, val = _kelley(_vertex_pieces(mu, pi, nbrs, r, sigma2), len(mu), A_ub=B if anytime and len(B) else None)
    return 1.0 / val if val > 0 else INF


def t_known_vertex_eps(mu, B, eps, sigma2=1.0, anytime=True):
    """Known constraints, eps-optimal VERTEX answers (Carlsson et al., App. D; target of the relaxed CTnS /
    CGE): min over eps-good vertices pi of 1 / max_omega min_{v != pi} (mu^T (pi - v) + eps)^2 / (2 sigma^2
    ||pi - v||^2_{W^-1}), omega in F (anytime) or in the simplex."""
    mu = np.asarray(mu, float)
    V = PolytopeVertices(len(mu), B.shape[0])(B)[0]
    vals = V @ mu
    best = 0.0
    for k in np.flatnonzero(vals > vals.max() - eps):
        _, v = _kelley(_vertex_pieces(mu, V[k], np.delete(V, k, axis=0), eps, sigma2), len(mu),
                       A_ub=B if anytime and len(B) else None)
        best = max(best, v)
    return 1.0 / best if best > 0 else INF


def t_lagrangian(mu, B, r, sigma2=1.0):
    mu = np.asarray(mu, float)
    pi, nbrs = _optimum_and_neighbours(mu, B, allow_ties=r > 0)
    if pi is None:
        return INF
    gamma = np.min(-(B @ pi))
    pieces = _vertex_pieces(mu, pi, nbrs, r, sigma2)
    w_f, val_f = _kelley(pieces, len(mu), A_ub=B)       # omega in F: the value when gamma = 0
    if gamma <= 1e-9:
        return 1.0 / val_f if val_f > 0 else INF
    # gamma > 0: min_l [D(omega) - l^T B omega] = D(omega) (1 - max_i (B omega)_+ / gamma), omega in simplex
    K = len(mu)

    def neg(w):
        w = np.maximum(w, 1e-12)
        return -pieces(w)[0].min() * max(1.0 - max((B @ w).max(), 0.0) / gamma, 0.0)
    best = val_f
    for w0 in (w_f, np.full(K, 1.0 / K)):
        res = minimize(neg, w0, bounds=[(0, 1)] * K, method="SLSQP",
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        best = max(best, -res.fun)
    return 1.0 / best if best > 0 else INF


class KnownEpsSolver:
    """F(omega, p) for eps-optimal identification with known constraints (same interface as AltSolver).

    With B known the alternatives only move mu, and the best witness q of an alternative is a vertex v of
    F = {q in simplex : B q <= 0}:  F(omega, p) = min_v (eps - mu^T (v - p))_+^2 / (2 sigma^2 ||v - p||^2_{W^-1})
    for p in F (0 when p is infeasible).
    """

    p_feasible = True        # answers may sit on the constraint boundary: keep B p <= 0 in the ascent LP

    def __init__(self, K, d, eps, sigma2=1.0):
        self.K, self.eps, self.sigma2 = K, float(eps), sigma2
        self.Sigma = np.diag(np.r_[sigma2, np.ones(d)])          # only used for starting answers
        self._enum = PolytopeVertices(K, d)
        self._cache = (None, None)

    def _vertices(self, B):
        key = B.tobytes()
        if self._cache[0] != key:
            self._cache = (key, self._enum(B)[0])
        return self._cache[1]

    def evaluate(self, mu, B, omega, p):
        K = self.K
        omega = np.maximum(omega, 1e-12)
        V = self._vertices(B)
        D = V - p
        keep = (D * D).sum(1) > 1e-18
        D, V = D[keep], V[keep]
        if len(D) == 0 or np.any(B @ p > 1e-9):
            zero = np.zeros((1, K))
            return Evaluation(F=0.0, vals=np.zeros(1), grad_w=zero, grad_p=zero, kind=np.zeros(1, int), q=V)
        x = np.maximum(self.eps - D @ mu, 0.0)
        S = (D * D / omega).sum(1)
        vals = x * x / (2 * self.sigma2 * S)
        grad_w = (x * x / (2 * self.sigma2 * S * S))[:, None] * D * D / (omega * omega)
        grad_p = (x / (self.sigma2 * S))[:, None] * mu[None, :] + (x * x / (self.sigma2 * S * S))[:, None] * D / omega
        return Evaluation(F=float(max(vals.min(), 0.0)), vals=vals, grad_w=grad_w, grad_p=grad_p,
                          kind=np.zeros(len(vals), int), q=V)


def t_known_eps(mu, B, eps, sigma2=1.0):
    """eps-optimal safe policy with known constraints and unrestricted allocation (omega in simplex)."""
    mu = np.asarray(mu, float)
    solver = KnownEpsSolver(len(mu), B.shape[0], eps, sigma2)
    return characteristic_time(mu, B, solver.Sigma, eps, solver=solver)["T"]


def t_prune(mu, B, eps, Sigma):
    return characteristic_time(np.asarray(mu, float), B, np.asarray(Sigma, float), eps)["T"]


def correlated_sigma(rho, d, sigma_r=1.0, sigma_c=1.0):
    """Reward-cost correlation rho with every cost, independent costs (PD iff |rho| < 1/sqrt(d))."""
    S = np.diag(np.r_[sigma_r ** 2, np.full(d, sigma_c ** 2)])
    S[0, 1:] = S[1:, 0] = rho * sigma_r * sigma_c
    return S
