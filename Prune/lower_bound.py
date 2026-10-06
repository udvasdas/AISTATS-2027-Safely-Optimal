"""Lower-bound machinery of PRUNE for unstructured Gaussian bandits with d unknown linear constraints.

Model (Section 3.1 / Corollary 1): pulling arm a returns y = (r, c) in R^{d+1} ~ N(m_a, Sigma) with
m_a = (mu_a, B_a) and Sigma known. For an allocation omega and a candidate answer p,

    F(omega, p) = min{ f_0(omega, p), f_1(omega, p), ..., f_d(omega, p) }                      (Thm 1)

* Cost branch (Eq. 32):  f_j = (Bp)_j^2 / (2 (Sigma_c)_jj ||p||^2_{W^-1})   if (Bp)_j < 0, else 0.
* Reward branch:         f_0 = min_{q in simplex} g(q),

      g(q) = min_{mu', B'} sum_a omega_a KL(m_a || m'_a)  s.t.  mu'^T (q - p) >= eps,  B' q <= 0.

  For fixed q this is a convex QP whose dual is a (d+1)-dimensional non-negative QP

      g(q) = max_{lam >= 0} lam^T b - 1/2 lam^T M lam,   M = Sigma o G,

  with b = (eps - mu^T(q-p), Bq) and G the Gram matrix of c_a = (-(q_a - p_a), q_a 1_d) under W^-1.
  (The closed form of Corollary 1 / Eq. (26) is the special case where every multiplier is positive.)
  We solve the dual exactly by enumerating active sets, which is cheap for small d.

  The minimising q can be taken to be a vertex of {q in simplex : B'q <= 0}, whose support has at most
  d + 1 arms. Hence min_q is searched exactly over the (d+1)-sparse faces of the simplex, with a
  vectorised coarse-to-fine grid on each face. Each face minimiser is one concave piece of F in omega,
  which gives the sub-differential set of Eq. (8) (Remark 1: q* need not be unique).
"""

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

_INF = np.inf


def simplex_lattice(s, R):
    """Points of the (s-1)-simplex whose coordinates are multiples of 1/R, shape (L, s)."""
    if s == 1:
        return np.ones((1, 1))
    pts = [c for c in itertools.product(range(R + 1), repeat=s - 1) if sum(c) <= R]
    arr = np.array(pts, dtype=float).reshape(-1, s - 1)
    return np.hstack([arr, R - arr.sum(1, keepdims=True)]) / R


def local_offsets(s, R):
    """Offsets of a local grid inside the face {sum = 0}: first s-1 coords in {-1, ..., 1}, step 1/R."""
    grid = np.array(list(itertools.product(range(-R, R + 1), repeat=s - 1)), dtype=float) / R
    grid = grid.reshape(-1, s - 1)
    return np.hstack([grid, -grid.sum(1, keepdims=True)])


def solve_sop(mu, B, eps=0.0):
    """LP (SOP): max mu^T p s.t. Bp <= 0, p in simplex. Returns (p*, OPT) or (None, -inf) if infeasible."""
    K = mu.shape[0]
    res = linprog(-mu, A_ub=B, b_ub=np.zeros(B.shape[0]), A_eq=np.ones((1, K)), b_eq=[1.0],
                  bounds=[(0, None)] * K, method="highs")
    if res.status != 0:
        return None, -_INF
    return np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum(), -res.fun


def central_answer(mu, B, eps, Sigma):
    """Policy in Pi(nu, eps) maximising the smallest standardised slack of its d + 1 defining constraints.

    Used to (re-)initialise the candidate answer in the relative interior of Pi(nu, eps) (Prop. 6).
    """
    K, d = mu.shape[0], B.shape[0]
    p_lp, opt = solve_sop(mu, B)
    if p_lp is None:
        return None, None, -_INF
    # variables (p, s): max s  s.t.  (Bp)_j / sd_j + s <= 0,  -(mu^T p - opt + eps) / eps + s <= 0
    sd = np.sqrt(np.diag(Sigma)[1:])
    A = np.vstack([np.c_[B / sd[:, None], np.ones(d)], np.r_[-mu / eps, 1.0][None]])
    b = np.r_[np.zeros(d), (eps - opt) / eps]
    res = linprog(np.r_[np.zeros(K), -1.0], A_ub=A, b_ub=b, A_eq=np.r_[np.ones(K), 0.0][None],
                  b_eq=[1.0], bounds=[(0, None)] * K + [(None, 1.0)], method="highs")
    if res.status != 0 or res.x[-1] <= 0:
        return p_lp, None, opt
    p_c = np.clip(res.x[:K], 0, None)
    return p_lp, p_c / p_c.sum(), opt


def _small_solve(M, b):
    """Batched solve of M x = b for (n, s, s) systems; closed form for s <= 3 (much faster than LAPACK
    on many tiny systems). Singular systems return non-finite entries."""
    s = M.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        if s == 1:
            return b / M[:, :, 0]
        if s == 2:
            det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
            return np.column_stack([M[:, 1, 1] * b[:, 0] - M[:, 0, 1] * b[:, 1],
                                    M[:, 0, 0] * b[:, 1] - M[:, 1, 0] * b[:, 0]]) / det[:, None]
        if s == 3:
            r0, r1, r2 = M[:, 0], M[:, 1], M[:, 2]
            c0, c1, c2 = np.cross(r1, r2), np.cross(r2, r0), np.cross(r0, r1)
            det = np.einsum("ij,ij->i", r0, c0)
            return (b[:, :1] * c0 + b[:, 1:2] * c1 + b[:, 2:] * c2) / det[:, None]
    return np.linalg.solve(M, b[..., None])[..., 0]


@dataclass
class Evaluation:
    """F(omega, p) together with its concave pieces (values and gradients in omega and p)."""

    F: float
    vals: np.ndarray      # (n_pieces,)
    grad_w: np.ndarray    # (n_pieces, K)  = per-arm KL at the confusing instance (Prop. 3)
    grad_p: np.ndarray    # (n_pieces, K)
    kind: np.ndarray      # (n_pieces,) 0 = reward branch (one per face), j >= 1 = cost branch j
    q: np.ndarray         # (n_faces, K) minimising q per face of the reward branch


class AltSolver:
    """Computes F(omega, p) of Theorem 1 and its sub-gradients for Gaussian rewards-costs, known Sigma.

    Parameters
    ----------
    K : number of arms.
    Sigma : (d+1, d+1) known covariance of (reward, cost_1, ..., cost_d).
    eps : slack epsilon of the SeOP set.
    pure_q : restrict the alternative answer q to pure arms (recovers constrained BAI, used for checks).
    grid : coarse lattice resolution per face (default depends on the face dimension).
    zoom : resolution of each refinement level, levels : number of refinement levels.
    """

    def __init__(self, K, Sigma, eps, pure_q=False, grid=None, zoom=None, levels=None):
        self.K = K
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.m = self.Sigma.shape[0]
        self.d = self.m - 1
        self.eps = float(eps)
        self.sr2 = self.Sigma[0, 0]
        self.Src = self.Sigma[1:, 0]
        self.Sc = self.Sigma[1:, 1:]
        self.Sc_diag = np.diag(self.Sc).copy()

        s = 1 if pure_q else min(K, self.d + 1)
        self.s = s
        self.faces = np.array(list(itertools.combinations(range(K), s)), dtype=int)
        nf = len(self.faces)
        if grid is None:
            grid = {1: 1, 2: 32, 3: 16, 4: 8}.get(s, 5)
        if zoom is None:                      # refinement: final resolution 1 / (grid * zoom^levels)
            zoom, levels = (8, 2) if s <= 2 else (4, 3)
        base = simplex_lattice(s, grid)
        self.h0 = 1.0 / grid
        self.levels = 0 if s == 1 else levels
        self.offsets = None if s == 1 else local_offsets(s, zoom)
        self.zoom = zoom
        L = base.shape[0]
        Q = np.zeros((nf, L, K))
        Q[np.arange(nf)[:, None, None], np.arange(L)[None, :, None], self.faces[:, None, :]] = base[None]
        self.base_alpha = base
        self.Q0 = Q.reshape(nf * L, K)
        self.subsets = [np.array(S) for r in range(1, self.m + 1)
                        for S in itertools.combinations(range(self.m), r)]

    # ------------------------------------------------------------------ reward branch
    def _dual_qp(self, mu, B, omega, p, Q):
        """Solve g(q) for every row q of Q. Returns (values (n,), multipliers (n, m))."""
        v = Q - p
        W = 1.0 / omega
        b0 = self.eps - v @ mu
        bc = Q @ B.T
        brr = (v * v) @ W
        brc = (Q * v) @ W
        bcc = (Q * Q) @ W
        n = Q.shape[0]
        if self.m == 2:
            return self._dual_qp_2(b0, bc[:, 0], brr, brc, bcc, n)
        b = np.column_stack([b0, bc])
        M = np.empty((n, self.m, self.m))
        M[:, 0, 0] = self.sr2 * brr
        M[:, 0, 1:] = -brc[:, None] * self.Src[None]
        M[:, 1:, 0] = M[:, 0, 1:]
        M[:, 1:, 1:] = bcc[:, None, None] * self.Sc[None]
        M[:, 0, 0] += 1e-14 * (1.0 + M[:, 0, 0])      # q = p makes the dual unbounded (value +inf)
        best = np.zeros(n)
        lam = np.zeros((n, self.m))
        for S in self.subsets:
            Ms = M[:, S][:, :, S]
            bs = b[:, S]
            ls = _small_solve(Ms, bs)
            val = 0.5 * np.einsum("ij,ij->i", bs, ls)
            ok = np.isfinite(val) & (ls >= 0).all(1) & (val > best)
            best = np.where(ok, val, best)
            if ok.any():
                lam[ok] = 0.0
                lam[np.ix_(ok, S)] = ls[ok]
        return best, lam

    def _dual_qp_2(self, b0, b1, brr, brc, bcc, n):
        """Closed-form active-set solution of the 2-d dual (single constraint, d = 1)."""
        M00 = self.sr2 * brr
        M01 = -brc * self.Src[0]
        M11 = bcc * self.Sc[0, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            l0 = np.where(b0 > 0, b0 / M00, 0.0)
            v0 = np.where(b0 > 0, 0.5 * b0 * l0, 0.0)
            l1 = np.where(b1 > 0, b1 / M11, 0.0)
            v1 = np.where(b1 > 0, 0.5 * b1 * l1, 0.0)
            det = M00 * M11 - M01 * M01
            j0 = (M11 * b0 - M01 * b1) / det
            j1 = (M00 * b1 - M01 * b0) / det
            vj = 0.5 * (b0 * j0 + b1 * j1)
        okj = (j0 >= 0) & (j1 >= 0) & (det > 0)
        vj = np.where(okj, vj, 0.0)
        val = np.maximum(np.maximum(v0, v1), vj)
        lam = np.zeros((n, 2))
        use0 = (v0 >= v1) & (v0 >= vj) & (v0 > 0)
        use1 = ~use0 & (v1 >= vj) & (v1 > 0)
        usej = ~use0 & ~use1 & (vj > 0)
        lam[use0, 0] = l0[use0]
        lam[use1, 1] = l1[use1]
        lam[usej, 0] = j0[usej]
        lam[usej, 1] = j1[usej]
        # q = p: the reward constraint cannot be met, no alternative in this branch
        degenerate = (brr <= 1e-300) & (b0 > 0)
        val = np.where(degenerate | np.isnan(val), _INF, val)
        lam[~np.isfinite(val)] = 0.0
        return val, lam

    def _face_search(self, mu, B, omega, p):
        """min_q g(q) on every (d+1)-sparse face. Returns per-face (value, q, lam)."""
        nf, K = len(self.faces), self.K
        vals, lam = self._dual_qp(mu, B, omega, p, self.Q0)
        L = self.base_alpha.shape[0]
        vals = vals.reshape(nf, L)
        idx = vals.argmin(1)
        best = vals[np.arange(nf), idx]
        alpha = self.base_alpha[idx]                       # (nf, s)
        lam_best = lam.reshape(nf, L, -1)[np.arange(nf), idx]
        h = self.h0
        fi = np.arange(nf)[:, None, None]
        for _ in range(self.levels):
            A = alpha[:, None, :] + h * self.offsets[None]   # (nf, L2, s)
            bad = ((A < -1e-12) | (A > 1 + 1e-12)).any(2)
            A = np.where(bad[..., None], alpha[:, None, :], np.clip(A, 0.0, 1.0))
            L2 = A.shape[1]
            Q = np.zeros((nf, L2, K))
            Q[fi, np.arange(L2)[None, :, None], self.faces[:, None, :]] = A
            v2, l2 = self._dual_qp(mu, B, omega, p, Q.reshape(nf * L2, K))
            v2 = v2.reshape(nf, L2)
            j = v2.argmin(1)
            better = v2[np.arange(nf), j] < best
            best = np.where(better, v2[np.arange(nf), j], best)
            alpha = np.where(better[:, None], A[np.arange(nf), j], alpha)
            lam_best = np.where(better[:, None], l2.reshape(nf, L2, -1)[np.arange(nf), j], lam_best)
            h /= self.zoom
        q = np.zeros((nf, K))
        q[np.arange(nf)[:, None], self.faces] = alpha
        return best, q, lam_best

    def _reward_grads(self, mu, omega, p, q, lam):
        """Per-arm KL at the confusing instance (grad in omega) and grad in p, for rows of (q, lam)."""
        v = q - p
        l0 = lam[:, :1]
        lc = lam[:, 1:]
        a_rc = (lc @ self.Src)[:, None]
        a_cc = np.einsum("ij,jk,ik->i", lc, self.Sc, lc)[:, None]
        w2 = omega * omega
        kl = (l0 * l0 * self.sr2 * v * v - 2.0 * l0 * a_rc * v * q + a_cc * q * q) / (2.0 * w2)
        mu_alt = mu + (l0 * self.sr2 * v - q * a_rc) / omega
        return kl, l0 * mu_alt

    # ------------------------------------------------------------------ public API
    def evaluate(self, mu, B, omega, p):
        """F(omega, p) and all its pieces, for the instance (mu, B)."""
        omega = np.maximum(omega, 1e-12)
        # cost branch, closed form (Eq. 32)
        slack = -(B @ p)                                  # (d,)
        P = np.sum(p * p / omega)
        pos = slack > 0
        cval = np.where(pos, slack ** 2 / (2 * self.Sc_diag * P), 0.0)
        cgw = np.where(pos[:, None], (slack ** 2 / (2 * self.Sc_diag * P * P))[:, None]
                       * (p * p / (omega * omega))[None], 0.0)
        cgp = np.where(pos[:, None], -(slack / (self.Sc_diag * P))[:, None] * B
                       - (slack ** 2 / (self.Sc_diag * P * P))[:, None] * (p / omega)[None], 0.0)
        # reward branch, one piece per face
        rval, q, lam = self._face_search(mu, B, omega, p)
        finite = np.isfinite(rval)
        rgw, rgp = self._reward_grads(mu, omega, p, q, lam)
        rgw[~finite] = 0.0
        rgp[~finite] = 0.0
        vals = np.r_[rval, cval]
        return Evaluation(
            F=float(max(vals.min(), 0.0)),
            vals=vals,
            grad_w=np.vstack([rgw, cgw]),
            grad_p=np.vstack([rgp, cgp]),
            kind=np.r_[np.zeros(len(rval), dtype=int), np.arange(1, self.d + 1)],
            q=q,
        )


# ---------------------------------------------------------------------- answer (p) optimisation
def ascend_answer(solver, mu, B, omega, p, ev, rho, iters=1, pieces_slack=np.inf):
    """Trust-region sequential-LP ascent of the quasi-concave map p -> F(omega, p) on the simplex.

    Each step maximises the piecewise-linear model min_i {vals_i + grad_p_i . dp} over
    {p + dp in simplex, |dp|_inf <= rho} and accepts it if F increases. Returns (p, ev, rho).
    """
    K = len(p)
    for _ in range(iters):
        keep = np.isfinite(ev.vals) & (ev.vals <= ev.F + pieces_slack)
        G, v = ev.grad_p[keep], ev.vals[keep]
        # variables (dp, z): max z  s.t.  z - G_i dp <= v_i
        res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.c_[-G, np.ones(len(v))], b_ub=v,
                      A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[0.0],
                      bounds=[(max(-pa, -rho), rho) for pa in p] + [(None, None)], method="highs")
        if res.status != 0 or -res.fun <= ev.F * (1 + 1e-9):
            rho *= 0.5
            if rho < 1e-6:
                break
            continue
        p_new = np.clip(p + res.x[:K], 0.0, None)
        p_new /= p_new.sum()
        ev_new = solver.evaluate(mu, B, omega, p_new)
        if ev_new.F > ev.F:
            p, ev, rho = p_new, ev_new, min(2.0 * rho, 0.5)
        else:
            rho *= 0.5
            if rho < 1e-6:
                break
    return p, ev, rho


def init_answer(solver, mu, B, omega, n_grid=9):
    """Pick an interior starting answer on the segment from the LP optimum to the central answer."""
    p_lp, p_c, opt = central_answer(mu, B, solver.eps, solver.Sigma)
    if p_lp is None:
        return None, None
    if p_c is None:
        return p_lp, solver.evaluate(mu, B, omega, p_lp)
    best_p, best_ev = None, None
    for th in np.linspace(0.0, 1.0, n_grid)[1:]:
        pt = (1 - th) * p_lp + th * p_c
        ev = solver.evaluate(mu, B, omega, pt)
        if best_ev is None or ev.F > best_ev.F:
            best_p, best_ev = pt, ev
    return best_p, best_ev


# ---------------------------------------------------------------------- oracle (lower bound)
def max_over_omega(solver, mu, B, p, omega0=None, iters=300, tol=1e-6, w_min=1e-6):
    """max_omega F(omega, p) by Kelley's cutting planes (every piece is concave in omega).

    Returns (omega*, F*, upper bound).
    """
    K = solver.K
    omega = np.full(K, 1.0 / K) if omega0 is None else omega0.copy()
    cuts_A, cuts_b = [], []
    best_w, best_F, ub = omega, -1.0, np.inf
    for _ in range(iters):
        ev = solver.evaluate(mu, B, omega, p)
        if ev.F > best_F:
            best_w, best_F = omega.copy(), ev.F
        ok = np.isfinite(ev.vals)
        # z <= val_i + g_i . (w - omega)   <=>   z - g_i . w <= val_i - g_i . omega
        G = ev.grad_w[ok]
        cuts_A.append(np.c_[-G, np.ones(len(G))])
        cuts_b.append(ev.vals[ok] - G @ omega)
        res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.vstack(cuts_A), b_ub=np.concatenate(cuts_b),
                      A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1.0],
                      bounds=[(w_min, None)] * K + [(None, None)], method="highs")
        ub = min(ub, -res.fun)
        omega = res.x[:K]
        if ub - best_F <= tol * max(best_F, 1e-12):
            break
    return best_w, best_F, ub


def joint_ascent(solver, mu, B, omega, p, iters=400, rho=0.05, w_min=1e-5):
    """Trust-region sequential-LP ascent of (omega, p) -> F(omega, p), moving both jointly.

    Alternating maximisation (omega, then p) stalls at the kinks of F; the joint piecewise-linear
    model does not. Returns (omega, p, Evaluation).
    """
    K = len(p)
    ev = solver.evaluate(mu, B, omega, p)
    A_eq = np.zeros((2, 2 * K + 1))
    A_eq[0, :K] = 1.0
    A_eq[1, K:2 * K] = 1.0
    # solvers whose answers must stay feasible (known constraints) keep B (p + dp) <= 0 in the LP
    feasible_p = getattr(solver, "p_feasible", False)
    for _ in range(iters):
        ok = np.isfinite(ev.vals)
        v = ev.vals[ok]
        A = np.c_[-ev.grad_w[ok], -ev.grad_p[ok], np.ones(len(v))]
        b = v
        if feasible_p:
            A = np.vstack([A, np.c_[np.zeros((len(B), K)), B, np.zeros(len(B))]])
            b = np.r_[v, -(B @ p)]
        bounds = ([(max(w_min - w, -rho), rho) for w in omega] + [(max(-pa, -rho), rho) for pa in p]
                  + [(None, None)])
        res = linprog(np.r_[np.zeros(2 * K), -1.0], A_ub=A, b_ub=b, A_eq=A_eq, b_eq=[0.0, 0.0],
                      bounds=bounds, method="highs")
        if res.status != 0 or -res.fun <= ev.F * (1 + 1e-10):
            rho *= 0.5
            if rho < 1e-7:
                break
            continue
        w_new = np.maximum(omega + res.x[:K], w_min)
        p_new = np.clip(p + res.x[K:2 * K], 0.0, None)
        w_new, p_new = w_new / w_new.sum(), p_new / p_new.sum()
        ev_new = solver.evaluate(mu, B, w_new, p_new)
        if ev_new.F > ev.F:
            ratio = (ev_new.F - ev.F) / (-res.fun - ev.F)
            omega, p, ev = w_new, p_new, ev_new
            if ratio > 0.5:
                rho = min(2.0 * rho, 0.5)
        else:
            rho *= 0.5
            if rho < 1e-7:
                break
    return omega, p, ev


def answer_starts(solver, mu, B, n_random=4, rng=None):
    """Starting answers in the interior of Pi(nu, eps): along the LP-optimum -> central-answer segment,
    plus random mixtures of the central answer with points of the simplex."""
    rng = np.random.default_rng(0) if rng is None else rng
    p_lp, p_c, opt = central_answer(mu, B, solver.eps, solver.Sigma)
    if p_lp is None or p_c is None:
        return [] if p_lp is None else [p_lp]
    starts = [(1 - th) * p_lp + th * p_c for th in (0.25, 0.5, 1.0)]
    a, attempts = 0.5, 0
    while len(starts) < 3 + n_random and attempts < 200:
        # random perturbations of the central answer that stay inside Pi(nu, eps); shrink on failure
        x = (1 - a) * p_c + a * rng.dirichlet(np.ones(len(mu)))
        attempts += 1
        if np.all(B @ x < 0) and mu @ x > opt - solver.eps:
            starts.append(x)
        else:
            a *= 0.8
    return starts


def characteristic_time(mu, B, Sigma, eps, p=None, pure_q=False, n_random=4, solver=None):
    """Oracle T*(nu, eps) of Theorem 1: max_p max_omega F(omega, p).

    Multi-start joint ascent over (omega, p), then Kelley's cutting planes over omega at the best
    answer (which also certifies max_omega F(., p) through an upper bound).
    If p is given, only max_omega is solved (e.g. p = e_{i*} with pure_q=True reproduces constrained
    BAI of Lardy et al. 2025 when eps -> 0). Returns dict(T, omega, p, F, starts).
    """
    solver = solver or AltSolver(len(mu), Sigma, eps, pure_q=pure_q)
    K = len(mu)
    if p is not None:
        w, F, _ = max_over_omega(solver, mu, B, p)
        return dict(T=1.0 / F, omega=w, p=p, F=F)
    best, found = None, []
    for p0 in answer_starts(solver, mu, B, n_random):
        w, p1, ev = joint_ascent(solver, mu, B, np.full(K, 1.0 / K), p0)
        found.append(1.0 / ev.F if ev.F > 0 else np.inf)
        if best is None or ev.F > best[2].F:
            best = (w, p1, ev)
    w, p, _ = best
    w, F, _ = max_over_omega(solver, mu, B, p, omega0=w)
    return dict(T=1.0 / F, omega=w, p=p, F=F, starts=found)
