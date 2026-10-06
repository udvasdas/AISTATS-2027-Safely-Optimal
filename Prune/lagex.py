"""LaGEx (LAgrangian Gamified EXplorer), Algorithm 3 of Das & Basu (2026), "Learning to explore with
Lagrangians for bandits under unknown linear constraints".

Setting of that paper: Gaussian rewards with known variance sigma_r^2 and d linear constraints
A pi <= 0 whose costs are observed with independent noise. The alternatives perturb only the mean
rewards; the uncertainty in A enters through an optimistic feasible set F_hat_t and a Lagrangian
penalty on the allocation. Works for any number of constraints d. On the correlated instances of this
repository, LaGEx uses the marginal variances and ignores the reward-cost correlation, as its model does.

Per round t (after one pull of each arm):
 1. Estimates mu_hat, A_hat (empirical means), Sigma_t = I + diag(N).
 2. Optimistic constraints (Lemma 1): A_tilde_i = argmin_{A' in C_t} A'_i pi_ref
        = A_hat_i - f(t, delta) sigma_i Sigma_t^-1 pi_ref / ||pi_ref||_{Sigma_t^-1},
    with C_t the ellipsoid of Eq. (2), f(t, delta) = 1 + sqrt(log(K/delta)/2 + log det(Sigma_t)/4), and
    pi_ref the LP solution under A_hat. F_hat_t = {pi in simplex : A_tilde pi <= 0}.
 3. pi_t = argmax_{pi in F_hat_t} mu_hat^T pi (a vertex) and its neighbouring vertices nu(pi_t)
    (Prop. 1).
 4. Stopping (Thm 5): min_{pi' in nu(pi_t)} (gap + r)^2 / (2 sigma_r^2 sum_a (pi_t - pi')_a^2 / N_a)
    > beta(t, delta), with gap = mu_hat^T (pi_t - pi'). Recommend pi_t.
 5. Allocation player: AdaGrad on the simplex gives omega_t.
 6. Instance player: best response lambda_t over the neighbours (closed-form Gaussian projection).
 7. Lagrange multiplier l_t = argmin_{l in L} [D(omega) - l^T A_tilde omega],
    L = {l >= 0, ||l||_1 <= D(omega) / Gamma}: all mass on the most violated row if A_tilde omega has a
    positive entry, else 0. Gamma is the Slater slack max_pi min_i (-A_tilde_i pi).
 8. Optimistic gains U_a = max(g(t)/N_a, d(alpha_a, lambda_a), d(beta_a, lambda_a)),
    [alpha_a, beta_a] = {x : N_a d(mu_hat_a, x) <= g(t)}, g(t) = log t. AdaGrad ascends
    U - A_tilde^T l_t, the gradient of <omega, U> - l_t^T A_tilde omega (line 11).
 9. C-tracking of the cumulative allocations, forced exploration when min N_a < sqrt(t) - K/2.

Where the text is ambiguous, `~/Codes/constraint-pure-exploration-main/CGE_Lag.py` was followed
(g(t) = log t, forced exploration, AdaGrad step 1/sqrt(2), bounded mean domain). See the README for the
remaining choices.
"""

import itertools

import numpy as np
from scipy.optimize import linprog

from .algorithms import PureExplorationAlgorithm


class PolytopeVertices:
    """Vertices of {pi in simplex : G pi <= 0} for G of shape (d, K), by enumerating bases."""

    def __init__(self, K, d):
        self.K = K
        self.bases = np.array(list(itertools.combinations(range(d + K), K - 1)), dtype=int).reshape(-1, K - 1)

    def __call__(self, G, tol=1e-9):
        """Returns (V (n, K), tight (n, d+K) boolean, Ineq (d+K, K)); Ineq = [G; -I]."""
        K = self.K
        Ineq = np.vstack([G, -np.eye(K)])
        m = len(self.bases)
        M = np.empty((m, K, K))
        M[:, 0, :] = 1.0
        M[:, 1:, :] = Ineq[self.bases]
        ok = np.abs(np.linalg.det(M)) > 1e-12
        rhs = np.zeros((int(ok.sum()), K, 1))
        rhs[:, 0] = 1.0
        V = np.linalg.solve(M[ok], rhs)[..., 0]
        V = np.clip(V[(V @ Ineq.T <= tol).all(1)], 0.0, None)
        if len(V) == 0:
            return V, np.zeros((0, len(Ineq)), dtype=bool), Ineq
        _, first = np.unique(np.round(V, 9), axis=0, return_index=True)
        V = V[np.sort(first)]
        return V, np.abs(V @ Ineq.T) <= tol, Ineq


def neighbours(k, V, tight, Ineq):
    """Vertices adjacent to V[k]: they share K-1 linearly independent tight constraints (an edge)."""
    K = V.shape[1]
    out = []
    for j in range(len(V)):
        if j == k:
            continue
        shared = tight[k] & tight[j]
        if np.linalg.matrix_rank(np.vstack([np.ones(K), Ineq[shared]])) >= K - 1:
            out.append(j)
    return V[out]


def gaussian_projections(omega, mu, pi, nbrs, r, sigma2):
    """Closed-form projections of mu onto {lambda : lambda^T (pi - pi') <= -r} for every alternative pi'.

    Returns (values (n,), lambdas (n, K)); value = (gap + r)_+^2 / (2 sigma^2 ||pi - pi'||^2_{diag(1/omega)})
    (zero, with lambda = mu, when pi' already beats pi by r under mu).
    """
    v = pi - nbrs                                  # (n, K)
    norm = (v * v / omega).sum(1)
    excess = np.maximum(v @ mu + r, 0.0)           # (gap + r)_+
    lam = mu - (excess / norm)[:, None] * v / omega
    return excess ** 2 / (2 * sigma2 * norm), lam


def project_simplex_weighted(y, h):
    """argmin_{x in simplex} sum_a h_a (x_a - y_a)^2: x_a = max(0, y_a - tau / h_a), exact via sorting."""
    order = np.argsort(-y * h)
    ys, hs = y[order], h[order]
    taus = (np.cumsum(ys) - 1.0) / np.cumsum(1.0 / hs)
    k = np.flatnonzero(ys * hs > taus)[-1]
    return np.maximum(y - taus[k] / h, 0.0)


class AdaGrad:
    """Diagonal AdaGrad ascent on the simplex (allocation player)."""

    def __init__(self, K, eta=1 / np.sqrt(2), eps=1e-2):
        self.eta, self.eps = eta, eps
        self.w = np.full(K, 1.0 / K)
        self.G2 = np.zeros(K)

    def update(self, gain):
        self.G2 += gain * gain
        h = np.sqrt(self.G2) + self.eps
        self.w = project_simplex_weighted(self.w + self.eta * gain / h, h)


class LaGEx(PureExplorationAlgorithm):
    """LaGEx of Das & Basu (2026) for Gaussian rewards and d unknown linear constraints A pi <= 0.

    Parameters
    ----------
    r : optimality tolerance (r-optimal policy); use r = eps to target the same set Pi(nu, eps) as PRUNE.
    variant : "corrected" (default) or "paper".
        "paper": Algorithm 3 as written. pi_t is the LP optimum on the optimistic set F_hat_t, the
        stopping rule tests reward alternatives among its neighbours with F_hat_t held fixed, and
        pi_t is recommended. When a constraint is tight at the optimum this recommends infeasible policies.
        "corrected": optimistic-pessimistic sandwich from per-arm anytime confidence boxes
        A_hat -/+ W. Recommend pi_check_t = argmax mu_hat^T pi on the pessimistic set
        F_check_t (subset of F); stop when no vertex of the optimistic set F_hat_t (superset of F) beats
        pi_check_t by more than r under any plausible reward vector (GLR > beta(t, delta/2)). On the
        confidence event pi_check_t is feasible and r-optimal, so the recommendation is (1-delta)-correct.
    recommend : "paper" variant only: "optimistic" (pi_t, the paper) or "empirical" (LP on A_hat, as
        CGE_Lag.py).
    mu_range : bounded domain D of the means (Assumption 1); bounds the instance player's lambda.
    """

    name = "LaGEx"

    def __init__(self, K, Sigma, delta, eps, r=None, threshold="stylized", variant="corrected",
                 recommend="optimistic", g=np.log, mu_range=(-1.0, 10.0), eta=1 / np.sqrt(2)):
        self.r = eps if r is None else r
        self.variant = variant
        self.recommend_mode = recommend
        self.g = g
        self.mu_range = mu_range
        self.eta = eta
        super().__init__(K, Sigma, delta, eps, threshold)
        self.sigma2 = self.Sigma[0, 0]
        self.sigma_c = np.sqrt(np.diag(self.Sigma)[1:])
        self.vertices = PolytopeVertices(K, self.d)

    def reset(self):
        super().reset()
        self.ada = AdaGrad(self.K, self.eta)
        self.W_sum = None
        self.pi = None          # candidate answer tested by the GLR (pi_t, or pi_check_t when corrected)
        self.nbrs = None        # its alternative policies
        self.A_tilde = None     # optimistic constraints used by the Lagrangian penalty
        self.certified = False  # corrected: a pessimistically feasible candidate exists

    def kl(self, x, y):
        return (x - y) ** 2 / (2 * self.sigma2)

    # ------------------------------------------------------------------ feasible sets
    def _lp(self, mu, G):
        """argmax mu^T pi over {pi in simplex : G pi <= 0} with its neighbours, by vertex enumeration."""
        V, tight, Ineq = self.vertices(G)
        if len(V) == 0:
            return None, None
        k = int(np.argmax(V @ mu))
        return V[k], neighbours(k, V, tight, Ineq)

    def _optimistic(self):
        """Paper: A_tilde = argmin_{A' in C_t} A' pi_ref with the ellipsoid of Eq. (2) (Lemma 1)."""
        mu, A_hat, N = self.mu_hat, self.B_hat, self.N
        s_inv = 1.0 / (1.0 + N)                                         # Sigma_t^-1 (diagonal)
        f = 1.0 + np.sqrt(0.5 * np.log(self.K / self.delta) + 0.25 * np.sum(np.log1p(N)))
        pi_ref, _ = self._lp(mu, A_hat)
        if pi_ref is None:
            pi_ref = np.full(self.K, 1.0 / self.K)
        direction = s_inv * pi_ref / np.sqrt(np.sum(s_inv * pi_ref ** 2))
        return A_hat - f * self.sigma_c[:, None] * direction[None, :]

    def _widths(self):
        """Confidence widths W (d, K) of the cost means, at level delta/2 overall.

        threshold="theory": anytime sub-Gaussian bound (Laplace method, union over the d K coordinates),
            |A_hat_ia - A_ia| <= sigma_i sqrt(2 (N+1) log(2 dK sqrt(N+1) / (delta/2))) / N.
        otherwise: the same threshold as the reward GLR, W_ia = sigma_i sqrt(2 beta(t, delta/2) / N_a), so
            that reward and cost evidence are calibrated alike (as in PRUNE's joint GLR).
        """
        N = self.N
        if self.threshold == "theory":
            dc = 0.5 * self.delta / (2 * self.d * self.K)
            return self.sigma_c[:, None] * (np.sqrt(2 * (N + 1) * np.log(np.sqrt(N + 1) / dc)) / N)[None, :]
        return self.sigma_c[:, None] * np.sqrt(2 * self._beta_reward() / N)[None, :]

    def _beta_reward(self):
        """Threshold of the reward GLR at level delta/2 (the other delta/2 covers the cost boxes)."""
        delta = self.delta
        self.delta = delta / 2
        try:
            return self.beta()
        finally:
            self.delta = delta

    # ------------------------------------------------------------------ interface
    def should_stop(self):
        if self.variant == "paper":
            self.A_tilde = self._optimistic()
            self.pi, self.nbrs = self._lp(self.mu_hat, self.A_tilde)
            if self.pi is None:
                return False
            if len(self.nbrs) == 0:              # F_hat is a single point: nothing to distinguish
                return True
            vals, _ = gaussian_projections(self.N, self.mu_hat, self.pi, self.nbrs, self.r, self.sigma2)
            return vals.min() > self.beta()      # sum_a N_a d(mu_hat_a, lambda_a) at the best response
        W = self._widths()
        self.W = W
        self.A_tilde = self.B_hat - W                                   # optimistic: F subset F_hat
        V, tight, self.Ineq = self.vertices(self.A_tilde)
        pi_check, _ = self._lp(self.mu_hat, self.B_hat + W)             # pessimistic: F_check subset F
        self.certified = pi_check is not None
        if len(V) == 0:
            self.pi = None
            return False
        self.pi = pi_check if self.certified else V[int(np.argmax(V @ self.mu_hat))]
        other = np.abs(V - self.pi).sum(1) > 1e-9                       # every other optimistic vertex
        self.nbrs, self.nbr_tight = V[other], tight[other]
        if not self.certified:
            return False
        if len(self.nbrs) == 0:
            return True
        vals, _ = gaussian_projections(self.N, self.mu_hat, self.pi, self.nbrs, self.r, self.sigma2)
        return vals.min() > self._beta_reward()

    def next_arm(self):
        t, K = self.t, self.K
        if self.W_sum is None:
            self.W_sum = self.N.copy()           # warm-up rounds count as tracked
        omega = self.ada.w
        if self.pi is not None and len(self.nbrs) > 0:
            w = np.maximum(omega, 1e-12)
            vals, lams = gaussian_projections(w, self.mu_hat, self.pi, self.nbrs, self.r, self.sigma2)
            k = int(np.argmin(vals))
            lam = np.clip(lams[k], *self.mu_range)
            # Lagrange multiplier on the most violated optimistic constraint of the allocation
            viol = self.A_tilde @ omega
            gain_lag = np.zeros(K)
            if viol.max() > 0:
                l_norm = vals[k] / max(self._slater_slack(), 1e-12)
                gain_lag = -l_norm * self.A_tilde[int(np.argmax(viol))]
            # optimistic gains
            g = self.g(t)
            width = np.sqrt(2 * self.sigma2 * g / self.N)
            U = np.maximum(g / self.N, np.maximum(self.kl(self.mu_hat - width, lam),
                                                  self.kl(self.mu_hat + width, lam)))
            self.ada.update(U + gain_lag)
            if self.variant == "corrected" and self.certified:
                omega = self._mix_constraint_allocation(omega)
            elif self.variant == "corrected":
                omega = self._uncertified_allocation(omega)
        self.W_sum += omega
        if self.N.min() < np.sqrt(t + 1) - K / 2:                  # forced exploration
            return int(np.argmin(self.N))
        return int(np.argmin(self.N - self.W_sum))                 # C-tracking

    def _lp_dual(self, G):
        """argmax mu_hat^T pi on {pi in simplex : G pi <= 0} and the dual multipliers y >= 0 of G."""
        res = linprog(-self.mu_hat, A_ub=G, b_ub=np.zeros(self.d), A_eq=np.ones((1, self.K)), b_eq=[1.0],
                      bounds=[(0, None)] * self.K, method="highs")
        if res.status != 0:
            return None, None, -np.inf
        return res.x, -res.ineqlin.marginals, -res.fun

    def _uncertified_allocation(self, omega_r):
        """No pessimistically feasible policy yet: shrink the widths of the optimistic LP optimum's arms."""
        pi_hat, y_hat, _ = self._lp_dual(self.A_tilde)
        if pi_hat is None:
            return omega_r
        dx = ((y_hat[:, None] * pi_hat[None, :]) * self.W).sum(0) / (2 * self.N)
        return dx / dx.sum() if dx.sum() > 0 else omega_r

    def _mix_constraint_allocation(self, omega_r):
        """Mix LaGEx's allocation with the allocation that shrinks the cost uncertainty where it binds.

        The stopping statistic Z = x^2 / (2 sigma^2 sum_a v_a^2 / N_a), with v = pi_check - v* (closest
        alternative vertex) and margin x = mu_hat^T v + r, grows through two channels: reward samples
        (dZ/dN_a = d(mu_hat_a, lambda_a) with x fixed; LaGEx's game targets these) and cost samples, which
        raise x (dZ/dN_a = x / (sigma^2 sum v^2 / N) dx/dN_a). The two allocations are mixed in proportion
        to their marginal gain in Z; while x <= 0 (Z is flat) only the cost channel is used.
        """
        vals, _ = gaussian_projections(self.N, self.mu_hat, self.pi, self.nbrs, self.r, self.sigma2)
        k = int(np.argmin(vals))
        dx = self._margin_sensitivity(k)
        if dx is None:
            return omega_r
        omega_c = dx / dx.sum()
        v = self.pi - self.nbrs[k]
        x = v @ self.mu_hat + self.r
        if x <= 0:
            return omega_c
        S = np.sum(v * v / self.N)
        gain_r = omega_r @ (x * x * v * v / (2 * self.sigma2 * S * S * self.N ** 2))
        gain_c = omega_c @ (x / (self.sigma2 * S) * dx)
        rho = gain_c / (gain_c + gain_r)
        return (1 - rho) * omega_r + rho * omega_c

    def _vertex_dual(self, k):
        """y = M^-T mu_hat for the basis M (ones row + K-1 independent tight rows) of alternative vertex k.

        For v = M^-1 e_1, a change dG in entry (i, a) of a basis row of G moves mu_hat^T v by -y_r v_a dG.
        Returns (y over the rows of Ineq, 0 off the basis), or None if no basis is found.
        """
        K = self.K
        rows, M = [], [np.ones(K)]
        for j in np.flatnonzero(self.nbr_tight[k]):
            cand = np.vstack(M + [self.Ineq[j]])
            if np.linalg.matrix_rank(cand) == len(cand):
                rows.append(j)
                M.append(self.Ineq[j])
                if len(M) == K:
                    break
        if len(M) < K:
            return None
        y = np.linalg.solve(np.array(M).T, self.mu_hat)
        y_rows = np.zeros(len(self.Ineq))
        y_rows[rows] = y[1:]
        return y_rows

    def _margin_sensitivity(self, k):
        """d x / d N_a for the margin x = mu_hat^T (pi_check - v) + r of alternative vertex v = nbrs[k].

        The cost widths move both ends: d (mu_hat^T pi_check) / d W_ia = -y_check_i pi_check_a (dual of the
        pessimistic LP on A_hat + W) and d (mu_hat^T v) / d W_ia = y_i v_a (basis dual of the optimistic
        vertex v, G = A_hat - W), while one more sample of arm a shrinks W_ia by about W_ia / (2 N_a).
        Returns None when x does not depend on W.
        """
        v = self.nbrs[k]
        sens = np.zeros((self.d, self.K))
        y_v = self._vertex_dual(k)
        if y_v is not None:
            sens += y_v[:self.d, None] * v[None, :]
        if self.certified:
            _, y_chk, _ = self._lp_dual(self.B_hat + self.W)
            if y_chk is not None:
                sens += y_chk[:, None] * self.pi[None, :]
        dx = np.maximum((sens * self.W).sum(0) / (2 * self.N), 0.0)
        return dx if dx.sum() > 0 else None

    def _slater_slack(self):
        """Gamma = max_{pi in simplex} min_i (-A_tilde_i pi): bounds the Lagrange multiplier (Thm 1)."""
        K, d = self.K, self.d
        res = linprog(np.r_[np.zeros(K), -1.0], A_ub=np.c_[self.A_tilde, np.ones(d)], b_ub=np.zeros(d),
                      A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1.0],
                      bounds=[(0, None)] * K + [(None, None)], method="highs")
        return -res.fun if res.status == 0 else 0.0

    def recommend(self):
        if self.variant == "paper" and self.recommend_mode == "empirical":
            p, _ = self._lp(self.mu_hat, self.B_hat)
        else:
            p = self.pi
        return p.copy() if p is not None else np.full(self.K, 1.0 / self.K)
