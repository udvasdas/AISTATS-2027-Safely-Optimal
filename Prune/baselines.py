"""Baselines of Lardy, Katsimerou & Koolen (2025), "Constrained best arm identification".

Both identify the best *feasible arm* i*(nu) = argmax_{a : B_a <= 0} mu_a (a single constraint,
d = 1, in the B = cost - gamma coordinates of `instances.py`), or None, not a mixed policy. They
share the GLR stopping rule of their Eq. (13) and recommend e_{i_hat}.

* TaS (Section 3, their algorithm): plug-in oracle weights w_n = w*(nu_hat(n)) computed as in their
  Theorem 2.2, C-tracking I_n = argmin_i N_i - sum_s w_{s,i}, forced exploration N_i(n) >= sqrt(n).
* TopTwo-TCI (Section 4, their best performing sampling rule): leader i_hat = i*(nu_hat); challenger
  argmin_j Lambda_{n,j} + log N_j over the GLR terms (N_None := N_{i_hat}); sample the leader with
  probability 1/2 (the usual Top Two choice), else the challenger; if either is None, the other.

Two arm models: "known" (Gaussian with known Sigma, Prop. 2.2, the model of PRUNE) and "unknown"
(Gaussian with unknown per-arm covariance, Prop. 2.3, the model of their Table 1).

eps-relaxed versions (`TrackAndStop_relaxed`, `TopTwoTCI_relaxed`; not in Lardy et al.): the correct
answers are the eps-good feasible arms {i : B_i <= 0, mu_i >= mu_{i*} - eps} (None iff no arm is feasible).
An alternative to answer i makes i infeasible or makes some j feasible with lambda_j > lambda_i + eps.
Shifting the reward of j by -eps turns the latter into the event of the exact problem, and both arm
models are translation invariant, so its cost is c1(nu_i, nu_j - eps e_1, w). Hence
  * stopping: max over candidate answers i (empirically feasible and eps-good) of min_j Lambda^eps_{n,i,j}
    > beta, recommend the maximiser (the GLR rule for multiple correct answers);
  * TaS_relaxed: Theorem 2.2 oracle with leader i and eps-shifted challengers, for every candidate i; track
    the weights of the candidate with the smallest T*_eps (the best answer, not sticky);
  * TopTwoTCI_relaxed: EB-TC_eps of Jourdan et al. (2023) in the CBAI model: leader = empirical best feasible
    arm, challenger = argmin_j Lambda^eps_{n,i_hat,j} + log N_j.
"""

import numpy as np

from .algorithms import PureExplorationAlgorithm
from .lower_bound import AltSolver

NONE = -1                     # the answer "no arm is feasible"


def _ell_known(x):
    return 0.5 * x


def _ell_unknown(x):
    return 0.5 * np.log1p(x)


# ---------------------------------------------------------------------- transportation costs
def _proj_feasible_better(mu, S, theta):
    """Squared Mahalanobis distance ||mu - lam||^2_{S^-1} from mu to {lam_1 >= theta, lam_2 <= 0}.

    mu (..., 2), S (..., 2, 2), theta (..., n) -> (..., n). Exact active-set solution of the 2-d QP.
    """
    S11, S12, S22 = S[..., 0, 0][..., None], S[..., 0, 1][..., None], S[..., 1, 1][..., None]
    b0 = theta - mu[..., 0][..., None]                     # reward constraint violated if > 0
    b1 = np.broadcast_to(mu[..., 1][..., None], b0.shape)  # feasibility constraint violated if > 0
    v0 = np.where(b0 > 0, b0 * b0 / S11, 0.0)
    v1 = np.where(b1 > 0, b1 * b1 / S22, 0.0)
    det = S11 * S22 - S12 * S12
    l0 = (S22 * b0 + S12 * b1) / det                       # multipliers with both constraints active
    l1 = (S11 * b1 + S12 * b0) / det
    vb = np.where((l0 >= 0) & (l1 >= 0), b0 * l0 + b1 * l1, 0.0)
    return np.maximum(np.maximum(v0, v1), vb)


def c1_theta(mu_i, S_i, N_i, mu_j, S_j, N_j, ell, grid=64, zoom=8, levels=2, return_parts=False):
    """N_i c1(nu_i, nu_j, N_j / N_i) of Prop. 2.2 / 2.3 for one leader i and J rows (j, N_j).

    min over theta of N_i ell(a(theta)) + N_j ell(b(theta)), with a(theta) the cost of pushing the
    reward of i below theta and b(theta) the cost of making j feasible with reward above theta.
    The minimiser lies in [min(lam*_j1, mu_i1), mu_i1], lam*_j the feasibility projection of j.
    With return_parts, also returns the unweighted costs ell(a), ell(b) at the minimiser.
    """
    shift = np.where(mu_j[:, 1] > 0, S_j[:, 0, 1] / S_j[:, 1, 1] * mu_j[:, 1], 0.0)
    hi = np.full(len(mu_j), mu_i[0])
    lo = np.minimum(mu_j[:, 0] - shift, hi)
    rows = np.arange(len(lo))

    def parts(theta):
        a = np.maximum(mu_i[0] - theta, 0.0) ** 2 / S_i[0, 0]
        return ell(a), ell(_proj_feasible_better(mu_j, S_j, theta))

    def objective(theta):
        pa, pb = parts(theta)
        return N_i * pa + N_j[:, None] * pb

    u = np.linspace(0.0, 1.0, grid + 1)[None, :]
    theta = lo[:, None] + (hi - lo)[:, None] * u
    val = objective(theta)
    k = val.argmin(1)
    best, th = val[rows, k], theta[rows, k]
    h = (hi - lo) / grid
    for _ in range(levels):
        cand = np.clip(th[:, None] + h[:, None] * np.linspace(-1, 1, 2 * zoom + 1)[None], lo[:, None], hi[:, None])
        v = objective(cand)
        k = v.argmin(1)
        better = v[rows, k] < best
        best = np.where(better, v[rows, k], best)
        th = np.where(better, cand[rows, k], th)
        h = h / zoom
    if not return_parts:
        return best
    pa, pb = parts(th[:, None])
    return best, pa[:, 0], pb[:, 0]


class _PairCost:
    """c1(nu_i, nu_j, w) = c1_i + w c1_j (Interface 2.1 of Lardy et al.) for many (j, w) rows."""

    def __init__(self, Sigma, covariance):
        self.covariance = covariance
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.qp = AltSolver(2, Sigma, 0.0) if covariance == "known" else None

    def __call__(self, mu_i, S_i, mu_j, S_j, w):
        """Rows: mu_j (n, 2), S_j (n, 2, 2), w (n,). Returns (c1, c1_i, c1_j), each (n,)."""
        if self.covariance == "unknown":
            return c1_theta(mu_i, S_i, 1.0, mu_j, S_j, w, _ell_unknown, return_parts=True)
        # known Sigma: exact 2-constraint dual QP (reward order + feasibility of j), eps = 0
        s = self.Sigma
        inv_w = 1.0 / w
        val, lam = self.qp._dual_qp_2(mu_i[0] - mu_j[:, 0], mu_j[:, 1], 1.0 + inv_w, inv_w, inv_w, len(w))
        l0, l1 = lam[:, 0], lam[:, 1]
        ci = 0.5 * l0 * l0 * s[0, 0]
        cj = 0.5 * (l0 * l0 * s[0, 0] - 2 * l0 * l1 * s[0, 1] + l1 * l1 * s[1, 1]) * inv_w * inv_w
        return val, ci, cj


def cbai_oracle(means, covs, pair_cost, ell, n_grid=256, w_range=(1e-6, 1e6), leader=None, eps=0.0):
    """Oracle weights w*(nu) and T*(nu) of constrained BAI via Theorem 2.2 of Lardy et al.

    Maximises h(C) = min(C, c2(nu_i*)) / (1 + sum_j w_j(C)) over C, where w_j(C) inverts the increasing
    map w -> c1(nu_i*, nu_j, w). Each map is tabulated on a log grid of w in one vectorised call
    (the inner binary search becomes interpolation), h is maximised on a grid that is zoomed once,
    and the w_j at the maximiser are refined on a local grid. Returns (i*, w*, T*).

    leader, eps: weights that certify the feasible arm `leader` (default: the best feasible arm) as an
    eps-good answer; the challengers' rewards are shifted by -eps.
    """
    K = len(means)
    c2 = ell(means[:, 1] ** 2 / covs[:, 1, 1])
    feasible = means[:, 1] <= 0
    if not feasible.any():
        inv = 1.0 / c2
        return NONE, inv / inv.sum(), inv.sum()
    i = int(np.flatnonzero(feasible)[np.argmax(means[feasible, 0])]) if leader is None else int(leader)
    J = np.array([j for j in range(K) if j != i])
    if eps:
        means = means.copy()
        means[J, 0] -= eps
    lw = np.linspace(np.log(w_range[0]), np.log(w_range[1]), n_grid)
    W = np.tile(np.exp(lw), len(J))
    rows = np.repeat(J, n_grid)
    curve = pair_cost(means[i], covs[i], means[rows], covs[rows], W)[0].reshape(len(J), n_grid)
    curve = np.maximum.accumulate(curve, axis=1)         # c1 is non-decreasing in w

    def w_of(C):
        """w_j(C) by interpolation in log w; 0 below the grid, inf above it. C (m,) -> (J, m)."""
        out = np.empty((len(J), len(C)))
        for r in range(len(J)):
            out[r] = np.exp(np.interp(C, curve[r], lw, left=-np.inf, right=np.inf))
        return out

    def h(C):
        return np.minimum(C, c2[i]) / (1.0 + w_of(C).sum(0))

    C_max = min(c2[i], curve[:, -1].min())
    C = np.linspace(0.0, C_max, 513)[1:]
    k = int(np.argmax(h(C)))
    C = np.linspace(C[max(k - 1, 0)], C[min(k + 1, len(C) - 1)], 257)
    C_star = C[int(np.argmax(h(C)))]
    # refine w_j(C*) inside its grid bracket
    wt = w_of(np.array([C_star]))[:, 0]
    finite = np.isfinite(wt) & (wt > 0)
    if finite.any():
        Jf = J[finite]
        k = np.clip(np.searchsorted(lw, np.log(wt[finite])), 1, n_grid - 1)
        fine = np.linspace(lw[k - 1], lw[k], 65, axis=1)                   # (Jf, 65)
        rows = np.repeat(Jf, 65)
        cf = pair_cost(means[i], covs[i], means[rows], covs[rows], np.exp(fine.ravel()))[0].reshape(len(Jf), 65)
        cf = np.maximum.accumulate(cf, axis=1)
        wt[finite] = [np.exp(np.interp(C_star, cf[r], fine[r])) for r in range(len(Jf))]
    w = np.zeros(K)
    w[i] = 1.0
    w[J] = wt
    total = w.sum()
    return i, w / total, total / min(C_star, c2[i])


# ---------------------------------------------------------------------- algorithms
class _ConstrainedBAI(PureExplorationAlgorithm):
    """Estimates, GLR stopping (Eq. 13) and recommendation shared by TaS and TopTwo-TCI."""

    def __init__(self, K, Sigma, delta, eps=0.0, covariance="known", n_init=1, threshold="stylized"):
        if np.asarray(Sigma).shape[0] != 2:
            raise ValueError("constrained BAI baselines handle a single constraint (d = 1)")
        self.covariance = covariance
        self.n_init = n_init
        self.ell = _ell_known if covariance == "known" else _ell_unknown
        self.pair_cost = _PairCost(Sigma, covariance)
        super().__init__(K, Sigma, delta, eps, threshold)

    def reset(self):
        super().reset()
        self.S2 = np.zeros((self.K, 2, 2))
        self.leader, self.Lam = NONE, None

    def initial_arms(self):
        return [a for a in range(self.K) for _ in range(self.n_init)]

    def observe(self, a, y):
        super().observe(a, y)
        self.S2[a] += np.outer(y, y)

    def beta(self):
        if self.threshold == "lardy":
            return np.log(1.0 / self.delta) + np.log(np.log(self.t))
        return super().beta()

    def _covs(self):
        if self.covariance == "known":
            return np.broadcast_to(self.Sigma, (self.K, 2, 2))
        # unbiased empirical covariance: with it TopTwo-TCI reproduces Table 1 of Lardy et al.
        m, N = self.means, self.N[:, None, None]
        C = (self.S2 - N * m[:, :, None] * m[:, None, :]) / np.maximum(N - 1, 1)
        return C + 1e-12 * np.eye(2)

    def _glr_terms(self, i=None, eps=0.0):
        """Leader and Lambda_{n,j} for every alternative answer j (arms, then None at index K).

        Default leader: i_hat, the empirical best feasible arm. With eps > 0, the terms certify the
        feasible arm i as an eps-good answer (challenger rewards shifted by -eps).
        """
        m, N, C = self.means, self.N, self._covs()
        feasible = m[:, 1] <= 0
        Lam = np.full(self.K + 1, np.inf)
        if not feasible.any():
            # i_hat = None: every alternative makes one arm feasible
            Lam[:self.K] = N * self.ell(m[:, 1] ** 2 / C[:, 1, 1])
            return NONE, Lam
        if i is None:
            i = int(np.flatnonzero(feasible)[np.argmax(m[feasible, 0])])
        others = np.array([j for j in range(self.K) if j != i])
        m_o = m[others].copy()
        m_o[:, 0] -= eps
        Lam[self.K] = N[i] * self.ell(m[i, 1] ** 2 / C[i, 1, 1])        # make i infeasible
        # N_i c1(nu_i, nu_j, N_j / N_i)
        Lam[others] = N[i] * self.pair_cost(m[i], C[i], m_o, C[others], N[others] / N[i])[0]
        return i, Lam

    def should_stop(self):
        self.leader, self.Lam = self._glr_terms()
        return self.Lam.min() > self.beta()

    def recommended_arm(self):
        return None if self.leader == NONE else self.leader

    def recommend(self):
        p = np.zeros(self.K)
        if self.leader != NONE:
            p[self.leader] = 1.0
        return p


class TrackAndStop(_ConstrainedBAI):
    """TaS of Lardy et al. (2025): C-tracking of the plug-in oracle weights, forced exploration."""

    name = "TaS"

    def reset(self):
        super().reset()
        self.W_sum = np.zeros(self.K)

    def next_arm(self):
        if self.t == self.K * self.n_init:
            self.W_sum = self.N.copy()                 # the warm-up rounds count as tracked
        _, w, _ = cbai_oracle(self.means, self._covs(), self.pair_cost, self.ell)
        self.W_sum += w
        # forced exploration N_i(n) >= sqrt(n), in the form of Garivier & Kaufmann (2016); with it TaS
        # reproduces Table 1 of Lardy et al. (with the plain sqrt(n) rule Easy takes 81.3 vs 76.3)
        if self.N.min() < np.sqrt(self.t + 1) - self.K / 2:
            return int(np.argmin(self.N))
        return int(np.argmin(self.N - self.W_sum))


class TopTwoTCI(_ConstrainedBAI):
    """TopTwo-TCI of Lardy et al. (2025), Section 4."""

    name = "TopTwo-TCI"

    def __init__(self, K, Sigma, delta, eps=0.0, covariance="known", beta_tt=0.5, n_init=1,
                 threshold="stylized", rng=None):
        self.beta_tt = beta_tt
        self.rng = rng if rng is not None else np.random.default_rng()
        super().__init__(K, Sigma, delta, eps, covariance, n_init, threshold)

    def next_arm(self):
        i, Lam = self.leader, self.Lam
        N_ans = np.r_[self.N, self.N[i] if i != NONE else np.inf]
        score = Lam + np.log(N_ans)
        if i != NONE:
            score[i] = np.inf
        j = int(np.argmin(score))
        if i == NONE:
            return j
        if j == self.K or self.rng.random() < self.beta_tt:
            return i
        return j


# ---------------------------------------------------------------------- eps-relaxed versions
class _Relaxed:
    """eps-good feasible arm identification: GLR stopping over all candidate answers (see module doc)."""

    def reset(self):
        super().reset()
        self.answer = NONE

    def _candidates(self):
        """Empirically feasible arms within eps of the empirical best feasible arm."""
        m = self.means
        feasible = np.flatnonzero(m[:, 1] <= 0)
        if len(feasible) == 0:
            return []
        return [int(i) for i in feasible if m[i, 0] > m[feasible, 0].max() - self.eps]

    def should_stop(self):
        self.leader, self.Lam = self._glr_terms(eps=self.eps)          # i_hat and its eps-terms
        self.answer, best = self.leader, self.Lam.min()
        if self.leader == NONE:
            return best > self.beta()
        for i in self._candidates():
            if i != self.leader:
                z = self._glr_terms(i, self.eps)[1].min()
                if z > best:
                    self.answer, best = i, z
        return best > self.beta()

    def recommended_arm(self):
        return None if self.answer == NONE else self.answer

    def recommend(self):
        p = np.zeros(self.K)
        if self.answer != NONE:
            p[self.answer] = 1.0
        return p


class TrackAndStop_relaxed(_Relaxed, TrackAndStop):
    """TaS for eps-good feasible arms: tracks the oracle weights of the easiest candidate answer."""

    name = "TaS_relaxed"

    def next_arm(self):
        if self.t == self.K * self.n_init:
            self.W_sum = self.N.copy()
        means, covs = self.means, self._covs()
        w, T = None, np.inf
        for i in self._candidates() or [None]:
            _, wi, Ti = cbai_oracle(means, covs, self.pair_cost, self.ell, leader=i, eps=self.eps if i is not None else 0.0)
            if np.all(np.isfinite(wi)) and (Ti < T or w is None):
                w, T = wi, Ti
        if w is None:                                   # no candidate can be certified yet (ties)
            w = np.full(self.K, 1.0 / self.K)
        self.W_sum += w
        if self.N.min() < np.sqrt(self.t + 1) - self.K / 2:
            return int(np.argmin(self.N))
        return int(np.argmin(self.N - self.W_sum))


class TopTwoTCI_relaxed(_Relaxed, TopTwoTCI):
    """TopTwo-TCI with eps-shifted challengers (EB-TC_eps of Jourdan et al., 2023, in the CBAI model).

    `should_stop` leaves self.leader = i_hat and self.Lam = its eps-terms, which next_arm uses unchanged.
    """

    name = "TopTwo-TCI_relaxed"


class UniformCBAI(_ConstrainedBAI):
    """Uniform sampling with Lardy et al.'s exact GLR stopping rule (Eq. 13) and recommendation: identifies
    the best feasible arm without eps-relaxation; only the sampling rule differs from TaS / TopTwo-TCI."""

    name = "Uniform"

    def reset(self):
        super().reset()
        self.rng = np.random.default_rng()

    def next_arm(self):
        return int(self.rng.integers(self.K))


class UniformCBAI_relaxed(_Relaxed, UniformCBAI):
    """Uniform sampling with the eps-relaxed GLR stopping rule of TaS_relaxed / TopTwoTCI_relaxed
    (eps-good feasible arm)."""

    name = "Uniform_relaxed"
