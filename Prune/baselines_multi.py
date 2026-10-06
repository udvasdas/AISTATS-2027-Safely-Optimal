"""TopTwo-TCI of Lardy et al. (2025) extended to d >= 1 unknown constraints (our extension; theirs has d = 1).

Answer: the best feasible arm (all d constraints satisfied), eps-relaxed as in `TopTwoTCI_relaxed`. Gaussian
(reward, d costs) per arm with known covariance Sigma. GLR terms for a leader i:
  * i infeasible:  N_i min_k (B_ki)^2 / (2 Sigma_kk)   (closed form, B_ki <= 0);
  * j feasible and better than i (by eps):  min over (lambda_i, lambda_j) of
        N_i KL(nu_i || lambda_i) + N_j KL(nu_j || lambda_j)  s.t.  mu'_j - eps >= mu'_i,  B'_kj <= 0 for all k,
    a QP with d + 1 linear constraints whose dual is the (d+1)-dimensional non-negative QP
        max_{eta >= 0} eta^T c - 1/2 eta^T G eta,
        c = (mu_i - mu_j + eps, B_1j, ..., B_dj),
        G = A Sigma A^T / N_j + sigma_r^2 e_0 e_0^T / N_i,   A = [e_0; -e_1; ...; -e_d],
    solved by Cholesky + NNLS: polynomial in d. For d = 1 this is the known-covariance pair cost of Lardy et al.
  * no feasible leader (answer None): each j made feasible, the same QP without the reward row.
Sampling: leader = empirical best feasible arm, challenger = argmin_j Lambda_j + log N_j, leader w.p. 1/2.
Stopping: max over eps-good empirically feasible candidates of min_j Lambda > beta (as TopTwoTCI_relaxed).
"""

import numpy as np
from scipy.optimize import nnls

from .algorithms import PureExplorationAlgorithm

NONE = -1


def _nnqp(c, G):
    """max_{eta >= 0} eta^T c - 1/2 eta^T G eta for one (m,) / (m, m) pair (G positive definite)."""
    if np.all(c <= 0):
        return 0.0
    m = len(c)
    G = G + 1e-12 * (1.0 + np.trace(G)) * np.eye(m)
    L = np.linalg.cholesky(G)
    eta = nnls(L.T, np.linalg.solve(L, c))[0]
    return float(eta @ c - 0.5 * eta @ G @ eta)


class TopTwoTCIMulti(PureExplorationAlgorithm):
    """TopTwo-TCI with d unknown constraints, eps-relaxed answer (eps = 0: exact best feasible arm)."""

    name = "TopTwo-TCI"

    def __init__(self, K, Sigma, delta, eps=0.0, beta_tt=0.5, threshold="stylized"):
        self.beta_tt = beta_tt
        super().__init__(K, Sigma, delta, eps, threshold)
        S = self.Sigma
        A = np.vstack([np.eye(self.d + 1)[0], -np.eye(self.d + 1)[1:]])
        self.G_j = A @ S @ A.T                       # / N_j
        self.G_i = np.zeros_like(self.G_j)
        self.G_i[0, 0] = S[0, 0]                     # / N_i

    def reset(self):
        super().reset()
        self.rng = np.random.default_rng()
        self.leader, self.answer, self.Lam = NONE, NONE, None

    def _feasible(self):
        return np.all(self.means[:, 1:] <= 0, axis=1)

    def _terms(self, i, eps):
        """Lambda_j for every alternative answer j (arms, then 'i infeasible' / None at index K)."""
        m, N, S, K = self.means, self.N, self.Sigma, self.K
        Lam = np.full(K + 1, np.inf)
        if i == NONE:
            for j in range(K):
                Lam[j] = _nnqp(m[j, 1:], N[j] ** -1 * S[1:, 1:])
            return Lam
        Bi = m[i, 1:]
        Lam[K] = N[i] * np.min(Bi ** 2 / (2 * np.diag(S)[1:]))
        for j in range(K):
            if j != i:
                c = np.r_[m[i, 0] - m[j, 0] + eps, m[j, 1:]]
                Lam[j] = _nnqp(c, self.G_j / N[j] + self.G_i / N[i])
        return Lam

    def _best_feasible(self):
        f = np.flatnonzero(self._feasible())
        return NONE if len(f) == 0 else int(f[np.argmax(self.means[f, 0])])

    def should_stop(self):
        self.leader = self._best_feasible()
        self.Lam = self._terms(self.leader, self.eps if self.leader != NONE else 0.0)
        self.answer, best = self.leader, self.Lam.min()
        if self.leader != NONE and self.eps > 0:
            f = np.flatnonzero(self._feasible())
            for i in f[self.means[f, 0] > self.means[self.leader, 0] - self.eps]:
                if i != self.leader:
                    z = self._terms(int(i), self.eps).min()
                    if z > best:
                        self.answer, best = int(i), z
        return best > self.beta()

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

    def recommended_arm(self):
        return None if self.answer == NONE else self.answer

    def recommend(self):
        p = np.zeros(self.K)
        if self.answer != NONE:
            p[self.answer] = 1.0
        return p
