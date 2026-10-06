"""Bandit environments: K arms with jointly Gaussian (reward, d costs), known covariance.

The constraint of the paper is B p <= 0. For a cost threshold gamma (Lardy et al., 2025) we use
B_a = E[C_a] - gamma, so B p <= 0  <=>  sum_a p_a E[C_a] <= gamma.
"""

from dataclasses import dataclass, field

import numpy as np

from .lower_bound import solve_sop


@dataclass
class ConstrainedGaussianBandit:
    name: str
    mu: np.ndarray        # (K,) mean rewards
    B: np.ndarray         # (d, K) mean costs minus thresholds
    Sigma: np.ndarray     # (d+1, d+1) covariance of (reward, costs), shared by the arms
    reward_std: np.ndarray = field(default=None)   # optional (K,) per-arm reward std (costs keep Sigma)

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=float)
        self.B = np.atleast_2d(np.asarray(self.B, dtype=float))
        self.Sigma = np.asarray(self.Sigma, dtype=float)
        self.K, self.d = self.mu.shape[0], self.B.shape[0]
        if self.reward_std is None:
            self.reward_var = np.full(self.K, self.Sigma[0, 0])
            self._chols = np.broadcast_to(np.linalg.cholesky(self.Sigma), (self.K, self.d + 1, self.d + 1))
        else:
            # per-arm reward variances: arm a has covariance Sigma_a (Sigma with its reward row/column
            # rescaled to std_a, correlations kept). The shared Sigma handed to the algorithms that assume
            # one covariance for all arms uses the largest std: conservative (sub-Gaussian) for every arm.
            std = np.asarray(self.reward_std, dtype=float)
            self.reward_var = std ** 2
            scale = np.ones((self.K, self.d + 1))
            scale[:, 0] = std / np.sqrt(self.Sigma[0, 0])
            self._chols = np.linalg.cholesky(scale[:, :, None] * self.Sigma[None] * scale[:, None, :])
            s_max = np.ones(self.d + 1)
            s_max[0] = std.max() / np.sqrt(self.Sigma[0, 0])
            self.Sigma = s_max[:, None] * self.Sigma * s_max[None, :]
        self.means = np.vstack([self.mu, self.B]).T          # (K, d+1)
        self.p_star, self.opt = solve_sop(self.mu, self.B)
        feasible = np.flatnonzero(np.all(self.B <= 0, axis=0))
        # best feasible arm, the answer of constrained BAI (None if no arm is feasible)
        self.best_feasible_arm = int(feasible[np.argmax(self.mu[feasible])]) if len(feasible) else None

    def sample(self, a, rng):
        return self.means[a] + self._chols[a] @ rng.standard_normal(self.d + 1)

    def is_eps_good_arm(self, arm, eps, tol=1e-9):
        """arm (or None) is a correct answer of eps-relaxed constrained BAI: feasible and eps-good."""
        if self.best_feasible_arm is None:
            return arm is None
        return arm is not None and bool(np.all(self.B[:, arm] <= tol)
                                        and self.mu[arm] >= self.mu[self.best_feasible_arm] - eps - tol)

    def is_correct(self, p, eps, tol=1e-9):
        """p in Pi(nu, eps): safe and eps-optimal under the true instance."""
        return bool(np.all(self.B @ p <= tol) and self.mu @ p >= self.opt - eps - tol)


# Figure 2 of Lardy, Katsimerou & Koolen (2025), "Constrained best arm identification".
# Means are (reward, cost) read off the figure; Sigma = [0.1 0.05; 0.05 0.09] for (reward, cost).
CBAI_SIGMA = np.array([[0.10, 0.05],
                       [0.05, 0.09]])


def cbai_instance(name, reward, cost, gamma, Sigma=CBAI_SIGMA):
    reward, cost = np.asarray(reward, float), np.asarray(cost, float)
    return ConstrainedGaussianBandit(name, reward, (cost - gamma)[None, :], Sigma)


def easy():
    """Easy: the best arm (1) is feasible with slack 0.25, so the SOP is the pure arm 1."""
    return cbai_instance("Easy",
                         reward=[0.60, 0.20, 0.19, 0.10, 0.05],
                         cost=[0.50, 0.65, 0.70, 1.00, 1.00],
                         gamma=0.75)


def hard():
    """Hard: the highest-reward arm (1) is slightly infeasible; the SOP mixes it with a safe arm."""
    return cbai_instance("Hard",
                         reward=[1.00, 0.85, 0.80, 0.70, 0.65],
                         cost=[0.95, 0.80, 0.75, 0.60, 0.50],
                         gamma=0.90)


# Figure 3 of Carlsson, Basu, Johansson & Dubhashi (2024), "Pure exploration in bandits with linear
# constraints", with the unknown-constraint observation model of Das & Basu (2026): rewards N(mu_a, 1),
# costs N(A_a, I), constraints A pi <= b with A = [[1,1,0,0,0],[0,0,1,1,0]], b = (0.5, 0.5).
# With sum(pi) = 1 this is B pi <= 0 for B = A - b 1^T.
CARLSSON_A = np.array([[1.0, 1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 1.0, 0.0]])
CARLSSON_b = np.array([0.5, 0.5])


def carlsson_instance(name, mu):
    return ConstrainedGaussianBandit(name, mu, CARLSSON_A - CARLSSON_b[:, None], np.eye(3))


def carlsson_easy():
    """Star of Fig. 3 (BAI hard, constrained problem easy); `emil_easy` in the LaGEx experiments."""
    return carlsson_instance("Carlsson-Easy", [1.0, 0.5, 0.4, 0.95, 0.8])


def carlsson_hard():
    """Triangle of Fig. 3 (BAI easy, constrained problem hard); `emil_hard` in the LaGEx experiments."""
    return carlsson_instance("Carlsson-Hard", [1.0, 0.5, 0.4, 0.4, 0.5])


# IMDB-50K movie recommendation instance of Carlsson et al. (2024), Section 5 / Fig. 5 ("12 movies"), as
# built by their code (IMDB/imdb_env.py, n_actions = 12, delta = 0.1): the first 12 movies of
# user_reviews_uncensored.csv (600 simulated users). Movie a gives reward N(mu_a, std_a^2), mu_a and std_a
# the mean and (ddof = 1) std of its 600 user ratings (1-5 stars), as in their GaussianBandit.
# Constraints: at most 0.3 on action movies, at least 0.3 on drama and at least 0.3 on family movies,
# A pi <= b with the ">" rows negated. For unknown constraints, costs N(A_a, I) independent of the reward,
# as for Carlsson-Easy/-Hard (Das & Basu, 2026). Algorithms that assume one covariance for all arms get
# env.Sigma with the largest std (1.324); env.reward_var holds the per-arm variances.
IMDB_MOVIES = ["The Net", "Happily N'Ever After", "Tomorrowland", "American Hero", "Das Boot",
               "Final Destination 3", "Licence to Kill", "The Hundred-Foot Journey", "The Matrix", "Creature",
               "The Basket", "Star Trek: The Motion Picture"]
IMDB_MU = np.array([2203, 1783, 1764, 2112, 1907, 1214, 1673, 1780, 1424, 1519, 1529, 1522]) / 600.0
IMDB_STD = np.array([1.2595666195179558, 1.3065382859717756, 1.3099337938386064, 1.3238281112416055,
                     1.299748488147981, 0.9314333949192747, 1.2162768755312845, 1.3121579502354743,
                     1.1428615152154227, 1.1950079339314672, 1.190248591010673, 1.1536349848859508])
#                      Net Hap Tom Ame Das Fin Lic Hun Mat Cre Bas Sta
IMDB_A = np.array([[1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0.],       # action <= 0.3
                   [-1., 0., 0., -1., -1., 0., 0., -1., 0., 0., -1., 0.],  # drama  >= 0.3
                   [0., -1., -1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])    # family >= 0.3
IMDB_b = np.array([0.3, -0.3, -0.3])


def imdb():
    """IMDB-50K instance of Carlsson et al. (2024): K = 12 movies, d = 3 genre constraints."""
    return ConstrainedGaussianBandit("IMDB", IMDB_MU, IMDB_A - IMDB_b[:, None], np.eye(4), reward_std=IMDB_STD)


INSTANCES = {"easy": easy, "hard": hard, "carlsson-easy": carlsson_easy, "carlsson-hard": carlsson_hard,
             "imdb": imdb}
