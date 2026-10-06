"""Run PRUNE and baselines on the Easy / Hard instances of Lardy et al. (2025).

Examples:
    python run_experiments.py --algos prune lagex tas toptwo --eps 0.05 --delta 0.01 --runs 100
    python run_experiments.py --algos tas toptwo --cbai-cov unknown --cbai-init 3 --threshold lardy
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker process

import argparse
import functools
import json

import numpy as np

from prune.algorithms import PRUNE, PRUNEUniform
from prune.fw_solver import FWAltSolver, PRUNEFW
from prune.baselines import TopTwoTCI, TopTwoTCI_relaxed, TrackAndStop, TrackAndStop_relaxed, UniformCBAI, UniformCBAI_relaxed
from prune.carlsson import CGE, CTnS, Uniform
from prune.experiments import run_many, summarise
from prune.hardness import t_known_exact, t_known_vertex_eps
from prune.instances import INSTANCES
from prune.lagex import LaGEx
from prune.lower_bound import characteristic_time


def make_prune(env, args):
    return PRUNE(env.K, env.Sigma, args.delta, args.eps, threshold=args.threshold,
                 recommend=args.recommend, p_every=args.p_every, lazy=args.lazy)


def make_prune_fw(env, args, stopping="fw"):
    # PRUNE with Frank-Wolfe over the witness q; stopping on the FW value or on the certified lower bound
    return PRUNEFW(env.K, env.Sigma, args.delta, args.eps, stopping=stopping, threshold=args.threshold,
                   recommend=args.recommend, p_every=args.p_every, lazy=args.lazy)


def make_uniform_prune(env, args):
    # uniform sampling with PRUNE's stopping rule and recommendation (unknown constraints)
    return PRUNEUniform(env.K, env.Sigma, args.delta, args.eps, threshold=args.threshold)


def make_uniform_cbai(env, args):
    # uniform sampling with Lardy et al.'s exact stopping rule (best feasible arm, no eps-relaxation)
    return UniformCBAI(env.K, env.Sigma, args.delta, args.eps, covariance=args.toptwo_cov,
                       n_init=args.toptwo_init, threshold=args.threshold)


def make_uniform_cbai_relaxed(env, args):
    return UniformCBAI_relaxed(env.K, env.Sigma, args.delta, args.eps, covariance=args.toptwo_cov,
                               n_init=args.toptwo_init, threshold=args.threshold)


def make_toptwo(env, args):
    return TopTwoTCI(env.K, env.Sigma, args.delta, args.eps, covariance=args.toptwo_cov,
                     n_init=args.toptwo_init, threshold=args.threshold)


def make_tas(env, args):
    return TrackAndStop(env.K, env.Sigma, args.delta, args.eps, covariance=args.toptwo_cov,
                        n_init=args.toptwo_init, threshold=args.threshold)


def make_lagex(env, args):
    # r = eps: LaGEx targets the same set of eps-optimal safe policies as PRUNE
    return LaGEx(env.K, env.Sigma, args.delta, args.eps, threshold=args.threshold, variant="corrected")


def make_lagex_paper(env, args):
    return LaGEx(env.K, env.Sigma, args.delta, args.eps, threshold=args.threshold, variant="paper",
                 recommend=args.lagex_recommend)


def make_tas_relaxed(env, args):
    # eps-good feasible arm (not in Lardy et al.): Theorem 2.2 oracle for the easiest eps-good answer
    return TrackAndStop_relaxed(env.K, env.Sigma, args.delta, args.eps, covariance=args.toptwo_cov,
                                n_init=args.toptwo_init, threshold=args.threshold)


def make_toptwo_relaxed(env, args):
    return TopTwoTCI_relaxed(env.K, env.Sigma, args.delta, args.eps, covariance=args.toptwo_cov,
                             n_init=args.toptwo_init, threshold=args.threshold)


def make_known(cls, env, args, relaxed=False):
    # Carlsson et al. (2024): constraints known, exact optimal policy (relaxed: eps-optimal vertex);
    # Pi = F for anytime constraints
    extra = dict(lazy=args.lazy) if cls is CTnS else {}
    return cls(env.K, env.Sigma, args.delta, args.eps, env.B, exploration=args.exploration, threshold=args.threshold,
               relaxed=relaxed, reward_var=env.reward_var, **extra)


ALGOS = {"prune": make_prune, "lagex": make_lagex, "lagex-paper": make_lagex_paper, "tas": make_tas, "toptwo": make_toptwo,
         "ctns": functools.partial(make_known, CTnS), "cge": functools.partial(make_known, CGE),
         "uniform": functools.partial(make_known, Uniform),
         "prune-fw": make_prune_fw, "prune-fw-cert": functools.partial(make_prune_fw, stopping="certified"),
         "uniform-prune": make_uniform_prune, "uniform-cbai": make_uniform_cbai,
         "uniform-cbai-relaxed": make_uniform_cbai_relaxed, "tas-relaxed": make_tas_relaxed, "toptwo-relaxed": make_toptwo_relaxed,
         "ctns-relaxed": functools.partial(make_known, CTnS, relaxed=True),
         "cge-relaxed": functools.partial(make_known, CGE, relaxed=True),
         "uniform-relaxed": functools.partial(make_known, Uniform, relaxed=True)}
KNOWN = {"ctns", "cge", "uniform", "ctns-relaxed", "cge-relaxed", "uniform-relaxed"}
LAZY = {"prune", "prune-fw", "prune-fw-cert", "ctns", "ctns-relaxed"}                     # algorithms that take --lazy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algos", nargs="+", default=["prune", "lagex", "tas", "toptwo"], choices=sorted(ALGOS))
    ap.add_argument("--instances", nargs="+", default=["easy", "hard"], choices=sorted(INSTANCES))
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--threshold", default="stylized", choices=["stylized", "theory", "lardy"],
                    help="stylized: log((1+log t)/delta); theory: Lemma 2; lardy: log(1/delta)+log log t")
    ap.add_argument("--recommend", default="certified", choices=["certified", "lp"], help="PRUNE only")
    ap.add_argument("--p-every", type=int, default=1, help="PRUNE: rounds between answer (p) updates")
    ap.add_argument("--toptwo-cov", "--cbai-cov", dest="toptwo_cov", default="known", choices=["known", "unknown"],
                    help="arm model of TaS / TopTwo-TCI (unknown = Table 1 of Lardy et al.)")
    ap.add_argument("--toptwo-init", "--cbai-init", dest="toptwo_init", type=int, default=1,
                    help="initial pulls per arm of TaS / TopTwo-TCI")
    ap.add_argument("--lagex-recommend", default="optimistic", choices=["optimistic", "empirical"],
                    help="lagex-paper: pi_t on the optimistic set (paper) or LP on A_hat (CGE_Lag.py)")
    ap.add_argument("--exploration", default="anytime", choices=["anytime", "end-of-time"],
                    help="ctns / cge / uniform: allocations restricted to F (anytime, Fig. 3c/3d) or the simplex")
    ap.add_argument("--fw-oracle", action="store_true", help="compute the oracle T* with the Frank-Wolfe solver")
    ap.add_argument("--lazy", type=int, default=1,
                    help="PRUNE / CTnS: recompute the expensive updates every LAZY rounds (file tag _lazyK)")
    ap.add_argument("--max-steps", type=int, default=10 ** 6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in args.instances:
        env = INSTANCES[name]()
        # oracle T*(nu, eps): grid solver by default; --fw-oracle uses the Frank-Wolfe solver (fast for large d)
        lb = characteristic_time(env.mu, env.B, env.Sigma, args.eps,
                                 solver=FWAltSolver(env.K, env.Sigma, args.eps) if args.fw_oracle else None)
        # known-constraint, exact-identification bound of Carlsson et al. (the target of ctns / cge / uniform)
        known = {e: t_known_exact(env.mu, env.B, env.reward_var, anytime=e == "anytime") for e in ("anytime", "end-of-time")}
        known.update({f"{e}-relaxed": t_known_vertex_eps(env.mu, env.B, args.eps, env.reward_var, anytime=e == "anytime")
                      for e in ("anytime", "end-of-time")})
        print(f"\n== {env.name}: K={env.K}, d={env.d}, eps={args.eps}, delta={args.delta}, threshold={args.threshold}")
        arm = env.best_feasible_arm
        print(f"   SOP p* = {np.round(env.p_star, 4)}, OPT = {env.opt:.4f}; best feasible arm = "
              + (f"{arm} (reward {env.mu[arm]:.4f})" if arm is not None else "none (no single arm is feasible)"))
        print(f"   oracle T*(nu, eps) = {lb['T']:.2f},  T* log(1/delta) = {lb['T'] * np.log(1 / args.delta):.1f}")
        print(f"   known constraints, exact p*: T_F = {known['anytime']:.2f} (anytime), {known['end-of-time']:.2f} "
              f"(end-of-time);  T_F kl(delta||1-delta) = {known[args.exploration] * _kl_delta(args.delta):.1f} "
              f"({args.exploration})")
        print(f"   known constraints, eps-optimal vertex: T_F = {known['anytime-relaxed']:.2f} (anytime), "
              f"{known['end-of-time-relaxed']:.2f} (end-of-time)")
        # same seeds for every algorithm on an instance
        seeds = np.random.SeedSequence([args.seed, sorted(INSTANCES).index(name)]).generate_state(args.runs)
        for algo in args.algos:
            tag = f"_{args.exploration}" if algo in KNOWN else ""
            if args.lazy > 1 and algo in LAZY:
                tag += f"_lazy{args.lazy}"
            path = os.path.join(args.out, f"{algo.replace('-relaxed', '_relaxed')}{tag}_{name}_eps{args.eps}_delta{args.delta}_{args.threshold}.json")
            partial = path[:-5] + ".partial.jsonl"            # finished runs, appended as they complete
            if os.path.exists(partial):
                os.remove(partial)
            res = run_many(INSTANCES[name], functools.partial(ALGOS[algo], args=args), seeds,
                           args.max_steps, args.workers, partial_path=partial)
            s = summarise(res)
            arm = (f", best-arm error = {s['arm_error_rate']:.3f}, not eps-good arm = {s['arm_eps_error_rate']:.3f}"
                   if "arm_error_rate" in s else "")
            if algo in KNOWN:
                arm = f", != p* = {s['exact_error_rate']:.3f}"
            print(f"   {algo:11s} tau = {s['mean_tau']:8.1f} ± {s['se_tau']:6.1f} (median {s['median_tau']:.0f}) | "
                  f"not in Pi(nu,eps) = {s['error_rate']:.3f}{arm} | reward of p_hat = {s['mean_reward']:.4f} | "
                  f"{s['sec_per_run']:.2f}s/run")
            with open(path, "w") as f:
                json.dump(dict(args=vars(args), instance=name, oracle=dict(T=lb["T"], omega=lb["omega"].tolist(),
                               p=lb["p"].tolist()), oracle_known=known, summary=s, runs=res), f, indent=1)
            os.remove(partial)                                     # the full result file supersedes it


def _kl_delta(delta):
    return delta * np.log(delta / (1 - delta)) + (1 - delta) * np.log((1 - delta) / delta)


if __name__ == "__main__":
    main()
