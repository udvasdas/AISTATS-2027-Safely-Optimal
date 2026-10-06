"""Running pure-exploration algorithms on bandit instances, in parallel over seeds."""

import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def run_episode(algo, env, rng, max_steps=10 ** 6):
    """One fixed-confidence episode. Returns stopping time, correctness and the recommendation.

    `correct`: the recommended policy is in Pi(nu, eps) (safe and eps-optimal); `exact`: it is the optimal
    policy p*. For algorithms that recommend an arm (constrained BAI), `arm_correct`: it is the best feasible arm.
    """
    algo.reset()
    if hasattr(algo, "rng"):
        algo.rng = rng                          # randomised sampling rules share the episode stream
    for a in algo.initial_arms():
        algo.observe(a, env.sample(a, rng))
    stopped = True
    while not algo.should_stop():
        if algo.t >= max_steps:
            stopped = False
            break
        a = algo.next_arm()
        algo.observe(a, env.sample(a, rng))
    p = algo.recommend()
    out = dict(tau=algo.t, correct=env.is_correct(p, algo.eps), exact=bool(np.allclose(p, env.p_star, atol=1e-6)),
               stopped=stopped,
               reward=float(env.mu @ p), p=p.tolist(), N=algo.N.tolist())
    if hasattr(algo, "recommended_arm"):
        arm = algo.recommended_arm()
        out["arm"] = arm
        out["arm_correct"] = arm == env.best_feasible_arm
        out["arm_eps_correct"] = env.is_eps_good_arm(arm, algo.eps)
    return out


def _worker(args):
    make_env, make_algo, seed, max_steps = args
    env = make_env()
    algo = make_algo(env)
    t0 = time.perf_counter()
    out = run_episode(algo, env, np.random.default_rng(seed), max_steps)
    out["seed"] = seed
    out["seconds"] = time.perf_counter() - t0
    return out


def run_many(make_env, make_algo, seeds, max_steps=10 ** 6, workers=None, partial_path=None):
    """Run independent episodes. `make_env()` and `make_algo(env)` must be picklable (module-level).

    partial_path: append every finished episode to this JSON-lines file as soon as it completes, so that an
    interrupted batch keeps its finished runs. Results are returned in the order of `seeds`.
    """
    workers = workers or os.cpu_count()
    jobs = [(make_env, make_algo, int(s), max_steps) for s in seeds]
    if workers == 1:
        out = []
        for j in jobs:
            out.append(_worker(j))
            _append(partial_path, out[-1])
        return out
    out = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, j): i for i, j in enumerate(jobs)}
        for f in as_completed(futures):
            out[futures[f]] = f.result()
            _append(partial_path, out[futures[f]])
    return out


def _append(path, result):
    if path is not None:
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")


def summarise(results):
    tau = np.array([r["tau"] for r in results], dtype=float)
    out = dict(runs=len(results),
               mean_tau=float(tau.mean()),
               se_tau=float(tau.std(ddof=1) / np.sqrt(len(tau))) if len(tau) > 1 else 0.0,
               median_tau=float(np.median(tau)),
               error_rate=float(np.mean([not r["correct"] for r in results])),
               exact_error_rate=float(np.mean([not r.get("exact", r["correct"]) for r in results])),
               mean_reward=float(np.mean([r["reward"] for r in results])),
               not_stopped=int(np.sum([not r["stopped"] for r in results])),
               sec_per_run=float(np.mean([r["seconds"] for r in results])))
    if "arm_correct" in results[0]:
        out["arm_error_rate"] = float(np.mean([not r["arm_correct"] for r in results]))
        out["arm_eps_error_rate"] = float(np.mean([not r.get("arm_eps_correct", r["arm_correct"]) for r in results]))
    return out
