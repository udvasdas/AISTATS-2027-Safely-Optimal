"""Effect of the reward-cost correlation rho on sample complexity: PRUNE-FW vs TopTwo-TCI (eps-relaxed) on the
Hard instance of Lardy et al. (2025) with eps = 0.16, Sigma = [[0.1, rho sqrt(0.009)], [rho sqrt(0.009), 0.09]].
(rho = +-1 makes Sigma singular, so the grid stops at +-0.95.)

    python rho_sweep.py                       # run, save results/rho_sweep_hard_eps0.16.json, plot
    python rho_sweep.py --plot-only
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from prune.baselines import TopTwoTCI_relaxed, _ell_known, _PairCost, cbai_oracle  # noqa: E402
from prune.experiments import run_many, summarise  # noqa: E402
from prune.fw_solver import FWAltSolver, PRUNEFW  # noqa: E402
from prune.instances import cbai_instance  # noqa: E402
from prune.lower_bound import characteristic_time  # noqa: E402

RHOS = [-0.95, -0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
VAR_R, VAR_C = 0.10, 0.09
REWARD, COST, GAMMA = [1.00, 0.85, 0.80, 0.70, 0.65], [0.95, 0.80, 0.75, 0.60, 0.50], 0.90


def sigma(rho):
    c = rho * np.sqrt(VAR_R * VAR_C)
    return np.array([[VAR_R, c], [c, VAR_C]])


def hard_rho(rho):
    return cbai_instance(f"Hard (rho={rho})", REWARD, COST, GAMMA, Sigma=sigma(rho))


def make_prune_fw(env, delta, eps):
    return PRUNEFW(env.K, env.Sigma, delta, eps)


def make_toptwo(env, delta, eps):
    return TopTwoTCI_relaxed(env.K, env.Sigma, delta, eps)


def bounds(rho, eps):
    """PRUNE's T*(nu, eps) (Frank-Wolfe oracle) and the best eps-relaxed constrained-BAI T* (over leaders)."""
    env = hard_rho(rho)
    t_prune = characteristic_time(env.mu, env.B, env.Sigma, eps, solver=FWAltSolver(env.K, env.Sigma, eps))["T"]
    covs = np.broadcast_to(env.Sigma, (env.K, 2, 2))
    pc = _PairCost(env.Sigma, "known")
    feas = np.flatnonzero(env.B[0] <= 0)
    best = env.mu[feas].max()
    t_cbai = min(cbai_oracle(env.means, covs, pc, _ell_known, leader=i, eps=eps)[2]
                 for i in feas if env.mu[i] > best - eps)
    return t_prune, t_cbai


def run(args):
    out = dict(rhos=RHOS, eps=args.eps, delta=args.delta, runs=args.runs, algos={})
    seeds = np.random.SeedSequence([args.seed, 7]).generate_state(args.runs)      # same seeds for every rho
    for name, make in (("PRUNE-FW", make_prune_fw), ("TopTwo-TCI", make_toptwo)):
        rows = []
        for rho in RHOS:
            res = run_many(functools.partial(hard_rho, rho),
                           functools.partial(make, delta=args.delta, eps=args.eps), seeds, workers=args.workers)
            s = summarise(res)
            rows.append(dict(rho=rho, mean_tau=s["mean_tau"], se_tau=s["se_tau"], median_tau=s["median_tau"],
                             error_rate=s["error_rate"], taus=[r["tau"] for r in res]))
            print(f"{name:10s} rho={rho:+.2f}  tau = {s['mean_tau']:7.1f} ± {s['se_tau']:5.1f}  "
                  f"not in Pi(nu,eps) = {s['error_rate']:.3f}", flush=True)
        out["algos"][name] = rows
    out["bounds"] = [dict(zip(("prune", "cbai"), bounds(r, args.eps))) for r in RHOS]
    return out


def plot(d, path):
    from intro_illustration import TIMES, _embed_fonts
    SURFACE, INK_2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
    color = {"PRUNE-FW": "#2a78d6", "TopTwo-TCI": "#eb6834"}
    rho = np.array(d["rhos"])
    L = np.log(1 / d["delta"])
    with plt.rc_context(TIMES):
        fig, ax = plt.subplots(figsize=(5.4, 4.0), facecolor=SURFACE, constrained_layout=True)
        ax.set_facecolor(SURFACE)
        for name, rows in d["algos"].items():
            m = np.array([r["mean_tau"] for r in rows])
            se = np.array([r["se_tau"] for r in rows])
            label = "PRUNE" if name == "PRUNE-FW" else name
            ax.fill_between(rho, m - 2 * se, m + 2 * se, color=color[name], alpha=0.15, lw=0)
            ax.plot(rho, m, color=color[name], lw=2, marker="o", ms=4, label=label)
        ax.plot(rho, [b["prune"] * L for b in d["bounds"]], color=color["PRUNE-FW"], lw=1.2, ls=(0, (4, 3)),
                label="lower bound T*(ν,ε) log(1/δ), PRUNE")
        ax.set_xlabel("reward–cost correlation ρ", color=INK_2)
        ax.set_ylabel("stopping time τ", color=INK_2)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, None)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                  ncol=2)
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
    _embed_fonts(f"{path}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps", type=float, default=0.16)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    path = f"results/rho_sweep_hard_eps{args.eps}"
    if args.plot_only:
        d = json.load(open(path + ".json"))
    else:
        d = run(args)
        json.dump(d, open(path + ".json", "w"), indent=1)
    for b, r in zip(d["bounds"], d["rhos"]):
        print(f"rho={r:+.2f}  T*_PRUNE = {b['prune']:8.1f}   T*_eps-CBAI = {b['cbai']:8.1f}")
    plot(d, path)
    print(f"saved {path}.{{json,pdf,png}}")


if __name__ == "__main__":
    main()
