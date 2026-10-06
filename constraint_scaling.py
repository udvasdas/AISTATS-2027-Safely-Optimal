"""Computational cost of PRUNE-FW as the number of constraints d grows (fixed K).

Instance: K = 10 arms, a fixed pool of constraints added one at a time (nested instances). Arm 1 has the best
reward and a high cost in every constraint; thresholds make the uniform policy strictly feasible, so the safely
optimal policy is a mixture. Gaussian (reward, costs) with Sigma = diag(0.1, 0.09, ..., 0.09).

Measured for every d (single thread, idle machine):
  * time of one evaluation of F(omega, p) with all its pieces:
      PRUNE-FW, dual QP by Cholesky + NNLS        (polynomial in d)
      PRUNE-FW, dual QP by active-set enumeration (2^(d+1) active sets)
      grid PRUNE, C(K, d+1) faces x grid           (d <= 3 only)
  * per-round time of PRUNE-FW (NNLS) over a short episode,
  * F_FW / F_grid where the grid is feasible.

    python constraint_scaling.py               # results/constraint_scaling.{json,pdf,png}
    python constraint_scaling.py --plot-only
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from prune.experiments import run_episode  # noqa: E402
from prune.fw_solver import FWAltSolver, PRUNEFW  # noqa: E402
from prune.instances import ConstrainedGaussianBandit  # noqa: E402
from prune.baselines_multi import TopTwoTCIMulti  # noqa: E402
from prune.lower_bound import AltSolver, init_answer  # noqa: E402

K, D_MAX, EPS = 10, 14, 0.05


def pool(seed=3):
    rng = np.random.default_rng(seed)
    mu = np.r_[1.0, np.sort(rng.uniform(0.3, 0.9, K - 1))[::-1]]
    C = rng.uniform(0.0, 1.0, (D_MAX, K))
    C[:, 0] = rng.uniform(0.85, 1.0, D_MAX)          # the best arm is expensive in every constraint
    gamma = C.mean(1) + 0.05                           # uniform policy strictly feasible
    return mu, C - gamma[:, None]


def instance(d):
    mu, B = pool()
    return ConstrainedGaussianBandit(f"K={K}, d={d}", mu, B[:d], np.diag(np.r_[0.10, np.full(d, 0.09)]))


def timed(f, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def measure(d, rng, reps=3, rounds=150):
    env = instance(d)
    fw_n = FWAltSolver(K, env.Sigma, EPS, qp="nnls")
    fw_e = FWAltSolver(K, env.Sigma, EPS, qp="enum")
    w = rng.dirichlet(np.full(K, 2.0))
    p, _ = init_answer(fw_n, env.mu, env.B, w)
    row = dict(d=d, active=int(np.sum(np.abs(env.B @ env.p_star) < 1e-9)),
               support=int(np.sum(env.p_star > 1e-9)))
    row["fw_nnls"], ev = timed(lambda: fw_n.evaluate(env.mu, env.B, w, p), reps)
    row["pieces"] = int(np.isfinite(ev.vals).sum())
    F_fw = ev.F
    row["fw_enum"] = timed(lambda: fw_e.evaluate(env.mu, env.B, w, p), reps)[0] if d <= 8 else None
    if d <= 3:
        grid = AltSolver(K, env.Sigma, EPS)
        row["grid"], evg = timed(lambda: grid.evaluate(env.mu, env.B, w, p), 1)
        row["F_fw_over_grid"] = F_fw / evg.F if evg.F > 0 else None
    else:
        row["grid"], row["F_fw_over_grid"] = None, None
    # per-round cost of PRUNE-FW over a short episode (all operations: answer, FW step, tracking, stopping)
    algo = PRUNEFW(K, env.Sigma, 0.1, EPS)
    algo.solver.qp = "nnls"
    t0 = time.perf_counter()
    out = run_episode(algo, env, np.random.default_rng(d), max_steps=K + rounds)
    row["round"] = (time.perf_counter() - t0) / out["tau"]
    return row


def toptwo_round(d, rounds=150, episodes=3):
    """Per-round time of TopTwo-TCI extended to d constraints (eps-relaxed, eps = EPS), mean over short episodes."""
    env = instance(d)
    ts = []
    for k in range(episodes):
        algo = TopTwoTCIMulti(K, env.Sigma, 0.1, EPS)
        t0 = time.perf_counter()
        out = run_episode(algo, env, np.random.default_rng(100 * d + k), max_steps=K + rounds)
        ts.append((time.perf_counter() - t0) / out["tau"])
    return float(np.mean(ts))


def plot_rounds(rows, path, fixed_powers=()):
    """Per-round run time of PRUNE (PRUNE-FW) against the number of constraints d."""
    from intro_illustration import TIMES, _embed_fonts
    SURFACE, INK_2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
    d = np.array([r["d"] for r in rows])
    # rendered through LaTeX with mathptmx: Times text and math, including \mathcal{O}
    rc = dict(TIMES, **{"text.usetex": True, "text.latex.preamble": r"\usepackage{mathptmx}"})
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(5.4, 3.8), facecolor=SURFACE, constrained_layout=True)
        ax.set_facecolor(SURFACE)
        t = np.array([1000 * r["round"] for r in rows])
        ax.plot(d, t, color="#2a78d6", lw=2, marker="o", ms=4, label="PRUNE")
        # power law c d^alpha fitted by least squares on log t vs log d: a log(d)-shaped curve on this axis
        alpha, logc = np.polyfit(np.log(d), np.log(t), 1)
        dd = np.linspace(d.min(), d.max(), 200)
        ax.plot(dd, np.exp(logc) * dd ** alpha, color=INK_2, lw=1.5, ls=(0, (4, 3)),
                label=r"$\mathcal{O}(d^{\alpha})$")
        for k, ls in zip(fixed_powers, [(0, (1, 2)), (0, (6, 2, 1, 2))]):
            # c d^k with the exponent fixed, c by least squares on the log scale
            ck = np.exp(np.mean(np.log(t) - k * np.log(d)))
            ax.plot(dd, ck * dd ** k, color="#eb6834", lw=1.5, ls=ls, label=rf"$\mathcal{{O}}(d^{{{k}}})$")
        ax.set_yscale("log")
        ax.set_xlabel(r"number of constraints $d$", color=INK_2)
        ax.set_ylabel("run time per round (ms)", color=INK_2)
        ax.set_xticks(d)
        ax.grid(color=GRID, lw=0.6, which="both")
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=11, labelcolor=INK_2, loc="center right")
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
    _embed_fonts(f"{path}.pdf")


def plot(rows, path, counts=None):
    """(a) time per evaluation / round vs d (log scale); (b) if `counts` is given, the decomposition of one
    PRUNE-FW evaluation into the number of dual QPs (bounded by the Frank-Wolfe budget) and the cost of one QP."""
    from intro_illustration import TIMES, _embed_fonts
    SURFACE, INK_2, GRID = "#fcfcfb", "#52514e", "#e4e3df"
    d = np.array([r["d"] for r in rows])
    series = [("fw_nnls", "PRUNE-FW, one evaluation of F (NNLS dual)", "#2a78d6", "-"),
              ("round", "PRUNE-FW, one round", "#2a78d6", (0, (4, 3))),
              ("fw_enum", "PRUNE-FW, one evaluation (active-set enumeration)", "#eb6834", "-"),
              ("grid", "grid PRUNE, one evaluation", "#52514e", "-")]

    def style(ax, ylabel):
        ax.set_facecolor(SURFACE)
        ax.set_xlabel("number of constraints d", color=INK_2)
        ax.set_ylabel(ylabel, color=INK_2)
        ax.set_xticks(d)
        ax.grid(color=GRID, lw=0.6, which="both")
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK_2, labelsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    with plt.rc_context(TIMES):
        n = 1 if counts is None else 2
        fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 3.8), facecolor=SURFACE, constrained_layout=True)
        ax = np.atleast_1d(axes)[0]
        for key, label, c, ls in series:
            x = [r["d"] for r in rows if r.get(key) is not None]
            y = [1000 * r[key] for r in rows if r.get(key) is not None]
            ax.plot(x, y, color=c, ls=ls, lw=2, marker="o", ms=4, label=label)
        ax.set_yscale("log")
        style(ax, "time (ms)")
        ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower right")
        if counts is not None:
            ax2 = axes[1]
            dc = [c["d"] for c in counts]
            ax2.plot(dc, [c["us_per_row"] for c in counts], color="#2a78d6", lw=2, marker="o", ms=4,
                     label="time of one dual QP (μs)")
            ax2.set_ylim(0, None)
            style(ax2, "time of one dual QP (μs)")
            tw = ax2.twinx()
            tw.bar(dc, [c["rows"] for c in counts], color="#9ec5f4", alpha=0.6, width=0.6, zorder=0,
                   label="dual QPs per evaluation")
            tw.axhline(11 * 40 * 10 + 11, color=INK_2, lw=1, ls=(0, (4, 3)), label="Frank–Wolfe budget")
            tw.set_ylabel("dual QPs per evaluation", color=INK_2)
            tw.tick_params(colors=INK_2, labelsize=8)
            tw.set_ylim(0, 5200)
            ax2.set_zorder(tw.get_zorder() + 1)
            ax2.patch.set_visible(False)
            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = tw.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left")
        for ext in ("pdf", "png"):
            fig.savefig(f"{path}.{ext}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
    _embed_fonts(f"{path}.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--toptwo", action="store_true",
                    help="add the per-round time of TopTwo-TCI (d constraints) to the saved results")
    args = ap.parse_args()
    path = "results/constraint_scaling"
    if args.plot_only:
        rows = json.load(open(path + ".json"))
    else:
        rng = np.random.default_rng(0)
        rows = []
        for d in range(1, D_MAX + 1):
            rows.append(measure(d, rng))
            r = rows[-1]
            fmt = lambda v: "      -" if v is None else f"{1000 * v:9.1f}"
            print(f"d={d:2d}  active at p*={r['active']:2d} support={r['support']:2d} pieces={r['pieces']:3d} | "
                  f"FW-nnls {fmt(r['fw_nnls'])} ms  FW-enum {fmt(r['fw_enum'])} ms  grid {fmt(r['grid'])} ms  "
                  f"round {fmt(r['round'])} ms  F_fw/F_grid {r['F_fw_over_grid']}", flush=True)
        json.dump(rows, open(path + ".json", "w"), indent=1)
    if args.toptwo:
        for r in rows:
            r["toptwo_round"] = toptwo_round(r["d"])
            print(f"d={r['d']:2d}  PRUNE round {1000 * r['round']:8.1f} ms   TopTwo-TCI round "
                  f"{1000 * r['toptwo_round']:7.2f} ms", flush=True)
        json.dump(rows, open(path + ".json", "w"), indent=1)
    if all("round" in r for r in rows):
        plot_rounds(rows, path + "_rounds")
        print(f"saved {path}_rounds.{{pdf,png}}")
    counts = path + "_qp_counts.json"
    plot(rows, path, json.load(open(counts)) if os.path.exists(counts) else None)
    print(f"saved {path}.{{json,pdf,png}}")


if __name__ == "__main__":
    main()
