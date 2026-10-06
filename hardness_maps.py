"""Hardness maps over mu = (1, 0.5, 0.4, mu4, mu5), constraints pi1 + pi2 <= 0.5, pi3 + pi4 <= 0.5
(Figure 3(a)/(b) of Carlsson et al., 2024), for five formulations of the problem:

  bai        unconstrained best-arm identification
  known      known constraints, exact optimal policy, anytime constraints (Carlsson et al., Fig. 3b)
  known_eps  known constraints, eps-optimal safe policy, unrestricted allocation (reference for PRUNE)
  lagrange   unknown constraints, Lagrangian relaxation of Das & Basu (2026), r = eps
  prune_rho  unknown constraints, PRUNE's lower bound (Theorem 1) with reward-cost correlation rho

Examples:
    python hardness_maps.py --grid 50 --eps 0.05                 # compute (parallel) and plot
    python hardness_maps.py --plot-only results/hardness_eps0.05.npz
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

from prune.hardness import (correlated_sigma, t_bai, t_known_eps, t_known_exact, t_lagrangian,  # noqa: E402
                            t_prune)
from prune.instances import CARLSSON_A, CARLSSON_b  # noqa: E402

B = CARLSSON_A - CARLSSON_b[:, None]
STAR, TRIANGLE = (0.95, 0.8), (0.4, 0.5)


def _point(args):
    mu4, mu5, eps, rhos = args
    mu = np.array([1.0, 0.5, 0.4, mu4, mu5])
    out = dict(bai=t_bai(mu), known=t_known_exact(mu, B), known_eps=t_known_eps(mu, B, eps),
               lagrange=t_lagrangian(mu, B, eps))
    for rho in rhos:
        out[f"prune_{rho:+.2f}"] = t_prune(mu, B, eps, correlated_sigma(rho, 2))
    return out


def compute(grid, eps, rhos, workers):
    axis = np.linspace(0.0, 0.99, grid)
    jobs = [(m4, m5, eps, rhos) for m5 in axis for m4 in axis]          # row = mu5, column = mu4
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_point, jobs, chunksize=4))
    print(f"computed {len(jobs)} instances in {time.time() - t0:.0f}s")
    maps = {k: np.array([r[k] for r in res]).reshape(grid, grid) for k in res[0]}
    return axis, maps


def _point_key(args):
    mu4, mu5, eps, key = args
    mu = np.array([1.0, 0.5, 0.4, mu4, mu5])
    if key == "bai":
        return t_bai(mu)
    if key == "known":
        return t_known_exact(mu, B)
    if key == "known_eps":
        return t_known_eps(mu, B, eps)
    if key == "lagrange":
        return t_lagrangian(mu, B, eps)
    return t_prune(mu, B, eps, correlated_sigma(float(key.split("_")[1]), 2))


def recompute(data, keys, workers):
    """Recompute only the maps in `keys` on the grid of an existing result."""
    axis, eps = data["axis"], float(data["eps"])
    for key in keys:
        jobs = [(m4, m5, eps, key) for m5 in axis for m4 in axis]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            data[key] = np.array(list(ex.map(_point_key, jobs, chunksize=4))).reshape(len(axis), len(axis))
        print(f"recomputed {key}")
    return data


def rho_curves(eps, rhos, workers):
    """T*(rho) of PRUNE's bound at the star and the triangle."""
    jobs = [(m4, m5, rho, eps) for (m4, m5) in (STAR, TRIANGLE) for rho in rhos]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        vals = list(ex.map(_rho_point, jobs))
    return np.array(vals).reshape(2, len(rhos))


def _rho_point(args):
    m4, m5, rho, eps = args
    return t_prune(np.array([1.0, 0.5, 0.4, m4, m5]), B, eps, correlated_sigma(rho, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", type=int, default=50)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--rhos", type=float, nargs="+", default=[-0.6, -0.3, 0.0, 0.3, 0.6])
    ap.add_argument("--rho-curve", type=float, nargs="+", default=list(np.round(np.linspace(-0.65, 0.65, 14), 3)))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results")
    ap.add_argument("--plot-only", default=None, help="npz file of a previous run")
    ap.add_argument("--recompute", nargs="+", default=None,
                    help="with --plot-only: recompute these maps (e.g. lagrange) and overwrite the npz")
    args = ap.parse_args()

    if args.plot_only:
        data = dict(np.load(args.plot_only))
        if args.recompute:
            data = recompute(data, args.recompute, args.workers)
            np.savez(args.plot_only, **data)
            print(f"saved {args.plot_only}")
    else:
        axis, maps = compute(args.grid, args.eps, args.rhos, args.workers)
        curve = rho_curves(args.eps, args.rho_curve, args.workers)
        data = dict(axis=axis, eps=args.eps, rho_curve=np.array(args.rho_curve), curve_star=curve[0],
                    curve_triangle=curve[1], **maps)
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"hardness_eps{args.eps}.npz")
        np.savez(path, **data)
        print(f"saved {path}")
    from plot_hardness import plot_all
    plot_all(data, args.out)


if __name__ == "__main__":
    main()
