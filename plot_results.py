"""Box plots of stopping times per algorithm with the lower bound, in the style of Fig. 3(c)/(d) of
Carlsson et al. (2024).

Example:
    python plot_results.py --instances carlsson-easy carlsson-hard --algos prune lagex lagex-paper \\
        --eps 0.05 --delta 0.1 --out results/carlsson_fig3.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# categorical slots 1-3 of the reference palette, in fixed order; ink and surface tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8a5cd6",
          "#17a3b3", "#c73e55", "#8fae2a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"prune": "PRUNE", "lagex": "LaGEx",
          "tas": "TaS", "toptwo": "TopTwo-TCI", "cge_anytime": "CGE", 
          "cge_relaxed_anytime": "CGE", "ctns_anytime": "CTnS", 
          "ctns_relaxed_anytime": "CTnS", "uniform_anytime": "Uniform",
          "uniform_relaxed_anytime": "Uniform-relaxed", "tas_relaxed": "TaS",
          "toptwo_relaxed": "TopTwo-TCI", "uniform-prune": "Uniform", "uniform-cbai": "Uniform", "uniform-cbai_relaxed": "Uniform",
          "prune_lazy10": "PRUNE", "prune-fw": "PRUNE", "ctns_relaxed_anytime_lazy10": "CTnS"}
TITLES = {"carlsson-easy": "Easy: μ = (1, 0.5, 0.4, 0.95, 0.8)",
          "carlsson-hard": "Hard: μ = (1, 0.5, 0.4, 0.4, 0.5)",
          "easy": "Easy (Lardy et al.)", "hard": "Hard (Lardy et al.)", "imdb": "IMDB-50K, 12 movies"}


def load(folder, algo, inst, eps, delta, threshold):
    path = os.path.join(folder, f"{algo}_{inst}_eps{eps}_delta{delta}_{threshold}.json")
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", default=["carlsson-easy", "carlsson-hard"])
    ap.add_argument("--algos", nargs="+", default=["prune", "lagex", "lagex-paper"])
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--threshold", default="stylized")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/carlsson_fig3.png")
    ap.add_argument("--title", default="Stopping times")
    ap.add_argument("--no-titles", action="store_true", help="no figure or panel titles")
    ap.add_argument("--known-bound", default=None,
                    help="also draw the known-constraint bound oracle_known[KEY] log(1/delta) stored in the result "
                         "files of ctns / cge / uniform, e.g. anytime-relaxed (eps-optimal vertex, omega in F)")
    ap.add_argument("--no-errors", action="store_true", help="algorithm names only under the boxes")
    ap.add_argument("--times", action="store_true",
                    help="Times fonts (Nimbus Roman) and clean font embedding in a .pdf output, as for AISTATS")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    if args.times:
        from intro_illustration import TIMES
        plt.rcParams.update(TIMES)
    fig, axes = plt.subplots(1, len(args.instances), figsize=(4.6 * len(args.instances), 4.2),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, inst in zip(axes, args.instances):
        ax.set_facecolor(SURFACE)
        data = [load(args.results, a, inst, args.eps, args.delta, args.threshold) for a in args.algos]
        taus = [np.array([r["tau"] for r in d["runs"]]) for d in data]
        errors = [np.mean([not r["correct"] for r in d["runs"]]) for d in data]
        bp = ax.boxplot(taus, widths=0.5, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK_2, linewidth=1), capprops=dict(color=INK_2, linewidth=1),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                                        markeredgecolor=INK_2, alpha=0.5))
        for patch, color in zip(bp["boxes"], SERIES):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(2)
        lb = data[0]["oracle"]["T"] * np.log(1 / args.delta)
        if args.known_bound is None:
            ax.axhline(lb, color=INK_2, linestyle="--", linewidth=1.2, label="lower bound T*(ν,ε) log(1/δ)")
        else:
            ax.axhline(lb, color=INK_2, linestyle="--", linewidth=1.2,
                       label="lower bound T*(ν,ε) log(1/δ), unknown constraints")
            known = next(d["oracle_known"][args.known_bound] for d in data if "oracle_known" in d)
            ax.axhline(known * np.log(1 / args.delta), color=INK_2, linestyle=":", linewidth=1.5,
                       label="lower bound T*(ν,ε) log(1/δ), known constraints")
        ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_2)
        ax.set_xticks(range(1, len(args.algos) + 1),
                      [LABELS.get(a, a) + ("" if args.no_errors else f"\nerror {e:.1%}")
                       for a, e in zip(args.algos, errors)], rotation=45)
        # ax.set_xticks(range(1, len(args.algos) + 1),[LABELS.get(a, a) for a in args.algos],rotation=45)
        if not args.no_titles:
            ax.set_title(TITLES.get(inst, inst), fontsize=10.5, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(0, 1.15 * max(t.max() for t in taus))      # headroom for the legend
    axes[0].set_ylabel("stopping time τ")
    n = len(data[0]["runs"])
    if not args.no_titles:
        fig.suptitle(f"{args.title} (ε = {args.eps}, δ = {args.delta}, {n} runs)",
                     fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    if args.times and args.out.endswith(".pdf"):
        from intro_illustration import _embed_fonts
        _embed_fonts(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
