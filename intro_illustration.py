"""Opening illustration: a 3-arm, 1-constraint instance on which NOT knowing the constraint does not
increase the sample complexity.

Instance (in the spirit of the Hard instance of Lardy et al., 2025): arm 1 has the highest reward but is
infeasible (cost above gamma), arms 2 and 3 are feasible; the safely optimal policy mixes arms 1 and 2 on
the constraint boundary. Rewards and costs are jointly Gaussian with Lardy et al.'s variances
(0.1, 0.09) and a strong reward-cost correlation rho = 0.9.

Panel (c) compares, at the same eps = 0.05 (eps = 0 is impossible here: with the optimum on the constraint
boundary the unknown-constraint bound is infinite, Lemma 1 of the PRUNE paper):
  * BAI           unconstrained eps-good arm identification, reward observations only;
  * known         known constraint, eps-optimal safe policy, reward observations only (the model of
                  Carlsson et al., 2024), allocation omega in the simplex;
  * unknown       PRUNE's lower bound (Theorem 1): eps-optimal safe policy, (reward, cost) observations.
With the same targets, allocations AND observations, unknown >= known always (the alternative set only
grows). Here the unknown-constraint learner also observes the costs, and the correlated cost noise acts
as a control variate for the reward: that is why its bound can be below the known-constraint one. The
script also reports the known-constraint bound with cost feedback (reward variance sigma_r^2 (1 - rho^2)),
which is below PRUNE's, as the theory requires.

    python intro_illustration.py --eps 0.05                # results/intro_illustration_eps0.05.{json,png,pdf}
    python intro_illustration.py --eps 0.01
    python intro_illustration.py --eps 0.05 --plot-only    # re-plot from the json
    python intro_illustration.py --eps 0.05 --rho-curve    # also the rho sweep, intro_correlation_eps0.05.*

Panel (c) also shows PRUNE's bound at rho = 0. The rho sweep plots the known- and unknown-constraint bounds
of the same instance against the reward-cost correlation rho.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from prune.hardness import KnownEpsSolver, _kelley, _vertex_pieces, t_known_eps  # noqa: E402
from prune.lower_bound import AltSolver, characteristic_time, solve_sop  # noqa: E402

MU = np.array([0.80, 0.79, 0.50])          # mean rewards
COST = np.array([1.00, 0.75, 0.25])        # mean costs
GAMMA = 0.9                                # constraint: sum_a p_a E[C_a] <= gamma
VAR_R, VAR_C, RHO = 0.10, 0.09, 0.9        # Lardy et al.'s variances, strong reward-cost correlation
EPS = 0.05
B = (COST - GAMMA)[None, :]
RHOS = np.round(np.arange(-0.95, 0.951, 0.05), 2)   # correlation sweep of the second figure


def sigma_of(rho):
    return np.array([[VAR_R, rho * np.sqrt(VAR_R * VAR_C)], [rho * np.sqrt(VAR_R * VAR_C), VAR_C]])


SIGMA = sigma_of(RHO)

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
BLUE, BLUE_LIGHT = "#2a78d6", "#cde2fb"
POLICIES = "#f9d4bf"                                   # (cost, reward) of all policies, panel (a)
P_STAR = r"$\mathbfit{p}^*(\nu)$"
BAR = {"bai": "#b7b5ae", "known": "#6f6e69", "unknown_rho0": "#9ec5f4", "unknown": BLUE}
INFEASIBLE = "#ecebe7"


# ---------------------------------------------------------------------- bounds
def t_bai_eps(mu, eps, var):
    """eps-good arm identification (answers: pure arms), reward observations only. Returns (T, omega, arm)."""
    K = len(mu)
    E = np.eye(K)
    best = (0.0, None, None)
    for i in np.flatnonzero(mu > mu.max() - eps):
        w, v = _kelley(_vertex_pieces(mu, E[i], np.delete(E, i, axis=0), eps, var), K)
        if v > best[0]:
            best = (v, w, int(i))
    return 1.0 / best[0], best[1], best[2]


def _unknown(rho, eps, n_random=16):
    S = sigma_of(rho)
    return characteristic_time(MU, B, S, eps, solver=AltSolver(3, S, eps), n_random=n_random)


def compute(eps=EPS, n_random=16):
    p_star, opt = solve_sop(MU, B)
    T_bai, w_bai, arm = t_bai_eps(MU, eps, VAR_R)
    ks = KnownEpsSolver(3, 1, eps, VAR_R)
    known = characteristic_time(MU, B, ks.Sigma, eps, solver=ks, n_random=n_random)
    with ProcessPoolExecutor(max_workers=2) as ex:
        unknown, unknown0 = ex.map(_unknown, (RHO, 0.0), (eps, eps))
    return dict(mu=MU.tolist(), cost=COST.tolist(), gamma=GAMMA, var_r=VAR_R, var_c=VAR_C, rho=RHO, eps=eps,
                p_star=p_star.tolist(), opt=float(opt),
                bai=dict(T=T_bai, omega=w_bai.tolist(), answer=arm),
                known=dict(T=known["T"], omega=known["omega"].tolist(), p=known["p"].tolist()),
                unknown=dict(T=unknown["T"], omega=unknown["omega"].tolist(), p=unknown["p"].tolist()),
                unknown_rho0=dict(T=unknown0["T"], omega=unknown0["omega"].tolist(), p=unknown0["p"].tolist()),
                known_with_cost_feedback=t_known_eps(MU, B, eps, VAR_R * (1 - RHO ** 2)))


def _t_unknown(args):
    return _unknown(*args)["T"]


def rho_curve(eps=EPS, rhos=RHOS, workers=None):
    """PRUNE's bound for every rho; the known-constraint bounds need no sweep: reward-only feedback does not
    depend on rho, and with cost feedback the reward variance is sigma_r^2 (1 - rho^2), so T scales by it."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        unknown = list(ex.map(_t_unknown, [(float(x), eps) for x in rhos]))
    return dict(rho=[float(x) for x in rhos], unknown=unknown)


# ---------------------------------------------------------------------- simplex geometry
# barycentric -> plane: arm 1 on top, arm 2 bottom left, arm 3 bottom right
CORNERS = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])


def to_xy(p):
    return np.asarray(p) @ CORNERS


def clip(poly, a, b):
    """Sutherland-Hodgman: part of the barycentric polygon `poly` with a^T p <= b."""
    out = []
    for k in range(len(poly)):
        P, Q = poly[k], poly[(k + 1) % len(poly)]
        fp, fq = a @ P - b, a @ Q - b
        if fp <= 0:
            out.append(P)
        if fp * fq < 0:
            out.append(P + fp / (fp - fq) * (Q - P))
    return out


# ---------------------------------------------------------------------- plot
def _clean(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


CAPTIONS = {"a": "(a) Arms in the (cost, reward) plane", "b": "(b) Policies on the simplex",
            "c": "(c) Lower bound on stopping time"}


def _panel_a(ax, r):
    _clean(ax)
    mu, cost, gamma, eps = np.array(r["mu"]), np.array(r["cost"]), r["gamma"], r["eps"]
    xlo, xhi = 0.1, 1.15
    ylo, yhi = mu.min() - 0.08, mu.max() + 0.06
    ax.axvspan(gamma, xhi, color=INFEASIBLE, lw=0, zorder=0)
    ax.axvline(gamma, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text((xlo + gamma) / 2, yhi - 0.012, "feasible region", color=INK_2, fontsize=9, ha="center", va="top")
    hull = [np.array([cost[a], mu[a]]) for a in range(3)]              # (cost, reward) of all policies
    correct = clip(clip(hull, np.array([1.0, 0.0]), gamma), np.array([0.0, -1.0]), -(r["opt"] - eps))
    ax.add_patch(Polygon(np.array(hull), closed=True, facecolor=POLICIES, edgecolor="none", alpha=0.8, zorder=1))
    ax.add_patch(Polygon(np.array(correct), closed=True, facecolor=BLUE_LIGHT, edgecolor=BLUE, lw=0.6,
                         zorder=2))                                   # image of Pi(nu, eps)
    ps = np.array(r["p_star"])
    ax.plot(ps @ cost, ps @ mu, marker="*", ms=12, color=INK, mec=SURFACE, mew=0.8, zorder=4)
    ax.annotate(P_STAR, (ps @ cost, ps @ mu), xytext=(-7, 6), textcoords="offset points", fontsize=10, color=INK,
                ha="right")
    ax.scatter(cost, mu, s=46, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    offsets = [(6, 5), (-8, -4), (6, -12)]
    for a in range(3):
        ax.annotate(f"arm {a + 1}", (cost[a], mu[a]), xytext=offsets[a], textcoords="offset points",
                    fontsize=9, color=INK, ha="left" if a != 1 else "right")
    ticks = [t for t in (0.2, 0.4, 0.6, 0.8, 1.0) if abs(t - gamma) > 0.05] + [gamma]
    ax.set_xticks(ticks, [f"{t:.1f}" for t in ticks[:-1]] + ["γ"])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("mean cost", color=INK_2)
    ax.set_ylabel("mean reward", color=INK_2)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

def _panel_b(ax, r):
    _clean(ax)
    mu, cost, gamma, eps = np.array(r["mu"]), np.array(r["cost"]), r["gamma"], r["eps"]
    ps = np.array(r["p_star"])
    ax.set_xticks([])
    ax.set_yticks([])
    tri = [np.eye(3)[k] for k in range(3)]
    Bv = cost - gamma
    feasible = clip(tri, Bv, 0.0)
    infeasible = clip(tri, -Bv, 0.0)
    answers = clip(feasible, -mu, -(r["opt"] - eps))           # Pi(nu, eps)
    ax.add_patch(Polygon(CORNERS, facecolor=POLICIES, edgecolor="none", alpha=0.8, zorder=0.5))  # as in (a)
    ax.add_patch(Polygon(to_xy(np.array(infeasible)), facecolor=INFEASIBLE, edgecolor="none", zorder=0.7))
    ax.add_patch(Polygon(to_xy(np.array(answers)), facecolor=BLUE_LIGHT, edgecolor="none", zorder=1))
    ax.add_patch(Polygon(CORNERS, fill=False, edgecolor=INK_2, lw=1, zorder=2))
    line = to_xy(np.array([q for q in infeasible if abs(Bv @ q) < 1e-12]))
    ax.plot(line[:, 0], line[:, 1], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=2)
    for a, (dx, dy, ha) in enumerate([(0, 0.05, "center"), (-0.03, -0.07, "center"), (0.03, -0.07, "center")]):
        ax.text(CORNERS[a, 0] + dx, CORNERS[a, 1] + dy, f"arm {a + 1}", fontsize=9, color=INK, ha=ha)
    anchor = to_xy(np.array(answers)).mean(0)                   # inside Pi(nu, eps) for any eps
    ax.annotate("Π(ν, ε)", anchor, xytext=(40, -40), textcoords="offset points",
                fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))
    ax.annotate("infeasible", to_xy(np.array([0.9, 0.06, 0.04])), xytext=(26, 4), textcoords="offset points",
                fontsize=8, color=INK_2, arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8))
    ax.plot(*to_xy(ps), marker="*", ms=12, color=INK, mec=SURFACE, mew=0.8, zorder=5)
    ax.annotate(P_STAR, to_xy(ps), xytext=(7, 2), textcoords="offset points", fontsize=10, color=INK)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.12, np.sqrt(3) / 2 + 0.1)
    ax.set_aspect("equal")


def _panel_c(ax, r, ylabel=None):
    _clean(ax)
    eps = r["eps"]
    if "unknown_rho0" not in r:                  # PRUNE's bound without correlation, from the rho sweep
        c = r["rho_curve"]
        r["unknown_rho0"] = dict(T=c["unknown"][int(np.argmin(np.abs(np.array(c["rho"]))))])
    keys = ["bai", "known", "unknown_rho0", "unknown"]
    labels = ["BAI\n(no\nconstraint)", "known\nconstraint", "unknown\nconstraint\nρ = 0",
              f"unknown\nconstraint\nρ = {r['rho']}"]
    vals = [r[k]["T"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, vals, width=0.6, color=[BAR[k] for k in keys], edgecolor=SURFACE, linewidth=2, zorder=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.015, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, labels, fontsize=8, color=INK_2)
    ax.set_ylabel(ylabel or f"Lower Bound", color=INK_2, fontsize=10)
    ax.set_ylim(0, max(vals) * 1.12)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)


def plot(r, out):
    eps = r["eps"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), facecolor=SURFACE, constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.05, 1.0, 0.85]))
    for ax, key, panel in zip(axes, "abc", (_panel_a, _panel_b, _panel_c)):
        panel(ax, r)
        ax.set_title(CAPTIONS[key], fontsize=10, color=INK, loc="left")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"intro_illustration_eps{eps}.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


# Times for the paper (AISTATS): Times New Roman is not installed here, so Nimbus Roman, the URW Times clone
# that LaTeX's times / mathptmx packages use, with STIX (Times-style) as fallback for missing glyphs.
TIMES = {"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
         "mathtext.fontset": "custom", "mathtext.rm": "Nimbus Roman", "mathtext.it": "Nimbus Roman:italic",
         "mathtext.bf": "Nimbus Roman:bold", "mathtext.bfit": "Nimbus Roman:bold:italic",
         "mathtext.fallback": "stix", "pdf.fonttype": 42}
SEPARATE_SIZE = {"a": (3.9, 3.4), "b": (3.6, 3.4), "c": (3.3, 3.4)}


def _embed_fonts(path):
    """Rewrite the PDF with Ghostscript: fonts embedded and subset as clean Type 1C (no Type 3)."""
    tmp = path + ".tmp.pdf"
    ok = os.system(f'gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress '
                   f'-dEmbedAllFonts=true -dSubsetFonts=true -dCompatibilityLevel=1.5 '
                   f'-sOutputFile="{tmp}" "{path}"') == 0
    if ok:
        os.replace(tmp, path)
    else:
        print(f"warning: Ghostscript failed, {path} keeps matplotlib's font embedding")


def plot_separate(r, out):
    """The three panels as separate files (Times fonts), each with its caption below instead of a title."""
    eps = r["eps"]
    with plt.rc_context(TIMES):
        for key, panel in zip("abc", (_panel_a, _panel_b, _panel_c)):
            fig, ax = plt.subplots(figsize=SEPARATE_SIZE[key], facecolor=SURFACE, constrained_layout=True)
            if key == "c":
                panel(ax, r, ylabel="Lower Bound")
            else:
                panel(ax, r)
            fig.supxlabel(CAPTIONS[key], fontsize=10, color=INK)
            base = os.path.join(out, f"intro_illustration_eps{eps}_{key}")
            fig.savefig(base + ".png", dpi=200, facecolor=SURFACE)
            fig.savefig(base + ".pdf", facecolor=SURFACE)
            plt.close(fig)
            _embed_fonts(base + ".pdf")


def plot_rho(r, out):
    with plt.rc_context(TIMES):
        _plot_rho(r, out)
    _embed_fonts(os.path.join(out, f"intro_correlation_eps{r['eps']}.pdf"))


def _plot_rho(r, out):
    c = r["rho_curve"]
    rho, unknown = np.array(c["rho"]), np.array(c["unknown"])
    known = r["known"]["T"]
    feedback = known * (1 - rho ** 2)
    fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor=SURFACE, constrained_layout=True)
    _clean(ax)
    easier = unknown < known
    #ax.fill_between(rho, unknown, known, where=easier, interpolate=True, color=BLUE_LIGHT, lw=0, zorder=1,
                    #label="unknown below known")
    ax.plot(rho, feedback, color=MUTED, lw=1.5, ls=(0, (4, 3)), zorder=2,
            label="known constraint, reward + cost feedback")
    ax.axhline(known, color=BAR["known"], lw=2, zorder=3, label="known constraint, reward feedback")
    ax.plot(rho, unknown, color=BLUE, lw=2, zorder=4, label="unknown constraint")
    i = int(np.argmin(np.abs(rho - r["rho"])))
    ax.plot(rho[i], unknown[i], marker="o", ms=8, color=BLUE, mec=SURFACE, mew=1.5, ls="none", zorder=5,
            label=f"instance of Fig. 1 (ρ = {r['rho']})")
    top = max(unknown.max(), known) * 1.08
    h, l = ax.get_legend_handles_labels()
    order = [2, 1, 0, 3]                            # unknown, known, known + cost feedback, then the marker
    ax.legend([h[k] for k in order], [l[k] for k in order], loc="lower center", frameon=False, fontsize=13,
              labelcolor=INK_2, handlelength=2.2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, top)
    ax.set_xlabel("reward–cost correlation ρ", color=INK_2,fontsize=10)
    ax.set_ylabel("Lower Bound", color=INK_2,fontsize=10)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    #ax.set_title(f"Known vs unknown constraint across correlations (BAI: {r['bai']['T']:.0f})", fontsize=10,
                 #color=INK, loc="left")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"intro_correlation_eps{r['eps']}.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--rho-curve", action="store_true", help="also sweep rho (second figure)")
    args = ap.parse_args()
    path = os.path.join(args.out, f"intro_illustration_eps{args.eps}.json")
    if args.plot_only:
        with open(path) as f:
            r = json.load(f)
    else:
        r = compute(args.eps)
    if args.rho_curve and "rho_curve" not in r:
        r["rho_curve"] = rho_curve(args.eps)
    os.makedirs(args.out, exist_ok=True)
    with open(path, "w") as f:
        json.dump(r, f, indent=1)
    print(f"p* = {np.round(r['p_star'], 3)}, OPT = {r['opt']:.4f}")
    for k in ("bai", "known", "unknown_rho0", "unknown"):
        if k in r:
            print(f"{k:12s} T* = {r[k]['T']:9.2f}  omega* = {np.round(r[k]['omega'], 3)}")
    print(f"known constraint WITH cost feedback: T* = {r['known_with_cost_feedback']:.2f}")
    plot(r, args.out)
    print(f"saved {args.out}/intro_illustration_eps{args.eps}.{{json,png,pdf}}")
    plot_separate(r, args.out)
    print(f"saved {args.out}/intro_illustration_eps{args.eps}_{{a,b,c}}.{{png,pdf}} (Times, captions below)")
    if "rho_curve" in r:
        c = r["rho_curve"]
        easier = [x for x, u in zip(c["rho"], c["unknown"]) if u < r["known"]["T"]]
        print(f"unknown < known (reward feedback) for rho in {easier}")
        plot_rho(r, args.out)
        print(f"saved {args.out}/intro_correlation_eps{args.eps}.{{png,pdf}}")


if __name__ == "__main__":
    main()
