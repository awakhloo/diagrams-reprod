"""Figures 2-4 of the paper: errors of the fixed-connectivity formulas against network size N.
  fig2_phi.pdf, fig3_x.pdf : off-diagonal errors of C_ij(omega = 0), C_ij(tau = 0), R_ij(tau = 2) and of the four-point function
                             (all sites distinct, equal times) for phi and x: per-N median over the draws of J (black) and every draw (grey).
  fig4_mixed.pdf           : diagonal errors; errors at tau != 0; errors at omega != 0; each median series divided by its value at N = 10000.
Every plotted value is a raw RMSE, sqrt of the mean squared difference between simulation and theory (nothing subtracted), read from
the per-draw metrics files results/{twopoint,response,fourpoint,spectral_direct}_metrics_N*_seed*_sigma2_9.json. The plotted numbers
are also written to figures/figure_data_figs2-4.txt. Each figure is 14 in wide, to be included at 7.0 in.
usage: python figures/make_figs2-4.py      (run from anywhere)"""
import os, sys, glob, re, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker, matplotlib.lines

# ---------------- parameters ----------------
OUT = "figures"
SIGMA2 = 9.0
N_MIN = 250                    # smallest N shown (response, four-point and Fourier-domain tests start at 250)
SEED_MAX_8000 = int(sys.argv[1]) if len(sys.argv) > 1 else 3   # keep only seeds <= this at N >= 8000 (three draws per N there in every experiment); 0 keeps every seed
TAU_R = 2.0                    # lag of the response panels (R_ij(0) = 0 off the diagonal; the run stored tau = 1, 2, 4)
TAUS_C = (2, 4)                # nonzero lags of C in the tau != 0 panel (the runs stored 0, 2, 4)
TAUS_R = (1.0, 2.0, 4.0)       # lags of R in the tau != 0 panel (all stored lags)
LAGS_4 = (0.0, 2.0)            # four-point lags in the tau != 0 panel (all stored lags)
WINDOW, TAU_MAX_SPEC = "rect", 100.0   # Fourier-domain panels: rectangular window on the simulation, lag horizon
OMEGAS = (0.25, 0.5)           # nonzero frequencies of the omega != 0 panel (all stored nonzero frequencies)
PANEL = 3.5                    # inches per panel
DPI = 200
FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_PANEL = 14, 16, 16, 10.75, 17   # font sizes: ticks, axis labels, titles, legends, panel letters
YLABEL_PAD = 7
REF_STYLE = {-1.0: "--", -0.5: "-.", -1.5: ":"}   # dashed N^-1, dash-dot N^-1/2, dotted N^-3/2
BLACK, LIGHT = "black", "0.75"
SERIES_COLORS = {"C": "#fa8072", "R": "#191970", "4": "#808000"}       # tau != 0 panels: salmon, midnight blue, olive
DIAG_COLORS = {"phi": "#808080", "x": "#6a4fbf"}                       # diagonal panel: grey (phi), purple (x)
OMEGA_COLORS = {"phi": "#fa8072", "x": "#191970"}                      # omega != 0 panel: salmon (phi), midnight blue (x), as C and R in B-C
MARKERS = ("o", "s", "^")
XLIM = (180, 13000)
# --------------------------------------------

plt.rcParams.update({"font.size": FS_TICK, "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
                     "legend.frameon": False, "figure.dpi": DPI, "savefig.dpi": DPI, "lines.linewidth": 1.4,
                     "xtick.direction": "in", "ytick.direction": "in"})
SYM = {"x": "x", "phi": r"\phi"}
LOG = []


def log(s):
    print(s); LOG.append(s)


# ---------------- data ----------------
def keep(rows, name):
    dropped = [r for r in rows if SEED_MAX_8000 and r["N"] >= 8000 and r["seed"] > SEED_MAX_8000]
    out = [r for r in rows if r["N"] >= N_MIN and r not in dropped]
    log(f"{name}: {len(rows)} draws in the files, {len(out)} with N >= {N_MIN}" + (f" and seeds <= {SEED_MAX_8000} at N >= 8000" if dropped else "") + "; all of them used; status of the used draws: "
        + ", ".join(f"N={r['N']} seed={r['seed']} {r['status']}" for r in out if r["status"] != "ok") + (" (all others ok)" if any(r["status"] != "ok" for r in out) else "all ok"))
    if dropped:
        log(f"    seeds > {SEED_MAX_8000} at N >= 8000 set aside: " + ", ".join(f"N={r['N']} seed={r['seed']}" for r in dropped))
    return out


def metrics(prefix):
    rows = []
    for fn in sorted(glob.glob(f"results/{prefix}_metrics_N*_seed*_sigma2_{SIGMA2:g}.json")):
        if re.search(rf"{prefix}_metrics_N(\d+)_seed(\d+)_sigma2_{SIGMA2:g}\.json$", fn):
            r = json.load(open(fn))
            if "status" not in r:    # the two-point files record the status only when the theory is undefined; same rule as the other analyses
                r["status"] = "non-chaotic" if (r["lyap_min"] <= 0 or r["decorr"] > 0.5) else ("theory undefined" if r["strip_width"] <= 0 else "ok")
            rows.append(r)
    return keep(rows, prefix)


TWO, RESP, FOUR, SPECT = metrics("twopoint"), metrics("response"), metrics("fourpoint"), metrics("spectral_direct")


def series(rows, key, name):
    """Per-N median and per-draw values of sqrt(rows[key]) over the draws whose file holds the key; everything logged."""
    have = [r for r in rows if key in r]
    Ns = sorted({r["N"] for r in have})
    draws = {N: [np.sqrt(r[key]) for r in have if r["N"] == N] for N in Ns}
    med = np.array([np.median(draws[N]) for N in Ns])
    log(f"{name:40s} key {key}")
    for N, m in zip(Ns, med):
        log(f"    N = {N:5d}  n = {len(draws[N])}  median = {m:.4e}   draws: " + " ".join(f"{v:.3e}" for v in draws[N]))
    missing = [r for r in rows if key not in r]
    if missing:
        log("    no such metric in the file (no theory: strip <= 0, or analysis skipped): " + ", ".join(f"N={r['N']} seed={r['seed']} ({r['status']})" for r in missing))
    assert all(v >= 0 for N in Ns for v in draws[N]), "raw MSE must be >= 0"
    return np.array(Ns, float), med, draws


def two_key(f, tau, part):             # two-point metrics (scripts/analyze_twopoint.py), raw
    return f"{f}_tau{tau}_mse_{part}_plain"


def resp_key(f, tau, part):            # response metrics (scripts/analyze_response.py), raw; theory with the DMFT gain
    return f"{f}_tau{tau:g}_mse_{part}_plain"


def four_key(f, lag):                  # four-point metrics (scripts/run_fourpoint.py), raw, Wick term from the THEORY two-point functions
    return f"{f}_lag{0 if lag == 0 else int(round(lag / FOUR[0]['dt']))}_mse_raw_th"


def spec_key(f, omega):                # Fourier-domain metrics (scripts/analyze_spectral_direct.py), raw, off diagonal
    return f"{f}_{WINDOW}_t{TAU_MAX_SPEC:g}_w{omega:g}_mse_off_plain"


# ---------------- panels ----------------
def ref_line(ax, Ns, y0, alpha, N0):
    x = np.array([Ns[0], Ns[-1]], float)
    txt = {-1.0: r"$N^{-1}$", -0.5: r"$N^{-1/2}$", -1.5: r"$N^{-3/2}$"}[alpha]
    ax.plot(x, y0 * (x / N0) ** alpha, REF_STYLE[alpha], color=BLACK, lw=1.0, label=txt, zorder=2)


def style(ax, title, ylabel, ncol=1, **legend_kw):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*XLIM)
    lo, hi = ax.get_ylim()
    if np.log10(hi / lo) < 1.5:      # short range: plain 1-2-5 labels instead of matplotlib's "4 x 10^-1" minor labels
        ticks = [m * 10.0 ** e for e in range(-6, 3) for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$N$", fontsize=FS_LABEL); ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=YLABEL_PAD)
    if title:
        ax.set_title(title, fontsize=FS_TITLE)
    ax.set_box_aspect(1)
    legend_kw.setdefault("loc", "upper right")
    ax.legend(fontsize=FS_LEGEND if ncol == 1 else FS_LEGEND - 1, ncol=ncol, handlelength=1.8, labelspacing=0.3, columnspacing=0.8, **legend_kw)


def scaling_panel(ax, rows, key, title, alpha, ylabel):
    Ns, med, draws = series(rows, key, title)
    for N in Ns:
        ax.plot([N] * len(draws[N]), draws[N], "o", ms=3, color=LIGHT, mec="none", zorder=1)
    ax.plot(Ns, med, "o-", color=BLACK, ms=4.5, zorder=3)
    ref_line(ax, Ns, med[-1], alpha, Ns[-1])
    style(ax, title, ylabel)


def normalized_panel(ax, items, alphas, title, ylabel, ncol):
    """items = (rows, key, label, colour, marker): each median series divided by its value at the largest N; reference lines
    through (N_max, 1)."""
    first = last = None
    for rows, key, label, colour, mk in items:
        Ns, med, _ = series(rows, key, f"{title}: {label}")
        ax.plot(Ns, med / med[-1], mk + "-", color=colour, ms=4.5, label=label, zorder=3)
        first, last = (Ns[0], Ns[-1]) if first is None else (first, last)
        assert Ns[0] == first and Ns[-1] == last, "all series must span the same N range for a common anchor"
    for alpha in alphas:
        ref_line(ax, np.array([first, last]), 1.0, alpha, last)
    ax.set_yscale("log"); ax.set_ylim(top=ax.get_ylim()[1] * 4)      # room for the legend above the curves
    style(ax, title, ylabel, ncol=ncol, loc="upper right")


def panel_letters(fig, axes):
    """Bold A, B, ... at the top left of each panel, at the left edge of the panel's tick labels / y label, on the title baseline."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    for ax, letter in zip(axes, "ABCD"):
        left_pt = (ax.get_tightbbox(ren).x0 - ax.get_window_extent(ren).x0) * 72.0 / fig.dpi     # <= 0
        ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(left_pt, plt.rcParams["axes.titlepad"]),
                    textcoords="offset points", ha="left", va="baseline", fontsize=FS_PANEL, fontweight="bold")


# ---------------- figures ----------------
def fig_field(f):
    """fig2 (phi) / fig3 (x): omega-domain, time-domain, response and four-point off-diagonal errors; y label on the first panel."""
    F = SYM[f]
    specs = [(SPECT, spec_key(f, 0.0), rf"$C^{{{F}}}_{{ij}}(\omega)$ (off diag)", -1.0),
             (TWO, two_key(f, 0, "off"), rf"$C^{{{F}}}_{{ij}}(\tau)$ (off diag)", -1.0),
             (RESP, resp_key(f, TAU_R, "off"), rf"$R^{{{F}}}_{{ij}}(\tau)$ (off diag)", -1.0),
             (FOUR, four_key(f, 0.0), rf"$C^{{{F}}}_{{ijkl}}(\tau_1,\tau_2,\tau_3)$ (all distinct)", -1.5)]
    return [lambda ax, s=s, i=i: scaling_panel(ax, *s, "Error" if i == 0 else "") for i, s in enumerate(specs)]


def fig_mixed():
    diag = [(TWO, two_key("phi", 0, "diag"), r"$C^{\phi}_{ii}(0)$", DIAG_COLORS["phi"], "o"),
            (TWO, two_key("x", 0, "diag"), r"$C^{x}_{ii}(0)$", DIAG_COLORS["x"], "o"),
            (RESP, resp_key("phi", TAU_R, "diag"), rf"$R^{{\phi}}_{{ii}}({TAU_R:g})$", DIAG_COLORS["phi"], "s"),
            (RESP, resp_key("x", TAU_R, "diag"), rf"$R^{{x}}_{{ii}}({TAU_R:g})$", DIAG_COLORS["x"], "s")]
    def tau_ne_0(f):
        F = SYM[f]
        return ([(TWO, two_key(f, t, "off"), rf"$C^{{{F}}}_{{ij}}({t:g})$", SERIES_COLORS["C"], mk) for t, mk in zip(TAUS_C, MARKERS)]
                + [(RESP, resp_key(f, t, "off"), rf"$R^{{{F}}}_{{ij}}({t:g})$", SERIES_COLORS["R"], mk) for t, mk in zip(TAUS_R, MARKERS)]
                + [(FOUR, four_key(f, l), rf"$C^{{{F}}}_{{ijkl}}({l:g})$", SERIES_COLORS["4"], mk) for l, mk in zip(LAGS_4, MARKERS)])
    omega = [(SPECT, spec_key(f, w), rf"$C^{{{SYM[f]}}}_{{ij}}({w:g})$", OMEGA_COLORS[f], mk) for f in ("phi", "x") for w, mk in zip(OMEGAS, MARKERS)]
    return [lambda ax: normalized_panel(ax, diag, (-0.5, -1.0), "Diagonal errors", "Error (normalized)", 1),
            lambda ax: normalized_panel(ax, tau_ne_0("phi"), (-1.0, -1.5), r"Errors with $\tau \neq 0$ ($\phi$)", "", 2),
            lambda ax: normalized_panel(ax, tau_ne_0("x"), (-1.0, -1.5), r"Errors with $\tau \neq 0$ ($x$)", "", 2),
            lambda ax: normalized_panel(ax, omega, (-1.0,), r"Errors with $\omega \neq 0$", "", 2)]


def render(panels, name):
    fig, axes = plt.subplots(1, 4, figsize=(4 * PANEL, PANEL), constrained_layout=True, squeeze=False)
    for ax, p in zip(axes.ravel(), panels):
        p(ax)
    panel_letters(fig, axes.ravel())
    for ext in ("png", "pdf"):       # this order: the layout is recomputed at every save, so it is part of the figure
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    log(f"saved {OUT}/{name}.png / .pdf")


for f, name in (("phi", "fig2_phi"), ("x", "fig3_x")):
    log(f"\n################ {name} ################")
    render(fig_field(f), name)
log("\n################ fig4_mixed ################")
render(fig_mixed(), "fig4_mixed")
open(os.path.join(OUT, "figure_data_figs2-4.txt"), "w").write("\n".join(LOG) + "\n")
print(f"\nnumbers written to {OUT}/figure_data_figs2-4.txt")
