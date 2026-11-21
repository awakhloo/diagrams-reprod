"""Figure 1 of the paper: three example pairs (i, j) of one network (N = 10000, seed 1, sigma^2 = 9). Panel A: the correlation function
C^phi_ij(tau) = <phi_i(t + tau) phi_j(t)> on [-40, 40]; panel B: the response function R^phi_ij(tau) on [0, 10] (kick on j, read-out on i).
Simulation solid, theory dashed, one colour per pair; panel B starts at the exact value R_ij(0) = 0.
Data: results/dense_cc_sigma2_9.npz (scripts/dense_cc.py) and results/dense_rc_N10000_seed1_sigma2_9.npz (scripts/dense_rc.py); the stored
unordered pair is flipped in tau when needed, C_ij(tau) = C_ji(-tau). Numbers logged to figures/figure_data_fig1.txt.
The file is 6.8 in wide (to be included at 3.4 in) and cropped vertically to the ink.
usage: python figures/make_fig1.py      (run from anywhere)"""
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines, matplotlib.transforms, matplotlib.ticker

# ---------------- parameters ----------------
OUT, NAME = "figures", "fig1_examples_rc_v2_noylabel"
CC, RC = "results/dense_cc_sigma2_9.npz", "results/dense_rc_N10000_seed1_sigma2_9.npz"
N, FIELD = 10000, "phi"
COLORS = ("#6a4fbf", "#2e8b57", "#e8a0bf")                       # purple, green, soft pink
SITE_PAIRS = ((4880, 8515), (9772, 4880), (4762, 7130))          # the plotted pairs as sites (i, j)
LEGEND_LOC_R = "lower right"                                     # panel B legend
TAU_MAX_C = 40.0                                                 # panel A window, +- TAU_MAX_C
PANEL = 3.4                                                      # inches per panel: the figure is 6.8 in wide
DPI = 200
FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND, FS_PANEL = 14, 16, 16, 10.75, 17   # font sizes: ticks, axis labels, titles, legends, panel letters
Y_NBINS = 6                                                      # y ticks
W_PAD = 0.02                                                     # constrained-layout pad in inches around each panel
X_TICKS = {"C": (-25, 0, 25), "R": (0, 5, 10)}                  # x ticks
RIGHT_TO_LEFT = 0.5                                              # right margin = RIGHT_TO_LEFT x left margin (the left one is mostly tick numbers)
V_CROP_PAD = 0.04                                                # the saved figure is cropped vertically to the ink plus this pad (inches)
BLACK = "black"
# --------------------------------------------

plt.rcParams.update({"font.size": FS_TICK, "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
                     "legend.frameon": False, "figure.dpi": DPI, "savefig.dpi": DPI, "lines.linewidth": 1.4,
                     "xtick.direction": "in", "ytick.direction": "in"})
SYM = {"x": "x", "phi": r"\phi"}[FIELD]
LOG = []


def log(s):
    print(s); LOG.append(s)


cc = np.load(CC); rc = np.load(RC)
taus_c, rows, pairs = cc[f"N{N}_taus"], cc[f"N{N}_rows"], cc[f"N{N}_pairs"]
sim_c = 0.5 * (cc[f"N{N}_{FIELD}_h1"].astype(float) + cc[f"N{N}_{FIELD}_h2"].astype(float))
th_c = cc[f"N{N}_{FIELD}_th"].astype(float)
sd_c = cc[f"N{N}_{FIELD}_sd"]
meta_c, meta_r = json.loads(str(cc[f"N{N}_meta"])), json.loads(str(rc["meta"]))
taus_r, sites, rows_r = rc["taus"], rc["sites"], rc["rows"]
assert np.array_equal(rows, rows_r), "the response data must use the dense rows of the correlation run"
assert meta_r["N"] == N == meta_c["N"] and meta_r["seed"] == meta_c["seed"] and meta_r["sigma2"] == meta_c["sigma2"], "same draw required"
sim_r = 0.5 * (rc[f"{FIELD}_h1"].astype(float) + rc[f"{FIELD}_h2"].astype(float))     # [k, a, b] = R_{rows[b], sites[a]}(taus_r[k])
th_r, sd_r = rc[f"{FIELD}_th"].astype(float), rc[f"{FIELD}_sd"]
pair_index = {(int(a), int(b)): p for p, (a, b) in enumerate(pairs)}                   # positions in rows, a <= b
pos = {int(r): k for k, r in enumerate(rows)}
tau_max_r = float(taus_r[-1])
log(f"C data: N = {N}, seed {meta_c['seed']}, T_total = {meta_c['T_total']:.3g}, B = {meta_c['B']}; R data: {meta_r['n_kicks']} kicks per site, "
    f"eps = {meta_r['eps']:g}, B = {meta_r['B']}, tau_win = {meta_r['tau_win']:g}, sites {sites.tolist()}")

win_c = np.abs(taus_c) <= TAU_MAX_C + 1e-9
t_c = taus_c[win_c]
win_r = taus_r <= tau_max_r + 1e-9
t_r = np.concatenate([[0.0], taus_r[win_r]])                     # R_ij(0) = 0 off the diagonal, exactly

log(f"\n################ {NAME} ################")
fig, (axA, axB) = plt.subplots(1, 2, figsize=(2 * PANEL, PANEL), constrained_layout=True)
fig.get_layout_engine().set(w_pad=W_PAD, h_pad=W_PAD, wspace=0, hspace=0)
loA, hiA, loB, hiB = np.inf, -np.inf, np.inf, -np.inf
for (i, j), c in zip(SITE_PAIRS, COLORS):
    assert i != j and i in pos and j in pos, (i, j)
    a_i, a_j = pos[i], pos[j]
    if a_i <= a_j:                                            # stored pair is C_{rows[a], rows[b]} with a <= b
        p, flipped = pair_index[(a_i, a_j)], False; sC, tC = sim_c[p, win_c], th_c[p, win_c]
    else:
        p, flipped = pair_index[(a_j, a_i)], True; sC, tC = sim_c[p, win_c][::-1], th_c[p, win_c][::-1]     # C_ij(tau) = C_ji(-tau)
    a, b = int(np.where(sites == j)[0][0]), int(np.where(rows == i)[0][0])      # kick j (column), read i (row)
    sR = np.concatenate([[0.0], sim_r[win_r, a, b]]); tR = np.concatenate([[0.0], th_r[win_r, a, b]])
    log(f"pair {p}{' (flipped)' if flipped else ''}: i = {i}, j = {j}, colour {c}; C: amplitude {np.abs(sC).max():.4f} at tau {t_c[np.argmax(np.abs(sC))]:+.2f}, "
        f"RMS(sim - th) {np.sqrt(np.mean((sC - tC) ** 2)):.2e}, noise sd {sd_c[win_c].mean():.2e}; R (kick {j}, read {i}): amplitude sim {np.abs(sR).max():.4e} "
        f"th {np.abs(tR).max():.4e} at tau {t_r[np.argmax(np.abs(tR))]:.2f}, RMS(sim - th) {np.sqrt(np.mean((sR[1:] - tR[1:]) ** 2)):.2e}, "
        f"noise sd {sd_r[win_r][0]:.1e} (tau = {taus_r[0]:g}) .. {sd_r[win_r][-1]:.1e} (tau = {tau_max_r:g})")
    axA.plot(t_c, sC, "-", color=c); axA.plot(t_c, tC, "--", color=c)
    axB.plot(t_r, sR, "-", color=c); axB.plot(t_r, tR, "--", color=c)
    loA, hiA = min(loA, sC.min(), tC.min()), max(hiA, sC.max(), tC.max())
    loB, hiB = min(loB, sR.min(), tR.min()), max(hiB, sR.max(), tR.max())
for ax, lo, hi, t, lab in ((axA, loA, hiA, t_c, "C"), (axB, loB, hiB, t_r, "R")):
    pad = 0.08 * (hi - lo)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_title(rf"Example ${lab}^{{{SYM}}}_{{ij}}(\tau)$", fontsize=FS_TITLE)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=Y_NBINS, steps=[1, 2.5, 5, 10]))
    ax.set_xticks(X_TICKS[lab])
    log(f"panel {lab}: y ticks " + " ".join(f"{v:g}" for v in ax.get_yticks() if lo - pad <= v <= hi + pad))
    if lab == "R":
        ax.legend(handles=[matplotlib.lines.Line2D([], [], color=BLACK, ls="-", label="sim."),
                           matplotlib.lines.Line2D([], [], color=BLACK, ls="--", label="theory")],
                  fontsize=FS_LEGEND, loc=LEGEND_LOC_R, handlelength=1.8, labelspacing=0.3)
    ax.set_box_aspect(1)
# right margin: shrink the layout rect on the right until the right margin is RIGHT_TO_LEFT times the left one (two corrections)
W = fig.get_size_inches()[0]; r = 0.0
for _ in range(2):
    fig.canvas.draw()
    bA, bB = axA.get_window_extent(fig.canvas.get_renderer()), axB.get_window_extent(fig.canvas.get_renderer())
    left, right = bA.x0 / fig.dpi, W - bB.x1 / fig.dpi
    r = max(0.0, r + (RIGHT_TO_LEFT * left - right) / W)
    fig.get_layout_engine().set(rect=(0, 0, 1 - r, 1))
fig.canvas.draw()
ren = fig.canvas.get_renderer()
bA, bB = axA.get_window_extent(ren), axB.get_window_extent(ren)
log(f"layout: right pad {r:.4f} of the width; panel boxes {bA.width / fig.dpi:.3f} in (right margin = {RIGHT_TO_LEFT} x left); margins left {bA.x0 / fig.dpi:.2f} in, "
    f"right {W - bB.x1 / fig.dpi:.2f} in, gap {(bB.x0 - bA.x1) / fig.dpi:.2f} in")
for ax, letter in zip((axA, axB), "AB"):                     # bold panel letters at the left edge of the tick labels, on the title baseline
    left_pt = (ax.get_tightbbox(ren).x0 - ax.get_window_extent(ren).x0) * 72.0 / fig.dpi
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(left_pt, plt.rcParams["axes.titlepad"]),
                textcoords="offset points", ha="left", va="baseline", fontsize=FS_PANEL, fontweight="bold")
fig.canvas.draw()
ink = fig.get_tightbbox(fig.canvas.get_renderer())            # inches, figure coordinates
W, H = fig.get_size_inches()
crop = matplotlib.transforms.Bbox([[0, max(0.0, ink.y0 - V_CROP_PAD)], [W, min(H, ink.y1 + V_CROP_PAD)]])
log(f"saved height {crop.height:.2f} in of {H:.2f} (ink rows {ink.y0:.2f}-{ink.y1:.2f} in), full width {W:.1f} in")
for ext in ("png", "pdf"):           # this order: the layout is recomputed at every save, so it is part of the figure
    fig.savefig(os.path.join(OUT, f"{NAME}.{ext}"), bbox_inches=crop)
log(f"saved {OUT}/{NAME}.png / .pdf")
open(os.path.join(OUT, "figure_data_fig1.txt"), "w").write("\n".join(LOG) + "\n")
