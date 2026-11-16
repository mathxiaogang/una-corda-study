"""Step 18 - figures formatted to the Music Perception author guidelines (July 2024).

Usage:
    python -I 18_mp_figures.py <data_dir> <results_dir> <fig_dir>

Rules applied: sans-serif text between 8 and 14 pt at final size (figures are drawn at a 6.5-in print width);
plot symbols no larger than 4 pt; axis labels centred, in capital-then-lowercase, units in parentheses;
greyscale with line styles and marker shapes so that the figures print clearly in black and white.
Writes Figure1-Figure6 and FigureS1 as PNG (300 dpi), TIFF (300 dpi, LZW) and PDF.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir, res_dir, fig_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
fin, rev, rev2 = res_dir / "final", res_dir / "revision", res_dir / "revision2"
fig_dir.mkdir(parents=True, exist_ok=True)
K, DARK, MID, LIGHT, PALE = "#000000", "#4d4d4d", "#8c8c8c", "#c4c4c4", "#ececec"
MS = 4  # marker size in points (guideline: symbols no larger than 4 pt)
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.edgecolor": K, "axes.labelcolor": K, "xtick.color": K, "ytick.color": K,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "savefig.bbox": "tight",
                     "figure.facecolor": "white", "lines.markersize": MS})


def save(fig, name):
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {}), ("tif", {"dpi": 300, "pil_kwargs": {"compression": "tiff_lzw"}})):
        fig.savefig(fig_dir / f"{name}.{ext}", **kw)
    plt.close(fig)


def grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=PALE, lw=0.6)
    ax.set_axisbelow(True)


D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")
R = json.load(open(rev / "revision_summary.json"))

# ---------------- Figure 1
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.hist(D.soft_frac * 100, bins=np.arange(0, 101, 5), color=MID, edgecolor=K, linewidth=0.5)
med = D.soft_frac.median() * 100
ax.axvline(med, color=K, lw=1, ls="--")
ax.text(med + 1.5, ax.get_ylim()[1] * 0.92, f"Mdn = {med:.1f}%", fontsize=8.5)
ax.set_xlabel("Una corda time share (% of performance)"); ax.set_ylabel("Number of performances"); grid(ax)
save(fig, "Figure1")

# ---------------- Figure 2
QS = pd.read_csv(fin / "h1a_quartile_curves.csv"); SL = pd.read_csv(fin / "h1a_performance_slopes.csv")
fig = plt.figure(figsize=(6.5, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.62, wspace=0.38)
a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
for ped, ls, mk, mfc, nm in [("soft", "-", "o", K, "Una corda"), ("sus", "--", "s", "white", "Sustain")]:
    x = QS[(QS.group == "all") & (QS.pedal == ped)].sort_values("quartile")
    a.errorbar(x.quartile, x["mean"] * 100, yerr=[(x["mean"] - x.lo) * 100, (x.hi - x["mean"]) * 100], color=K, ls=ls,
               marker=mk, ms=MS, mfc=mfc, mec=K, lw=1.3, capsize=2, label=nm)
a.set_xticks([1, 2, 3, 4]); a.set_xticklabels(["Q1\nquietest", "Q2", "Q3", "Q4\nloudest"])
a.set_xlim(0.6, 4.4); a.set_ylim(0, 85); a.set_ylabel("Windows with pedal down (%)")
a.set_xlabel("Consensus loudness quartile"); a.legend(frameon=False, loc="lower left"); grid(a); a.set_title("A")
bins = np.linspace(-5, 3, 33)
b.hist(SL.soft_slope.dropna(), bins=bins, color=DARK, edgecolor="white", linewidth=0.4, label="Una corda")
b.hist(SL.sus_slope.dropna(), bins=bins, histtype="step", color=K, linewidth=1.0, label="Sustain")
b.axvline(0, color=K, lw=0.6)
b.set_xlabel("Slope (log-odds per SD louder)"); b.set_ylabel("Number of performances")
b.legend(frameon=False, loc="upper left"); grid(b); b.set_title("B")
r4 = R["R4"]; models = [("m0", "No controls"), ("m1", "+ Texture"), ("m2", "+ Own loudness")]
for ped, off, mk, mfc, nm in [("soft", -0.12, "o", K, "Una corda"), ("sus", 0.12, "s", "white", "Sustain")]:
    for j, (m, _) in enumerate(models):
        e = r4[f"{ped}_{m}"]
        c.errorbar(e["mean"], j + off, xerr=1.96 * e["se"], fmt=mk, color=K, mfc=mfc, mec=K, ms=MS, capsize=2, lw=1.1,
                   label=nm if j == 0 else None)
c.axvline(0, color=K, lw=0.6); c.set_yticks(range(3)); c.set_yticklabels([m[1] for m in models]); c.set_ylim(2.5, -0.7)
c.set_xlim(-2, 0.5); c.set_xlabel("Consensus-loudness coefficient (log-odds per SD)"); grid(c, "x")
c.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02)); c.set_title("C")
save(fig, "Figure2")

# ---------------- Figure 3
G = pd.read_csv(fin / "h1c_profiles_mean.csv"); G5 = pd.read_csv(rev / "R5_event_profiles_mean.csv")
E = pd.read_csv(rev2 / "S5_event_locked_consensus.csv")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.9), gridspec_kw={"wspace": 0.42})
spec = [(G[G.kind == "press"], K, "-", "Una corda press", None), (G[G.kind == "release"], K, "--", "Una corda release", None),
        (G5[G5.kind == "sustain press"], MID, "-", "Sustain press", None), (G[G.kind == "random"], MID, ":", "Random time", None)]
for x, col, ls, nm, _ in spec:
    ax.fill_between(x.t, x.lo, x.hi, color=LIGHT, alpha=0.45, linewidth=0)
    ax.plot(x.t, x["mean"], color=col, ls=ls, lw=1.4, label=nm)
ax.axvline(0, color=K, lw=0.6, ls="-"); ax.set_xlim(-4, 4)
ax.set_xlabel("Time from pedal event (s)"); ax.set_ylabel("Note velocity (z within performance)"); grid(ax)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=2.2); ax.set_title("A")
for cc, col, ls, nm in [("own", K, "-", "This pianist"), ("cons", MID, "--", "Other pianists")]:
    bx.fill_between(E.t, E[f"{cc}_mean"] - 1.96 * E[f"{cc}_se"], E[f"{cc}_mean"] + 1.96 * E[f"{cc}_se"], color=LIGHT, alpha=0.45, linewidth=0)
    bx.plot(E.t, E[f"{cc}_mean"], color=col, ls=ls, lw=1.4, label=nm)
bx.axvline(0, color=K, lw=0.6); bx.set_xlim(-4, 4)
bx.set_xlabel("Time from una corda press (s)"); bx.set_ylabel("Loudness (z within performance)"); grid(bx)
bx.legend(frameon=False, loc="upper right", handlelength=2.2); bx.set_title("B")
save(fig, "Figure3")

# ---------------- Figure 4
V = pd.read_csv(rev / "R1_h2a_variance_shares_B1000.csv")
NAMES = {"soft_frac": "Una corda: time share", "soft_note_share": "Una corda: note share",
         "soft_episodes_pm": "Una corda: presses per min", "sus_frac": "Sustain: time share",
         "sus_partial": "Sustain: partial depth", "sus_presses_pm": "Sustain: presses per min",
         "press_lag_ms": "Sustain: press lag", "vel_mean": "Mean velocity", "dyn_range": "Velocity range",
         "nps": "Notes per second"}
V = V.sort_values("share_pianist")
fig, ax = plt.subplots(figsize=(6.5, 3.6))
y = np.arange(len(V)); left = np.zeros(len(V))
for col, cc, nm, hatch in [("share_pianist", DARK, "Pianist", None), ("share_composer", MID, "Composer", None),
                           ("share_edition", LIGHT, "Edition", None)]:
    ax.barh(y, V[col] * 100, left=left, color=cc, height=0.62, label=nm, edgecolor="white", linewidth=0.6, hatch=hatch)
    left += V[col].values * 100
ax.barh(y, 100 - left, left=left, color="white", height=0.62, label="Residual", edgecolor=MID, linewidth=0.6)
for i, (_, r) in enumerate(V.iterrows()):
    ax.errorbar(r.share_pianist * 100, i, xerr=[[max(0, (r.share_pianist - r.boot_low) * 100)], [max(0, (r.boot_high - r.share_pianist) * 100)]],
                fmt="none", ecolor=K, elinewidth=1.0, capsize=2)
ax.set_yticks(y); ax.set_yticklabels([NAMES[f] for f in V.feature]); ax.set_xlim(0, 100)
ax.set_xlabel("Share of variance (%)")
ax.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0))
save(fig, "Figure4")

# ---------------- Figure 5
def round_table(col):
    x = D.assign(res=D[col] - D.groupby("composer")[col].transform("mean"))
    yy = x[x["round"].isin(["R1", "R2", "R3"])].assign(stage=lambda d: np.where(d["round"] == "R1", "R1", "later"))
    return yy.groupby(["year", "pianist", "stage"]).res.mean().unstack().dropna()
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw={"wspace": 0.38})
for ax, col, nm, fc, lab in [(axes[0], "soft_frac", "Una corda", K, "A"), (axes[1], "sus_frac", "Sustain", "white", "B")]:
    g = round_table(col) * 100; r = np.corrcoef(g.R1, g.later)[0, 1]
    ax.scatter(g.R1, g.later, s=12, facecolor=fc, edgecolor=K, linewidth=0.5)
    lim = [min(g.min().min(), -40) - 5, max(g.max().max(), 40) + 5]
    ax.plot(lim, lim, color=MID, lw=0.8, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Round 1 (pp from composer mean)"); ax.set_ylabel("Later rounds (pp from composer mean)")
    ax.text(0.03, 0.95, f"{nm} time share\nr({len(g) - 2}) = {r:.2f}".replace("0.", "."), transform=ax.transAxes, va="top", fontsize=8.5)
    ax.set_title(lab); grid(ax); grid(ax, "x")
save(fig, "Figure5")

# ---------------- Figure 6
CM = pd.read_csv(fin / "h2c_commonality_individuality_map.csv")
fig, ax = plt.subplots(figsize=(5.0, 3.8))
sty = {"una corda": ("o", K), "sustain": ("s", "white"), "loudness": ("^", MID)}
for _, r in CM.iterrows():
    mk, fc = sty[r.parameter]
    ax.errorbar(r.placement_agreement, r.extent_pianist_share * 100,
                xerr=[[r.placement_agreement - r.pa_lo], [r.pa_hi - r.placement_agreement]],
                yerr=[[(r.extent_pianist_share - r.ex_lo) * 100], [(r.ex_hi - r.extent_pianist_share) * 100]],
                fmt=mk, color=K, mfc=fc, mec=K, ms=MS, capsize=2, lw=1.0)
    lab = f"{r.parameter.capitalize()}\n({'κ' if r.agreement_measure == 'kappa' else 'r'} = {r.placement_agreement:.2f})".replace("0.", ".")
    ax.text(r.placement_agreement + 0.03, r.extent_pianist_share * 100 + 2, lab, fontsize=8.5)
ax.set_xlim(0, 1); ax.set_ylim(0, 70)
ax.set_xlabel("Placement agreement between pianists"); ax.set_ylabel("Pianist share of variance in extent (%)")
grid(ax); grid(ax, "x"); save(fig, "Figure6")

# ---------------- Figure S1
R2 = pd.DataFrame(R["R2"]["table"])
EN = {"main": "Main model", "rank-normal scores": "Rank-based normal scores", "una corda users only (time share >= 1%)": "Una corda users only",
      "musical covariates": "+ Repertoire-related covariates", "title random effect": "+ Piece random effect",
      "title random effect + covariates": "+ Piece random effect and covariates", "junior editions only": "Junior editions only",
      "senior editions only": "Senior editions only"}
fig, ax = plt.subplots(figsize=(6.5, 3.2)); yy = np.arange(len(R2))
for i, r in R2.iterrows():
    ax.plot([r.sus_frac * 100, r.soft_frac * 100], [i, i], color=LIGHT, lw=1.5, zorder=1)
ax.scatter(R2.soft_frac * 100, yy, s=14, color=K, label="Una corda time share", zorder=3)
ax.scatter(R2.sus_frac * 100, yy, s=14, facecolor="white", edgecolor=K, linewidth=0.8, label="Sustain time share", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([f"{EN[v]} (n = {n:,})" for v, n in zip(R2.variant, R2.n)]); ax.invert_yaxis()
ax.set_xlim(0, 70); ax.set_xlabel("Pianist share of variance (%)"); grid(ax, "x")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "FigureS1")
print("figures:", sorted(p.name for p in fig_dir.glob("Figure*.png")))
