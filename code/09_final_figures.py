"""Step 9 - final publication figures (English labels) for the paper.

Usage:
    python -I 09_final_figures.py <data_dir> <results_dir> <fig_dir>

Reads results/final/*.csv (step 8). Writes PNG (300 dpi) + PDF:
  Fig1_prevalence, Fig2_placement, Fig3_event_locked, Fig4_commonality_individuality,
  Fig5_variance_decomposition, Fig6_test_retest
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir, res_dir, fig_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
fin = res_dir / "final"
fig_dir.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#9b9a94", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "semibold",
                     "axes.titlelocation": "left", "savefig.bbox": "tight", "figure.facecolor": "white"})
SRC = "Source: MAESTRO v3.0.0; authors' analysis."


def finish(fig, name, drop_in=0.5):
    fig.text(0.01, -drop_in / fig.get_figheight(), SRC, fontsize=7, color=INK2)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")

# ---------------- Fig 1 prevalence
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.hist(D.soft_frac * 100, bins=np.arange(0, 101, 5), color=BLUE, edgecolor="white", linewidth=1.2)
med = D.soft_frac.median() * 100
ax.axvline(med, color=INK, lw=1, ls="--")
ax.text(med + 1.5, ax.get_ylim()[1] * 0.92, f"median {med:.0f}%", fontsize=8, color=INK)
share_any = (D.soft_frac > 0.01).mean() * 100
hours = (D.soft_frac * D.duration).sum() / 3600; tot = D.duration.sum() / 3600
ax.set_xlabel("Time with una corda down (% of performance)")
ax.set_ylabel("Performances")
grid(ax)
ax.set_title(f"The una corda is down for {hours / tot * 100:.0f}% of all playing time", pad=16)
ax.text(0, 1.02, f"{len(D):,} performances; {share_any:.0f}% use it at least 1% of the time; "
        f"{hours:.0f} of {tot:.0f} hours with una corda", transform=ax.transAxes, fontsize=8, color=INK2)
finish(fig, "Fig1_prevalence")

# ---------------- Fig 2 placement follows consensus loudness
QS = pd.read_csv(fin / "h1a_quartile_curves.csv")
SL = pd.read_csv(fin / "h1a_performance_slopes.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
labels = ["Quietest\n25%", "", "", "Loudest\n25%"]
for ped, col, nm in [("soft", BLUE, "Una corda"), ("sus", GREY, "Sustain")]:
    x = QS[(QS.group == "all") & (QS.pedal == ped)].sort_values("quartile")
    a.errorbar(x.quartile, x["mean"] * 100, yerr=[(x["mean"] - x.lo) * 100, (x.hi - x["mean"]) * 100], color=col,
               marker="o", ms=5, lw=2, capsize=3)
    a.text(4.12, x["mean"].iloc[-1] * 100, nm, va="center", fontsize=8.5, color=INK, fontweight="semibold")
a.set_xticks([1, 2, 3, 4]); a.set_xticklabels(["Q1\nquietest", "Q2", "Q3", "Q4\nloudest"])
a.set_xlim(0.7, 4.9); a.set_ylim(0, 85)
a.set_ylabel("Time pedal is down (%)"); a.set_xlabel("Consensus loudness of other pianists (quartile)")
grid(a); a.set_title("A  Pedal use by consensus loudness")
bins = np.linspace(-5, 3, 33)
b.hist(SL.soft_slope.dropna(), bins=bins, color=BLUE, alpha=0.85, label="Una corda", edgecolor="white")
b.hist(SL.sus_slope.dropna(), bins=bins, color=GREY, alpha=0.6, label="Sustain", edgecolor="white")
b.axvline(0, color=INK2, lw=0.8)
b.set_xlabel("Log-odds change per SD louder (per performance)"); b.set_ylabel("Performances")
b.legend(frameon=False, fontsize=8); grid(b); b.set_title("B  Within-performance slopes")
fig.suptitle("Where the una corda goes is shared: it tracks the passages all pianists play quietly",
             x=0.01, ha="left", fontsize=10.5, fontweight="semibold", y=1.05)
finish(fig, "Fig2_placement", 0.6)

# ---------------- Fig 3 event-locked velocity
G = pd.read_csv(fin / "h1c_profiles_mean.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for kind, col, nm in [("press", BLUE, "Una corda pressed"), ("release", ORANGE, "Una corda released"), ("random", GREY, "Random time")]:
    x = G[G.kind == kind]
    ax.fill_between(x.t, x.lo, x.hi, color=col, alpha=0.15, linewidth=0)
    ax.plot(x.t, x["mean"], color=col, lw=2)
    dy = {"press": 0, "release": 0.03, "random": -0.03}[kind]
    ax.text(4.1, x["mean"].iloc[-1] + dy, nm, va="center", fontsize=8, color=INK)
ax.axvline(0, color=INK2, lw=0.8, ls="--"); ax.axhline(0, color=GRID, lw=0.8)
ax.set_xlim(-4, 6.2); ax.set_xlabel("Time from pedal event (s)"); ax.set_ylabel("Note velocity (z within performance)")
grid(ax)
ax.set_title("Pressing the una corda coincides with a drop in loudness; releasing it, with a rise", pad=8)
finish(fig, "Fig3_event_locked")

# ---------------- Fig 4 commonality x individuality map
CM = pd.read_csv(fin / "h2c_commonality_individuality_map.csv")
fig, ax = plt.subplots(figsize=(5.2, 4.0))
cols = {"una corda": BLUE, "sustain": GREY, "loudness": ORANGE}
for _, r in CM.iterrows():
    c = cols[r.parameter]
    ax.errorbar(r.placement_agreement, r.extent_pianist_share * 100,
                xerr=[[r.placement_agreement - r.pa_lo], [r.pa_hi - r.placement_agreement]],
                yerr=[[(r.extent_pianist_share - r.ex_lo) * 100], [(r.ex_hi - r.extent_pianist_share) * 100]],
                fmt="o", color=c, ms=8, capsize=3, lw=1.5)
    lab = f"{r.parameter.capitalize()}\n({'κ' if r.agreement_measure == 'kappa' else 'r'} = {r.placement_agreement:.2f})"
    ax.text(r.placement_agreement + 0.02, r.extent_pianist_share * 100 + 2, lab, fontsize=8, color=INK)
ax.set_xlim(0, 1); ax.set_ylim(0, 70)
ax.set_xlabel("Placement commonality (agreement between pianists, same piece)")
ax.set_ylabel("Individuality of extent (% variance from pianist)")
grid(ax); grid(ax, "x")
ax.set_title("Only the una corda combines shared placement\nwith performer-specific extent", pad=8)
finish(fig, "Fig4_commonality_individuality", 0.6)

# ---------------- Fig 5 variance decomposition
V = pd.read_csv(fin / "h2a_variance_shares.csv")
NAMES = {"soft_frac": "Una corda: time down", "soft_note_share": "Una corda: share of notes",
         "soft_episodes_pm": "Una corda: presses per min", "sus_frac": "Sustain: time down",
         "sus_partial": "Sustain: partial depth", "sus_presses_pm": "Sustain: presses per min",
         "press_lag_ms": "Sustain: press timing", "vel_mean": "Mean velocity", "dyn_range": "Velocity range",
         "nps": "Notes per second"}
V = V.sort_values("share_pianist")
fig, ax = plt.subplots(figsize=(6.8, 3.8))
y = np.arange(len(V))
left = np.zeros(len(V))
for col, c, nm in [("share_pianist", BLUE, "Pianist"), ("share_composer", ORANGE, "Composer"), ("share_edition", "#c9c7bf", "Edition")]:
    ax.barh(y, V[col] * 100, left=left, color=c, height=0.62, label=nm, edgecolor="white", linewidth=1)
    left += V[col].values * 100
ax.barh(y, 100 - left, left=left, color="#ecebe6", height=0.62, label="Residual", edgecolor="white", linewidth=1)
for i, (_, r) in enumerate(V.iterrows()):
    if not np.isnan(r.get("boot_low", np.nan)):
        ax.errorbar(r.share_pianist * 100, i, xerr=[[(r.share_pianist - r.boot_low) * 100], [(r.boot_high - r.share_pianist) * 100]],
                    fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)
ax.set_yticks(y); ax.set_yticklabels([NAMES[f] for f in V.feature])
ax.set_xlim(0, 100); ax.set_xlabel("Share of variance (%)")
ax.text(0, -0.25, "Black whiskers: pianist-cluster bootstrap 95% CI of the pianist share (60 replicates).",
        transform=ax.transAxes, fontsize=7.5, color=INK2)
ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, 1.13))
ax.set_title("The pianist accounts for more of una corda extent than of any other measure", pad=26)
finish(fig, "Fig5_variance_decomposition", 0.85)

# ---------------- Fig 6 test-retest across rounds
def round_table(col):
    x = D.assign(res=D[col] - D.groupby("composer")[col].transform("mean"))
    y = x[x["round"].isin(["R1", "R2", "R3"])].assign(stage=lambda d: np.where(d["round"] == "R1", "R1", "later"))
    return y.groupby(["year", "pianist", "stage"]).res.mean().unstack().dropna()
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharex=False)
for ax, col, nm, c in [(axes[0], "soft_frac", "Una corda", BLUE), (axes[1], "sus_frac", "Sustain", GREY)]:
    g = round_table(col) * 100
    r = np.corrcoef(g.R1, g.later)[0, 1]
    ax.scatter(g.R1, g.later, s=14, color=c, alpha=0.8, edgecolor="white", linewidth=0.5)
    lim = [min(g.min().min(), -40) - 5, max(g.max().max(), 40) + 5]
    ax.plot(lim, lim, color=GRID, lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Round 1 (composer-adjusted, pp)"); ax.set_ylabel("Later rounds")
    ax.set_title(f"{nm}: r = {r:.2f} (n = {len(g)})", fontsize=9)
    grid(ax); grid(ax, "x")
fig.suptitle("The same pianist uses the una corda similarly on different days; not so the sustain pedal",
             x=0.01, ha="left", fontsize=10.5, fontweight="semibold", y=1.04)
finish(fig, "Fig6_test_retest", 0.55)
print("figures:", sorted(p.name for p in fig_dir.glob("Fig[1-6]_*.png")))
