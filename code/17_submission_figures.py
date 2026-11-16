"""Step 17 - figures for the Music Perception submission (as step 15; "piece" instead of "work" in Figure S1 labels).

Usage:
    python -I 17_submission_figures.py <data_dir> <results_dir> <fig_dir>

Writes Figure1 ... Figure7 (PNG 300 dpi + PDF) numbered as in revision 2.
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
fin, rev = res_dir / "final", res_dir / "revision"
fig_dir.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY, INK, INK2, GRID, GREEN = "#2a78d6", "#eb6834", "#9b9a94", "#0b0b0b", "#52514e", "#e6e5e0", "#3a9a5b"
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
                     "axes.titlelocation": "left", "savefig.bbox": "tight", "figure.facecolor": "white"})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def grid(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")
R = json.load(open(rev / "revision_summary.json"))

# ---------------- Figure 1 prevalence
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.hist(D.soft_frac * 100, bins=np.arange(0, 101, 5), color=BLUE, edgecolor="white", linewidth=1.2)
med = D.soft_frac.median() * 100
ax.axvline(med, color=INK, lw=1, ls="--")
ax.text(med + 1.5, ax.get_ylim()[1] * 0.92, f"median {med:.1f}%", fontsize=8, color=INK)
ax.set_xlabel("Time with una corda down (% of performance)"); ax.set_ylabel("Performances"); grid(ax)
save(fig, "Figure1")

# ---------------- Figure 2 placement (A quartiles, B slopes, C controls)
QS = pd.read_csv(fin / "h1a_quartile_curves.csv"); SL = pd.read_csv(fin / "h1a_performance_slopes.csv")
fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [1.05, 1, 0.9], "wspace": 0.62})
for ped, col, nm in [("soft", BLUE, "Una corda"), ("sus", GREY, "Sustain")]:
    x = QS[(QS.group == "all") & (QS.pedal == ped)].sort_values("quartile")
    a.errorbar(x.quartile, x["mean"] * 100, yerr=[(x["mean"] - x.lo) * 100, (x.hi - x["mean"]) * 100], color=col,
               marker="o", ms=4.5, lw=2, capsize=3)
    a.text(4.12, x["mean"].iloc[-1] * 100, nm, va="center", fontsize=8, color=INK)
a.set_xticks([1, 2, 3, 4]); a.set_xticklabels(["Q1\nquietest", "Q2", "Q3", "Q4\nloudest"])
a.set_xlim(0.7, 5.1); a.set_ylim(0, 85); a.set_ylabel("Windows with pedal down (%)")
a.set_xlabel("Consensus loudness quartile"); grid(a); a.set_title("A")
bins = np.linspace(-5, 3, 33)
b.hist(SL.soft_slope.dropna(), bins=bins, color=BLUE, alpha=0.85, label="Una corda", edgecolor="white")
b.hist(SL.sus_slope.dropna(), bins=bins, color=GREY, alpha=0.6, label="Sustain", edgecolor="white")
b.axvline(0, color=INK2, lw=0.8); b.set_xlabel("Slope (log-odds per SD louder)"); b.set_ylabel("Performances")
b.legend(frameon=False, fontsize=7.5, loc="upper left"); grid(b); b.set_title("B")
r4 = R["R4"]; models = [("m0", "No controls"), ("m1", "+ texture"), ("m2", "+ own loudness")]
for k, (ped, col, off) in enumerate([("soft", BLUE, -0.12), ("sus", GREY, 0.12)]):
    for j, (m, _) in enumerate(models):
        e = r4[f"{ped}_{m}"]
        c.errorbar(e["mean"], j + off, xerr=1.96 * e["se"], fmt="o", color=col, ms=5, capsize=3, lw=1.5,
                   label=("Una corda" if ped == "soft" else "Sustain") if j == 0 else None)
c.axvline(0, color=INK2, lw=0.8); c.set_yticks(range(3)); c.set_yticklabels([m[1] for m in models]); c.invert_yaxis()
c.set_xlabel("Consensus-loudness coefficient\n(log-odds per SD, 95% CI)"); grid(c, "x"); c.set_xlim(-2, 0.5)
c.set_ylim(2.45, -1.05); c.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2, handletextpad=0.3, columnspacing=1.0); c.set_title("C")
save(fig, "Figure2")

# ---------------- Figure 3 event-locked velocity: A full corpus with sustain control; B own vs others (aligned set)
G = pd.read_csv(fin / "h1c_profiles_mean.csv"); G5 = pd.read_csv(rev / "R5_event_profiles_mean.csv")
E = pd.read_csv(res_dir / "revision2" / "S5_event_locked_consensus.csv")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.2), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.5})
lines = [(G[G.kind == "press"], BLUE, "Una corda press", None), (G[G.kind == "release"], ORANGE, "Una corda release", 0.025),
         (G5[G5.kind == "sustain press"], GREEN, "Sustain press", 0.085), (G[G.kind == "random"], GREY, "Random time", -0.04)]
for x, col, nm, ly in lines:
    ax.fill_between(x.t, x.lo, x.hi, color=col, alpha=0.15, linewidth=0)
    ax.plot(x.t, x["mean"], color=col, lw=2 if "Una" in nm else 1.6)
    ax.text(4.1, x["mean"].iloc[-1] if ly is None else ly, nm, va="center", fontsize=7.8, color=col if col != GREY else INK2)
ax.axvline(0, color=INK2, lw=0.8, ls="--"); ax.axhline(0, color=GRID, lw=0.8)
ax.set_xlim(-4, 7.2); ax.set_xlabel("Time from pedal event (s)"); ax.set_ylabel("Note velocity (z within performance)"); grid(ax)
ax.set_title("A")
for c, col, nm, ly in [("own", BLUE, "This pianist", None), ("cons", INK2, "Other pianists,\nsame moment", None)]:
    bx.fill_between(E.t, E[f"{c}_mean"] - 1.96 * E[f"{c}_se"], E[f"{c}_mean"] + 1.96 * E[f"{c}_se"], color=col, alpha=0.15, linewidth=0)
    bx.plot(E.t, E[f"{c}_mean"], color=col, lw=2, ls="-" if c == "own" else "--")
    bx.text(4.15, E[f"{c}_mean"].iloc[-1] + (0.06 if c == "own" else -0.08), nm, va="center", fontsize=7.8, color=col)
bx.axvline(0, color=INK2, lw=0.8, ls="--"); bx.axhline(0, color=GRID, lw=0.8)
bx.set_xlim(-4, 7.2); bx.set_xlabel("Time from una corda press (s)"); bx.set_ylabel("Loudness (z within performance)"); grid(bx)
bx.set_title("B")
save(fig, "Figure3")

# ---------------- Figure 4 variance decomposition (B = 1000)
V = pd.read_csv(rev / "R1_h2a_variance_shares_B1000.csv")
NAMES = {"soft_frac": "Una corda: time share", "soft_note_share": "Una corda: note share",
         "soft_episodes_pm": "Una corda: presses per min", "sus_frac": "Sustain: time share",
         "sus_partial": "Sustain: partial depth", "sus_presses_pm": "Sustain: presses per min",
         "press_lag_ms": "Sustain: press lag", "vel_mean": "Mean velocity", "dyn_range": "Velocity range",
         "nps": "Notes per second"}
V = V.sort_values("share_pianist")
fig, ax = plt.subplots(figsize=(6.8, 3.7))
y = np.arange(len(V)); left = np.zeros(len(V))
for col, cc, nm in [("share_pianist", BLUE, "Pianist"), ("share_composer", ORANGE, "Composer"), ("share_edition", "#c9c7bf", "Edition")]:
    ax.barh(y, V[col] * 100, left=left, color=cc, height=0.62, label=nm, edgecolor="white", linewidth=1)
    left += V[col].values * 100
ax.barh(y, 100 - left, left=left, color="#ecebe6", height=0.62, label="Residual", edgecolor="white", linewidth=1)
for i, (_, r) in enumerate(V.iterrows()):
    ax.errorbar(r.share_pianist * 100, i, xerr=[[max(0, (r.share_pianist - r.boot_low) * 100)], [max(0, (r.boot_high - r.share_pianist) * 100)]],
                fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)
ax.set_yticks(y); ax.set_yticklabels([NAMES[f] for f in V.feature]); ax.set_xlim(0, 100)
ax.set_xlabel("Share of variance (%)")
ax.legend(ncol=4, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.4, 1.0))
save(fig, "Figure4")

# ---------------- Figure 5 test-retest
def round_table(col):
    x = D.assign(res=D[col] - D.groupby("composer")[col].transform("mean"))
    y = x[x["round"].isin(["R1", "R2", "R3"])].assign(stage=lambda d: np.where(d["round"] == "R1", "R1", "later"))
    return y.groupby(["year", "pianist", "stage"]).res.mean().unstack().dropna()
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
for ax, col, nm, cc, lab in [(axes[0], "soft_frac", "Una corda", BLUE, "A"), (axes[1], "sus_frac", "Sustain", GREY, "B")]:
    g = round_table(col) * 100; r = np.corrcoef(g.R1, g.later)[0, 1]
    ax.scatter(g.R1, g.later, s=14, color=cc, alpha=0.8, edgecolor="white", linewidth=0.5)
    lim = [min(g.min().min(), -40) - 5, max(g.max().max(), 40) + 5]
    ax.plot(lim, lim, color=GRID, lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Round 1 (pp from composer mean)"); ax.set_ylabel("Later rounds (pp)")
    ax.set_title(f"{lab}  {nm} time share: r = {r:.2f}, n = {len(g)}", fontsize=9); grid(ax); grid(ax, "x")
fig.tight_layout(); save(fig, "Figure5")

# ---------------- Figure 6 commonality x individuality map
CM = pd.read_csv(fin / "h2c_commonality_individuality_map.csv")
fig, ax = plt.subplots(figsize=(5.0, 3.8)); cols = {"una corda": BLUE, "sustain": GREY, "loudness": ORANGE}
for _, r in CM.iterrows():
    ax.errorbar(r.placement_agreement, r.extent_pianist_share * 100,
                xerr=[[r.placement_agreement - r.pa_lo], [r.pa_hi - r.placement_agreement]],
                yerr=[[(r.extent_pianist_share - r.ex_lo) * 100], [(r.ex_hi - r.extent_pianist_share) * 100]],
                fmt="o", color=cols[r.parameter], ms=8, capsize=3, lw=1.5)
    lab = f"{r.parameter.capitalize()}\n({'κ' if r.agreement_measure == 'kappa' else 'r'} = {r.placement_agreement:.2f})"
    ax.text(r.placement_agreement + 0.02, r.extent_pianist_share * 100 + 2, lab, fontsize=8, color=INK)
ax.set_xlim(0, 1); ax.set_ylim(0, 70)
ax.set_xlabel("Placement agreement between pianists (same piece)"); ax.set_ylabel("Pianist share of variance in extent (%)")
grid(ax); grid(ax, "x"); save(fig, "Figure6")

# ---------------- Figure 7 robustness forest plot (pianist share, una corda vs sustain time share)
R2 = pd.DataFrame(R["R2"]["table"])
fig, ax = plt.subplots(figsize=(6.6, 3.3)); yy = np.arange(len(R2))
ax.scatter(R2.soft_frac * 100, yy, color=BLUE, s=36, label="Una corda time share", zorder=3)
ax.scatter(R2.sus_frac * 100, yy, color=GREY, s=36, label="Sustain time share", zorder=3)
for i, r in R2.iterrows():
    ax.plot([r.sus_frac * 100, r.soft_frac * 100], [i, i], color=GRID, lw=2, zorder=1)
EN = {"main": "Main model", "rank-normal scores": "Rank-normal scores", "una corda users only (time share >= 1%)": "Una corda users only",
      "musical covariates": "+ musical covariates", "title random effect": "+ piece random effect",
      "title random effect + covariates": "+ piece random effect and covariates", "junior editions only": "Junior editions only",
      "senior editions only": "Senior editions only"}
ax.set_yticks(yy); ax.set_yticklabels([f"{EN[v]} (n = {n:,})" for v, n in zip(R2.variant, R2.n)]); ax.invert_yaxis()
ax.set_xlim(0, 70); ax.set_xlabel("Pianist share of variance (%)"); grid(ax, "x")
ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "Figure7")
print("figures:", sorted(p.name for p in fig_dir.glob("Figure*.png")))
