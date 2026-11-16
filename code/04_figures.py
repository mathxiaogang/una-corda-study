"""Step 4 - publication figures (English labels, for journal submission).

Usage:
    python -I 04_figures.py <data_dir> <results_dir> <fig_dir>

Writes PNG (300 dpi) and PDF versions of Fig1-Fig4 and FigS1.
Colour: senior = accent blue (#2a78d6), junior = neutral grey; identity is also given by text labels.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir, res_dir, fig_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
fig_dir.mkdir(parents=True, exist_ok=True)

BLUE, GREY, INK, INK2, GRID = "#2a78d6", "#9b9a94", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight",
})
SRC = "Source: MAESTRO v3.0.0 (1,276 performances, 201 pianists); authors' analysis."


def finish(fig, name, drop_in=0.5):
    # source line sits drop_in inches below the axes area, clear of tick and axis labels
    fig.text(0.01, -drop_in / fig.get_figheight(), SRC, fontsize=7, color=INK2, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=300)
    plt.close(fig)


meta = pd.read_csv(data_dir / "metadata_clean.csv")
feat = pd.read_csv(data_dir / "features.csv")
D = meta.merge(feat, on="midi_filename")
col = lambda g: BLUE if g == "senior" else GREY

# ---------------- Fig 1: every performance, by edition
fig, ax = plt.subplots(figsize=(7.2, 3.6))
years = sorted(D.year.unique())
rng = np.random.default_rng(1)
for i, y in enumerate(years):
    x = D[D.year == y]
    g = x.group.iloc[0]
    ax.scatter(i + rng.uniform(-0.28, 0.28, len(x)), x.soft_frac * 100, s=6, color=col(g), alpha=0.35,
               linewidths=0, zorder=2)
    med = x.soft_frac.median() * 100
    ax.plot([i - 0.34, i + 0.34], [med, med], color=INK if g == "senior" else INK2, lw=2, zorder=3,
            solid_capstyle="round")
ax.set_xticks(range(len(years)))
ax.set_xticklabels([f"{y}\n{'Senior' if D[D.year == y].group.iloc[0] == 'senior' else 'Junior'}" for y in years])
ax.set_ylabel("Time with una corda down (% of performance)")
ax.set_ylim(-3, 103)
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("Every senior edition used the una corda more than every junior edition", pad=18)
ax.text(0, 1.02, "Dots = performances (blue senior, grey junior); bars = edition medians",
        transform=ax.transAxes, fontsize=8, color=INK2)
finish(fig, "Fig1_una_corda_by_edition", 0.45)

# ---------------- Fig 2: specificity forest plot
spec = pd.read_csv(res_dir / "specificity_standardized.csv")
EN = {"soft_frac": "Una corda: time down", "soft_note_share": "Una corda: share of notes",
      "soft_episodes_pm": "Una corda: presses per minute", "log_episode_s": "Una corda: episode length (log)",
      "vel_gap": "Una corda: velocity gap (off - on)", "soft_at_mf": "Una corda: share of notes at velocity >= 60",
      "sus_frac": "Sustain: time down", "sus_partial": "Sustain: partial-depth share",
      "sus_presses_pm": "Sustain: presses per minute", "press_lag_ms": "Sustain: press lag after onset",
      "synco_share": "Sustain: syncopated presses", "vel_mean": "Dynamics: mean velocity",
      "dyn_range": "Dynamics: velocity range (P95-P5)", "nps": "Texture: notes per second"}
spec = spec.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for i, r in spec.iterrows():
    c = BLUE if r.family == "una corda" else GREY
    ax.plot([r.boot_low, r.boot_high], [i, i], color=c, lw=1.6, solid_capstyle="round")
    ax.scatter(r.d_senior_minus_junior, i, s=34, color=c, edgecolor="white", linewidth=1, zorder=3)
    lab = f"p = {r.perm_p:.3f}" + (" *" if r.perfect_separation else "")
    ax.text(1.02, i, lab, va="center", fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(range(len(spec)))
ax.set_yticklabels([EN[f] for f in spec.feature])
ax.set_xlabel("Senior minus junior (SD units, composer-adjusted, edition level)")
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("Group differences concentrate in how often the una corda is used", pad=18)
ax.text(0, 1.02, "Points = difference; bars = bootstrap 95% CI over editions; p = exact permutation (252); "
        "* = perfect separation", transform=ax.transAxes, fontsize=7.5, color=INK2)
finish(fig, "Fig2_specificity_forest", 0.55)

# ---------------- Fig 3: by style period
era = pd.read_csv(res_dir / "supp_era_by_group.csv")
order = ["Baroque", "Classical", "Romantic", "Modern"]
fig, ax = plt.subplots(figsize=(6.2, 2.8))
for i, e in enumerate(order):
    j = era[(era.era == e) & (era.group == "junior")].iloc[0]
    s = era[(era.era == e) & (era.group == "senior")].iloc[0]
    ax.plot([j.soft_median * 100, s.soft_median * 100], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
    ax.scatter(j.soft_median * 100, i, s=46, color=GREY, zorder=2, edgecolor="white")
    ax.scatter(s.soft_median * 100, i, s=46, color=BLUE, zorder=3, edgecolor="white")
    ax.text(j.soft_median * 100 - 1.5, i, f"Junior {j.soft_median*100:.0f}% (n={j.n})", ha="right", va="center",
            fontsize=7.5, color=INK2)
    ax.text(s.soft_median * 100 + 1.5, i, f"Senior {s.soft_median*100:.0f}% (n={s.n})", ha="left", va="center",
            fontsize=7.5, color=INK)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xlim(-22, 72)
ax.set_xticks([0, 20, 40, 60])
ax.set_xlabel("Median time with una corda down (%)")
ax.xaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("The senior-junior gap appears in every style period", pad=8)
finish(fig, "Fig3_by_style_period", 0.6)

# ---------------- Fig 4: repeat entrants
ch = pd.read_csv(res_dir / "h1b_changes.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
TYPES = [("junior->junior", "Junior to junior"), ("junior->senior", "Junior to senior"), ("senior->senior", "Senior to senior")]
for ax, (t, lab) in zip(axes, TYPES):
    sub = ch[ch.repeat_type == t]
    c = BLUE if t == "junior->senior" else GREY
    for _, r in sub.iterrows():
        ax.plot([0, 1], [r.soft_first * 100, r.soft_last * 100], color=c, lw=1.4, marker="o", ms=4,
                markeredgecolor="white")
    if t == "junior->senior":
        lab_y, last = [], -1e9
        for _, r in sub.sort_values("soft_last").iterrows():
            yv = max(r.soft_last * 100, last + 4.5)
            lab_y.append((yv, r.pianist.split()[-1]))
            last = yv
        for yv, name in lab_y:
            ax.text(1.06, yv, name, fontsize=7, va="center", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["First entry", "Second entry"])
    ax.set_xlim(-0.2, 1.45)
    ax.set_title(f"{lab} (n={len(sub)})", fontsize=9, pad=4)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Median una corda time (%)")
fig.suptitle("Repeat entrants: too few to confirm a within-person rise", x=0.01, ha="left",
             fontsize=10.5, fontweight="semibold", y=1.04)
finish(fig, "Fig4_repeat_entrants", 0.45)

# ---------------- Fig S1: raw sensor values by year
raw = pd.read_csv(data_dir / "cc67_raw_by_year.csv")
fig, ax = plt.subplots(figsize=(6.2, 3.2))
for y, g in raw.groupby("year"):
    g = g[g.value > 0].sort_values("value")
    cdf = g.n.cumsum() / g.n.sum()
    grp = D[D.year == y].group.iloc[0]
    ax.step(g.value, cdf, where="post", color=col(grp), lw=1.1, alpha=0.9)
    if g.value.min() > 15:  # edition with no low (shallow) values recorded
        ax.text(g.value.min() + 30, 0.06, f"{y}: no values below {g.value.min()}", fontsize=7.5, color=INK)
ax.axvline(64, color=INK2, lw=0.8, ls="--")
ax.text(65, 0.05, "threshold 64", fontsize=7.5, color=INK2)
ax.set_xlabel("Raw soft-pedal controller value (non-zero events)")
ax.set_ylabel("Cumulative share of events")
ax.yaxis.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.set_title("Raw soft-pedal values look alike across editions, except shallow values in 2006", pad=18)
ax.text(0, 1.02, "One line per edition: blue senior, grey junior", transform=ax.transAxes, fontsize=8, color=INK2)
finish(fig, "FigS1_sensor_audit", 0.6)
print("figures:", sorted(p.name for p in fig_dir.iterdir()))
