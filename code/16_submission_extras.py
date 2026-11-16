"""Step 16 - small additions for the submission version (Music Perception).

Usage:
    python -I 16_submission_extras.py <data_dir> <results_dir>

Completes Table 4: round-to-round correlations of una corda presses per minute and velocity range within junior and
senior editions and after regressing out the musical covariates (same procedure as step 12, part R3).
Writes results/submission/test_retest_completion.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
out = res_dir / "submission"; out.mkdir(exist_ok=True)
D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")
D = D.merge(pd.read_csv(res_dir / "revision" / "performance_musical_covariates.csv"), on="midi_filename", how="left")
for e in ["Baroque", "Classical", "Modern"]:
    D[f"era_{e}"] = (D.era == e).astype(float)
MUS = ["pitch_mean", "pitch_sd", "high_share", "low_share", "chord_share", "nps", "log_dur", "era_Baroque", "era_Classical", "era_Modern"]
d0 = D.dropna(subset=MUS).copy()


def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


rows = []
for col in ["soft_frac", "soft_episodes_pm", "sus_frac", "sus_partial", "vel_mean", "dyn_range"]:
    for adj in ["composer", "composer + musical covariates"]:
        x = d0.copy(); x["res"] = x[col] - x.groupby("composer")[col].transform("mean")
        if adj != "composer":
            Xc = np.column_stack([np.ones(len(x))] + [((x[c] - x[c].mean()) / (x[c].std() + 1e-12)).values for c in MUS])
            ok = x.res.notna().values
            beta, *_ = np.linalg.lstsq(Xc[ok], x.res.values[ok], rcond=None)
            x["res"] = x.res - Xc @ beta
        y = x[x["round"].isin(["R1", "R2", "R3"])].assign(stage=lambda d: np.where(d["round"] == "R1", "R1", "later"))
        for sub, yy in [("all", y), ("junior", y[y.group == "junior"]), ("senior", y[y.group == "senior"])]:
            g = yy.groupby(["year", "pianist", "stage"]).res.mean().unstack().dropna()
            r = np.corrcoef(g.R1, g.later)[0, 1]
            rows.append({"feature": col, "adjustment": adj, "subset": sub, "r": r, "n": len(g), "ci": fisher_ci(r, len(g))})
R = pd.DataFrame(rows); R.to_csv(out / "test_retest_completion.csv", index=False)
print(R.assign(r=R.r.round(3)).drop(columns="ci").to_string(index=False))
