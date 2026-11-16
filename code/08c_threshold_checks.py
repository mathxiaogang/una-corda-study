"""Step 8c - H2a/H2b under alternative pedal-down thresholds (32, 96) and without transcriptions / pieces < 2 min.

Usage: python -I 08c_threshold_checks.py <data_dir> <results_dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import linalg, optimize

data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")

def share(d, col):
    d = d.dropna(subset=[col]).reset_index(drop=True)
    y = ((d[col] - d[col].mean()) / d[col].std()).values
    X = np.column_stack([np.ones(len(d)), (d.group == "senior").astype(float)])
    G = [(g[:, None] == g[None, :]).astype(float) for g in [pd.factorize(d.pianist)[0], pd.factorize(d.composer)[0], pd.factorize(d.year)[0]]]
    n = len(y)
    def nll(t):
        s = np.exp(t); V = s[0] * np.eye(n) + sum(s[k + 1] * Gk for k, Gk in enumerate(G)); c = linalg.cho_factor(V, lower=True)
        ViX = linalg.cho_solve(c, X); XtViX = X.T @ ViX; b = np.linalg.solve(XtViX, X.T @ linalg.cho_solve(c, y)); r = y - X @ b
        return 0.5 * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(XtViX)[1] + r @ linalg.cho_solve(c, r))
    o = optimize.minimize(nll, np.log(np.full(4, .25)), method="L-BFGS-B", bounds=[(-25, 5)] * 4)
    v = np.exp(o.x); return v[1] / v.sum()

def rounds_r(d, col):
    x = d.assign(res=d[col] - d.groupby("composer")[col].transform("mean"))
    x = x[x["round"].isin(["R1", "R2", "R3"])]
    a = x[x["round"] == "R1"].groupby(["year", "pianist"]).res.mean(); b = x[x["round"] != "R1"].groupby(["year", "pianist"]).res.mean()
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna(); return j.a.corr(j.b)

rows = []
for lab, d, s_col, u_col in [("threshold 64 (main)", D, "soft_frac", "sus_frac"), ("threshold 32", D, "soft_frac32", "sus_frac32"),
                             ("threshold 96", D, "soft_frac96", "sus_frac96"),
                             ("no transcriptions, >= 2 min", D[~D.is_transcription & (D.duration >= 120)], "soft_frac", "sus_frac")]:
    rows.append({"check": lab, "pianist_share_una_corda": share(d, s_col), "pianist_share_sustain": share(d, u_col),
                 "r_rounds_una_corda": rounds_r(d, s_col), "r_rounds_sustain": rounds_r(d, u_col), "n": len(d)})
T = pd.DataFrame(rows); T.to_csv(res_dir / "final" / "robustness_thresholds.csv", index=False)
print(T.round(3).to_string(index=False))
