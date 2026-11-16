"""Step 8b - pianist-cluster bootstrap for H2a (variance share from pianist), run as parallel workers.

Usage:
    python -I -u 08b_h2a_bootstrap_worker.py <data_dir> <results_dir> seed=1 B=30

Each worker resamples pianists with replacement (duplicates relabelled as distinct pianists), refits the
crossed random-effects model (pianist, composer, edition; group fixed) and appends the pianist share for
each feature to results/final/h2a_boot_seed<seed>.csv after every replicate (safe to stop and resume).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize

data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
OPTS = dict(a.split("=", 1) for a in sys.argv[3:])
seed, B = int(OPTS.get("seed", 1)), int(OPTS.get("B", 30))
FEATS = ["soft_frac", "sus_frac", "vel_mean"]
out = res_dir / "final" / f"h2a_boot_seed{seed}.csv"
out.parent.mkdir(exist_ok=True)

D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")


def reml_shares(y, X, groups):
    n = len(y)
    G = [(g[:, None] == g[None, :]).astype(float) for g in groups]
    def nll(t):
        s = np.exp(t)
        V = s[0] * np.eye(n) + sum(s[k + 1] * Gk for k, Gk in enumerate(G))
        c = linalg.cho_factor(V, lower=True)
        ViX = linalg.cho_solve(c, X); XtViX = X.T @ ViX
        b = np.linalg.solve(XtViX, X.T @ linalg.cho_solve(c, y)); r = y - X @ b
        return 0.5 * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(XtViX)[1] + r @ linalg.cho_solve(c, r))
    t0 = np.log(np.full(len(G) + 1, 0.25))
    o = optimize.minimize(nll, t0, method="L-BFGS-B", bounds=[(-25, 5)] * len(t0))
    v = np.exp(o.x)
    return v / v.sum()


done = len(pd.read_csv(out)) if out.exists() else 0
rng = np.random.default_rng(1000 + seed)
pianists = D.pianist.unique()
for b in range(B):
    pick = rng.choice(pianists, len(pianists))  # drawn every time so a resumed run continues the same stream
    if b < done:
        continue
    Db = pd.concat([D[D.pianist == p].assign(pid=f"{p}#{k}") for k, p in enumerate(pick)], ignore_index=True)
    rec = {"seed": seed, "rep": b}
    for col in FEATS:
        d = Db.dropna(subset=[col]).reset_index(drop=True)
        y = ((d[col] - d[col].mean()) / d[col].std()).values
        X = np.column_stack([np.ones(len(d)), (d.group == "senior").astype(float)])
        rec[col] = reml_shares(y, X, [pd.factorize(d.pid)[0], pd.factorize(d.composer)[0], pd.factorize(d.year)[0]])[1]
    pd.DataFrame([rec]).to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"seed {seed}: replicate {b + 1}/{B}", flush=True)
