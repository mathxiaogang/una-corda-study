"""Step 10 - independent re-computation of headline numbers in results/final (different code paths).

Usage:
    python -I 10_verify_final_numbers.py <data_dir> <results_dir>
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
S = json.load(open(res_dir / "final" / "final_summary.json"))
D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")
ok = []

def check(name, mine, reported, tol):
    good = abs(mine - reported) <= tol
    ok.append(good)
    print(f"{'OK ' if good else 'BAD'} {name}: recomputed {mine:.4f} vs reported {reported:.4f}")

# prevalence
check("share of playing time with una corda", (D.soft_frac * D.duration).sum() / D.duration.sum(), 68.726 / 198.653, 0.001)

# quartile curve and kappa from the saved frames, written without numpy reshaping tricks
q1, q4, ks, ku = [], [], [], []
for f in sorted((res_dir / "aligned_frames").glob("*.npz")):
    z = np.load(f, allow_pickle=True)
    n, T = z["soft"].shape
    nw = T // 5
    soft = np.array([[z["soft"][i, w * 5:(w + 1) * 5].mean() > 0.5 for w in range(nw)] for i in range(n)])
    sus = np.array([[z["sus"][i, w * 5:(w + 1) * 5].mean() > 0.5 for w in range(nw)] for i in range(n)])
    loud = pd.DataFrame(z["loud"][:, : nw * 5].reshape(n, nw, 5).transpose(0, 2, 1).reshape(n * 5, nw)).T
    lw = np.array([np.nanmean(z["loud"][i, : nw * 5].reshape(nw, 5), axis=1) for i in range(n)])
    lw = np.where(np.isnan(lw), np.nanmean(lw, axis=1, keepdims=True), lw)
    lz = (lw - lw.mean(1, keepdims=True)) / lw.std(1, keepdims=True)
    piece_q1, piece_q4 = [], []
    for i in range(n):
        others = np.delete(lz, i, axis=0).mean(0)
        r = pd.Series(others).rank(method="first")
        q = pd.qcut(r, 4, labels=False).values
        piece_q1.append(soft[i][q == 0].mean()); piece_q4.append(soft[i][q == 3].mean())
    q1.append(np.mean(piece_q1)); q4.append(np.mean(piece_q4))
    kk_s, kk_u = [], []
    for a, b in combinations(range(n), 2):
        if z["pianist"][a] == z["pianist"][b]:
            continue
        for arr, store in ((soft, kk_s), (sus, kk_u)):
            x, y = arr[a], arr[b]
            if 0.05 < x.mean() < 0.95 and 0.05 < y.mean() < 0.95:
                po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
                store.append((po - pe) / (1 - pe))
    if kk_s: ks.append(np.mean(kk_s))
    if kk_u: ku.append(np.mean(kk_u))
qs = pd.read_csv(res_dir / "final" / "h1a_quartile_curves.csv")
rep = qs[(qs.group == "all") & (qs.pedal == "soft")].set_index("quartile")["mean"]
check("una corda share, quietest quartile (piece mean)", np.mean(q1), rep.loc[1], 0.005)
check("una corda share, loudest quartile (piece mean)", np.mean(q4), rep.loc[4], 0.005)
check("kappa una corda (piece mean)", np.mean(ks), S["H1b"]["k_soft"]["mean"], 0.005)
check("kappa sustain (piece mean)", np.mean(ku), S["H1b"]["k_sus"]["mean"], 0.005)

# test-retest across rounds
def rounds_r(col):
    x = D.assign(res=D[col] - D.groupby("composer")[col].transform("mean"))
    x = x[x["round"].isin(["R1", "R2", "R3"])]
    a = x[x["round"] == "R1"].groupby(["year", "pianist"]).res.mean()
    b = x[x["round"] != "R1"].groupby(["year", "pianist"]).res.mean()
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return j.a.corr(j.b)
tab = {r["feature"]: r for r in S["H2b"]["table"]}
check("round-to-round r, una corda", rounds_r("soft_frac"), tab["soft_frac"]["r_rounds"], 0.001)
check("round-to-round r, sustain", rounds_r("sus_frac"), tab["sus_frac"]["r_rounds"], 0.001)
print(f"\n{sum(ok)} of {len(ok)} checks passed")
