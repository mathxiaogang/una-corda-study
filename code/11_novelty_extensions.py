"""Step 11 - three extensions that sharpen the paper's contribution, using only the existing data.

Usage:
    python -I -u 11_novelty_extensions.py <data_dir> <results_dir>

A. Decision-rule model: within each aligned performance, P(una corda) = logistic(a + b * consensus quietness).
   Is the slope b (the rule) shared and the intercept a (the threshold) personal?
B. Signature test: can a pianist be picked out from the other pianists of the SAME edition (same piano, same
   competition) using only una corda features, only sustain features, or only dynamics features?
C. Style independence: does a pianist's una corda habit carry across style periods?

Writes results/final/ext_*.csv and ext_summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
out = res_dir / "final"
rng = np.random.default_rng(11)
D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")
D["log_episode_s"] = np.log(D.soft_episode_med_s)
SUM = {}


def reml(y, X, groups):
    n = len(y); G = [(g[:, None] == g[None, :]).astype(float) for g in groups]
    def nll(t):
        s = np.exp(t); V = s[0] * np.eye(n) + sum(s[k + 1] * Gk for k, Gk in enumerate(G)); c = linalg.cho_factor(V, lower=True)
        ViX = linalg.cho_solve(c, X); XtViX = X.T @ ViX; b = np.linalg.solve(XtViX, X.T @ linalg.cho_solve(c, y)); r = y - X @ b
        return 0.5 * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(XtViX)[1] + r @ linalg.cho_solve(c, r))
    o = optimize.minimize(nll, np.log(np.full(len(G) + 1, max(np.var(y), 1e-6) / (len(G) + 1))), method="L-BFGS-B",
                          bounds=[(-25, 5)] * (len(G) + 1))
    v = np.exp(o.x); return v / v.sum()


# ============================================================ A. shared rule, personal threshold
WIN = 5
rows = []
for f in sorted((res_dir / "aligned_frames").glob("*.npz")):
    z = np.load(f, allow_pickle=True)
    n, T = z["soft"].shape; nw = T // WIN
    soft = z["soft"][:, : nw * WIN].reshape(n, nw, WIN).mean(2) > 0.5
    with np.errstate(invalid="ignore"):
        loud = np.nanmean(z["loud"][:, : nw * WIN].reshape(n, nw, WIN), axis=2)
    loud = np.where(np.isnan(loud), np.nanmean(loud, axis=1, keepdims=True), loud)
    lz = (loud - loud.mean(1, keepdims=True)) / (loud.std(1, keepdims=True) + 1e-9)
    loo = (lz.sum(0, keepdims=True) - lz) / (n - 1)
    loo = (loo - loo.mean(1, keepdims=True)) / (loo.std(1, keepdims=True) + 1e-9)
    quiet = -loo  # higher = quieter according to the other pianists
    for i in range(n):
        y = soft[i].astype(float)
        rec = {"piece": str(z["piece"]), "perf": z["perf"][i], "pianist": z["pianist"][i], "group": z["group"][i],
               "year": int(z["year"][i]), "share": y.mean()}
        if 0.05 < y.mean() < 0.95:
            X = np.column_stack([np.ones(len(y)), quiet[i]])
            nll = lambda b: -(y * (X @ b) - np.logaddexp(0, X @ b)).sum() + 0.01 * (b ** 2).sum()
            b = optimize.minimize(nll, np.zeros(2), method="BFGS").x
            rec["threshold_a"], rec["rule_b"] = b[0], b[1]
        rows.append(rec)
A = pd.DataFrame(rows)
A.to_csv(out / "ext_A_rule_threshold_by_performance.csv", index=False)
U = A.dropna(subset=["rule_b"]).reset_index(drop=True)
resA = {"n_perf": len(U), "n_pianists": U.pianist.nunique(), "n_pieces": U.piece.nunique()}
for col in ["threshold_a", "rule_b"]:
    y = ((U[col] - U[col].mean()) / U[col].std()).values
    X = np.column_stack([np.ones(len(U)), (U.group == "senior").astype(float)])
    sh = reml(y, X, [pd.factorize(U.pianist)[0], pd.factorize(U.piece)[0], pd.factorize(U.year)[0]])
    bs = []
    pian = U.pianist.unique()
    for _ in range(200):
        pick = rng.choice(pian, len(pian))
        Ub = pd.concat([U[U.pianist == p].assign(pid=f"{p}#{k}") for k, p in enumerate(pick)], ignore_index=True)
        yb = ((Ub[col] - Ub[col].mean()) / Ub[col].std()).values
        Xb = np.column_stack([np.ones(len(Ub)), (Ub.group == "senior").astype(float)])
        bs.append(reml(yb, Xb, [pd.factorize(Ub.pid)[0], pd.factorize(Ub.piece)[0], pd.factorize(Ub.year)[0]])[1])
    resA[col] = {"mean": float(U[col].mean()), "sd": float(U[col].std()), "share_pianist": float(sh[1]),
                 "share_piece": float(sh[2]), "share_edition": float(sh[3]), "share_resid": float(sh[0]),
                 "pianist_boot95": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}
d = np.array([0])  # placeholder to keep names short
SUM["A_rule_threshold"] = resA
print("== A", json.dumps(resA, indent=1))


# ============================================================ B. signature test within edition
SETS = {"una corda": ["soft_frac", "soft_note_share", "soft_episodes_pm"],
        "sustain": ["sus_frac", "sus_partial", "sus_presses_pm"],
        "dynamics": ["vel_mean", "vel_sd", "dyn_range"]}
SETS["all nine"] = SETS["una corda"] + SETS["sustain"] + SETS["dynamics"]
F = D.copy()
for c in SETS["all nine"]:
    F[c] = F[c].fillna(F[c].median())
    F[c] = F[c] - F.groupby("composer")[c].transform("mean")       # remove repertoire
    F[c] = (F[c] - F[c].mean()) / F[c].std()

def identify(df, cols, shuffle=False):
    hits, chance = [], []
    for y, g in df.groupby("year"):
        g = g[g.groupby("pianist").pianist.transform("size") >= 3]
        if g.pianist.nunique() < 3:
            continue
        lab = g.pianist.values.copy()
        if shuffle:
            rng.shuffle(lab)
        Xg = g[cols].values
        for i in range(len(g)):
            m = np.ones(len(g), bool); m[i] = False
            cents = {p: Xg[m & (lab == p)].mean(0) for p in np.unique(lab[m])}
            pred = min(cents, key=lambda p: np.sum((Xg[i] - cents[p]) ** 2))
            hits.append(pred == lab[i]); chance.append(1 / len(cents))
    return float(np.mean(hits)), float(np.mean(chance)), len(hits)

resB = []
for name, cols in SETS.items():
    acc, ch, n = identify(F, cols)
    null = [identify(F, cols, shuffle=True)[0] for _ in range(100)]
    resB.append({"feature_set": name, "n_features": len(cols), "accuracy": acc, "chance": ch, "lift": acc / ch,
                 "perm_null_mean": float(np.mean(null)), "perm_p": float((np.sum(np.array(null) >= acc) + 1) / 101), "n_tests": n})
    print(name, round(acc, 3), "chance", round(ch, 3))
RB = pd.DataFrame(resB)
# paired comparison una corda vs dynamics / sustain by bootstrap over editions is overkill; use McNemar on per-test hits
def hits_vector(cols):
    out_ = []
    for y, g in F.groupby("year"):
        g = g[g.groupby("pianist").pianist.transform("size") >= 3]
        if g.pianist.nunique() < 3:
            continue
        Xg, lab = g[cols].values, g.pianist.values
        for i in range(len(g)):
            m = np.ones(len(g), bool); m[i] = False
            cents = {p: Xg[m & (lab == p)].mean(0) for p in np.unique(lab[m])}
            out_.append(min(cents, key=lambda p: np.sum((Xg[i] - cents[p]) ** 2)) == lab[i])
    return np.array(out_)
hu, hs, hd = hits_vector(SETS["una corda"]), hits_vector(SETS["sustain"]), hits_vector(SETS["dynamics"])
def mcnemar(a, b):
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
SUM["B_signature"] = {"table": RB.round(4).to_dict("records"),
                      "una_vs_sustain_mcnemar_p": mcnemar(hu, hs), "una_vs_dynamics_mcnemar_p": mcnemar(hu, hd)}
RB.to_csv(out / "ext_B_signature_identification.csv", index=False)
print("== B\n", RB.round(3).to_string(index=False), "\n", SUM["B_signature"])


# ============================================================ C. style independence
resC = []
for col in ["soft_frac", "soft_episodes_pm", "sus_frac", "sus_partial", "vel_mean", "dyn_range"]:
    x = D.assign(res=D[col] - D.groupby("composer")[col].transform("mean"))
    x["style"] = np.where(x.era == "Romantic", "Romantic", "other")
    g = x.groupby(["year", "pianist", "style"]).res.mean().unstack().dropna()
    r = np.corrcoef(g.Romantic, g.other)[0, 1]
    zz = np.arctanh(r); se = 1 / np.sqrt(len(g) - 3)
    resC.append({"feature": col, "r_romantic_vs_other": r, "ci_low": np.tanh(zz - 1.96 * se), "ci_high": np.tanh(zz + 1.96 * se), "n": len(g)})
RC = pd.DataFrame(resC)
RC.to_csv(out / "ext_C_style_independence.csv", index=False)
SUM["C_style"] = RC.round(4).to_dict("records")
print("== C\n", RC.round(3).to_string(index=False))

with open(out / "ext_summary.json", "w") as fh:
    json.dump(SUM, fh, ensure_ascii=False, indent=1, default=float)


# ============================================================ A2. paired bootstrap: threshold vs rule (same resamples)
def paired_diff(U, B=200, seed=12):
    r = np.random.default_rng(seed); pian = U.pianist.unique(); diffs = []
    for _ in range(B):
        pick = r.choice(pian, len(pian))
        Ub = pd.concat([U[U.pianist == p].assign(pid=f"{p}#{k}") for k, p in enumerate(pick)], ignore_index=True)
        Xb = np.column_stack([np.ones(len(Ub)), (Ub.group == "senior").astype(float)])
        g = [pd.factorize(Ub.pid)[0], pd.factorize(Ub.piece)[0], pd.factorize(Ub.year)[0]]
        sa = reml(((Ub.threshold_a - Ub.threshold_a.mean()) / Ub.threshold_a.std()).values, Xb, g)[1]
        sb = reml(((Ub.rule_b - Ub.rule_b.mean()) / Ub.rule_b.std()).values, Xb, g)[1]
        diffs.append(sa - sb)
    return np.array(diffs)
dd = paired_diff(U)
SUM["A_rule_threshold"]["threshold_minus_rule"] = {"point": resA["threshold_a"]["share_pianist"] - resA["rule_b"]["share_pianist"],
                                                   "boot95": [float(x) for x in np.percentile(dd, [2.5, 97.5])],
                                                   "p_one_sided": float((dd <= 0).mean())}
print("== A2", SUM["A_rule_threshold"]["threshold_minus_rule"])
with open(out / "ext_summary.json", "w") as fh:
    json.dump(SUM, fh, ensure_ascii=False, indent=1, default=float)
