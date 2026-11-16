"""Step 14 - supplementary analyses requested in the second review round (revision 2).

Usage:
    python -I -u 14_revision2_analyses.py <data_dir> <results_dir>

S1  Circularity check: is consensus loudness merely an echo of the OTHER pianists' una corda use (they play
    softer under the pedal, H1c)? Three pedal-purged references: (a) others' loudness residualised on their own
    una corda state; (b) others' loudness with their una corda windows masked; (c) consensus from non-users only.
S2  Agreement explained by the placement model: model-implied kappa for every pair under (i) each performance's own
    logistic fit on consensus loudness and (ii) a shared slope with personal intercepts; prevalence-adjusted
    agreement (PABAK) and pedal prevalence.
S3  Trait vs edition-specific: variance components among the 16 pianists who played two editions
    (pianist, pianist x edition, composer, edition), pianist-cluster bootstrap.
S4  Threshold vs extent: how closely the intercept tracks overall una corda share; slope-intercept correlation;
    pianist share of the slope after adjusting for the intercept.
S5  Event-locked consensus: around una corda presses in the aligned set, do the OTHER pianists also get softer
    (musical decrescendo) and by how much does the pianist's own loudness drop beyond them?

Writes results/revision2/*.csv and revision2_summary.json.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
out = res_dir / "revision2"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(20261116)
SUM = {}
WIN = 5


def dump():
    with open(out / "revision2_summary.json", "w") as fh:
        json.dump(SUM, fh, ensure_ascii=False, indent=1, default=float)


def reml_fast(y, X, groups):
    n = len(y); Zs = []
    for g in groups:
        g = pd.factorize(g)[0]; Z = np.zeros((n, g.max() + 1)); Z[np.arange(n), g] = 1; Zs.append(Z)
    Z = np.hstack(Zs); sizes = [z.shape[1] for z in Zs]; q = Z.shape[1]
    ZtZ, ZtX, Zty, XtX, Xty, yty = Z.T @ Z, Z.T @ X, Z.T @ y, X.T @ X, X.T @ y, y @ y

    def nll(t):
        s = np.exp(t); se = s[0]
        d = np.concatenate([np.full(k, s[i + 1]) for i, k in enumerate(sizes)])
        try:
            c = linalg.cho_factor(se * np.diag(1 / d) + ZtZ, lower=True)
        except linalg.LinAlgError:
            return 1e12
        MZX, MZy = linalg.cho_solve(c, ZtX), linalg.cho_solve(c, Zty)
        XVX, XVy, yVy = (XtX - ZtX.T @ MZX) / se, (Xty - ZtX.T @ MZy) / se, (yty - Zty @ MZy) / se
        logdetV = n * np.log(se) + np.log(d).sum() + 2 * np.log(np.diag(c[0])).sum() - q * np.log(se)
        b = np.linalg.solve(XVX, XVy)
        return 0.5 * (logdetV + np.linalg.slogdet(XVX)[1] + yVy - b @ XVy)

    t0 = np.log(np.full(len(groups) + 1, max(np.var(y), 1e-6) / (len(groups) + 1)))
    return np.exp(optimize.minimize(nll, t0, method="L-BFGS-B", bounds=[(-25, 5)] * len(t0)).x)


def pooled_mean(d, col):
    """Mixed-model mean of per-performance coefficients (crossed pianist and piece intercepts), GLS SE."""
    d = d.dropna(subset=[col]).reset_index(drop=True); n = len(d); y = d[col].values
    v = reml_fast(y, np.ones((n, 1)), [d.pianist.values, d.piece.values])
    P = (pd.factorize(d.pianist)[0][:, None] == pd.factorize(d.pianist)[0][None, :]).astype(float)
    Q = (pd.factorize(d.piece)[0][:, None] == pd.factorize(d.piece)[0][None, :]).astype(float)
    Vi1 = np.linalg.solve(v[0] * np.eye(n) + v[1] * P + v[2] * Q, np.ones(n)); se = 1 / np.sqrt(Vi1.sum()); b = (Vi1 @ y) * se ** 2
    return {"n": n, "pieces": int(d.piece.nunique()), "mean": float(b), "se": float(se), "z": float(b / se),
            "p": float(2 * stats.norm.sf(abs(b / se))), "share_negative": float((y < 0).mean())}


def logit_fit(y, X, lam=0.01):
    def nll(b):
        eta = X @ b
        return -(y * eta - np.logaddexp(0, eta)).sum() + lam * (b[1:] ** 2).sum()
    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b))); g = -X.T @ (y - p); g[1:] += 2 * lam * b[1:]; return g
    return optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x


def zs(a, axis=None):
    return (a - np.nanmean(a, axis=axis, keepdims=axis is not None)) / (np.nanstd(a, axis=axis, keepdims=axis is not None) + 1e-9)


def kappa(a, b):
    po = (a == b).mean(); pa, pb = a.mean(), b.mean(); pe = pa * pb + (1 - pa) * (1 - pb)
    return np.nan if pe >= 1 - 1e-9 else (po - pe) / (1 - pe)


# ------------------------------------------------------------------ load aligned pieces
pieces = {}
for f in sorted((res_dir / "aligned_frames").glob("*.npz")):
    z = np.load(f, allow_pickle=True); n, T = z["soft"].shape; nw = T // WIN
    soft = z["soft"][:, : nw * WIN].reshape(n, nw, WIN).mean(2) > 0.5
    sus = z["sus"][:, : nw * WIN].reshape(n, nw, WIN).mean(2) > 0.5
    with np.errstate(invalid="ignore"):
        loud = np.nanmean(z["loud"][:, : nw * WIN].reshape(n, nw, WIN), axis=2)
    loud = np.where(np.isnan(loud), np.nanmean(loud, axis=1, keepdims=True), loud)
    lz = zs(loud, axis=1)
    loo = zs((lz.sum(0, keepdims=True) - lz) / (n - 1), axis=1)
    pieces[str(z["piece"])] = dict(perf=z["perf"], pianist=z["pianist"], group=z["group"], year=z["year"], soft=soft, sus=sus,
                                   lz=lz, loo=loo, frame_soft=z["soft"], frame_loud=z["loud"])

# ================================================================== S1 pedal-purged consensus
rows = []
for pc, P in pieces.items():
    n = len(P["perf"]); soft, lz = P["soft"], P["lz"]
    share = soft.mean(1)
    # (a) residualise each performance's loudness on its own una corda state
    lres = lz.copy()
    for k in range(n):
        if 0 < share[k] < 1:
            m1, m0 = lz[k][soft[k]].mean(), lz[k][~soft[k]].mean()
            lres[k] = lz[k] - np.where(soft[k], m1, m0) + lz[k].mean()
    lres = zs(lres, axis=1)
    # (b) mask the others' una corda windows
    lmask = np.where(soft, np.nan, lz)
    for i in range(n):
        y = soft[i].astype(float)
        if not 0.05 < y.mean() < 0.95:
            continue
        others = np.arange(n) != i
        r = {"piece": pc, "perf": P["perf"][i], "pianist": P["pianist"][i]}
        one = np.ones(len(y))
        r["main"] = logit_fit(y, np.column_stack([one, P["loo"][i]]))[1]
        ca = zs(lres[others].mean(0)); r["residualised"] = logit_fit(y, np.column_stack([one, ca]))[1]
        with np.errstate(invalid="ignore"):
            cb = np.nanmean(lmask[others], axis=0)
        ok = ~np.isnan(cb)
        r["masked_windows"] = ok.mean()
        if ok.sum() > 50 and 0 < y[ok].mean() < 1:
            r["masked"] = logit_fit(y[ok], np.column_stack([np.ones(ok.sum()), zs(cb[ok])]))[1]
        non = others & (share <= 0.05)
        r["n_nonusers"] = int(non.sum())
        if non.sum() >= 1:
            r["nonusers"] = logit_fit(y, np.column_stack([one, zs(lz[non].mean(0))]))[1]
            r["main_same_targets"] = r["main"]
        rows.append(r)
S1 = pd.DataFrame(rows); S1.to_csv(out / "S1_purged_consensus_slopes.csv", index=False)
res = {k: pooled_mean(S1, k) for k in ["main", "residualised", "masked", "nonusers", "main_same_targets"]}
res["masked_median_window_coverage"] = float(S1.masked_windows.median())
d2 = S1.dropna(subset=["nonusers"]); res["nonusers_ge2"] = pooled_mean(d2[d2.n_nonusers >= 2], "nonusers")
res["nonusers_ge2_main_same_targets"] = pooled_mean(d2[d2.n_nonusers >= 2], "main")
SUM["S1"] = res; print("== S1", json.dumps(res, indent=1)); dump()

# ================================================================== S2 model-implied agreement
pair_rows = []
fits = {}
for pc, P in pieces.items():
    n = len(P["perf"])
    for ped in ("soft", "sus"):
        for i in range(n):
            y = P[ped][i].astype(float)
            if 0.05 < y.mean() < 0.95:
                fits[(pc, ped, i)] = logit_fit(y, np.column_stack([np.ones(len(y)), P["loo"][i]]))
b_mean = {ped: float(np.mean([v[1] for k, v in fits.items() if k[1] == ped])) for ped in ("soft", "sus")}


def intercept_for_rate(x, b, rate):
    """Intercept a such that mean(logistic(a + b x)) equals the observed rate."""
    f = lambda a: (1 / (1 + np.exp(-(a + b * x)))).mean() - rate
    return optimize.brentq(f, -30, 30)


for pc, P in pieces.items():
    n = len(P["perf"])
    for a_, b_ in itertools.combinations(range(n), 2):
        if P["pianist"][a_] == P["pianist"][b_]:
            continue
        for ped in ("soft", "sus"):
            if (pc, ped, a_) not in fits or (pc, ped, b_) not in fits:
                continue
            xa, xb = P[ped][a_], P[ped][b_]
            r = {"piece": pc, "pedal": ped, "kappa": kappa(xa, xb), "po": (xa == xb).mean(), "prev": (xa.mean() + xb.mean()) / 2}
            r["pabak"] = 2 * r["po"] - 1
            pe = xa.mean() * xb.mean() + (1 - xa.mean()) * (1 - xb.mean())
            for model in ("own", "shared"):
                ps = []
                for k in (a_, b_):
                    x = P["loo"][k]
                    if model == "own":
                        bb = fits[(pc, ped, k)]; p = 1 / (1 + np.exp(-(bb[0] + bb[1] * x)))
                    else:
                        aa = intercept_for_rate(x, b_mean[ped], P[ped][k].mean()); p = 1 / (1 + np.exp(-(aa + b_mean[ped] * x)))
                    ps.append(p)
                epo = (ps[0] * ps[1] + (1 - ps[0]) * (1 - ps[1])).mean()
                r[f"kappa_model_{model}"] = (epo - pe) / (1 - pe)
            pair_rows.append(r)
S2 = pd.DataFrame(pair_rows); S2.to_csv(out / "S2_model_implied_agreement_pairs.csv", index=False)
pm = S2.groupby(["pedal", "piece"])[["kappa", "kappa_model_own", "kappa_model_shared", "pabak", "prev"]].mean().reset_index()
res = {"b_mean": b_mean}
for ped in ("soft", "sus"):
    x = pm[pm.pedal == ped]
    ent = {"n_pairs": int((S2.pedal == ped).sum()), "n_pieces": len(x)}
    for c in ["kappa", "kappa_model_own", "kappa_model_shared", "pabak", "prev"]:
        bs = [rng.choice(x[c].values, len(x)).mean() for _ in range(2000)]
        ent[c] = {"mean": float(x[c].mean()), "boot95": [float(v) for v in np.percentile(bs, [2.5, 97.5])]}
    ent["ratio_shared_to_observed"] = float(x.kappa_model_shared.mean() / x.kappa.mean())
    ent["ratio_own_to_observed"] = float(x.kappa_model_own.mean() / x.kappa.mean())
    ent["wilcoxon_observed_vs_shared_p"] = float(stats.wilcoxon(x.kappa, x.kappa_model_shared).pvalue)
    res[ped] = ent
SUM["S2"] = res; print("== S2", json.dumps(res, indent=1)); dump()

# ================================================================== S3 trait vs edition-specific variance
meta = pd.read_csv(data_dir / "metadata_clean.csv"); feat = pd.read_csv(data_dir / "features.csv")
D = meta.merge(feat, on="midi_filename")
R = D[D.repeat_type != "single"].copy()
R["pe"] = R.pianist + "|" + R.year.astype(str)
res = {"n_perf": len(R), "n_pianists": int(R.pianist.nunique()), "n_pianist_editions": int(R.pe.nunique())}
rows = []
for col in ["soft_frac", "soft_episodes_pm", "sus_frac", "sus_partial", "vel_mean"]:
    def fit(d):
        d = d.dropna(subset=[col]).reset_index(drop=True)
        y = ((d[col] - d[col].mean()) / d[col].std()).values
        v = reml_fast(y, np.ones((len(d), 1)), [d.pid.values, d.pe.values, d.composer.values, d.year.values])
        return v / v.sum()
    R["pid"] = R.pianist
    pt = fit(R)
    bs = []
    pian = R.pianist.unique()
    for _ in range(300):
        pick = rng.choice(pian, len(pian))
        Rb = pd.concat([R[R.pianist == p].assign(pid=f"{p}#{k}", pe=lambda d, k=k: d.pe + f"#{k}") for k, p in enumerate(pick)], ignore_index=True)
        v = fit(Rb); bs.append([v[1], v[2], v[1] / (v[1] + v[2] + 1e-12)])
    bs = np.array(bs)
    rows.append({"feature": col, "share_pianist_stable": pt[1], "share_pianist_edition": pt[2], "share_composer": pt[3],
                 "share_edition": pt[4], "share_resid": pt[0], "stable_fraction": pt[1] / (pt[1] + pt[2] + 1e-12),
                 "stable_boot95": list(np.percentile(bs[:, 0], [2.5, 97.5])),
                 "stable_fraction_boot95": list(np.percentile(bs[:, 2], [2.5, 97.5]))})
S3 = pd.DataFrame(rows); S3.to_csv(out / "S3_trait_vs_edition.csv", index=False)
res["table"] = S3.round(4).to_dict("records")
SUM["S3"] = res; print("== S3\n", S3.round(3).to_string(index=False)); dump()

# ================================================================== S4 threshold vs extent
A = pd.read_csv(res_dir / "final" / "ext_A_rule_threshold_by_performance.csv").dropna(subset=["rule_b"]).reset_index(drop=True)
res = {"n": len(A),
       "r_threshold_logit_share": float(stats.pearsonr(A.threshold_a, np.log(A.share / (1 - A.share)))[0]),
       "r_threshold_share_spearman": float(stats.spearmanr(A.threshold_a, A.share).correlation),
       "r_rule_share_spearman": float(stats.spearmanr(A.rule_b, A.share).correlation),
       "r_threshold_rule_spearman": float(stats.spearmanr(A.threshold_a, A.rule_b).correlation)}
# pianist share of the rule after adjusting for the threshold (fixed effect)
y = ((A.rule_b - A.rule_b.mean()) / A.rule_b.std()).values
X = np.column_stack([np.ones(len(A)), (A.group == "senior").astype(float), zs(A.threshold_a.values)])
v = reml_fast(y, X, [A.pianist.values, A.piece.values, A.year.values]); res["rule_pianist_share_adj_threshold"] = float(v[1] / v.sum())
res["rule_piece_share_adj_threshold"] = float(v[2] / v.sum())
SUM["S4"] = res; print("== S4", json.dumps(res, indent=1)); dump()

# ================================================================== S5 event-locked consensus vs own loudness (aligned set, 10 fps)
lag = np.arange(-40, 41)  # +-4 s at 10 fps
prof = []
for pc, P in pieces.items():
    fl = P["frame_loud"]; n, T = fl.shape
    fl = np.where(np.isnan(fl), np.nanmean(fl, axis=1, keepdims=True), fl)
    fz = zs(fl, axis=1)
    for i in range(n):
        own = fz[i]; cons = zs(np.delete(fz, i, axis=0).mean(0))
        s = P["frame_soft"][i] > 0.5
        presses = np.where(np.diff(s.astype(int)) == 1)[0] + 1
        presses = presses[(presses >= 40) & (presses < T - 40)]
        if len(presses) < 3:
            continue
        idx = presses[:, None] + lag[None, :]
        prof.append({"piece": pc, "perf": P["perf"][i], "pianist": P["pianist"][i],
                     "own": own[idx].mean(0), "cons": cons[idx].mean(0)})
own = np.array([p["own"] for p in prof]); cons = np.array([p["cons"] for p in prof]); t = lag / 10
pre, post = t < -2, (t > 0) & (t < 2)
d_own = own[:, post].mean(1) - own[:, pre].mean(1); d_cons = cons[:, post].mean(1) - cons[:, pre].mean(1)
pd.DataFrame({"t": t, "own_mean": own.mean(0), "own_se": own.std(0) / np.sqrt(len(own)),
              "cons_mean": cons.mean(0), "cons_se": cons.std(0) / np.sqrt(len(cons))}).to_csv(out / "S5_event_locked_consensus.csv", index=False)
dd = pd.DataFrame({"pianist": [p["pianist"] for p in prof], "piece": [p["piece"] for p in prof], "d_own": d_own, "d_cons": d_cons,
                   "d_excess": d_own - d_cons})
res = {"n_perf": len(prof), "own_change": pooled_mean(dd, "d_own"), "consensus_change": pooled_mean(dd, "d_cons"),
       "excess_own_minus_consensus": pooled_mean(dd, "d_excess"),
       "share_consensus_of_own": float(d_cons.mean() / d_own.mean())}
# time at which each curve first falls 0.1 SD below its -4..-2 s baseline
for nm, curve in (("own", own.mean(0)), ("cons", cons.mean(0))):
    base = curve[pre].mean(); below = np.where((curve < base - 0.1) & (t > -2))[0]
    res[f"{nm}_onset_s"] = float(t[below[0]]) if len(below) else None
SUM["S5"] = res; print("== S5", json.dumps(res, indent=1)); dump()
