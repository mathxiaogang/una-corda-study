"""Step 8 - final analyses for the paper
"Commonality in Placement, Individuality in Extent: A Corpus Study of Una Corda Use in 1,276 Expert Piano Performances".

Usage:
    python -I -u 08_final_analyses.py <maestro-v3.0.0 dir> <data_dir> <results_dir> [B=100]

Needs (from earlier steps): data/metadata_clean.csv, data/features.csv,
results/aligned_frames/*.npz and results/align_pairs_quality.csv (07 run with save_frames=1),
results/sensitivity/align_*_<tag>.csv (07 sensitivity runs), results/*.csv from 03 and 06.

Writes everything to results/final/ and prints a log. Random seeds are fixed.
"""
import itertools
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

root, data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OPTS = dict(a.split("=", 1) for a in sys.argv[4:])
B = int(OPTS.get("B", 100))
out = res_dir / "final"
out.mkdir(exist_ok=True)
rng = np.random.default_rng(20261115)
SUM = {}
WIN = 5  # frames per 0.5-s window at 10 fps

meta = pd.read_csv(data_dir / "metadata_clean.csv")
feat = pd.read_csv(data_dir / "features.csv")
D = meta.merge(feat, on="midi_filename")


def save(df, name):
    df.to_csv(out / name, index=False)


def reml(y, X, groups, return_beta=False):
    """Crossed random-intercept LMM by REML (dense Cholesky). Returns variances [resid, g1, g2, ...]."""
    n = len(y)
    G = [(g[:, None] == g[None, :]).astype(float) for g in groups]
    def parts(t):
        s = np.exp(t)
        V = s[0] * np.eye(n) + sum(s[k + 1] * Gk for k, Gk in enumerate(G))
        c = linalg.cho_factor(V, lower=True)
        ViX = linalg.cho_solve(c, X); XtViX = X.T @ ViX
        b = np.linalg.solve(XtViX, X.T @ linalg.cho_solve(c, y)); r = y - X @ b
        nll = 0.5 * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(XtViX)[1] + r @ linalg.cho_solve(c, r))
        return nll, b, XtViX
    t0 = np.log(np.full(len(G) + 1, max(np.var(y), 1e-6) / (len(G) + 1)))
    o = optimize.minimize(lambda t: parts(t)[0], t0, method="L-BFGS-B", bounds=[(-25, 5)] * len(t0))
    v = np.exp(o.x)
    if return_beta:
        _, b, XtViX = parts(o.x)
        return v, b, np.sqrt(np.diag(np.linalg.inv(XtViX)))
    return v


def shares(v, names):
    t = v.sum()
    return {f"share_{n}": v[i] / t for i, n in enumerate(names)}


# =====================================================================================
# A. load aligned frames -> 0.5-s windows with leave-one-out consensus loudness
# =====================================================================================
win_rows = []
pieces = {}
for f in sorted((res_dir / "aligned_frames").glob("*.npz")):
    z = np.load(f, allow_pickle=True)
    pc = str(z["piece"])
    n_perf, n_frames = z["soft"].shape
    nw = n_frames // WIN
    def wmean(a):
        return a[:, : nw * WIN].reshape(a.shape[0], nw, WIN).mean(axis=2)
    soft = wmean(z["soft"]) > 0.5
    sus = wmean(z["sus"]) > 0.5
    with np.errstate(invalid="ignore"):
        loud = np.nanmean(z["loud"][:, : nw * WIN].reshape(n_perf, nw, WIN), axis=2)
    loud = np.where(np.isnan(loud), np.nanmean(loud, axis=1, keepdims=True), loud)
    lz = (loud - loud.mean(axis=1, keepdims=True)) / (loud.std(axis=1, keepdims=True) + 1e-9)
    loo = (lz.sum(axis=0, keepdims=True) - lz) / (n_perf - 1)          # consensus of the OTHER pianists
    loo = (loo - loo.mean(axis=1, keepdims=True)) / (loo.std(axis=1, keepdims=True) + 1e-9)
    pieces[pc] = dict(perf=z["perf"], group=z["group"], year=z["year"], pianist=z["pianist"],
                      composer=z["composer"], soft=soft, sus=sus, loud=loud, lz=lz, loo=loo)
print("aligned pieces:", len(pieces), "| performances:", sum(len(p["perf"]) for p in pieces.values()))

Q = pd.read_csv(res_dir / "align_pairs_quality.csv")
counts = D.assign(piece=(D.composer + " | " + D.canonical_title.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
                         .str.replace(r"\s+", " ", regex=True).str.strip())).groupby("piece").size()
SUM["alignment"] = {
    "pieces_with_3plus": int((counts >= 3).sum()), "perfs_in_those_pieces": int(counts[counts >= 3].sum()),
    "candidates_vs_reference": len(Q), "excluded_duration_ratio": int((Q.status != "ok").sum()),
    "aligned_ok": int((Q.status == "ok").sum()), "cost_median": float(Q[Q.status == "ok"].cost.median()),
    "final_pieces": len(pieces), "final_perfs": int(sum(len(p["perf"]) for p in pieces.values())),
    "final_pianists": int(len({x for p in pieces.values() for x in p["pianist"]})),
    "final_windows": int(sum(p["soft"].size for p in pieces.values())),
}
print("== alignment", SUM["alignment"])


# =====================================================================================
# B. H1a - placement follows consensus loudness (per-performance logistic slopes)
# =====================================================================================
def logit_slope(y, x, lam=0.01):
    X = np.column_stack([np.ones(len(x)), x])
    def nll(b):
        eta = X @ b
        return -(y * eta - np.logaddexp(0, eta)).sum() + lam * (b[1] ** 2)
    b = optimize.minimize(nll, np.zeros(2), method="BFGS").x
    return b[1]

slopes = []
for pc, P in pieces.items():
    for i in range(len(P["perf"])):
        r = {"piece": pc, "perf": P["perf"][i], "pianist": P["pianist"][i], "group": P["group"][i], "year": int(P["year"][i])}
        for ped in ("soft", "sus"):
            y = P[ped][i].astype(float)
            r[f"{ped}_share"] = y.mean()
            r[f"{ped}_slope"] = logit_slope(y, P["loo"][i]) if 0.05 < y.mean() < 0.95 else np.nan
        slopes.append(r)
SL = pd.DataFrame(slopes)
save(SL, "h1a_performance_slopes.csv")

def piece_boot_mean(df, col, B=2000):
    pm = df.groupby("piece")[col].mean().dropna()
    bs = [rng.choice(pm.values, len(pm)).mean() for _ in range(B)]
    return pm.mean(), np.percentile(bs, [2.5, 97.5])

h1a = {}
for ped in ("soft", "sus"):
    d = SL.dropna(subset=[f"{ped}_slope"]).reset_index(drop=True)
    v, b, se = reml(d[f"{ped}_slope"].values, np.ones((len(d), 1)),
                    [pd.factorize(d.pianist)[0], pd.factorize(d.piece)[0]], return_beta=True)
    m, ci = piece_boot_mean(d, f"{ped}_slope")
    h1a[ped] = {"n_perf": len(d), "n_pieces": d.piece.nunique(), "lmm_mean_slope": float(b[0]), "lmm_se": float(se[0]),
                "z": float(b[0] / se[0]), "p": float(2 * stats.norm.sf(abs(b[0] / se[0]))),
                "odds_ratio_per_sd_louder": float(np.exp(b[0])), "piece_mean": float(m), "piece_boot95": [float(x) for x in ci],
                "share_negative": float((d[f"{ped}_slope"] < 0).mean())}
both = SL.dropna(subset=["soft_slope", "sus_slope"])
w = stats.wilcoxon(both.soft_slope, both.sus_slope)
h1a["soft_vs_sus_paired"] = {"n": len(both), "median_soft": float(both.soft_slope.median()),
                             "median_sus": float(both.sus_slope.median()), "wilcoxon_p": float(w.pvalue)}
SUM["H1a"] = h1a
print("== H1a", json.dumps(h1a, indent=1))

# quartile curves with piece-cluster bootstrap
qrows = []
for pc, P in pieces.items():
    for i in range(len(P["perf"])):
        qcut = pd.qcut(pd.Series(P["loo"][i]).rank(method="first"), 4, labels=[1, 2, 3, 4]).values
        for q in (1, 2, 3, 4):
            m = qcut == q
            qrows.append({"piece": pc, "perf": P["perf"][i], "group": P["group"][i], "quartile": q,
                          "soft": P["soft"][i][m].mean(), "sus": P["sus"][i][m].mean()})
QR = pd.DataFrame(qrows)
qsum = []
for g, x in [("all", QR), ("junior", QR[QR.group == "junior"]), ("senior", QR[QR.group == "senior"])]:
    for q in (1, 2, 3, 4):
        xx = x[x.quartile == q]
        for ped in ("soft", "sus"):
            m, ci = piece_boot_mean(xx, ped, 1000)
            qsum.append({"group": g, "quartile": q, "pedal": ped, "mean": m, "lo": ci[0], "hi": ci[1], "n_perf": len(xx)})
QS = pd.DataFrame(qsum)
save(QS, "h1a_quartile_curves.csv")
print(QS[QS.group == "all"].round(3).to_string(index=False))


# =====================================================================================
# C. H1b - placement agreement vs chance (circular-shift baseline)
# =====================================================================================
def kappa(a, b):
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return np.nan if pe >= 1 - 1e-9 else (po - pe) / (1 - pe)

pairs = []
for pc, P in pieces.items():
    n = len(P["perf"]); nw = P["soft"].shape[1]
    for a, b in itertools.combinations(range(n), 2):
        if P["pianist"][a] == P["pianist"][b]:
            continue
        rec = {"piece": pc, "same_edition": P["year"][a] == P["year"][b],
               "pair_type": "-".join(sorted([str(P["group"][a]), str(P["group"][b])]))}
        shifts = rng.integers(int(0.1 * nw), int(0.9 * nw), size=20)
        for ped in ("soft", "sus"):
            xa, xb = P[ped][a], P[ped][b]
            ok = (0.05 < xa.mean() < 0.95) and (0.05 < xb.mean() < 0.95)
            rec[f"k_{ped}"] = kappa(xa, xb) if ok else np.nan
            rec[f"k_{ped}_chance"] = np.mean([kappa(xa, np.roll(xb, s)) for s in shifts]) if ok else np.nan
        la, lb = P["lz"][a], P["lz"][b]
        rec["r_loud"] = np.corrcoef(la, lb)[0, 1]
        rec["r_loud_chance"] = np.mean([np.corrcoef(la, np.roll(lb, s))[0, 1] for s in shifts])
        pairs.append(rec)
PR = pd.DataFrame(pairs)
save(PR, "h1b_pairs.csv")
pm = PR.groupby("piece")[["k_soft", "k_soft_chance", "k_sus", "k_sus_chance", "r_loud", "r_loud_chance"]].mean()
h1b = {}
for col in pm.columns:
    x = pm[col].dropna()
    bs = [rng.choice(x.values, len(x)).mean() for _ in range(2000)]
    h1b[col] = {"mean": float(x.mean()), "boot95": [float(v) for v in np.percentile(bs, [2.5, 97.5])], "n_pieces": len(x)}
cmp = pm[["k_soft", "k_soft_chance"]].dropna(); h1b["soft_vs_chance_p"] = float(stats.wilcoxon(cmp.k_soft, cmp.k_soft_chance).pvalue)
cmp = pm[["k_soft", "k_sus"]].dropna()
h1b["soft_vs_sus"] = {"n_pieces": len(cmp), "diff_mean": float((cmp.k_soft - cmp.k_sus).mean()),
                      "pieces_soft_lower": int((cmp.k_soft < cmp.k_sus).sum()), "wilcoxon_p": float(stats.wilcoxon(cmp.k_soft, cmp.k_sus).pvalue)}
for lab, sub in [("same_edition", PR[PR.same_edition]), ("different_edition", PR[~PR.same_edition])]:
    h1b[f"k_soft_{lab}"] = float(sub.groupby("piece").k_soft.mean().dropna().mean())
SUM["H1b"] = h1b
print("== H1b", json.dumps(h1b, indent=1))


# =====================================================================================
# D. H1c - event-locked velocity profiles (full corpus)
# =====================================================================================
def _vlq(b, i):
    v = 0
    while True:
        c = b[i]; i += 1; v = (v << 7) | (c & 0x7F)
        if c < 0x80:
            return v, i

def load(path):
    b = path.read_bytes(); hl = struct.unpack(">I", b[4:8])[0]; ntrk, div = struct.unpack(">HH", b[10:14])
    i, ev = 8 + hl, []
    for _ in range(ntrk):
        L = struct.unpack(">I", b[i + 4:i + 8])[0]; j, end, tick, rs = i + 8, i + 8 + L, 0, None
        while j < end:
            d, j = _vlq(b, j); tick += d; s = b[j]
            if s == 0xFF:
                l, j = _vlq(b, j + 2); j += l; continue
            if s in (0xF0, 0xF7):
                l, j = _vlq(b, j + 1); j += l; continue
            if s & 0x80:
                rs = s; j += 1
            st = rs & 0xF0
            if st in (0xC0, 0xD0):
                j += 1; continue
            ev.append((tick, st, b[j], b[j + 1])); j += 2
        i = end
    ev = np.array(sorted(ev), dtype=float); ev[:, 0] *= 0.5 / div
    return ev

edges = np.arange(-4.0, 4.0001, 0.25)
centres = (edges[:-1] + edges[1:]) / 2
prof = []
PF_CACHE = out / "h1c_profiles_by_performance.csv"
for f, grp, pianist in ([] if PF_CACHE.exists() else zip(D.midi_filename, D.group, D.pianist)):
    ev = load(root / f)
    on = ev[(ev[:, 1] == 0x90) & (ev[:, 3] > 0)]
    t, v = on[:, 0], on[:, 3]
    vz = (v - v.mean()) / (v.std() + 1e-9)
    c = ev[(ev[:, 1] == 0xB0) & (ev[:, 2] == 67)]
    x = (c[:, 3] >= 64).astype(int)
    ev_t = {"press": c[np.where(np.diff(x) == 1)[0] + 1, 0], "release": c[np.where(np.diff(x) == -1)[0] + 1, 0]}
    if len(ev_t["press"]) < 3 or t.max() < 10:
        continue
    ev_t["random"] = rng.uniform(4, t.max() - 4, size=max(len(ev_t["press"]), 20))
    for kind, times in ev_t.items():
        times = times[(times > 4) & (times < t.max() - 4)]
        if len(times) < 3:
            continue
        rel = (t[None, :] - times[:, None])
        sums = np.zeros(len(centres)); cnts = np.zeros(len(centres))
        m = (rel >= -4) & (rel < 4)
        idx = np.floor((rel[m] + 4) / 0.25).astype(int)
        np.add.at(sums, idx, np.broadcast_to(vz, rel.shape)[m]); np.add.at(cnts, idx, 1)
        p = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        prof.append(pd.DataFrame({"perf": f, "group": grp, "pianist": pianist, "kind": kind, "t": centres, "vz": p}))
if PF_CACHE.exists():  # reuse the profiles computed in an earlier (interrupted) run
    PF = pd.read_csv(PF_CACHE)
else:
    PF = pd.concat(prof)
    save(PF, "h1c_profiles_by_performance.csv")
G = PF.groupby(["kind", "t"]).vz.agg(["mean", "std", "count"]).reset_index()
G["lo"] = G["mean"] - 1.96 * G["std"] / np.sqrt(G["count"]); G["hi"] = G["mean"] + 1.96 * G["std"] / np.sqrt(G["count"])
save(G, "h1c_profiles_mean.csv")
Gg = PF.groupby(["group", "kind", "t"]).vz.mean().reset_index()
save(Gg, "h1c_profiles_by_group.csv")
pr = G[G.kind == "press"].set_index("t")["mean"]
pre = pr[(pr.index < -2)].mean(); post = pr[(pr.index > 0) & (pr.index < 2)].mean()
half = pre + (post - pre) / 2
cross = pr[pr.index < 2]
t_half = float(cross[cross <= half].index.min())
h1c = {"n_perf": int(PF[PF.kind == "press"].perf.nunique()), "press_pre_mean_sd": float(pre), "press_post_mean_sd": float(post),
       "press_drop_sd": float(post - pre), "time_half_drop_s": t_half,
       "release_change_sd": float(G[(G.kind == "release") & (G.t > 0) & (G.t < 2)]["mean"].mean() - G[(G.kind == "release") & (G.t < -2)]["mean"].mean()),
       "random_change_sd": float(G[(G.kind == "random") & (G.t > 0) & (G.t < 2)]["mean"].mean() - G[(G.kind == "random") & (G.t < -2)]["mean"].mean())}
# per-performance drop, group comparison
pp = PF[PF.kind == "press"].pivot_table(index=["perf", "group"], columns="t", values="vz")
drop = (pp.loc[:, (pp.columns > 0) & (pp.columns < 2)].mean(axis=1) - pp.loc[:, pp.columns < -2].mean(axis=1)).reset_index(name="drop")
h1c["share_perf_with_drop"] = float((drop["drop"] < 0).mean())
h1c["drop_by_group"] = drop.groupby("group")["drop"].mean().to_dict()
h1c["group_mwu_p"] = float(stats.mannwhitneyu(drop[drop.group == "junior"]["drop"], drop[drop.group == "senior"]["drop"]).pvalue)
SUM["H1c"] = h1c
print("== H1c", json.dumps(h1c, indent=1))


# =====================================================================================
# E. H2a - variance decomposition with pianist-cluster bootstrap
# =====================================================================================
FEATS = ["soft_frac", "soft_note_share", "soft_episodes_pm", "sus_frac", "sus_partial", "sus_presses_pm",
         "press_lag_ms", "vel_mean", "dyn_range", "nps"]
def decomp(d, col):
    d = d.dropna(subset=[col]).reset_index(drop=True)
    y = ((d[col] - d[col].mean()) / d[col].std()).values
    X = np.column_stack([np.ones(len(d)), (d.group == "senior").astype(float)])
    v = reml(y, X, [pd.factorize(d.pid)[0], pd.factorize(d.composer)[0], pd.factorize(d.year)[0]])
    return v[1] / v.sum(), v[2] / v.sum(), v[3] / v.sum()

D["pid"] = D.pianist
point = {c: decomp(D, c) for c in FEATS}
# bootstrap replicates come from the parallel workers (08b_h2a_bootstrap_worker.py)
bfiles = sorted(out.glob("h2a_boot_seed*.csv"))
BT = pd.concat([pd.read_csv(f) for f in bfiles], ignore_index=True) if bfiles else pd.DataFrame()
boot = {c: BT[c].dropna().tolist() for c in ["soft_frac", "sus_frac", "vel_mean"]} if len(BT) else {}
B = len(BT)
print("H2a bootstrap replicates found:", B)
h2a_rows = []
for c in FEATS:
    r = {"feature": c, "share_pianist": point[c][0], "share_composer": point[c][1], "share_edition": point[c][2]}
    if c in boot and len(boot[c]):
        r["boot_low"], r["boot_high"] = np.percentile(boot[c], [2.5, 97.5])
    h2a_rows.append(r)
H2A = pd.DataFrame(h2a_rows)
save(H2A, "h2a_variance_shares.csv")
diffs = {}
for c in ["sus_frac", "vel_mean"]:
    d = np.array(boot["soft_frac"]) - np.array(boot[c])
    diffs[f"soft_minus_{c}"] = {"point": point["soft_frac"][0] - point[c][0], "boot95": [float(x) for x in np.percentile(d, [2.5, 97.5])],
                                "p_one_sided": float((d <= 0).mean())}
SUM["H2a"] = {"table": H2A.round(4).to_dict("records"), "differences": diffs, "B": B}
print("== H2a\n", H2A.round(3).to_string(index=False), "\n", json.dumps(diffs, indent=1))


# =====================================================================================
# F. H2b - test-retest stability (rounds within edition; editions)
# =====================================================================================
def adj(col):
    return D[col] - D.groupby("composer")[col].transform("mean")

def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]

h2b_rows, round_tables = [], {}
for col in ["soft_frac", "soft_episodes_pm", "sus_frac", "sus_partial", "vel_mean", "dyn_range"]:
    x = D.assign(res=adj(col))
    y = x[x["round"].isin(["R1", "R2", "R3"])].assign(stage=lambda d: np.where(d["round"] == "R1", "R1", "later"))
    g = y.groupby(["year", "pianist", "stage"]).res.mean().unstack().dropna()
    round_tables[col] = g
    rep = x[x.repeat_type != "single"].groupby(["pianist", "year"]).res.mean().reset_index().sort_values("year")
    a, b = rep.groupby("pianist").res.first(), rep.groupby("pianist").res.last()
    r1, r2 = np.corrcoef(g.R1, g.later)[0, 1], np.corrcoef(a, b)[0, 1]
    h2b_rows.append({"feature": col, "r_rounds": r1, "n_rounds": len(g), "ci_rounds": fisher_ci(r1, len(g)),
                     "r_editions": r2, "n_editions": len(a), "ci_editions": fisher_ci(r2, len(a))})
H2B = pd.DataFrame(h2b_rows)
save(H2B, "h2b_test_retest.csv")
gs, gu = round_tables["soft_frac"], round_tables["sus_frac"]
common = gs.index.intersection(gu.index)
bs = []
for _ in range(5000):
    idx = rng.choice(len(common), len(common))
    s_, u_ = gs.loc[common].iloc[idx], gu.loc[common].iloc[idx]
    bs.append(np.corrcoef(s_.R1, s_.later)[0, 1] - np.corrcoef(u_.R1, u_.later)[0, 1])
SUM["H2b"] = {"table": H2B.round(4).to_dict("records"),
              "rounds_soft_minus_sus": {"point": float(H2B.loc[0, "r_rounds"] - H2B.loc[2, "r_rounds"]),
                                        "boot95": [float(v) for v in np.percentile(bs, [2.5, 97.5])],
                                        "p_one_sided": float((np.array(bs) <= 0).mean())}}
print("== H2b\n", H2B.round(3).to_string(index=False), "\n", SUM["H2b"]["rounds_soft_minus_sus"])


# =====================================================================================
# G. H2c - who decides "whether" at matched passages + commonality/individuality map
# =====================================================================================
cov = []
for pc, P in pieces.items():
    for i in range(len(P["perf"])):
        quiet = P["loo"][i] <= np.quantile(P["loo"][i], 0.25)
        cov.append({"piece": pc, "perf": P["perf"][i], "pianist": P["pianist"][i], "group": P["group"][i],
                    "year": int(P["year"][i]), "soft_quiet": P["soft"][i][quiet].mean(), "soft_all": P["soft"][i].mean(),
                    "sus_all": P["sus"][i].mean(), "loud_level": np.nanmean(P["loud"][i])})
CV = pd.DataFrame(cov)
save(CV, "h2c_quiet_coverage.csv")

def decomp_aligned(d, col):
    d = d.reset_index(drop=True)
    y = ((d[col] - d[col].mean()) / (d[col].std() + 1e-12)).values
    X = np.column_stack([np.ones(len(d)), (d.group == "senior").astype(float)])
    v = reml(y, X, [pd.factorize(d.pid)[0], pd.factorize(d.piece)[0], pd.factorize(d.year)[0]])
    return v / v.sum()

CV["pid"] = CV.pianist
h2c = {}
pian = CV.pianist.unique()
for col in ["soft_quiet", "soft_all", "sus_all", "loud_level"]:
    pt = decomp_aligned(CV, col)
    bsv = []
    for _ in range(200):
        pick = rng.choice(pian, len(pian))
        Cb = pd.concat([CV[CV.pianist == p].assign(pid=f"{p}#{k}") for k, p in enumerate(pick)], ignore_index=True)
        bsv.append(decomp_aligned(Cb, col)[1])
    h2c[col] = {"share_pianist": float(pt[1]), "share_piece": float(pt[2]), "share_edition": float(pt[3]),
                "share_resid": float(pt[0]), "pianist_boot95": [float(x) for x in np.percentile(bsv, [2.5, 97.5])]}
SUM["H2c"] = h2c
print("== H2c", json.dumps(h2c, indent=1))

cmap = pd.DataFrame([
    {"parameter": "una corda", "placement_agreement": h1b["k_soft"]["mean"], "pa_lo": h1b["k_soft"]["boot95"][0],
     "pa_hi": h1b["k_soft"]["boot95"][1], "agreement_measure": "kappa", "extent_pianist_share": h2c["soft_all"]["share_pianist"],
     "ex_lo": h2c["soft_all"]["pianist_boot95"][0], "ex_hi": h2c["soft_all"]["pianist_boot95"][1]},
    {"parameter": "sustain", "placement_agreement": h1b["k_sus"]["mean"], "pa_lo": h1b["k_sus"]["boot95"][0],
     "pa_hi": h1b["k_sus"]["boot95"][1], "agreement_measure": "kappa", "extent_pianist_share": h2c["sus_all"]["share_pianist"],
     "ex_lo": h2c["sus_all"]["pianist_boot95"][0], "ex_hi": h2c["sus_all"]["pianist_boot95"][1]},
    {"parameter": "loudness", "placement_agreement": h1b["r_loud"]["mean"], "pa_lo": h1b["r_loud"]["boot95"][0],
     "pa_hi": h1b["r_loud"]["boot95"][1], "agreement_measure": "r", "extent_pianist_share": h2c["loud_level"]["share_pianist"],
     "ex_lo": h2c["loud_level"]["pianist_boot95"][0], "ex_hi": h2c["loud_level"]["pianist_boot95"][1]}])
save(cmap, "h2c_commonality_individuality_map.csv")
print(cmap.round(3).to_string(index=False))


# =====================================================================================
# H. Robustness of alignment-based results
# =====================================================================================
rob = []
def row_from(tag, folder):
    sfx = f"_{tag}" if tag else ""
    try:
        S = pd.read_csv(folder / f"align_agreement_summary{sfx}.csv").set_index("subset")
        QT = pd.read_csv(folder / f"align_by_loudness_quartile{sfx}.csv")
        VD = pd.read_csv(folder / f"align_variance_quiet_coverage{sfx}.csv").set_index("outcome")
        Qq = pd.read_csv(folder / f"align_pairs_quality{sfx}.csv")
    except FileNotFoundError:
        return None
    a = QT[QT.group == "all"].set_index("quartile")
    return {"setting": tag or "main (10 fps, band 12%, median reference)",
            "aligned_ok": int((Qq.status == "ok").sum()), "kappa_soft": S.loc["all pairs", "kappa_soft_mean"],
            "kappa_sus": S.loc["all pairs", "kappa_sus_mean"], "r_loud": S.loc["all pairs", "r_loud_mean"],
            "soft_Q1": a.loc["Q1", "soft"], "soft_Q4": a.loc["Q4", "soft"], "sus_Q1": a.loc["Q1", "sustain"], "sus_Q4": a.loc["Q4", "sustain"],
            "pianist_share_quiet_coverage": VD.loc["soft_in_consensus_quiet", "share_pianist"],
            "edition_share_quiet_coverage": VD.loc["soft_in_consensus_quiet", "share_edition"]}
for tag, folder in [("", res_dir), ("fps5", res_dir / "sensitivity"), ("fps20", res_dir / "sensitivity"),
                    ("band08", res_dir / "sensitivity"), ("band20", res_dir / "sensitivity"), ("reflongest", res_dir / "sensitivity")]:
    r = row_from(tag, folder)
    if r:
        rob.append(r)
# 1-s windows from the saved frames
k1 = []
for pc, P in pieces.items():
    s2 = P["soft"][:, : (P["soft"].shape[1] // 2) * 2].reshape(len(P["perf"]), -1, 2).mean(axis=2) > 0.5
    u2 = P["sus"][:, : (P["sus"].shape[1] // 2) * 2].reshape(len(P["perf"]), -1, 2).mean(axis=2) > 0.5
    for a, b in itertools.combinations(range(len(P["perf"])), 2):
        if P["pianist"][a] == P["pianist"][b]:
            continue
        ok = 0.05 < s2[a].mean() < 0.95 and 0.05 < s2[b].mean() < 0.95
        k1.append({"piece": pc, "k_soft": kappa(s2[a], s2[b]) if ok else np.nan, "k_sus": kappa(u2[a], u2[b])})
K1 = pd.DataFrame(k1).groupby("piece").mean()
rob.append({"setting": "1-s windows", "kappa_soft": K1.k_soft.mean(), "kappa_sus": K1.k_sus.mean()})
ROB = pd.DataFrame(rob)
save(ROB, "robustness_alignment.csv")
print("== robustness\n", ROB.round(3).to_string(index=False))


# =====================================================================================
# I. RQ3 exploratory (summarised from earlier steps)
# =====================================================================================
s03 = json.load(open(res_dir / "summary.json"))
mixed = pd.read_csv(res_dir / "align_mixed_pieces_group_contrast.csv").iloc[0].to_dict()
rq4 = pd.read_csv(res_dir / "rq4_advancement.csv")
SUM["RQ3"] = {"group_composer_adjusted_pp": s03["H1a"]["diff_pp"], "group_perm_p": s03["H1a"]["perm_p"],
              "matched_pieces_wholepiece": s03["matched_pieces"], "matched_passages": mixed,
              "advancement": rq4.round(4).to_dict("records")}

# =====================================================================================
# J. Holm correction for the three confirmatory tests
# =====================================================================================
conf = {"H1a (una corda slope < 0)": h1a["soft"]["p"],
        "H2a (pianist share: una corda > sustain)": max(diffs["soft_minus_sus_frac"]["p_one_sided"], 1 / (B + 1)),
        "H2b (round-to-round r: una corda > sustain)": max(SUM["H2b"]["rounds_soft_minus_sus"]["p_one_sided"], 1 / 5001)}
order = sorted(conf, key=conf.get)
holm, running = {}, 0
for k, name in enumerate(order):
    adjp = min(1.0, (len(order) - k) * conf[name]); running = max(running, adjp); holm[name] = {"p": conf[name], "p_holm": running}
SUM["confirmatory_holm"] = holm
print("== Holm", json.dumps(holm, indent=1))

with open(out / "final_summary.json", "w") as fh:
    json.dump(SUM, fh, ensure_ascii=False, indent=1, default=float)
print("wrote", sorted(p.name for p in out.iterdir()))
