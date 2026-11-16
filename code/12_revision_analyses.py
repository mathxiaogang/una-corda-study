"""Step 12 - supplementary analyses requested in peer review (revision 1).

Usage:
    python -I -u 12_revision_analyses.py <maestro-v3.0.0 dir> <data_dir> <results_dir> [part=all|R1|...] [B=1000]

R1  Fast REML (Woodbury identity) - checks that it reproduces step 8, then a 1,000-replicate
    pianist-cluster bootstrap for all ten H2a measures (replaces the 60-replicate bootstrap).
R2  H2a robustness: rank-normal scores; una corda users only; performance-level musical covariates
    (register, texture, density, duration, style); title-level random effect; junior-only and senior-only fits.
R3  H2b replication within junior and within senior editions; covariate-adjusted test-retest.
R4  H1a controls: per-performance logistic models adding texture covariates from the reference performance
    (register, note density) and the pianist's own loudness.
R5  H1c control: velocity profile around SUSTAIN presses with the same pipeline.
R6  Rule-threshold reliability: split-half (interleaved 10-s blocks) reliability of a and b; disattenuated shares.
R7  Partial una corda depth and identification among una corda users only.

Writes results/revision/*.csv and revision_summary_<part>.json; run part=merge to combine them into revision_summary.json.
"""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

root, data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OPTS = dict(a.split("=", 1) for a in sys.argv[4:])
PART = OPTS.get("part", "all")
B = int(OPTS.get("B", 1000))
out = res_dir / "revision"
out.mkdir(exist_ok=True)
SUMF = out / f"revision_summary_{PART}.json"  # one file per part, merged by part=merge
SUM = {}
rng = np.random.default_rng(20261115)

meta = pd.read_csv(data_dir / "metadata_clean.csv")
feat = pd.read_csv(data_dir / "features.csv")
D = meta.merge(feat, on="midi_filename")
FEATS = ["soft_frac", "soft_note_share", "soft_episodes_pm", "sus_frac", "sus_partial", "sus_presses_pm",
         "press_lag_ms", "vel_mean", "dyn_range", "nps"]


def run(p):
    return PART in ("all", p)


def dump():
    with open(SUMF, "w") as fh:
        json.dump(SUM, fh, ensure_ascii=False, indent=1, default=float)


# ------------------------------------------------------------------ fast REML (Woodbury)
def reml_fast(y, X, groups):
    """Crossed random-intercept LMM by REML using the Woodbury identity.
    groups: list of integer code arrays. Returns variance components [resid, g1, g2, ...]."""
    n = len(y)
    Zs = []
    for g in groups:
        g = pd.factorize(g)[0]
        Z = np.zeros((n, g.max() + 1)); Z[np.arange(n), g] = 1; Zs.append(Z)
    Z = np.hstack(Zs); sizes = [z.shape[1] for z in Zs]; q = Z.shape[1]
    ZtZ, ZtX, Zty = Z.T @ Z, Z.T @ X, Z.T @ y
    XtX, Xty, yty = X.T @ X, X.T @ y, y @ y
    p = X.shape[1]

    def nll(t):
        s = np.exp(t); se = s[0]
        d = np.concatenate([np.full(k, s[i + 1]) for i, k in enumerate(sizes)])
        M = se * np.diag(1 / d) + ZtZ
        try:
            c = linalg.cho_factor(M, lower=True)
        except linalg.LinAlgError:
            return 1e12
        MZX, MZy = linalg.cho_solve(c, ZtX), linalg.cho_solve(c, Zty)
        XVX = (XtX - ZtX.T @ MZX) / se
        XVy = (Xty - ZtX.T @ MZy) / se
        yVy = (yty - Zty @ MZy) / se
        logdetV = n * np.log(se) + np.log(d).sum() + 2 * np.log(np.diag(c[0])).sum() - q * np.log(se)
        b = np.linalg.solve(XVX, XVy)
        return 0.5 * (logdetV + np.linalg.slogdet(XVX)[1] + yVy - b @ XVy)

    t0 = np.log(np.full(len(groups) + 1, max(np.var(y), 1e-6) / (len(groups) + 1)))
    o = optimize.minimize(nll, t0, method="L-BFGS-B", bounds=[(-25, 5)] * len(t0))
    return np.exp(o.x)


def shares(d, col, gcols, X_extra=None, transform=None):
    d = d.dropna(subset=[col]).reset_index(drop=True)
    v = d[col].astype(float)
    if transform == "rank":
        v = pd.Series(stats.norm.ppf((stats.rankdata(v) - 0.375) / (len(v) + 0.25)))
    y = ((v - v.mean()) / v.std()).values
    X = [np.ones(len(d))]
    if "group" in d and d.group.nunique() > 1:
        X.append((d.group == "senior").astype(float).values)
    if X_extra is not None:
        for c in X_extra:
            x = d[c].astype(float); X.append(((x - x.mean()) / (x.std() + 1e-12)).values)
    v = reml_fast(y, np.column_stack(X), [d[g].values for g in gcols])
    return v / v.sum()


# ------------------------------------------------------------------ MIDI parser (as in step 8)
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


# ================================================================== musical covariates per performance
COVF = out / "performance_musical_covariates.csv"
if (run("R2") or run("R3") or run("R7")) and not COVF.exists():
    rows = []
    for f in D.midi_filename:
        ev = load(root / f)
        on = ev[(ev[:, 1] == 0x90) & (ev[:, 3] > 0)]
        p = on[:, 2]; T = ev[:, 0].max()
        # simultaneity: onsets within 50 ms of the previous onset belong to the same chord
        gaps = np.diff(on[:, 0]); chord_share = float((gaps < 0.05).mean()) if len(gaps) else 0.0
        c = ev[(ev[:, 1] == 0xB0) & (ev[:, 2] == 67)]
        t67, v67 = c[:, 0], c[:, 3]
        dur = np.diff(np.append(t67, T)) if len(t67) else np.array([])
        eng = dur[v67 >= 10].sum() if len(t67) else 0.0
        part = dur[(v67 >= 10) & (v67 < 100)].sum() if len(t67) else 0.0
        rows.append({"midi_filename": f, "pitch_mean": p.mean(), "pitch_sd": p.std(),
                     "high_share": (p >= 84).mean(), "low_share": (p <= 48).mean(), "chord_share": chord_share,
                     "log_dur": np.log(T), "soft_partial": part / eng if eng > 0 else np.nan, "soft_engaged": eng / T})
    pd.DataFrame(rows).to_csv(COVF, index=False)
if COVF.exists():
    D = D.merge(pd.read_csv(COVF), on="midi_filename", how="left")
    for e in ["Baroque", "Classical", "Modern"]:
        D[f"era_{e}"] = (D.era == e).astype(float)
MUS = ["pitch_mean", "pitch_sd", "high_share", "low_share", "chord_share", "nps", "log_dur",
       "era_Baroque", "era_Classical", "era_Modern"]
GC = ["pianist", "composer", "year"]


# ================================================================== R1 fast REML + 1,000-replicate bootstrap
if run("R1"):
    t = time.time()
    pt = {c: shares(D, c, GC) for c in FEATS}
    print("R1 point estimates (pianist, composer, edition):", {c: np.round(v[1:], 3).tolist() for c, v in pt.items()},
          f"{time.time() - t:.1f}s")
    ref = pd.read_csv(res_dir / "final" / "h2a_variance_shares.csv").set_index("feature")
    check = {c: float(abs(pt[c][1] - ref.loc[c, "share_pianist"])) for c in FEATS}
    print("max |fast - dense| pianist share:", max(check.values()))
    pian = D.pianist.unique(); BT = []
    t = time.time()
    for bi in range(B):
        pick = rng.choice(pian, len(pian))
        Db = pd.concat([D[D.pianist == p].assign(pianist=f"{p}#{k}") for k, p in enumerate(pick)], ignore_index=True)
        BT.append({c: shares(Db, c, GC)[1] for c in FEATS})
        if bi % 100 == 99:
            print(f"  bootstrap {bi + 1}/{B}  {time.time() - t:.0f}s", flush=True)
    BT = pd.DataFrame(BT); BT.to_csv(out / "R1_h2a_bootstrap_B1000.csv", index=False)
    rows = []
    for c in FEATS:
        lo, hi = np.percentile(BT[c], [2.5, 97.5])
        rows.append({"feature": c, "share_pianist": pt[c][1], "share_composer": pt[c][2], "share_edition": pt[c][3],
                     "share_resid": pt[c][0], "boot_low": lo, "boot_high": hi})
    R1 = pd.DataFrame(rows); R1.to_csv(out / "R1_h2a_variance_shares_B1000.csv", index=False)
    diffs = {}
    for c in ["sus_frac", "vel_mean", "sus_partial", "dyn_range"]:
        d = BT["soft_frac"] - BT[c]
        diffs[f"soft_minus_{c}"] = {"point": float(pt["soft_frac"][1] - pt[c][1]),
                                    "boot95": [float(x) for x in np.percentile(d, [2.5, 97.5])],
                                    "p_one_sided": float(((d <= 0).sum() + 1) / (len(d) + 1))}
    SUM["R1"] = {"B": B, "max_abs_diff_vs_dense": max(check.values()), "table": R1.round(4).to_dict("records"),
                 "differences": diffs}
    print("== R1\n", R1.round(3).to_string(index=False), "\n", json.dumps(diffs, indent=1)); dump()


# ================================================================== R2 H2a robustness
if run("R2"):
    KEY = ["soft_frac", "soft_episodes_pm", "sus_frac", "sus_partial", "vel_mean", "dyn_range"]
    D["title"] = D.composer + " | " + D.canonical_title.str.lower()
    variants = {
        "main": dict(d=D, g=GC),
        "rank-normal scores": dict(d=D, g=GC, transform="rank"),
        "una corda users only (time share >= 1%)": dict(d=D[D.soft_frac >= 0.01], g=GC),
        "musical covariates": dict(d=D, g=GC, X=MUS),
        "title random effect": dict(d=D, g=["pianist", "composer", "title", "year"]),
        "title random effect + covariates": dict(d=D, g=["pianist", "composer", "title", "year"], X=MUS),
        "junior editions only": dict(d=D[D.group == "junior"], g=GC),
        "senior editions only": dict(d=D[D.group == "senior"], g=GC),
    }
    rows = []
    for name, v in variants.items():
        t = time.time()
        r = {"variant": name, "n": len(v["d"]), "pianists": v["d"].pianist.nunique()}
        for c in KEY:
            sh = shares(v["d"], c, v["g"], v.get("X"), v.get("transform"))
            r[c] = sh[1]
            if "title" in v["g"]:
                r[c + "_title"] = sh[3]
        rows.append(r); print(name, {k: round(x, 3) for k, x in r.items() if isinstance(x, float)}, f"{time.time() - t:.0f}s", flush=True)
    R2 = pd.DataFrame(rows); R2.to_csv(out / "R2_h2a_robustness.csv", index=False)
    # bootstrap of the una corda - sustain difference within each group (independent pianists)
    grp = {}
    for g in ["junior", "senior"]:
        Dg = D[D.group == g]; pian = Dg.pianist.unique(); dd = []
        for _ in range(500):
            pick = rng.choice(pian, len(pian))
            Db = pd.concat([Dg[Dg.pianist == p].assign(pianist=f"{p}#{k}") for k, p in enumerate(pick)], ignore_index=True)
            dd.append(shares(Db, "soft_frac", GC)[1] - shares(Db, "sus_frac", GC)[1])
        pt_ = R2.set_index("variant").loc[f"{g} editions only"]
        grp[g] = {"point": float(pt_.soft_frac - pt_.sus_frac), "boot95": [float(x) for x in np.percentile(dd, [2.5, 97.5])],
                  "p_one_sided": float(((np.array(dd) <= 0).sum() + 1) / (len(dd) + 1)), "B": 500}
    # covariate check: how much do musical covariates explain of una corda time share?
    d = D.dropna(subset=MUS + ["soft_frac"])
    Xc = np.column_stack([np.ones(len(d))] + [((d[c] - d[c].mean()) / (d[c].std() + 1e-12)).values for c in MUS])
    beta, *_ = np.linalg.lstsq(Xc, d.soft_frac.values, rcond=None)
    r2 = 1 - ((d.soft_frac.values - Xc @ beta) ** 2).sum() / ((d.soft_frac - d.soft_frac.mean()) ** 2).sum()
    SUM["R2"] = {"table": R2.round(4).to_dict("records"), "group_replication_soft_minus_sus": grp,
                 "musical_covariates_R2_soft_frac": float(r2)}
    print("== R2\n", R2.round(3).to_string(index=False), "\n", grp, "R2 covariates:", round(r2, 3)); dump()


# ================================================================== R3 H2b replication and covariate adjustment
def fisher_ci(r, n):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return [float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))]


if run("R3"):
    rows = []
    d0 = D.dropna(subset=MUS).copy()
    for col in ["soft_frac", "sus_frac", "sus_partial", "vel_mean"]:
        for adjname in ["composer", "composer + musical covariates"]:
            x = d0.copy()
            x["res"] = x[col] - x.groupby("composer")[col].transform("mean")
            if adjname != "composer":
                Xc = np.column_stack([np.ones(len(x))] + [((x[c] - x[c].mean()) / (x[c].std() + 1e-12)).values for c in MUS])
                ok = x.res.notna().values
                beta, *_ = np.linalg.lstsq(Xc[ok], x.res.values[ok], rcond=None)
                x["res"] = x.res - Xc @ beta
            y = x[x["round"].isin(["R1", "R2", "R3"])].assign(stage=lambda d: np.where(d["round"] == "R1", "R1", "later"))
            for grp_name, yy in [("all", y), ("junior", y[y.group == "junior"]), ("senior", y[y.group == "senior"])]:
                g = yy.groupby(["year", "pianist", "stage"]).res.mean().unstack().dropna()
                r = np.corrcoef(g.R1, g.later)[0, 1]
                rows.append({"feature": col, "adjustment": adjname, "subset": grp_name, "r": r, "n": len(g),
                             "ci": fisher_ci(r, len(g))})
    R3 = pd.DataFrame(rows); R3.to_csv(out / "R3_test_retest_replication.csv", index=False)
    SUM["R3"] = R3.round(4).to_dict("records")
    print("== R3\n", R3.round(3).to_string(index=False)); dump()


# ================================================================== R4 H1a with texture and own-loudness controls
WIN = 5
if run("R4"):
    Q = pd.read_csv(res_dir / "align_pairs_quality.csv")
    refs = Q.groupby("piece").ref.first().to_dict()

    def logit_fit(y, X, lam=0.01):
        def nll(b):
            eta = X @ b
            return -(y * eta - np.logaddexp(0, eta)).sum() + lam * (b[1:] ** 2).sum()
        def grad(b):
            p = 1 / (1 + np.exp(-(X @ b)))
            g = -X.T @ (y - p); g[1:] += 2 * lam * b[1:]; return g
        return optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x

    rows = []
    for f in sorted((res_dir / "aligned_frames").glob("*.npz")):
        z = np.load(f, allow_pickle=True); pc = str(z["piece"])
        n_perf, n_frames = z["soft"].shape; nw = n_frames // WIN
        ref = refs[pc]; assert ref in set(z["perf"]), pc
        ev = load(root / ref); on = ev[(ev[:, 1] == 0x90) & (ev[:, 3] > 0)]
        fr = np.clip((on[:, 0] * 10).astype(int), 0, n_frames - 1)
        cnt = np.bincount(fr, minlength=n_frames).astype(float)
        psum = np.bincount(fr, weights=on[:, 2], minlength=n_frames)
        k = np.ones(10)  # centred 1-s neighbourhood
        cnt1, psum1 = np.convolve(cnt, k, "same"), np.convolve(psum, k, "same")
        reg = np.where(cnt1 > 0, psum1 / np.maximum(cnt1, 1), np.nan)
        reg = pd.Series(reg).interpolate(limit_direction="both").values
        def w(a):
            return a[: nw * WIN].reshape(nw, WIN).mean(1)
        density, register = np.log1p(w(cnt1)), w(reg)
        soft = z["soft"][:, : nw * WIN].reshape(n_perf, nw, WIN).mean(2) > 0.5
        sus = z["sus"][:, : nw * WIN].reshape(n_perf, nw, WIN).mean(2) > 0.5
        with np.errstate(invalid="ignore"):
            loud = np.nanmean(z["loud"][:, : nw * WIN].reshape(n_perf, nw, WIN), axis=2)
        loud = np.where(np.isnan(loud), np.nanmean(loud, axis=1, keepdims=True), loud)
        lz = (loud - loud.mean(1, keepdims=True)) / (loud.std(1, keepdims=True) + 1e-9)
        loo = (lz.sum(0, keepdims=True) - lz) / (n_perf - 1)
        loo = (loo - loo.mean(1, keepdims=True)) / (loo.std(1, keepdims=True) + 1e-9)
        zs = lambda a: (a - a.mean()) / (a.std() + 1e-9)
        regz, denz = zs(register), zs(density)
        for i in range(n_perf):
            r = {"piece": pc, "perf": z["perf"][i], "pianist": z["pianist"][i], "group": z["group"][i],
                 "r_cons_register": np.corrcoef(loo[i], regz)[0, 1], "r_cons_density": np.corrcoef(loo[i], denz)[0, 1],
                 "r_cons_own": np.corrcoef(loo[i], lz[i])[0, 1]}
            for ped, Y in (("soft", soft), ("sus", sus)):
                y = Y[i].astype(float)
                if not 0.05 < y.mean() < 0.95:
                    continue
                one = np.ones(nw)
                r[f"{ped}_m0"] = logit_fit(y, np.column_stack([one, loo[i]]))[1]
                b1 = logit_fit(y, np.column_stack([one, loo[i], regz, denz]))
                r[f"{ped}_m1"], r[f"{ped}_m1_register"], r[f"{ped}_m1_density"] = b1[1], b1[2], b1[3]
                b2 = logit_fit(y, np.column_stack([one, loo[i], regz, denz, lz[i]]))
                r[f"{ped}_m2"], r[f"{ped}_m2_own"] = b2[1], b2[4]
            rows.append(r)
    R4 = pd.DataFrame(rows); R4.to_csv(out / "R4_h1a_controls_by_performance.csv", index=False)

    def pooled(col):
        d = R4.dropna(subset=[col]).reset_index(drop=True)
        n = len(d); y = d[col].values; X = np.ones((n, 1))
        v = reml_fast(y, X, [d.pianist.values, d.piece.values])
        # GLS mean and SE under the fitted variance components (dense; n <= 400)
        Zp = (pd.factorize(d.pianist)[0][:, None] == pd.factorize(d.pianist)[0][None, :]).astype(float)
        Zq = (pd.factorize(d.piece)[0][:, None] == pd.factorize(d.piece)[0][None, :]).astype(float)
        V = v[0] * np.eye(n) + v[1] * Zp + v[2] * Zq
        Vi1 = np.linalg.solve(V, np.ones(n)); se = 1 / np.sqrt(Vi1.sum()); b = (Vi1 @ y) * se ** 2
        return {"n": n, "mean": float(b), "se": float(se), "z": float(b / se), "p": float(2 * stats.norm.sf(abs(b / se))),
                "share_negative": float((y < 0).mean())}

    res = {}
    for ped in ("soft", "sus"):
        for m in ("m0", "m1", "m1_register", "m1_density", "m2", "m2_own"):
            res[f"{ped}_{m}"] = pooled(f"{ped}_{m}")
    res["cons_vs_covariates_median_r"] = {k: float(R4[k].median()) for k in ["r_cons_register", "r_cons_density", "r_cons_own"]}
    SUM["R4"] = res
    print("== R4", json.dumps(res, indent=1)); dump()


# ================================================================== R5 H1c control: sustain presses
if run("R5"):
    edges = np.arange(-4.0, 4.0001, 0.25); centres = (edges[:-1] + edges[1:]) / 2
    prof = []
    for f, grp in zip(D.midi_filename, D.group):
        ev = load(root / f)
        on = ev[(ev[:, 1] == 0x90) & (ev[:, 3] > 0)]
        t, v = on[:, 0], on[:, 3]; vz = (v - v.mean()) / (v.std() + 1e-9)
        if t.max() < 10:
            continue
        for kind, cc in (("una corda press", 67), ("sustain press", 64)):
            c = ev[(ev[:, 1] == 0xB0) & (ev[:, 2] == cc)]
            x = (c[:, 3] >= 64).astype(int)
            times = c[np.where(np.diff(x) == 1)[0] + 1, 0] if len(c) > 1 else np.array([])
            times = times[(times > 4) & (times < t.max() - 4)]
            if len(times) < 3:
                continue
            if len(times) > 400:
                times = rng.choice(times, 400, replace=False)
            sums = np.zeros(len(centres)); cnts = np.zeros(len(centres))
            for tt in np.array_split(times, max(1, len(times) // 50)):
                rel = t[None, :] - tt[:, None]; m = (rel >= -4) & (rel < 4)
                idx = np.floor((rel[m] + 4) / 0.25).astype(int)
                np.add.at(sums, idx, np.broadcast_to(vz, rel.shape)[m]); np.add.at(cnts, idx, 1)
            prof.append(pd.DataFrame({"perf": f, "group": grp, "kind": kind, "t": centres,
                                      "vz": np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)}))
    PF = pd.concat(prof); PF.to_csv(out / "R5_event_profiles_by_performance.csv", index=False)
    G = PF.groupby(["kind", "t"]).vz.agg(["mean", "std", "count"]).reset_index()
    G["lo"] = G["mean"] - 1.96 * G["std"] / np.sqrt(G["count"]); G["hi"] = G["mean"] + 1.96 * G["std"] / np.sqrt(G["count"])
    G.to_csv(out / "R5_event_profiles_mean.csv", index=False)
    res = {}
    pp = PF.pivot_table(index=["perf", "kind"], columns="t", values="vz")
    drop = (pp.loc[:, (pp.columns > 0) & (pp.columns < 2)].mean(1) - pp.loc[:, pp.columns < -2].mean(1)).rename("drop").reset_index()
    for kind in ["una corda press", "sustain press"]:
        d = drop[drop.kind == kind]["drop"]
        res[kind] = {"n_perf": int(len(d)), "mean_drop_sd": float(d.mean()), "share_negative": float((d < 0).mean())}
    both = drop.pivot(index="perf", columns="kind", values="drop").dropna()
    res["paired"] = {"n": len(both), "wilcoxon_p": float(stats.wilcoxon(both["una corda press"], both["sustain press"]).pvalue)}
    SUM["R5"] = res
    print("== R5", json.dumps(res, indent=1)); dump()


# ================================================================== R6 rule-threshold reliability
if run("R6"):
    def logit2(y, x, lam=0.01):
        X = np.column_stack([np.ones(len(x)), x])
        nll = lambda b: -(y * (X @ b) - np.logaddexp(0, X @ b)).sum() + lam * (b ** 2).sum()
        return optimize.minimize(nll, np.zeros(2), method="BFGS").x
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
        quiet = -loo
        half = (np.arange(nw) // 20) % 2  # interleaved 10-s blocks
        for i in range(n):
            y = soft[i].astype(float)
            if not 0.05 < y.mean() < 0.95:
                continue
            r = {"piece": str(z["piece"]), "pianist": z["pianist"][i], "group": z["group"][i], "year": int(z["year"][i])}
            for h in (0, 1):
                m = half == h
                if 0 < y[m].mean() < 1:
                    r[f"a{h}"], r[f"b{h}"] = logit2(y[m], quiet[i][m])
            rows.append(r)
    H = pd.DataFrame(rows).dropna().reset_index(drop=True)
    H.to_csv(out / "R6_split_half_coefficients.csv", index=False)
    res = {"n_perf": len(H)}
    ext = json.load(open(res_dir / "final" / "ext_summary.json"))["A_rule_threshold"]
    for k, nm in (("a", "threshold_a"), ("b", "rule_b")):
        # robust to outliers: Spearman split-half, Spearman-Brown corrected
        r_half = stats.spearmanr(H[f"{k}0"], H[f"{k}1"]).correlation
        rel = 2 * r_half / (1 + r_half)
        sh = ext[nm]["share_pianist"]
        res[nm] = {"split_half_r": float(r_half), "reliability_sb": float(rel), "share_pianist": sh,
                   "share_pianist_disattenuated": float(min(sh / rel, 1.0)) if rel > 0 else None,
                   "share_piece_disattenuated": float(min(ext[nm]["share_piece"] / rel, 1.0)) if rel > 0 else None}
    SUM["R6"] = res
    print("== R6", json.dumps(res, indent=1)); dump()


# ================================================================== R7 partial una corda; identification among users
if run("R7"):
    res = {}
    d = D.dropna(subset=["soft_partial"])
    d = d[d.soft_frac >= 0.01]
    res["soft_partial_median"] = float(d.soft_partial.median())
    res["soft_partial_iqr"] = [float(x) for x in d.soft_partial.quantile([.25, .75])]
    res["sus_partial_median"] = float(D.sus_partial.median())
    res["n"] = len(d)
    sh = shares(d, "soft_partial", GC)
    res["soft_partial_shares"] = {"pianist": float(sh[1]), "composer": float(sh[2]), "edition": float(sh[3])}
    # identification restricted to una corda users (time share >= 1%)
    SETS = {"una corda": ["soft_frac", "soft_note_share", "soft_episodes_pm"],
            "sustain": ["sus_frac", "sus_partial", "sus_presses_pm"], "dynamics": ["vel_mean", "vel_sd", "dyn_range"]}
    F = D[D.soft_frac >= 0.01].copy()
    for c in sum(SETS.values(), []):
        F[c] = F[c].fillna(F[c].median()); F[c] = F[c] - F.groupby("composer")[c].transform("mean"); F[c] = (F[c] - F[c].mean()) / F[c].std()
    def ident(cols):
        hits, ch = [], []
        for _, g in F.groupby("year"):
            g = g[g.groupby("pianist").pianist.transform("size") >= 3]
            if g.pianist.nunique() < 3:
                continue
            Xg, lab = g[cols].values, g.pianist.values
            for i in range(len(g)):
                m = np.ones(len(g), bool); m[i] = False
                cents = {p: Xg[m & (lab == p)].mean(0) for p in np.unique(lab[m])}
                hits.append(min(cents, key=lambda p: np.sum((Xg[i] - cents[p]) ** 2)) == lab[i]); ch.append(1 / len(cents))
        return np.array(hits), float(np.mean(ch))
    hv = {}
    for k, cols in SETS.items():
        h, ch = ident(cols); hv[k] = h
        res[f"ident_users_{k}"] = {"accuracy": float(h.mean()), "chance": ch, "n_tests": int(len(h))}
    def mcn(a, b):
        n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
        return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)
    res["ident_users_mcnemar_una_vs_sustain"] = mcn(hv["una corda"], hv["sustain"])
    res["ident_users_mcnemar_una_vs_dynamics"] = mcn(hv["una corda"], hv["dynamics"])
    SUM["R7"] = res
    print("== R7", json.dumps(res, indent=1)); dump()


if PART == "merge":
    allsum = {}
    for f in sorted(out.glob("revision_summary_R*.json")):
        allsum.update(json.load(open(f)))
    with open(out / "revision_summary.json", "w") as fh:
        json.dump(allsum, fh, ensure_ascii=False, indent=1, default=float)
    print("merged:", sorted(allsum))
