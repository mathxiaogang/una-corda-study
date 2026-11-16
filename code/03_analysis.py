"""Step 3 - all statistical analyses for the una corda study.

Usage:
    python -I 03_analysis.py <data_dir> <results_dir>

Reads data_dir/metadata_clean.csv, features.csv, cc67_raw_by_year.csv.
Writes CSV tables and summary.json to results_dir and prints a log.

Design note: group (junior/senior) is constant within an edition (year), so the unit of
inference for the group effect is the edition (10 editions: 5 senior, 5 junior).
The primary test is an exact permutation of group labels across editions (C(10,5) = 252).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2])
res_dir.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261114)

meta = pd.read_csv(data_dir / "metadata_clean.csv")
feat = pd.read_csv(data_dir / "features.csv")
D = meta.merge(feat, on="midi_filename")
D["log_dur"] = np.log(D.duration)
D["log_episode_s"] = np.log(D.soft_episode_med_s)
SUMMARY = {}


def save(df, name):
    df.to_csv(res_dir / name, index=False)


# ---------------------------------------------------------------- helpers
def year_means(d, col, adjust="composer"):
    """Composer-adjusted residuals averaged per edition."""
    d = d.dropna(subset=[col]).copy()
    if adjust == "composer":
        d["res"] = d[col] - d.groupby("composer")[col].transform("mean")
    elif adjust == "piece":
        d["res"] = d[col] - d.groupby("canonical_title_label")[col].transform("mean")
    else:
        d["res"] = d[col]
    return d.groupby("year").agg(res=("res", "mean"), group=("group", "first"))


def perm_test(Y):
    """Exact two-sided permutation of group labels across editions."""
    is_s = (Y.group == "senior").values
    obs = Y.res[is_s].mean() - Y.res[~is_s].mean()
    k, n = is_s.sum(), len(Y)
    null = []
    for comb in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(comb)] = True
        null.append(Y.res.values[m].mean() - Y.res.values[~m].mean())
    null = np.array(null)
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    sep = bool(Y.res[is_s].min() > Y.res[~is_s].max() or Y.res[is_s].max() < Y.res[~is_s].min())
    return float(obs), p, len(null), sep


def boot_ci(Y, B=10000):
    s = Y.res[Y.group == "senior"].values
    j = Y.res[Y.group == "junior"].values
    d = [rng.choice(s, len(s)).mean() - rng.choice(j, len(j)).mean() for _ in range(B)]
    return [float(x) for x in np.percentile(d, [2.5, 97.5])]


# ---------------------------------------------------------------- 1. sample description
t1 = []
for g, x in D.groupby("group"):
    era_share = x.groupby("era").duration.sum() / x.duration.sum()
    t1.append({"group": g, "performances": len(x), "pianists": x.pianist.nunique(),
               "editions": x.year.nunique(), "hours": round(x.duration.sum() / 3600, 1),
               "median_minutes": round(x.duration.median() / 60, 1),
               **{f"share_{e}": round(era_share.get(e, 0), 3) for e in ["Baroque", "Classical", "Romantic", "Modern"]},
               "top_composers": "; ".join(x.composer.value_counts().head(4).index)})
t1 = pd.DataFrame(t1)
save(t1, "table1_sample.csv")
print("== Table 1\n", t1.to_string(index=False))

# ---------------------------------------------------------------- 2. H1a primary test
Y = year_means(D, "soft_frac")
obs, p, nperm, sep = perm_test(Y)
ci = boot_ci(Y)
raw_med = D.groupby("year").agg(group=("group", "first"), median_soft=("soft_frac", "median"),
                                n=("soft_frac", "size"))
save(raw_med.reset_index().merge(Y.reset_index()[["year", "res"]], on="year")
     .rename(columns={"res": "composer_adjusted_mean"}), "h1a_by_edition.csv")
welch = stats.ttest_ind(Y.res[Y.group == "senior"], Y.res[Y.group == "junior"], equal_var=False)
SUMMARY["H1a"] = {"diff_pp": obs * 100, "perm_p": p, "n_perm": nperm, "perfect_separation": sep,
                  "boot95_pp": [c * 100 for c in ci], "welch_t": float(welch.statistic), "welch_p": float(welch.pvalue),
                  "senior_year_medians": raw_med[raw_med.group == "senior"].median_soft.round(3).tolist(),
                  "junior_year_medians": raw_med[raw_med.group == "junior"].median_soft.round(3).tolist()}
print("\n== H1a", json.dumps(SUMMARY["H1a"], indent=1))

# ---------------------------------------------------------------- 3. crossed random-effects LMM (REML)
def reml_lmm(y, X, groups):
    """Linear mixed model with crossed random intercepts, REML via dense Cholesky."""
    n = len(y)
    G = [(g[:, None] == g[None, :]).astype(float) for g in groups]

    def parts(theta):
        s = np.exp(theta)
        V = s[0] * np.eye(n)
        for k, Gk in enumerate(G):
            V += s[k + 1] * Gk
        c = linalg.cho_factor(V, lower=True)
        ViX = linalg.cho_solve(c, X)
        Viy = linalg.cho_solve(c, y)
        XtViX = X.T @ ViX
        beta = np.linalg.solve(XtViX, X.T @ Viy)
        r = y - X @ beta
        quad = r @ linalg.cho_solve(c, r)
        logdetV = 2 * np.log(np.diag(c[0])).sum()
        logdetX = np.linalg.slogdet(XtViX)[1]
        return 0.5 * (logdetV + logdetX + quad), beta, XtViX

    theta0 = np.log(np.full(len(G) + 1, np.var(y) / (len(G) + 1)))
    opt = optimize.minimize(lambda t: parts(t)[0], theta0, method="L-BFGS-B",
                            bounds=[(-25, 5)] * len(theta0))
    nll, beta, XtViX = parts(opt.x)
    se = np.sqrt(np.diag(np.linalg.inv(XtViX)))
    return beta, se, np.exp(opt.x), opt


def run_lmm(d, ycol, label):
    d = d.dropna(subset=[ycol]).reset_index(drop=True)
    z = lambda v: (v - v.mean()) / v.std()
    X = np.column_stack([np.ones(len(d)), (d.group == "senior").astype(float),
                         z(d.nps), z(d.vel_mean), z(d.log_dur)])
    groups = [pd.factorize(d.pianist)[0], pd.factorize(d.year)[0], pd.factorize(d.composer)[0]]
    beta, se, var, opt = reml_lmm(d[ycol].values.astype(float), X, groups)
    df_ = 8  # editions - 2 (group varies only between editions)
    t = beta[1] / se[1]
    out = {"model": label, "n": len(d), "beta_senior": beta[1], "se": se[1], "t": t,
           "df": df_, "p": 2 * stats.t.sf(abs(t), df_),
           "ci_low": beta[1] - stats.t.ppf(0.975, df_) * se[1], "ci_high": beta[1] + stats.t.ppf(0.975, df_) * se[1],
           "beta_nps": beta[2], "beta_vel": beta[3], "beta_logdur": beta[4],
           "var_resid": var[0], "var_pianist": var[1], "var_year": var[2], "var_composer": var[3],
           "converged": bool(opt.success)}
    return out


D["soft_logit"] = np.log(D.soft_frac.clip(0.005, 0.995) / (1 - D.soft_frac.clip(0.005, 0.995)))
lmm = pd.DataFrame([run_lmm(D, "soft_frac", "soft_frac (proportion)"),
                    run_lmm(D, "soft_logit", "soft_frac (logit)"),
                    run_lmm(D, "soft_note_share", "soft_note_share (proportion)")])
save(lmm, "h1a_lmm.csv")
print("\n== LMM\n", lmm.round(4).to_string(index=False))
SUMMARY["LMM"] = lmm.round(5).to_dict("records")

# ---------------------------------------------------------------- 4. specificity (RQ2) + mechanism (H1c, RQ3)
FEATURES = [
    ("soft_frac", "弱音踏板时间占比", "una corda"), ("soft_note_share", "弱音踏板下的音符占比", "una corda"),
    ("soft_episodes_pm", "弱音踏板每分钟次数", "una corda"), ("log_episode_s", "弱音踏板单次时长（对数）", "una corda"),
    ("vel_gap", "踩与不踩时的力度差", "una corda"), ("soft_at_mf", "弱音踏板下中强以上音符占比", "una corda"),
    ("sus_frac", "延音踏板时间占比", "sustain"), ("sus_partial", "延音踏板浅踩比例", "sustain"),
    ("sus_presses_pm", "延音踏板每分钟踩下次数", "sustain"), ("press_lag_ms", "延音踏板踩下时机", "sustain"),
    ("synco_share", "后踩（切分）比例", "sustain"),
    ("vel_mean", "平均力度", "dynamics"), ("dyn_range", "力度范围", "dynamics"), ("nps", "音符密度", "dynamics"),
]
spec = []
for col, zh, fam in FEATURES:
    d = D.dropna(subset=[col]).copy()
    d["res"] = d[col] - d.groupby("composer")[col].transform("mean")
    sd = d.res.std()
    Yf = d.groupby("year").agg(res=("res", "mean"), group=("group", "first"))
    Yf["res"] = Yf.res / sd
    o, pp, _, sp = perm_test(Yf)
    lo, hi = boot_ci(Yf, 5000)
    spec.append({"feature": col, "label_zh": zh, "family": fam, "d_senior_minus_junior": o,
                 "boot_low": lo, "boot_high": hi, "perm_p": pp, "perfect_separation": sp, "n": len(d)})
spec = pd.DataFrame(spec)
save(spec, "specificity_standardized.csv")
print("\n== Specificity (composer-adjusted, SD units)\n", spec.round(3).to_string(index=False))
SUMMARY["specificity"] = spec.round(4).to_dict("records")

# ---------------------------------------------------------------- 5. robustness of H1a
rob = []
def add(label, d, col="soft_frac", adjust="composer"):
    Yr = year_means(d, col, adjust)
    if Yr.group.nunique() < 2:
        return
    o, pp, nn, sp = perm_test(Yr)
    rob.append({"check": label, "diff_pp": o * 100, "perm_p": pp, "n_perm": nn, "perfect_separation": sp,
                "n_perf": int(d[col].notna().sum()), "n_editions": len(Yr)})

add("基线：作曲家调整", D)
add("不做调整（原始均值）", D, adjust="none")
add("阈值 32", D, "soft_frac32")
add("阈值 96", D, "soft_frac96")
add("剔除改编曲和 2 分钟以下", D[~D.is_transcription & (D.duration >= 120)])
add("只用第一轮", D[D["round"] == "R1"])
add("剔除全程踩弱音踏板（>95%）的演奏", D[D.soft_frac <= 0.95])
add("剔除室内乐轮次", D[D["round"] != "Chamber"])
# pianist-level aggregation: one value per pianist-edition
dp = D.copy()
dp["res"] = dp.soft_frac - dp.groupby("composer").soft_frac.transform("mean")
pe = dp.groupby(["year", "pianist"]).agg(res=("res", "mean"), group=("group", "first")).reset_index()
Yp = pe.groupby("year").agg(res=("res", "mean"), group=("group", "first"))
o, pp, nn, sp = perm_test(Yp)
rob.append({"check": "按钢琴家先平均", "diff_pp": o * 100, "perm_p": pp, "n_perm": nn, "perfect_separation": sp,
            "n_perf": len(pe), "n_editions": len(Yp)})
# recording-format strata
for tpq in (384, 480):
    add(f"只用 MIDI 精度 {tpq} 的年份", D[D.tpq == tpq])
# leave one edition out
loo = []
for y in sorted(D.year.unique()):
    Yr = year_means(D[D.year != y], "soft_frac")
    o, pp, nn, sp = perm_test(Yr)
    loo.append({"left_out": y, "diff_pp": o * 100, "perm_p": pp, "perfect_separation": sp})
loo = pd.DataFrame(loo)
save(loo, "robust_leave_one_edition_out.csv")
rob.append({"check": "逐届剔除（10 次中的范围）", "diff_pp": f"{loo.diff_pp.min():.1f}–{loo.diff_pp.max():.1f}",
            "perm_p": f"≤ {loo.perm_p.max():.3f}", "n_perm": 126, "perfect_separation": bool(loo.perfect_separation.all()),
            "n_perf": None, "n_editions": 9})
# matched pieces played in both groups
both = D.groupby("canonical_title_label").group.nunique()
M = D[D.canonical_title_label.isin(both[both == 2].index)]
pm = M.groupby(["canonical_title_label", "group"]).soft_frac.mean().unstack()
diff = pm.senior - pm.junior
w = stats.wilcoxon(diff)
rob.append({"check": "只用两组都弹过的曲子（逐曲配对）", "diff_pp": diff.mean() * 100, "perm_p": float(w.pvalue),
            "n_perm": None, "perfect_separation": None, "n_perf": len(M), "n_editions": f"{len(diff)} 首曲子"})
rob = pd.DataFrame(rob)
save(rob, "robustness_h1a.csv")
print("\n== Robustness\n", rob.to_string(index=False))
SUMMARY["robustness"] = rob.astype(str).to_dict("records")
SUMMARY["matched_pieces"] = {"n_pieces": int(len(diff)), "senior_higher": int((diff > 0).sum()),
                             "mean_diff_pp": float(diff.mean() * 100), "median_diff_pp": float(diff.median() * 100),
                             "wilcoxon_p": float(w.pvalue)}

# ---------------------------------------------------------------- 6. H1b longitudinal
dl = D.copy()
dl["res"] = dl.soft_frac - dl.groupby("composer").soft_frac.transform("mean")
L = dl[dl.repeat_type != "single"].groupby(["pianist", "year", "group", "repeat_type"]).agg(
    n=("soft_frac", "size"), soft_median=("soft_frac", "median"), soft_adj=("res", "mean")).reset_index()
save(L, "h1b_repeat_pianists_by_edition.csv")
ch = L.sort_values("year").groupby("pianist").agg(
    repeat_type=("repeat_type", "first"), first_year=("year", "first"), last_year=("year", "last"),
    soft_first=("soft_median", "first"), soft_last=("soft_median", "last"),
    adj_first=("soft_adj", "first"), adj_last=("soft_adj", "last")).reset_index()
ch["change_pp"] = (ch.soft_last - ch.soft_first) * 100
ch["change_adj_pp"] = (ch.adj_last - ch.adj_first) * 100
save(ch, "h1b_changes.csv")
summ = ch.groupby("repeat_type").agg(n=("change_pp", "size"), mean_change_pp=("change_pp", "mean"),
                                     mean_change_adj_pp=("change_adj_pp", "mean")).reset_index()
is_js = (ch.repeat_type == "junior->senior").values
def diff_js(vals, mask):
    return vals[mask].mean() - vals[~mask].mean()
for col in ["change_pp", "change_adj_pp"]:
    v = ch[col].values
    o = diff_js(v, is_js)
    null = []
    for comb in itertools.combinations(range(len(v)), is_js.sum()):
        m = np.zeros(len(v), bool); m[list(comb)] = True
        null.append(diff_js(v, m))
    pp = float((np.abs(null) >= abs(o) - 1e-12).mean())
    SUMMARY[f"H1b_{col}"] = {"jr_to_sr_minus_others_pp": float(o), "exact_perm_p": pp, "n_perm": len(null)}
print("\n== H1b\n", ch.round(2).to_string(index=False), "\n", summ.round(2).to_string(index=False))
print({k: v for k, v in SUMMARY.items() if k.startswith("H1b")})
SUMMARY["H1b_groups"] = summ.round(3).to_dict("records")

# ---------------------------------------------------------------- 7. RQ4 una corda and advancement (exploratory)
r1 = dl[dl["round"] == "R1"].groupby(["year", "pianist", "group"]).agg(
    soft=("res", "mean"), adv=("advanced_R2", "first")).reset_index()
r1["soft_z"] = r1.groupby("year").soft.transform(lambda x: (x - x.mean()) / x.std())
rq4 = []
for lab, sub in [("全部", r1), ("少年组", r1[r1.group == "junior"]), ("成人组", r1[r1.group == "senior"])]:
    a, b = sub[sub.adv == 1].soft_z, sub[sub.adv == 0].soft_z
    mw = stats.mannwhitneyu(a, b)
    rq4.append({"subset": lab, "n": len(sub), "n_adv": int(sub.adv.sum()), "z_adv": a.mean(), "z_not": b.mean(),
                "diff_sd": a.mean() - b.mean(), "mwu_p": float(mw.pvalue)})
rq4 = pd.DataFrame(rq4)
save(rq4, "rq4_advancement.csv")
print("\n== RQ4\n", rq4.round(3).to_string(index=False))
SUMMARY["RQ4"] = rq4.round(4).to_dict("records")

# ---------------------------------------------------------------- 8. supplementary: style period and Bach
era_t = D.groupby(["era", "group"]).agg(n=("soft_frac", "size"), soft_median=("soft_frac", "median"),
                                        sus_median=("sus_frac", "median"), sus_partial_median=("sus_partial", "median")).reset_index()
save(era_t, "supp_era_by_group.csv")
bach = D[D.composer == "Johann Sebastian Bach"]
bach_t = bach.groupby("group").agg(n=("sus_frac", "size"), no_sustain_share=("sus_frac", lambda x: (x < 0.05).mean()),
                                   sus_median=("sus_frac", "median"), soft_median=("soft_frac", "median")).reset_index()
save(bach_t, "supp_bach.csv")
print("\n== Era\n", era_t.round(3).to_string(index=False), "\n== Bach\n", bach_t.round(3).to_string(index=False))
SUMMARY["bach"] = bach_t.round(4).to_dict("records")
SUMMARY["bach_all"] = {"n": len(bach), "no_sustain_n": int((bach.sus_frac < 0.05).sum())}

# ---------------------------------------------------------------- 9. equipment audit
raw = pd.read_csv(data_dir / "cc67_raw_by_year.csv")
aud = []
for y, g in raw.groupby("year"):
    v = np.repeat(g.value.values, g.n.values)
    nz = v[v > 0]
    aud.append({"year": y, "group": D[D.year == y].group.iloc[0], "tpq": int(D[D.year == y].tpq.iloc[0]),
                "cc67_events": len(v), "distinct_values": int(g.value.nunique()),
                "median_nonzero": float(np.median(nz)), "share_ge64_of_nonzero": float((nz >= 64).mean()),
                "perf_soft_gt95": int((D[D.year == y].soft_frac > 0.95).sum())})
aud = pd.DataFrame(aud)
save(aud, "equipment_audit.csv")
print("\n== Equipment audit\n", aud.round(3).to_string(index=False))
SUMMARY["equipment"] = aud.round(4).to_dict("records")

with open(res_dir / "summary.json", "w") as fh:
    json.dump(SUMMARY, fh, ensure_ascii=False, indent=1, default=float)
print("\nwrote", sorted(p.name for p in res_dir.iterdir()))
