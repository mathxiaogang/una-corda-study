"""Step 6 - follow-up analyses for the discovery framing (prevalence, individuality, event-locked dynamics).

Usage:
    python -I 06_discovery_followups.py <maestro-v3.0.0 dir> <data_dir> <results_dir>

Writes:
    results/followup_prevalence.csv       how common the una corda is
    results/followup_individuality.csv    share of variance between pianists, per feature (crossed REML LMM)
                                          + split-half consistency within pianist
    results/followup_event_locked.csv     velocity just before vs just after una corda presses/releases,
                                          against random time points
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

root, data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rng = np.random.default_rng(7)
D = pd.read_csv(data_dir / "metadata_clean.csv").merge(pd.read_csv(data_dir / "features.csv"), on="midi_filename")
D["log_episode_s"] = np.log(D.soft_episode_med_s)

# ------------------------------------------------ A. prevalence
prev = []
for lab, x in [("all", D), ("junior", D[D.group == "junior"]), ("senior", D[D.group == "senior"])]:
    prev.append({"subset": lab, "n": len(x),
                 "share_perf_using_una_corda": (x.soft_frac > 0.01).mean(),
                 "median_time_share": x.soft_frac.median(), "q1": x.soft_frac.quantile(.25), "q3": x.soft_frac.quantile(.75),
                 "median_note_share": x.soft_note_share.median(),
                 "median_presses_per_min": x.soft_episodes_pm.median(),
                 "median_episode_s": x.soft_episode_med_s.median(),
                 "hours_under_una_corda": (x.soft_frac * x.duration).sum() / 3600,
                 "hours_total": x.duration.sum() / 3600})
prev = pd.DataFrame(prev)
prev.to_csv(res_dir / "followup_prevalence.csv", index=False)
print("== prevalence\n", prev.round(3).to_string(index=False))


# ------------------------------------------------ B. individuality
def reml(y, X, groups):
    n = len(y)
    G = [(g[:, None] == g[None, :]).astype(float) for g in groups]
    def nll(theta):
        s = np.exp(theta)
        V = s[0] * np.eye(n) + sum(s[k + 1] * Gk for k, Gk in enumerate(G))
        c = linalg.cho_factor(V, lower=True)
        ViX, Viy = linalg.cho_solve(c, X), linalg.cho_solve(c, y)
        XtViX = X.T @ ViX
        beta = np.linalg.solve(XtViX, X.T @ Viy)
        r = y - X @ beta
        return 0.5 * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(XtViX)[1] + r @ linalg.cho_solve(c, r))
    t0 = np.log(np.full(len(G) + 1, 0.25))
    opt = optimize.minimize(nll, t0, method="L-BFGS-B", bounds=[(-25, 5)] * len(t0))
    return np.exp(opt.x)

FEATS = [("soft_frac", "una corda", "弱音踏板：使用时间占比"), ("soft_note_share", "una corda", "弱音踏板：音符占比"),
         ("soft_episodes_pm", "una corda", "弱音踏板：每分钟次数"), ("sus_frac", "sustain", "延音踏板：使用时间占比"),
         ("sus_partial", "sustain", "延音踏板：浅踩比例"), ("sus_presses_pm", "sustain", "延音踏板：每分钟次数"),
         ("press_lag_ms", "sustain", "延音踏板：踩下时机"), ("vel_mean", "dynamics", "平均力度"),
         ("dyn_range", "dynamics", "力度范围"), ("nps", "dynamics", "音符密度")]
ind = []
for col, fam, zh in FEATS:
    d = D.dropna(subset=[col]).reset_index(drop=True)
    y = ((d[col] - d[col].mean()) / d[col].std()).values
    X = np.column_stack([np.ones(len(d)), (d.group == "senior").astype(float)])
    v = reml(y, X, [pd.factorize(d.pianist)[0], pd.factorize(d.year)[0], pd.factorize(d.composer)[0]])
    tot = v.sum()
    # split-half: composer-adjusted values, pianists with >= 4 performances, odd vs even performances
    d["res"] = d[col] - d.groupby("composer")[col].transform("mean")
    a, b = [], []
    for p, g in d.groupby("pianist"):
        if len(g) >= 4:
            g = g.sample(frac=1, random_state=1)
            a.append(g.res.values[0::2].mean()); b.append(g.res.values[1::2].mean())
    r = np.corrcoef(a, b)[0, 1]
    ind.append({"feature": col, "family": fam, "label_zh": zh, "share_pianist": v[1] / tot, "share_composer": v[3] / tot,
                "share_edition": v[2] / tot, "share_residual": v[0] / tot, "split_half_r": r,
                "spearman_brown": 2 * r / (1 + r), "n_pianists_split": len(a)})
    print(col, "pianist share", round(v[1] / tot, 3), "split-half", round(r, 3))
ind = pd.DataFrame(ind).sort_values("share_pianist", ascending=False)
ind.to_csv(res_dir / "followup_individuality.csv", index=False)
print("== individuality\n", ind.round(3).to_string(index=False))


# ------------------------------------------------ C. event-locked velocity around una corda presses
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
    ev = np.array(sorted(ev), dtype=float); ev[:, 0] *= 0.5 / div  # single 120 BPM tempo in MAESTRO
    return ev

W = 2.0  # seconds before / after
rows = []
for f, grp in zip(D.midi_filename, D.group):
    ev = load(root / f)
    on = ev[(ev[:, 1] == 0x90) & (ev[:, 3] > 0)]
    t, v = on[:, 0], on[:, 3]
    c = ev[(ev[:, 1] == 0xB0) & (ev[:, 2] == 67)]
    x = (c[:, 3] >= 64).astype(int)
    press = c[np.where(np.diff(x) == 1)[0] + 1, 0]
    release = c[np.where(np.diff(x) == -1)[0] + 1, 0]
    def delta(times):
        out = []
        for tt in times:
            pre = v[(t >= tt - W) & (t < tt)]; post = v[(t >= tt) & (t < tt + W)]
            if len(pre) >= 3 and len(post) >= 3:
                out.append(post.mean() - pre.mean())
        return out
    rnd = rng.uniform(W, t.max() - W, size=max(len(press), 20)) if t.max() > 2 * W else []
    dp, dr, dn = delta(press), delta(release), delta(rnd)
    rows.append({"midi_filename": f, "group": grp, "n_press": len(dp),
                 "dv_press": np.mean(dp) if len(dp) >= 3 else np.nan,
                 "dv_release": np.mean(dr) if len(dr) >= 3 else np.nan,
                 "dv_random": np.mean(dn) if len(dn) >= 3 else np.nan})
E = pd.DataFrame(rows).merge(D[["midi_filename", "year", "composer"]], on="midi_filename")
E.to_csv(res_dir / "followup_event_locked_by_performance.csv", index=False)
summ = []
for lab, x in [("all", E), ("junior", E[E.group == "junior"]), ("senior", E[E.group == "senior"])]:
    x = x.dropna(subset=["dv_press", "dv_random"])
    w = stats.wilcoxon(x.dv_press - x.dv_random)
    summ.append({"subset": lab, "n_perf": len(x), "press_minus_before": x.dv_press.mean(),
                 "release_minus_before": x.dv_release.mean(), "random_minus_before": x.dv_random.mean(),
                 "share_perf_drop_at_press": (x.dv_press < 0).mean(), "wilcoxon_press_vs_random_p": w.pvalue})
summ = pd.DataFrame(summ)
summ.to_csv(res_dir / "followup_event_locked.csv", index=False)
print("== event-locked velocity (post 2 s minus pre 2 s)\n", summ.round(4).to_string(index=False))
# group difference at edition level (composer-adjusted), exact permutation
import itertools
x = E.dropna(subset=["dv_press"]).copy()
x["res"] = x.dv_press - x.groupby("composer").dv_press.transform("mean")
Y = x.groupby("year").agg(res=("res", "mean"), group=("group", "first"))
s = (Y.group == "senior").values
obs = Y.res[s].mean() - Y.res[~s].mean()
null = [Y.res.values[list(cmb)].mean() - np.delete(Y.res.values, list(cmb)).mean() for cmb in itertools.combinations(range(10), 5)]
print("press drop senior-junior:", round(obs, 3), "perm p:", round(float((np.abs(null) >= abs(obs) - 1e-12).mean()), 4))

# ------------------------------------------------ D. stability of personal habits
# same pianist across editions (repeat entrants, 2-10 years apart, different instruments)
# and across rounds within an edition (different days), composer-adjusted
stab = []
for col in ["soft_frac", "soft_episodes_pm", "sus_frac", "sus_partial", "vel_mean", "dyn_range"]:
    x = D.copy()
    x["res"] = x[col] - x.groupby("composer")[col].transform("mean")
    rep = x[x.repeat_type != "single"].groupby(["pianist", "year"]).res.mean().reset_index().sort_values("year")
    a, b = rep.groupby("pianist").res.first(), rep.groupby("pianist").res.last()
    pr, sr = stats.pearsonr(a, b), stats.spearmanr(a, b)
    y = x[x["round"].isin(["R1", "R2", "R3"])].copy()
    y["stage"] = np.where(y["round"] == "R1", "R1", "later")
    g = y.groupby(["year", "pianist", "stage"]).res.mean().unstack().dropna()
    pr2 = stats.pearsonr(g.R1, g.later)
    stab.append({"feature": col, "cross_edition_r": pr.statistic, "cross_edition_p": pr.pvalue,
                 "cross_edition_rho": sr.statistic, "n_cross": len(a),
                 "R1_vs_laterRound_r": pr2.statistic, "R1_vs_later_p": pr2.pvalue, "n_rounds": len(g)})
stab = pd.DataFrame(stab)
stab.to_csv(res_dir / "followup_stability.csv", index=False)
print("== stability\n", stab.round(3).to_string(index=False))
