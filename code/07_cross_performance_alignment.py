"""Step 7 - use performances of the same piece as each other's "score".

No external data (no score, no ages): every performance of a piece is aligned to a reference
performance of that piece with banded DTW on chroma features, so pedal and loudness can be compared
at the same musical position across pianists.

Usage:
    python -I 07_cross_performance_alignment.py <maestro-v3.0.0 dir> <data_dir> <results_dir>

Writes:
    results/align_pairs_quality.csv        alignment quality for every performance vs its reference
    results/align_agreement_pairs.csv      pairwise agreement (una corda kappa, sustain kappa, loudness r)
    results/align_agreement_summary.csv    summary by measure and by pair type
    results/align_consensus_quiet.csv      una corda use in passages everyone plays softly vs loudly
"""
import hashlib
import re
import struct
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

root, data_dir, res_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
# optional key=value arguments after the three paths, e.g. fps=20 band=0.2 ref=longest tag=fps20 save_frames=1
OPTS = dict(a.split("=", 1) for a in sys.argv[4:])
FPS = int(OPTS.get("fps", 10))                 # alignment frame rate
WIN = int(OPTS.get("win", max(1, FPS // 2)))   # frames per analysis window (default 0.5 s)
BAND = float(OPTS.get("band", 0.12))           # Sakoe-Chiba band as a share of the longer sequence
REF = OPTS.get("ref", "median")                # reference performance: median | longest | shortest duration
TAG = OPTS.get("tag", "")
SAVE_FRAMES = OPTS.get("save_frames", "0") == "1"
SFX = f"_{TAG}" if TAG else ""
RATIO = (0.75, 1.33)  # allowed duration ratio to the reference (filters cuts / different versions)

meta = pd.read_csv(data_dir / "metadata_clean.csv")
feat = pd.read_csv(data_dir / "features.csv")
D = meta.merge(feat, on="midi_filename")
D["piece"] = (D.composer + " | " + D.canonical_title.str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
              .str.replace(r"\s+", " ", regex=True).str.strip())


# ------------------------------------------------------------ MIDI
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
    ev = np.array(sorted(ev), dtype=float); ev[:, 0] *= 0.5 / div  # one 120 BPM tempo in MAESTRO
    return ev


def performance_frames(path):
    ev = load(path)
    T = ev[:, 0].max()
    n = int(np.ceil(T * FPS)) + 1
    chroma = np.zeros((n, 12))
    onset_vel = np.zeros(n); onset_cnt = np.zeros(n)
    open_notes = {}
    for t, st, a, c in ev:
        if st == 0x90 and c > 0:
            open_notes.setdefault(int(a), []).append((t, c))
            f = int(t * FPS)
            onset_vel[f] += c; onset_cnt[f] += 1
            chroma[f, int(a) % 12] += 2 * c / 127  # onset emphasis
        elif st == 0x80 or (st == 0x90 and c == 0):
            lst = open_notes.get(int(a))
            if lst:
                t0, v0 = lst.pop(0)
                f0, f1 = int(t0 * FPS), max(int(t * FPS), int(t0 * FPS) + 1)
                chroma[f0:f1, int(a) % 12] += v0 / 127
    chroma = np.log1p(chroma)
    chroma /= np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9

    def state(k):
        c = ev[(ev[:, 1] == 0xB0) & (ev[:, 2] == k)]
        s = np.zeros(n)
        if len(c):
            idx = np.searchsorted(c[:, 0], np.arange(n) / FPS, side="right") - 1
            s = np.where(idx >= 0, c[np.clip(idx, 0, None), 3], 0)
        return (s >= 64).astype(float)

    return {"chroma": chroma, "soft": state(67), "sus": state(64), "onset_vel": onset_vel, "onset_cnt": onset_cnt}


# ------------------------------------------------------------ banded DTW (vectorised per row)
def dtw_path(X, Y, band=BAND):
    """Align reference X (N x 12) to Y (M x 12); return for each X frame the matched Y frame."""
    N, M = len(X), len(Y)
    W = int(band * max(N, M)) + 5
    lo = np.clip(np.round(np.arange(N) * (M - 1) / max(N - 1, 1)).astype(int) - W, 0, M - 1)
    hi = np.clip(np.round(np.arange(N) * (M - 1) / max(N - 1, 1)).astype(int) + W, 0, M - 1)
    INF = 1e18
    Drows = []
    prev, plo, phi = None, 0, -1
    for i in range(N):
        js = np.arange(lo[i], hi[i] + 1)
        c = 1.0 - Y[js] @ X[i]
        if prev is None:
            A = np.full(len(js), INF); A[0] = c[0] if js[0] == 0 else INF
            if js[0] == 0:
                row = np.cumsum(c)  # first row: only horizontal moves from (0,0)
            else:
                row = np.full(len(js), INF)
        else:
            def get(jj):
                out = np.full(len(jj), INF)
                m = (jj >= plo) & (jj <= phi)
                out[m] = prev[jj[m] - plo]
                return out
            diag = get(js - 1); up = get(js)
            A = c + np.minimum(diag, up)
            S = np.cumsum(c)
            row = S + np.minimum.accumulate(A - S)
        Drows.append(row); prev, plo, phi = row, lo[i], hi[i]
    # backtrack
    i, j = N - 1, M - 1
    if not (lo[i] <= j <= hi[i]) or not np.isfinite(Drows[i][j - lo[i]]):
        return None, np.inf
    cost = Drows[i][j - lo[i]] / (N + M)
    match = np.zeros(N, dtype=int)
    match[i] = j
    def val(ii, jj):
        if ii < 0 or jj < 0 or not (lo[ii] <= jj <= hi[ii]):
            return INF
        return Drows[ii][jj - lo[ii]]
    while i > 0 or j > 0:
        cands = [(val(i - 1, j - 1), i - 1, j - 1), (val(i - 1, j), i - 1, j), (val(i, j - 1), i, j - 1)]
        _, i, j = min(cands)
        match[i] = j if match[i] == 0 or j < match[i] else match[i]
    return match, cost


def windows(x, w=WIN):
    n = len(x) // w
    return x[: n * w].reshape(n, w).mean(axis=1)


def kappa(a, b):
    a, b = a.astype(bool), b.astype(bool)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return np.nan if pe >= 1 - 1e-9 else (po - pe) / (1 - pe)


# ------------------------------------------------------------ main loop over pieces
counts = D.groupby("piece").size()
pieces = counts[counts >= 3].index
quality, aligned_store = [], {}
for pc in pieces:
    P = D[D.piece == pc].copy()
    if REF == "longest":
        ref = P.loc[P.duration.idxmax()]
    elif REF == "shortest":
        ref = P.loc[P.duration.idxmin()]
    else:
        ref = P.iloc[(P.duration - P.duration.median()).abs().argsort().iloc[0]]
    R = performance_frames(root / ref.midi_filename)
    store = {ref.midi_filename: {"soft": R["soft"], "sus": R["sus"], "loud": None, "row": ref}}
    # reference loudness curve: mean onset velocity in 1 s neighbourhood
    def loud_curve(fr, match=None):
        v, n = fr["onset_vel"], fr["onset_cnt"]
        k = np.ones(FPS)
        vs, ns = np.convolve(v, k, "same"), np.convolve(n, k, "same")
        curve = np.where(ns > 0, vs / np.maximum(ns, 1), np.nan)
        return curve if match is None else curve[match]
    store[ref.midi_filename]["loud"] = loud_curve(R)
    for _, row in P.iterrows():
        if row.midi_filename == ref.midi_filename:
            continue
        ratio = row.duration / ref.duration
        q = {"piece": pc, "ref": ref.midi_filename, "perf": row.midi_filename, "ratio": ratio}
        if not (RATIO[0] <= ratio <= RATIO[1]):
            q.update(status="duration ratio out of range", cost=np.nan); quality.append(q); continue
        F = performance_frames(root / row.midi_filename)
        match, cost = dtw_path(R["chroma"], F["chroma"])
        q["cost"] = cost
        if match is None:
            q["status"] = "no path"; quality.append(q); continue
        q["status"] = "ok"; quality.append(q)
        store[row.midi_filename] = {"soft": F["soft"][match], "sus": F["sus"][match],
                                    "loud": loud_curve(F, match), "row": row}
    aligned_store[pc] = store
    print(pc[:70], "| perfs", len(P), "| aligned", len(store))

Q = pd.DataFrame(quality)
Q.to_csv(res_dir / f"align_pairs_quality{SFX}.csv", index=False)
cost_cut = Q[Q.status == "ok"].cost.quantile(0.9)  # drop the worst 10% of alignments
print("alignment cost: median", round(Q[Q.status == 'ok'].cost.median(), 4), "| 90th pct cut", round(cost_cut, 4))
bad = set(Q[(Q.status == "ok") & (Q.cost > cost_cut)].perf)

# ------------------------------------------------------------ pairwise agreement
pairs = []
cons_rows = []
for pc, store in aligned_store.items():
    keys = [k for k in store if k not in bad]
    if len(keys) < 3:
        continue
    if SAVE_FRAMES:  # frame-level aligned data on the reference timeline, for 08_final_analyses.py
        fdir = res_dir / "aligned_frames"; fdir.mkdir(exist_ok=True)
        np.savez_compressed(fdir / f"{hashlib.md5(pc.encode()).hexdigest()[:12]}.npz", piece=pc, fps=FPS,
                            perf=np.array(keys), group=np.array([store[k]["row"].group for k in keys]),
                            year=np.array([store[k]["row"].year for k in keys]),
                            pianist=np.array([store[k]["row"].pianist for k in keys]),
                            composer=np.array([store[k]["row"].composer for k in keys]),
                            soft=np.vstack([store[k]["soft"] for k in keys]),
                            sus=np.vstack([store[k]["sus"] for k in keys]),
                            loud=np.vstack([store[k]["loud"] for k in keys]))
    W_soft = {k: windows(store[k]["soft"]) > 0.5 for k in keys}
    W_sus = {k: windows(store[k]["sus"]) > 0.5 for k in keys}
    W_loud = {k: pd.Series(windows(np.nan_to_num(store[k]["loud"], nan=np.nanmean(store[k]["loud"])))) for k in keys}
    for a, b in combinations(keys, 2):
        ra, rb = store[a]["row"], store[b]["row"]
        both_use = W_soft[a].mean() > 0.05 and W_soft[b].mean() > 0.05
        pairs.append({"piece": pc, "a": a, "b": b,
                      "same_edition": ra.year == rb.year,
                      "pair_type": "-".join(sorted([ra.group, rb.group])),
                      "same_pianist": ra.pianist == rb.pianist,
                      "kappa_soft": kappa(W_soft[a], W_soft[b]) if both_use else np.nan,
                      "kappa_sus": kappa(W_sus[a], W_sus[b]),
                      "r_loud": W_loud[a].corr(W_loud[b])})
    # consensus loudness: z-score each performance's loudness, average across pianists
    Z = pd.DataFrame({k: (W_loud[k] - W_loud[k].mean()) / W_loud[k].std() for k in keys})
    consensus = Z.mean(axis=1)
    q1, q3 = consensus.quantile(0.25), consensus.quantile(0.75)
    quiet, loud = consensus <= q1, consensus >= q3
    quart = pd.qcut(consensus.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])  # Q1 = quietest
    for k in keys:
        s = pd.Series(W_soft[k].astype(float))
        u = pd.Series(W_sus[k].astype(float))
        own = Z[k]
        rec = {"piece": pc, "perf": k, "group": store[k]["row"].group, "year": store[k]["row"].year,
               "pianist": store[k]["row"].pianist,
               "soft_in_consensus_quiet": s[quiet].mean(), "soft_in_consensus_loud": s[loud].mean(),
               "soft_in_own_quiet": s[own <= own.quantile(0.25)].mean(), "soft_overall": s.mean()}
        for qq in ["Q1", "Q2", "Q3", "Q4"]:
            rec[f"soft_{qq}"] = s[quart == qq].mean()
            rec[f"sus_{qq}"] = u[quart == qq].mean()
        cons_rows.append(rec)

PA = pd.DataFrame(pairs)
PA.to_csv(res_dir / f"align_agreement_pairs{SFX}.csv", index=False)
CQ = pd.DataFrame(cons_rows)
CQ.to_csv(res_dir / f"align_consensus_quiet{SFX}.csv", index=False)

# ------------------------------------------------------------ summaries (piece-level means first)
def piece_mean(df, col):
    return df.groupby("piece")[col].mean().dropna()

summ = []
for lab, sub in [("all pairs", PA[~PA.same_pianist]), ("same edition", PA[PA.same_edition & ~PA.same_pianist]),
                 ("different edition", PA[~PA.same_edition & ~PA.same_pianist]),
                 ("junior-junior", PA[(PA.pair_type == "junior-junior") & ~PA.same_pianist]),
                 ("senior-senior", PA[(PA.pair_type == "senior-senior") & ~PA.same_pianist]),
                 ("junior-senior", PA[(PA.pair_type == "junior-senior") & ~PA.same_pianist])]:
    row = {"subset": lab, "n_pairs": len(sub), "n_pieces": sub.piece.nunique()}
    for col in ["kappa_soft", "kappa_sus", "r_loud"]:
        pm = piece_mean(sub, col)
        row[f"{col}_mean"] = pm.mean(); row[f"{col}_n_pieces"] = len(pm)
    both = sub.groupby("piece")[["kappa_soft", "kappa_sus"]].mean().dropna()
    if len(both) >= 5:
        w = stats.wilcoxon(both.kappa_soft, both.kappa_sus)
        row["soft_vs_sus_wilcoxon_p"] = w.pvalue
        row["pieces_soft_lower"] = int((both.kappa_soft < both.kappa_sus).sum()); row["pieces_compared"] = len(both)
    summ.append(row)
S = pd.DataFrame(summ)
S.to_csv(res_dir / f"align_agreement_summary{SFX}.csv", index=False)
print("\n== agreement (piece-level means)\n", S.round(3).to_string(index=False))

cq = CQ.groupby("group")[["soft_in_consensus_quiet", "soft_in_consensus_loud", "soft_in_own_quiet", "soft_overall"]].mean()
cq.loc["all"] = CQ[["soft_in_consensus_quiet", "soft_in_consensus_loud", "soft_in_own_quiet", "soft_overall"]].mean()
print("\n== una corda in passages everyone plays softly vs loudly\n", cq.round(3).to_string())
users = CQ[CQ.soft_overall > 0.05]
print("performances using una corda >5%:", len(users), "of", len(CQ))
print("share of consensus-quiet windows WITHOUT una corda, among users:", round(1 - users.soft_in_consensus_quiet.mean(), 3))
w = stats.wilcoxon(CQ.soft_in_consensus_quiet, CQ.soft_in_consensus_loud)
print("quiet vs loud within performance: Wilcoxon p =", w.pvalue)

# by consensus-loudness quartile (Q1 quietest), for una corda and sustain, by group
qt = []
for g, x in [("all", CQ), ("junior", CQ[CQ.group == "junior"]), ("senior", CQ[CQ.group == "senior"])]:
    for qq in ["Q1", "Q2", "Q3", "Q4"]:
        qt.append({"group": g, "quartile": qq, "soft": x[f"soft_{qq}"].mean(), "sustain": x[f"sus_{qq}"].mean(), "n_perf": len(x)})
QT = pd.DataFrame(qt)
QT.to_csv(res_dir / f"align_by_loudness_quartile{SFX}.csv", index=False)
print("\n== pedal use by consensus-loudness quartile\n", QT.round(3).to_string(index=False))

# ------------------------------------------------------------ who decides "whether": variance of quiet-passage coverage
from scipy import linalg, optimize

def reml(y, X, groups):
    n = len(y); G = [(g[:, None] == g[None, :]).astype(float) for g in groups]
    def nll(t):
        s = np.exp(t); V = s[0] * np.eye(n) + sum(s[k + 1] * Gk for k, Gk in enumerate(G)); c = linalg.cho_factor(V, lower=True)
        ViX = linalg.cho_solve(c, X); XtViX = X.T @ ViX; b = np.linalg.solve(XtViX, X.T @ linalg.cho_solve(c, y)); r = y - X @ b
        return 0.5 * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(XtViX)[1] + r @ linalg.cho_solve(c, r))
    o = optimize.minimize(nll, np.log(np.full(len(G) + 1, np.var(y) / 3)), method="L-BFGS-B", bounds=[(-25, 5)] * (len(G) + 1))
    return np.exp(o.x)

vd = []
for col in ["soft_in_consensus_quiet", "soft_overall"]:
    d = CQ.dropna(subset=[col]).reset_index(drop=True)
    X = np.column_stack([np.ones(len(d)), (d.group == "senior").astype(float)])
    v = reml(d[col].values, X, [pd.factorize(d.pianist)[0], pd.factorize(d.piece)[0], pd.factorize(d.year)[0]])
    t = v.sum()
    vd.append({"outcome": col, "share_pianist": v[1] / t, "share_piece": v[2] / t, "share_edition": v[3] / t,
               "share_residual": v[0] / t, "n_perf": len(d), "n_pianists": d.pianist.nunique(), "n_pieces": d.piece.nunique()})
VD = pd.DataFrame(vd)
VD.to_csv(res_dir / f"align_variance_quiet_coverage{SFX}.csv", index=False)
print("\n== variance of una corda use at matched passages\n", VD.round(3).to_string(index=False))

# junior vs senior at the SAME passages: only pieces played by both groups carry information
mix = CQ.groupby("piece").group.nunique()
M = CQ[CQ.piece.isin(mix[mix == 2].index)]
pm = M.groupby(["piece", "group"]).soft_in_consensus_quiet.mean().unstack()
wm = stats.wilcoxon(pm.senior - pm.junior)
mixed = {"n_pieces": len(pm), "senior_higher": int((pm.senior > pm.junior).sum()),
         "mean_junior": pm.junior.mean(), "mean_senior": pm.senior.mean(), "wilcoxon_p": wm.pvalue}
pd.DataFrame([mixed]).to_csv(res_dir / f"align_mixed_pieces_group_contrast{SFX}.csv", index=False)
print("\n== same passages, junior vs senior (mixed pieces)\n", mixed)
