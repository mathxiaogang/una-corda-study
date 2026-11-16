"""Step 2 - extract pedal and dynamics features from every MAESTRO MIDI file.

Usage:
    python -I 02_extract_features.py <maestro-v3.0.0 dir> <metadata_clean.csv> <out_dir>

Output:
    <out_dir>/features.csv   one row per performance
    <out_dir>/cc67_raw_by_year.csv   raw soft-pedal controller values (for the equipment audit)

Pedal "down" = controller value >= 64 unless stated otherwise (CC64 sustain, CC66 sostenuto, CC67 soft).
All MAESTRO files have a single tempo event (120 BPM), so ticks convert to seconds with one factor.
A small self-contained Standard MIDI File parser is used (no third-party MIDI library needed).
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root, meta_csv, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)


def _vlq(b, i):
    v = 0
    while True:
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if c < 0x80:
            return v, i


def load_midi(path):
    """Return (events array [time_s, status, data1, data2], ticks_per_quarter)."""
    b = path.read_bytes()
    assert b[:4] == b"MThd"
    hl = struct.unpack(">I", b[4:8])[0]
    _, ntrk, div = struct.unpack(">HHH", b[8:14])
    i, ev, tempo = 8 + hl, [], 500000
    for _ in range(ntrk):
        assert b[i:i + 4] == b"MTrk"
        L = struct.unpack(">I", b[i + 4:i + 8])[0]
        j, end, tick, rs = i + 8, i + 8 + L, 0, None
        while j < end:
            d, j = _vlq(b, j)
            tick += d
            s = b[j]
            if s == 0xFF:
                typ = b[j + 1]
                l, j = _vlq(b, j + 2)
                if typ == 0x51:
                    tempo = int.from_bytes(b[j:j + 3], "big")
                j += l
                continue
            if s in (0xF0, 0xF7):
                l, j = _vlq(b, j + 1)
                j += l
                continue
            if s & 0x80:
                rs = s
                j += 1
            st = rs & 0xF0
            if st in (0xC0, 0xD0):
                j += 1
                continue
            ev.append((tick, st, b[j], b[j + 1]))
            j += 2
        i = end
    ev = np.array(sorted(ev), dtype=float)
    ev[:, 0] *= tempo / 1e6 / div
    return ev, div


def frac_time(t, v, T, lo, hi=None):
    """Share of [0, T] during which a step function (times t, values v) lies in [lo, hi]."""
    if len(t) == 0:
        return 0.0
    dur = np.diff(np.append(t, T))
    m = v >= lo if hi is None else (v >= lo) & (v <= hi)
    return float(dur[m].sum() / T)


def episodes(t, v, T, thr=64):
    """Start/end times of spans where the controller is >= thr."""
    if len(t) == 0:
        return np.empty((0, 2))
    down = v >= thr
    starts, ends = [], []
    prev = False
    for ti, d in zip(t, down):
        if d and not prev:
            starts.append(ti)
        if prev and not d:
            ends.append(ti)
        prev = d
    if prev:
        ends.append(T)
    return np.column_stack([starts, ends]) if starts else np.empty((0, 2))


meta = pd.read_csv(meta_csv)
rows, raw67 = [], []
for f, year in zip(meta.midi_filename, meta.year):
    ev, tpq = load_midi(root / f)
    T = ev[:, 0].max()
    notes = ev[(ev[:, 1] == 0x90) & (ev[:, 3] > 0)]
    on, vel = notes[:, 0], notes[:, 3]
    r = {"midi_filename": f, "tpq": tpq, "midi_len_s": T, "n_notes": len(on), "nps": len(on) / T,
         "vel_mean": vel.mean(), "vel_sd": vel.std(),
         "dyn_range": np.percentile(vel, 95) - np.percentile(vel, 5)}
    cc = {k: ev[(ev[:, 1] == 0xB0) & (ev[:, 2] == k)] for k in (64, 66, 67)}

    # soft pedal (una corda)
    s = cc[67]
    st, sv = s[:, 0], s[:, 3]
    raw67.append(pd.DataFrame({"year": year, "value": sv.astype(int)}))
    r["soft_frac"] = frac_time(st, sv, T, 64)
    r["soft_frac32"] = frac_time(st, sv, T, 32)
    r["soft_frac96"] = frac_time(st, sv, T, 96)
    ep = episodes(st, sv, T)
    r["soft_episodes_pm"] = len(ep) / (T / 60)
    r["soft_episode_med_s"] = float(np.median(ep[:, 1] - ep[:, 0])) if len(ep) else np.nan
    k = np.searchsorted(st, on, side="right") - 1
    under = np.where(k >= 0, sv[np.clip(k, 0, None)], 0) >= 64 if len(st) else np.zeros(len(on), bool)
    r["soft_note_share"] = under.mean()
    if under.sum() >= 30 and (~under).sum() >= 30:
        r["vel_soft"] = np.median(vel[under])
        r["vel_nosoft"] = np.median(vel[~under])
        r["vel_gap"] = r["vel_nosoft"] - r["vel_soft"]
        r["soft_at_mf"] = (vel[under] >= 60).mean()

    # sustain pedal
    p = cc[64]
    pt, pv = p[:, 0], p[:, 3]
    r["sus_frac"] = frac_time(pt, pv, T, 64)
    r["sus_frac32"] = frac_time(pt, pv, T, 32)
    r["sus_frac96"] = frac_time(pt, pv, T, 96)
    down10 = frac_time(pt, pv, T, 10)
    r["sus_partial"] = frac_time(pt, pv, T, 10, 100) / down10 if down10 > 0 else np.nan
    x = (pv >= 64).astype(int)
    press_idx = np.where(np.diff(x) == 1)[0] + 1
    r["sus_presses_pm"] = len(press_idx) / (T / 60)
    press_t = pt[press_idx]
    kk = np.searchsorted(on, press_t) - 1
    lag = press_t[kk >= 0] - on[kk[kk >= 0]]
    if len(lag) > 10:
        r["press_lag_ms"] = np.median(lag) * 1000
        r["synco_share"] = ((lag > 0.03) & (lag < 0.4)).mean()

    # sostenuto
    q = cc[66]
    r["sost_frac"] = frac_time(q[:, 0], q[:, 3], T, 64)
    rows.append(r)

feat = pd.DataFrame(rows)
feat.to_csv(out_dir / "features.csv", index=False)
raw = pd.concat(raw67)
raw.groupby(["year", "value"]).size().rename("n").reset_index().to_csv(out_dir / "cc67_raw_by_year.csv", index=False)
print(feat.describe().T.round(3).to_string())
