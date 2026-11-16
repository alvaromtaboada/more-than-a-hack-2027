"""Per-frame features from (px, py, team, bx, by), identity-free (permutation invariant per team).

Team features are written for team 0 and team 1 in each team's attacking direction, so swapping
the teams (augmentation) is a column permutation. Ball features come only from the ball track.
"""
import numpy as np
import pandas as pd

L, W = 105.0, 68.0
FPS = 25


def _vel(a, k=2):
    v = np.full_like(a, np.nan)
    v[k:-k] = (a[2 * k:] - a[:-2 * k]) / (2 * k / FPS)
    return v


def attack_sign(px, team, period):
    """+1 if the team attacks towards x=L in that period (its centroid sits in the other half)."""
    s = np.ones((len(px), 2), np.float32)
    for p in np.unique(period[period > 0]):
        m = period == p
        for t in (0, 1):
            cx = np.nanmean(px[m][:, team == t])
            s[m, t] = 1.0 if cx < L / 2 else -1.0
    return s


def interp_gaps(a, max_gap=10):
    """Linear interpolation across gaps of at most max_gap frames (0.4 s)."""
    a = a.copy(); nan = np.isnan(a)
    if nan.all():
        return a
    i = np.arange(len(a))
    filled = np.interp(i, i[~nan], a[~nan])
    edges = np.diff(np.concatenate(([0], nan.astype(np.int8), [0])))
    for s0, s1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if s1 - s0 <= max_gap and s0 > 0 and s1 < len(a):
            a[s0:s1] = filled[s0:s1]
    return a


def roll(x, w):
    return pd.Series(x).rolling(2 * w + 1, center=True, min_periods=1).mean().values


def frame_features(d, period):
    px, py, team = d["px"], d["py"], d["team"]
    n = len(px)
    vx, vy = _vel(px), _vel(py)
    sp = np.hypot(vx, vy)
    sp[sp > 12] = np.nan  # identity switches create teleports
    sgn = attack_sign(px, team, period)
    F = {}
    vis_all = ~np.isnan(px)
    F["n_vis"] = vis_all.sum(1)
    F["sp_all"] = np.nanmean(sp, 1)
    F["sp_p90"] = np.nanpercentile(sp, 90, axis=1)
    F["slow_frac"] = np.nanmean(np.where(np.isnan(sp), np.nan, sp < 1.0), 1)
    for t in (0, 1):
        m = team == t
        x = (px[:, m] - L / 2) * sgn[:, t:t + 1]  # attacking direction = +x, centre = 0
        y = py[:, m] - W / 2
        F[f"t{t}_n"] = (~np.isnan(x)).sum(1)
        F[f"t{t}_cx"] = np.nanmean(x, 1)
        F[f"t{t}_sx"] = np.nanstd(x, 1)
        F[f"t{t}_sy"] = np.nanstd(np.abs(y), 1)
        F[f"t{t}_maxx"] = np.nanmax(x, 1)
        F[f"t{t}_minx"] = np.nanpercentile(x, 10, axis=1)
        F[f"t{t}_sp"] = np.nanmean(sp[:, m], 1)
        F[f"t{t}_vx"] = np.nanmean(vx[:, m] * sgn[:, t:t + 1], 1)
        F[f"t{t}_wide"] = np.nansum(np.abs(y) > W / 2 - 6, 1)
    # ball (may be NaN)
    bx, by = interp_gaps(d["bx"]), interp_gaps(d["by"])
    bvx, bvy = _vel(bx), _vel(by)
    bsp = np.hypot(bvx, bvy)
    dist = np.hypot(px - bx[:, None], py - by[:, None])
    F["b_seen"] = (~np.isnan(bx)).astype(np.float32)
    F["b_sp"] = bsp
    F["b_out"] = np.maximum.reduce([-bx, bx - L, -by, by - W])  # >0 outside the pitch
    for t in (0, 1):
        dt = np.where(team[None, :] == t, dist, np.nan)
        ds = np.sort(dt, 1)
        F[f"t{t}_d1"] = ds[:, 0]
        F[f"t{t}_d2"] = ds[:, 1]
        F[f"t{t}_n5"] = np.nansum(dt < 5, 1) * F["b_seen"]
        F[f"t{t}_bx"] = (bx - L / 2) * sgn[:, t]
    F["d_diff"] = F["t0_d1"] - F["t1_d1"]
    slow = bsp < 6
    for t in (0, 1):
        F[f"t{t}_ctrl"] = ((F[f"t{t}_d1"] < 1.5) & slow & (F[f"t{t}_d1"] < F[f"t{1-t}_d1"])).astype(np.float32)
    # last / next sighting of the ball: which team was nearer, and how long ago / until
    seen = ~np.isnan(F["d_diff"])
    i = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, i, -1))
    nxt = np.minimum.accumulate(np.where(seen, i, n)[::-1])[::-1]
    F["d_last"] = np.where(last >= 0, F["d_diff"][np.maximum(last, 0)], np.nan)
    F["d_next"] = np.where(nxt < n, F["d_diff"][np.minimum(nxt, n - 1)], np.nan)
    F["age_last"] = np.minimum(np.where(last >= 0, i - last, 9999), 9999)
    F["age_next"] = np.minimum(np.where(nxt < n, nxt - i, 9999), 9999)
    df = pd.DataFrame(F).astype(np.float32)
    # temporal context
    base = ["sp_all", "slow_frac", "sp_p90", "t0_sp", "t1_sp", "t0_vx", "t1_vx", "t0_cx", "t1_cx"]
    for w in (12, 50, 125):
        for c in base:
            df[f"{c}_r{w}"] = roll(df[c].values, w)
    for w in (6, 25, 75):
        for c in ("d_diff", "b_seen", "b_sp", "t0_n5", "t1_n5"):
            df[f"{c}_r{w}"] = pd.Series(df[c].values).rolling(2 * w + 1, center=True, min_periods=1).mean().values
    for c in ("cx", "sx", "sy", "maxx", "minx", "vx", "wide"):
        df[f"dt_{c}"] = df[f"t0_{c}"] - df[f"t1_{c}"]
    for w in (25, 75, 200):
        for c in ("dt_cx", "dt_sx", "dt_sy", "dt_maxx", "dt_minx", "dt_vx", "dt_wide", "t0_sx", "t1_sx",
                  "t0_maxx", "t1_maxx", "t0_minx", "t1_minx"):
            df[f"{c}_r{w}"] = roll(df[c].values, w)
        for c in ("t0_ctrl", "t1_ctrl"):
            df[f"{c}_r{w}"] = roll(df[c].values, w)
    df["sp_all_trend"] = roll(df.sp_all.values, 25) - roll(np.roll(df.sp_all.values, 75), 25)
    return df.astype(np.float32)


def swap_cols(cols):
    """Column order with team 0 <-> team 1 swapped (for symmetry augmentation)."""
    def sw(c):
        if c.startswith("t0_"): return "t1_" + c[3:]
        if c.startswith("t1_"): return "t0_" + c[3:]
        return c
    return [sw(c) for c in cols]
