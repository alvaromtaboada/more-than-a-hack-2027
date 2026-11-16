"""Train on other games, decode with an HMM, evaluate leave-one-game-out with the official metric."""
import os, sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from .features import frame_features, swap_cols  # noqa: E402
from . import external, atd, idsse  # noqa: E402

SEED = 0
HGB = dict(max_iter=int(os.environ.get("MTAH_ITER", 600)), learning_rate=0.05,
           max_leaf_nodes=int(os.environ.get("MTAH_LEAVES", 63)), l2_regularization=1.0, random_state=SEED)
BALL_COLS = ("b_", "d_diff", "_d1", "_d2", "_n5", "_bx", "_ctrl", "d_last", "d_next", "age_")


def is_ball(c):
    return any(k in c for k in BALL_COLS)


def target_masks():
    """Real missingness patterns of the provided game (ball, and player count), no labels."""
    d = atd.load()
    from games import half_video_blocks
    idx = np.concatenate([np.arange(a, b + 1) for _, a, b in half_video_blocks("BAR-ATH")])
    return np.isnan(d["bx"][idx]), idx


def augment(d, rng, ball_missing):
    d = dict(d)
    n, P = d["px"].shape
    off = rng.integers(len(ball_missing))
    m = ball_missing[(off + np.arange(n)) % len(ball_missing)]
    d["bx"] = np.where(m, np.nan, d["bx"]); d["by"] = np.where(m, np.nan, d["by"])
    # player dropout in 2 s chunks (~7%, the provided game shows 20.5 of 22 on average)
    chunks = rng.random(((n + 49) // 50, P)) < 0.07
    drop = np.repeat(chunks, 50, 0)[:n]
    d["px"] = np.where(drop, np.nan, d["px"]); d["py"] = np.where(drop, np.nan, d["py"])
    return d


def game_xy(game, rng, ball_missing):
    import pandas as pd
    d = external.load(game) if isinstance(game, int) else idsse.load(game)
    d = augment(d, rng, ball_missing)   # always consume rng so cached/uncached runs agree
    if not isinstance(game, int):
        d["label"] = idsse.harmonize(d["label"], int(os.environ.get("MTAH_HARM", 0)))
    cache = os.path.join(ROOT, "cache", f"feat_{'sample' if isinstance(game, int) else ''}{game}.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache), d["label"]
    X = frame_features(d, d["period"])
    X.to_pickle(cache)
    return X, d["label"]


def swapped(X, y):
    Xs = X[swap_cols(list(X.columns))].copy(); Xs.columns = X.columns
    for c in Xs.columns:
        if c.startswith(("d_diff", "d_last", "d_next", "dt_")):
            Xs[c] = -Xs[c]
    ys = y.copy(); ys[y == 0] = 1; ys[y == 1] = 0
    return Xs, ys


def fit(Xs, ys, step=3):
    import pandas as pd
    X = pd.concat(Xs); y = np.concatenate(ys)
    X2, y2 = swapped(X, y)
    X = pd.concat([X, X2]); y = np.concatenate([y, y2])
    keep = (y >= 0) & (np.arange(len(y)) % step == 0)
    X, y = X[keep], y[keep]
    dead_cols = [c for c in X.columns if not is_ball(c)]
    mdead = HistGradientBoostingClassifier(**HGB)
    mdead.fit(X[dead_cols], (y == 2).astype(int))
    alive = y < 2
    mteam = HistGradientBoostingClassifier(**HGB)
    mteam.fit(X[alive], y[alive])
    return dict(dead=mdead, dead_cols=dead_cols, team=mteam, cols=list(X.columns))


def predict(M, X):
    pd_ = M["dead"].predict_proba(X[M["dead_cols"]])[:, 1]
    ph = M["team"].predict_proba(X[M["cols"]])[:, 0]
    # symmetrise: average with the team-swapped view
    Xs, _ = swapped(X, np.zeros(len(X), np.int8))
    ph = 0.5 * (ph + M["team"].predict_proba(Xs[M["cols"]])[:, 1])
    pds = M["dead"].predict_proba(Xs[M["dead_cols"]])[:, 1]
    pd_ = 0.5 * (pd_ + pds)
    return np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_], 1)


def viterbi(P, alpha=0.3, p_switch=0.004, p_dead=0.0015, eps=1e-4):
    """3-state Viterbi. alpha scales emissions (frames are strongly correlated)."""
    logE = alpha * np.log(np.clip(P, eps, 1))
    T = np.log(np.array([[1 - p_switch - p_dead, p_switch, p_dead],
                         [p_switch, 1 - p_switch - p_dead, p_dead],
                         [p_dead, p_dead, 1 - 2 * p_dead]]))
    n = len(P)
    back = np.zeros((n, 3), np.int8)
    s = logE[0].copy()
    for t in range(1, n):
        c = s[:, None] + T
        back[t] = np.argmax(c, 0)
        s = c[back[t], np.arange(3)] + logE[t]
    path = np.zeros(n, np.int8); path[-1] = np.argmax(s)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def score(gt, pred):
    gt = np.where(gt < 0, 3, gt).astype(np.int8)
    pred = np.where(gt == 3, 3, pred).astype(np.int8)
    runs = pm.valid_runs(gt)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    sp = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    return dict(spell=sp["mean_f1"], macro=pm.macro_f1(cm3), n_gt=sp["n_gt"], n_pred=sp["n_pred"])


# ---------------- stage 2: temporal stacking over stage-1 probabilities ----------------
import pandas as _pd

S2_RAW = ["d_diff", "d_last", "d_next", "age_last", "age_next", "t0_ctrl", "t1_ctrl", "b_sp", "b_seen",
          "sp_all", "sp_all_r12", "b_out"]


def _side(x, w, future):
    s = _pd.Series(x[::-1] if future else x).rolling(w, min_periods=1).mean().values
    return s[::-1] if future else s


def stage2_feats(P, X):
    ph = P[:, 0] / np.clip(P[:, 0] + P[:, 1], 1e-6, None) - 0.5   # signed: + = team 0
    pdd = P[:, 2]
    F = {"ph": ph, "pd": pdd}
    for w in (6, 12, 25, 50, 100, 250):
        for nm, x in (("ph", ph), ("pd", pdd)):
            a, b = _side(x, w, False), _side(x, w, True)
            F[f"{nm}_p{w}"] = a; F[f"{nm}_f{w}"] = b; F[f"{nm}_j{w}"] = b - a
    for c in S2_RAW:
        F[c] = X[c].values
    return _pd.DataFrame(F).astype(np.float32)


def s2_swap(F, y=None):
    F = F.copy()
    for c in F.columns:
        if c.startswith(("ph", "d_diff", "d_last", "d_next")):
            F[c] = -F[c]
    F["t0_ctrl"], F["t1_ctrl"] = F["t1_ctrl"].values.copy(), F["t0_ctrl"].values.copy()
    if y is None:
        return F
    ys = y.copy(); ys[y == 0] = 1; ys[y == 1] = 0
    return F, ys


def fit2(Fs, ys, step=2):
    """Two heads, as in stage 1: dead-ball head sees no ball-derived input (ball visibility is
    confounded with dead ball in the training games but not in the target game)."""
    F = _pd.concat(Fs); y = np.concatenate(ys)
    F2, y2 = s2_swap(F, y)
    F = _pd.concat([F, F2]); y = np.concatenate([y, y2])
    keep = (y >= 0) & (np.arange(len(y)) % step == 0)
    F, y = F[keep], y[keep]
    dcols = [c for c in F.columns if c.startswith(("pd", "sp_all"))]
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=SEED)
    md = HistGradientBoostingClassifier(**kw).fit(F[dcols], (y == 2).astype(int))
    alive = y < 2
    mt = HistGradientBoostingClassifier(**kw).fit(F[alive], y[alive])
    return dict(dead=md, dcols=dcols, team=mt)


def predict2(m, F):
    Fs = s2_swap(F)
    pdd = 0.5 * (m["dead"].predict_proba(F[m["dcols"]])[:, 1] + m["dead"].predict_proba(Fs[m["dcols"]])[:, 1])
    ph = 0.5 * (m["team"].predict_proba(F)[:, 0] + m["team"].predict_proba(Fs)[:, 1])
    return np.stack([(1 - pdd) * ph, (1 - pdd) * (1 - ph), pdd], 1)


def viterbi_md(P, alpha=0.5, p_switch=0.01, p_dead=0.0015, k=0, eps=1e-4):
    """Viterbi where a team spell must last >= k frames before it can end (chain of k sub-states)."""
    if k <= 1:
        return viterbi(P, alpha, p_switch, p_dead, eps)
    logE = alpha * np.log(np.clip(P, eps, 1))
    n = len(P); S = 2 * k + 1; D = 2 * k
    lsw, ld, lstay = np.log(p_switch), np.log(p_dead), np.log(1 - p_switch - p_dead)
    ldd, lda = np.log(1 - 2 * p_dead), np.log(p_dead)
    emit_idx = np.array([0] * k + [1] * k + [2])
    back = np.zeros((n, S), np.int16)
    s = np.full(S, -np.inf); s[0] = logE[0, 0]; s[k] = logE[0, 1]; s[D] = logE[0, 2]
    for t in range(1, n):
        new = np.full(S, -np.inf); bp = np.zeros(S, np.int16)
        for team in (0, 1):
            b = team * k; o = (1 - team) * k
            # advance inside the chain
            new[b + 1:b + k] = s[b:b + k - 1]; bp[b + 1:b + k] = np.arange(b, b + k - 1)
            # last sub-state: stay
            cand = [(s[b + k - 1] + lstay, b + k - 1)]
            if k == 1:
                pass
            new[b + k - 1], bp[b + k - 1] = max(cand + [(new[b + k - 1], bp[b + k - 1])]) if k > 1 else cand[0]
            # entry: from the other team's mature state, or from dead
            e = [(s[o + k - 1] + lsw, o + k - 1), (s[D] + lda, D)]
            v, j = max(e)
            new[b], bp[b] = v, j
        dv = [(s[D] + ldd, D), (s[k - 1] + ld, k - 1), (s[2 * k - 1] + ld, 2 * k - 1)]
        new[D], bp[D] = max(dv)
        back[t] = bp
        s = new + logE[t, emit_idx]
    path = np.zeros(n, np.int16); path[-1] = int(np.argmax(s))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return emit_idx[path].astype(np.int8)
