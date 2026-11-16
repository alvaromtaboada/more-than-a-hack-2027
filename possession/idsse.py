"""IDSSE open dataset (Bassek et al. 2025, 7 Bundesliga matches, CC-BY 4.0) -> arrays + labels.

Labels come straight from the DFL position feed: BallStatus=0 -> D, else BallPossession 1/2 -> H/A,
i.e. ownership as named by corrected tracking -- the same derivation as the challenge ground truth.
"""
import glob, os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE = os.environ.get("MTAH_IDSSE_DIR", os.path.join(os.path.dirname(ROOT), "extdata", "idsse"))
CACHE = os.path.join(ROOT, "cache")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
FS = re.compile(r'GameSection="(\w+)".*TeamId="([^"]+)" PersonId="([^"]+)"')
FR = re.compile(r'N="(\d+)".*? X="([-\d.]+)" Y="([-\d.]+)"')
BL = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')
GAP = 50


def _parse(match):
    info = glob.glob(os.path.join(IDSSE, f"DFL_02_01_matchinformation_*{match}.xml"))[0]
    txt = open(info, encoding="utf-8").read()
    home = re.search(r'HomeTeamId="([^"]+)"', txt).group(1)
    pos = glob.glob(os.path.join(IDSSE, f"DFL_04_03_positions_raw_observed_*{match}.xml"))[0]
    tracks = {}  # (section, team, person) -> list of (N, x, y[, poss, status])
    cur = None
    with open(pos, encoding="utf-8") as f:
        for line in f:
            if line.startswith("<Frame "):
                if cur is None:
                    continue
                m = FR.search(line)
                row = [int(m.group(1)), float(m.group(2)), float(m.group(3))]
                if cur[1] == "BALL":
                    b = BL.search(line); row += [int(b.group(1)), int(b.group(2))]
                tracks[cur].append(row)
            elif line.startswith("<FrameSet"):
                m = FS.search(line)
                cur = None if m.group(2) == "referee" else (m.group(1), m.group(2), m.group(3))
                if cur is not None:
                    tracks.setdefault(cur, [])
    return home, tracks


def harmonize(label, min_frames):
    """Controlled possession: a team run shorter than min_frames inside live play (a touch, a
    deflection) is given back to the team that had the ball before it."""
    lab = label.copy()
    if min_frames <= 0:
        return lab
    edges = np.flatnonzero(np.diff(lab) != 0) + 1
    starts = np.concatenate(([0], edges)); stops = np.concatenate((edges, [len(lab)]))
    prev = -1
    for a, b in zip(starts, stops):
        v = lab[a]
        if v in (0, 1):
            if prev in (0, 1) and v != prev and b - a < min_frames:
                lab[a:b] = prev; v = prev
            prev = v
        else:
            prev = -1 if v == 2 else prev
    return lab


def load(match):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"idsse_{match}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    home, tracks = _parse(match)
    secs = ("firstHalf", "secondHalf")
    rng = {}
    for s in secs:
        b = np.array(tracks[[k for k in tracks if k[0] == s and k[1] == "BALL"][0]])
        rng[s] = (int(b[:, 0].min()), int(b[:, 0].max()))
    off = {secs[0]: 0, secs[1]: rng[secs[0]][1] - rng[secs[0]][0] + 1 + GAP}
    n = off[secs[1]] + rng[secs[1]][1] - rng[secs[1]][0] + 1
    people = sorted({(k[1], k[2]) for k in tracks if k[1] != "BALL"})
    team = np.array([0 if t == home else 1 for t, _ in people], np.int8)
    px = np.full((n, len(people)), np.nan, np.float32); py = px.copy()
    bx = np.full(n, np.nan, np.float32); by = bx.copy()
    label = np.full(n, -1, np.int8); period = np.zeros(n, np.int8)
    for (s, t, p), rows in tracks.items():
        a = np.array(rows); idx = a[:, 0].astype(int) - rng[s][0] + off[s]
        if t == "BALL":
            bx[idx] = a[:, 1] + 52.5; by[idx] = a[:, 2] + 34.0
            label[idx] = np.where(a[:, 4] == 0, 2, np.where(a[:, 3] == 1, 0, 1))
            period[idx] = 1 if s == secs[0] else 2
        else:
            j = people.index((t, p)); px[idx, j] = a[:, 1] + 52.5; py[idx, j] = a[:, 2] + 34.0
    out = dict(px=px, py=py, team=team, bx=bx, by=by, label=label, period=period)
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(4) as pool:
        pool.map(load, MATCHES)
    for m in MATCHES:
        d = load(m); lab = d["label"]
        ch = np.flatnonzero(np.diff(lab) != 0) + 1
        segs = np.split(lab, ch); poss = [s for s in segs if s[0] in (0, 1)]
        print(m, len(lab), d["px"].shape[1], {k: int((lab == k).sum()) for k in (-1, 0, 1, 2)},
              "poss", len(poss), "mean %.1fs" % np.mean([len(s) / 25 for s in poss]),
              "<2s %d" % sum(len(s) < 50 for s in poss), "dead segs", sum(s[0] == 2 for s in segs),
              "nvis %.1f" % np.mean(np.sum(~np.isnan(d["px"]), 1)[lab >= 0]))
