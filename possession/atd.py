"""Raw ATD -> numpy arrays (cached). Coordinates converted to metres, origin at pitch corner."""
import os, sys, xml.etree.ElementTree as ET
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, GAME, require, source_frames  # noqa: E402

CACHE = os.path.join(ROOT, "cache")


def teams(meta_path):
    root = ET.parse(meta_path).getroot()
    home = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], np.int8)


def load(source="tactical", game=GAME):
    """dict: px, py (n_frames, n_slots) metres; bx, by ball metres; team (n_slots,) 0=H 1=A."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"{game}_{source}_atd.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    meta, raw = require(["atd_meta", "atd_raw"], game, source=source)
    team = teams(meta)
    n = source_frames(source, game)
    L, W = GAMES[game]["pitch"]
    with open(raw) as f:
        txt = f.read().replace(":", ",").replace(";", ",")
    vals = np.array(txt.split(), dtype=object)  # one line per frame
    arr = np.array([np.array(l.split(","), dtype=np.float32) for l in vals])
    fid = arr[:, 0].astype(int) - 1
    P = arr[:, 1:1 + 3 * len(team)].reshape(len(arr), len(team), 3)
    B = arr[:, 1 + 3 * len(team):]
    out = dict(px=np.full((n, len(team)), np.nan, np.float32), py=None, team=team)
    out["px"][fid] = P[:, :, 0] * L
    out["py"] = np.full_like(out["px"], np.nan); out["py"][fid] = P[:, :, 1] * W
    out["ps"] = np.full_like(out["px"], np.nan); out["ps"][fid] = P[:, :, 2]
    for k, i, s in (("bx", 0, L), ("by", 1, W), ("bs", 2, 1)):
        out[k] = np.full(n, np.nan, np.float32); out[k][fid] = B[:, i] * s
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    d = load()
    print({k: v.shape for k, v in d.items()}, "ball seen", np.mean(~np.isnan(d["bx"])))
