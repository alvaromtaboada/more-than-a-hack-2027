"""Metrica public sample games 1-3 (other matches, human event data) -> arrays + frame labels.

Labels (0=H, 1=A, 2=D, -1=ungraded) are derived from the event stream:
  * dead ball: from the end of a BALL OUT / goal / offside, or the start of a FAULT RECEIVED,
    until the start of the next restart (SET PIECE or GOAL KICK);
  * alive: team of the latest on-ball event (PASS, CARRY, RECOVERY, SHOT, BALL LOST, SET PIECE,
    BALL OUT); CHALLENGE / FAULT / CARD do not move possession.
"""
import json, os
import numpy as np, pandas as pd
from kloppy import metrica

SAMPLE = os.environ.get("MTAH_SAMPLE_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "extdata", "sample-data", "data"))
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
PITCH = (105.0, 68.0)
ONBALL = {"PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE", "BALL OUT"}


def _events(game):
    g = os.path.join(SAMPLE, f"Sample_Game_{game}")
    if game in (1, 2):
        e = pd.read_csv(os.path.join(g, f"Sample_Game_{game}_RawEventsData.csv"))
        return pd.DataFrame(dict(team=np.where(e.Team == "Home", 0, 1), type=e.Type,
                                 sub=e.Subtype.fillna("").astype(str), start=e["Start Frame"],
                                 end=e["End Frame"], period=e.Period))
    ev = json.load(open(os.path.join(g, f"Sample_Game_{game}_events.json")))["data"]
    rows = []
    for x in ev:
        st = x.get("subtypes")
        st = " ".join(s["name"] for s in st) if isinstance(st, list) else (st or {}).get("name", "") if st else ""
        rows.append(dict(team=0 if x["team"]["id"] == "FIFATMA" else 1, type=x["type"]["name"], sub=st,
                         start=x["start"]["frame"], end=x["end"]["frame"], period=x["period"]))
    return pd.DataFrame(rows)


def labels_from_events(e, n):
    e = e.sort_values("start", kind="stable").reset_index(drop=True)
    lab = np.full(n + 2, -1, np.int8)
    restart = (e.type == "SET PIECE") | e["sub"].str.contains("GOAL KICK") & (e.type == "PASS")
    dead_from = None
    for p in sorted(e.period.unique()):
        ep = e[e.period == p]
        lo = int(ep.start.min()); hi = int(max(ep.end.max(), ep.start.max()))
        cur = None; last = lo
        for i, r in ep.iterrows():
            s = int(r.start)
            if s > last:
                lab[last:s] = 2 if dead_from is not None else (cur if cur is not None else -1)
                last = s
            if restart[i] and dead_from is not None:
                dead_from = None
            if r.type in ONBALL:
                cur = int(r.team)
            sub = str(r["sub"])
            ends_play = (r.type == "BALL OUT" or (r.type == "SHOT" and ("OUT" in sub or "GOAL" in sub))
                         or "OFFSIDE" in sub or "HAND BALL" in sub)
            if ends_play:
                lab[last:max(int(r.end), s) + 1] = cur; last = max(int(r.end), s) + 1; dead_from = last
            elif r.type == "FAULT RECEIVED":
                dead_from = s
        lab[last:hi + 1] = 2 if dead_from is not None else cur
        dead_from = None
    return lab[1:n + 1]  # event frames count from 1


def load(game):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f"sample{game}.npz")
    if os.path.exists(cache):
        return dict(np.load(cache))
    g = os.path.join(SAMPLE, f"Sample_Game_{game}")
    if game in (1, 2):
        ds = metrica.load_tracking_csv(
            home_data=os.path.join(g, f"Sample_Game_{game}_RawTrackingData_Home_Team.csv"),
            away_data=os.path.join(g, f"Sample_Game_{game}_RawTrackingData_Away_Team.csv"))
    else:
        ds = metrica.load_tracking_epts(meta_data=os.path.join(g, f"Sample_Game_{game}_metadata.xml"),
                                        raw_data=os.path.join(g, f"Sample_Game_{game}_tracking.txt"))
    df = ds.to_df()
    pids = [p.player_id for t in ds.metadata.teams for p in t.players if f"{p.player_id}_x" in df]
    team = np.array([0 if t.ground.value == "home" else 1 for t in ds.metadata.teams
                     for p in t.players if f"{p.player_id}_x" in df], np.int8)
    n = int(df.frame_id.max())
    fid = df.frame_id.values.astype(int) - 1
    out = dict(team=team, period=np.zeros(n, np.int8))
    out["period"][fid] = df.period_id.values
    for k, col, s in (("px", "_x", PITCH[0]), ("py", "_y", PITCH[1])):
        a = np.full((n, len(pids)), np.nan, np.float32)
        a[fid] = df[[p + col for p in pids]].values * s
        out[k] = a
    for k, col, s in (("bx", "ball_x", PITCH[0]), ("by", "ball_y", PITCH[1])):
        a = np.full(n, np.nan, np.float32); a[fid] = df[col].values * s; out[k] = a
    out["label"] = labels_from_events(_events(game), n)
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    for gm in (1, 2, 3):
        d = load(gm); lab = d["label"]
        ch = np.flatnonzero(np.diff(lab) != 0) + 1
        segs = np.split(lab, ch); poss = [s for s in segs if s[0] in (0, 1)]
        print(gm, "frames", len(lab), {k: int((lab == k).sum()) for k in (-1, 0, 1, 2)},
              "poss", len(poss), "mean dur %.1fs" % np.mean([len(s) / 25 for s in poss]),
              "dead segs", sum(s[0] == 2 for s in segs),
              "ball NaN | alive %.2f dead %.2f" % (np.isnan(d["bx"][lab < 2][lab[lab < 2] >= 0]).mean(),
                                                     np.isnan(d["bx"][lab == 2]).mean()))
