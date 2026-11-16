"""End-to-end: train on other games (Metrica sample + IDSSE), label BAR-ATH, write the submission.

    python -m possession.run --out evaluation/results/<team>.csv
"""
import argparse, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, half_video_blocks, source_frames  # noqa: E402
from . import atd  # noqa: E402
from .features import frame_features  # noqa: E402
from .idsse import MATCHES  # noqa: E402
from .kickoff import dead_before_kickoff  # noqa: E402
from .model import game_xy, fit, predict, viterbi, target_masks  # noqa: E402

TRAIN = [1, 2, 3] + list(MATCHES)
ALPHA, P_SWITCH, P_DEAD = 0.5, 0.01, 0.0015   # set by grouped CV on the training games


def game_period(n):
    per = np.zeros(n, np.int8)
    for h, a, b in half_video_blocks("BAR-ATH"):
        per[a:b + 1] = h
    return per


def main(out):
    rng = np.random.default_rng(0)
    bm, _ = target_masks()
    tr = [game_xy(g, rng, bm) for g in TRAIN]
    M = fit([x for x, _ in tr], [y for _, y in tr])
    d = atd.load()
    n = source_frames("tactical")
    P = predict(M, frame_features(d, game_period(n)))
    np.save(os.path.join(ROOT, "cache", "bar_ath_probs.npy"), P)
    labels = np.full(n, "X", dtype="<U1")
    for _, a, b in half_video_blocks("BAR-ATH"):
        labels[a:b + 1] = np.array(["H", "A", "D"])[viterbi(P[a:b + 1], ALPHA, P_SWITCH, P_DEAD)]
    # Law 8: nothing is in play before a half's kickoff, which the second block opens ~2 min ahead of.
    labels = dead_before_kickoff(labels, d, half_video_blocks("BAR-ATH"), GAMES["BAR-ATH"]["pitch"])
    with open(out, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{l}\n" for i, l in enumerate(labels))
    print("wrote", out, {k: int((labels == k).sum()) for k in "HADX"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "submission", "BecaData.csv"))
    main(ap.parse_args().out)
