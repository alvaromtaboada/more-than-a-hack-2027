"""Before each half's kickoff the ball is not in play (Laws of the Game, law 8).

The second graded block opens ~2 minutes before the second-half kickoff, and the frame model can
read players walking back onto the pitch as play. The kickoff is found in the tracking alone: the
first frame where both teams stand in their own halves, the ball is at the centre spot and it
starts to move. Every frame of the block before it is dead ball. No frame number is hard-coded.
"""
import numpy as np

FPS = 25
CENTRE_RADIUS = 5.0     # m; the tracked ball sits up to ~3.3 m from the spot at a kickoff here
MOVING = 2.5            # m/s; the kick
OWN_HALF = 0.7          # share of each team's tracked players in its own half (noisy near the line)
SEARCH = 5 * 60 * FPS   # only look this far into each graded block
MIN_LEAD = 2 * FPS      # a kickoff within 2 s of the block start is the block start itself


def _speed(x, y, k=2):
    v = np.full(len(x), np.nan)
    v[k:-k] = np.hypot(x[2 * k:] - x[:-2 * k], y[2 * k:] - y[:-2 * k]) / (2 * k / FPS)
    return v


def find_kickoff(d, first, last, pitch):
    """Frame of the kickoff inside [first, last], or None."""
    stop = min(last + 1, first + SEARCH)
    length, width = pitch
    bx, by = d["bx"][first:stop], d["by"][first:stop]
    px, team = d["px"][first:stop], d["team"]
    with np.errstate(invalid="ignore"):
        at_spot = np.hypot(bx - length / 2, by - width / 2) < CENTRE_RADIUS
        moving = _speed(bx, by) > MOVING
        left = px < length / 2
        tracked = ~np.isnan(px)
        share = {t: np.where(tracked[:, team == t].sum(1) > 0,
                             (left & tracked)[:, team == t].sum(1)
                             / np.maximum(tracked[:, team == t].sum(1), 1), np.nan)
                 for t in (0, 1)}
    # one team entirely left of halfway and the other entirely right, in either orientation
    split = (((share[0] >= OWN_HALF) & (share[1] <= 1 - OWN_HALF))
             | ((share[1] >= OWN_HALF) & (share[0] <= 1 - OWN_HALF)))
    # the formation must hold for the second before the kick
    held = np.convolve(split.astype(float), np.ones(FPS), "full")[:len(split)] >= FPS
    held = np.r_[False, held[:-1]]
    hits = np.flatnonzero(at_spot & moving & held)
    return first + int(hits[0]) if len(hits) else None


def dead_before_kickoff(labels, d, blocks, pitch):
    """labels (str array) with every frame of a graded block before its kickoff set to D."""
    labels = labels.copy()
    for _, first, last in blocks:
        k = find_kickoff(d, first, last, pitch)
        if k is not None and k - first > MIN_LEAD:
            labels[first:k] = "D"
    return labels
