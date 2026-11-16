"""Nested leave-one-game-out CV of the two-stage model."""
import warnings; warnings.filterwarnings("ignore")
import itertools, pickle, os
import numpy as np
from .model import game_xy, fit, predict, viterbi, score, target_masks, stage2_feats, fit2, predict2

rng = np.random.default_rng(0)
bm, _ = target_masks()
data = {g: game_xy(g, rng, bm) for g in (1, 2, 3)}
single = {g: fit([data[g][0]], [data[g][1]]) for g in (1, 2, 3)}       # stage-1 on one game
pairs = {g: fit([data[h][0] for h in (1, 2, 3) if h != g], [data[h][1] for h in (1, 2, 3) if h != g])
         for g in (1, 2, 3)}                                               # stage-1 on two games
out = {}
for g in (1, 2, 3):
    tr = [h for h in (1, 2, 3) if h != g]
    # stage-1 OOF on each training game from the other training game
    Fs = [stage2_feats(predict(single[tr[1 - i]], data[h][0]), data[h][0]) for i, h in enumerate(tr)]
    m2 = fit2(Fs, [data[h][1] for h in tr])
    P1 = predict(pairs[g], data[g][0])
    out[g] = predict2(m2, stage2_feats(P1, data[g][0]))
    print(g, "stage2 argmax", score(data[g][1], np.argmax(out[g], 1)))
pickle.dump((out, {g: data[g][1] for g in data}), open(os.path.join("cache", "cv2_probs.pkl"), "wb"))
for a, ps, pdd in itertools.product((0.1, 0.2, 0.4, 0.7, 1.0), (0.004, 0.01), (0.0015,)):
    r = [score(data[g][1], viterbi(out[g], a, ps, pdd)) for g in (1, 2, 3)]
    print(f"a={a} ps={ps} pd={pdd} spell={np.mean([x['spell'] for x in r]):.4f} "
          f"macro={np.mean([x['macro'] for x in r]):.4f} n={[x['n_pred'] for x in r]}")
