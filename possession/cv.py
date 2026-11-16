"""Leave-one-game-out CV on the three Metrica sample games."""
import warnings; warnings.filterwarnings("ignore")
import itertools, pickle, os
import numpy as np
from .model import game_xy, fit, predict, viterbi, score, target_masks

rng = np.random.default_rng(0)
bm, _ = target_masks()
data = {g: game_xy(g, rng, bm) for g in (1, 2, 3)}
probs = {}
for g in (1, 2, 3):
    tr = [h for h in (1, 2, 3) if h != g]
    M = fit([data[h][0] for h in tr], [data[h][1] for h in tr])
    probs[g] = predict(M, data[g][0])
    y = data[g][1]; a = y[(y >= 0) & (y < 2)]
    print(g, "argmax", score(y, np.argmax(probs[g], 1)))
pickle.dump((probs, {g: data[g][1] for g in data}), open(os.path.join("cache", "cv_probs.pkl"), "wb"))
for alpha, ps, pdd in itertools.product((0.1, 0.2, 0.4), (0.002, 0.004), (0.0005, 0.0015)):
    r = [score(data[g][1], viterbi(probs[g], alpha, ps, pdd)) for g in (1, 2, 3)]
    print(f"a={alpha} ps={ps} pd={pdd}  spell={np.mean([x['spell'] for x in r]):.4f} "
          f"macro={np.mean([x['macro'] for x in r]):.4f}  npred/ngt={[(x['n_pred'], x['n_gt']) for x in r]}")
