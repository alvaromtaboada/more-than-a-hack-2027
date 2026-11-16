"""Grouped CV over all 10 training games (3 Metrica sample + 7 IDSSE)."""
import warnings; warnings.filterwarnings("ignore")
import itertools, pickle, os, sys
import numpy as np
from .model import game_xy, fit, predict, viterbi, score, target_masks
from .idsse import MATCHES

GAMES = [1, 2, 3] + list(MATCHES)
FOLDS = [[1, "J03WMX"], [2, "J03WN1"], [3, "J03WOH"], ["J03WOY", "J03WPY"], ["J03WQQ", "J03WR9"]]
only = sys.argv[1] if len(sys.argv) > 1 else "all"   # "all" or "metrica" (train only on 1,2,3)
rng = np.random.default_rng(0)
bm, _ = target_masks()
data = {g: game_xy(g, rng, bm) for g in GAMES}
probs = {}
for fold in FOLDS:
    tr = [h for h in GAMES if h not in fold and (only == "all" or isinstance(h, int))]
    M = fit([data[h][0] for h in tr], [data[h][1] for h in tr])
    for g in fold:
        probs[g] = predict(M, data[g][0])
pickle.dump((probs, {g: data[g][1] for g in GAMES}), open(os.path.join("cache", f"cv3_{only}_{os.environ.get('MTAH_ITER',300)}_{os.environ.get('MTAH_LEAVES',31)}_h{os.environ.get('MTAH_HARM',0)}.pkl"), "wb"))
for a, ps, pdd in itertools.product((0.3, 0.5, 0.8), (0.004, 0.01), (0.0015,)):
    r = {g: score(data[g][1], viterbi(probs[g], a, ps, pdd)) for g in GAMES}
    met = np.mean([r[g]["spell"] for g in (1, 2, 3)]); dfl = np.mean([r[g]["spell"] for g in MATCHES])
    print(f"a={a} ps={ps} spell metrica={met:.4f} idsse={dfl:.4f} macro={np.mean([x['macro'] for x in r.values()]):.4f}")
