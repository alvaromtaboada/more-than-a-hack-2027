# Ball possession from automatic tracking — technical document

## 1. Approach

A frame classifier trained on **other matches** and decoded by a 3-state HMM.

1. **Features (per frame, permutation-invariant).** Raw ATD → metres. Track slots are not people,
   so every feature is a team aggregate or a nearest-to-ball statistic, with nothing per player:
   - *team shape* per side, in its own attacking direction (the side is inferred per half from the
     team centroid): centroid, spread, deepest and highest players, width, mean speed, forward
     velocity, and differences between the two teams;
   - *game tempo*: mean player speed, 90th percentile speed, share of slow players, speed trend;
   - *ball* (when seen): ball speed, distance outside the pitch, nearest and second-nearest
     player of each team, players within 5 m, a "control" flag (nearest player < 1.5 m, slow
     ball), and the team nearer the ball at the last and next sighting with the time gap;
   - rolling means over ±0.25 s … ±8 s.
   Ball gaps of up to 0.4 s are linearly interpolated.
2. **Two gradient-boosting heads** (scikit-learn `HistGradientBoostingClassifier`):
   - *dead-ball head*: P(D). It uses **player features only**. In the training games the ball
     disappears whenever it is dead, but in BAR-ATH it often stays visible (carried back for a
     throw-in, placed for a free kick), so ball visibility is a spurious cue for D;
   - *team head*: P(H | alive), trained on alive frames only.
   Both are made symmetric by training on team-swapped copies and averaging the prediction with
   the swapped view, so there is no home/away bias.
3. **Decoding.** Viterbi over {H, A, D} with emissions `alpha·log P` (alpha = 0.5), switch probability
   `p_switch` = 0.01 (H↔A) and `p_dead` = 0.0015 (alive↔dead), run separately inside each graded half. The
   segment structure is what `spellF1` scores, so frame-level argmax is never submitted. Argmax
   produces about 3× too many possessions.

## 2. Training data (none from the provided game)

| source | games | labels | who labelled |
|---|---|---|---|
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) games 1–3 | 3 | derived by code from the human event data: D from BALL OUT / goal / offside / FAULT RECEIVED to the next SET PIECE or goal kick; otherwise the team of the latest on-ball event (PASS, CARRY, RECOVERY, SHOT, BALL LOST, SET PIECE, BALL OUT) | Metrica analysts (events) |
| [IDSSE, Bassek et al. 2025, *Sci. Data*](https://doi.org/10.6084/m9.figshare.28196177) (CC-BY 4.0) | 7 Bundesliga | per frame from the DFL position feed: `BallStatus=0` → D, else `BallPossession` 1/2 → H/A. This is the same derivation as the challenge GT (ownership as named by corrected tracking) | DFL / Sportec (official data) |

About 1.4 M labelled frames and roughly 4,100 possessions. The provided game contributes no labels.

**Domain adaptation, all automatic.** The training data is clean, while BAR-ATH's ATD has the ball
missing on 41% of graded frames and about 1.5 players missing per frame. During training:
- each training game's ball track is masked with BAR-ATH's **real ball-missing pattern** (a
  circular slice of the graded frames' NaN mask; it carries no labels);
- players are dropped in 2 s chunks with probability 7%.

## 3. Assets used and dropped

- **Used:** tactical ATD (players and ball) and the ATD metadata (team of each slot, home = first team).
- **Dropped:**
  - *Smart Tagging*: set-piece windows are only accurate to ±5 s, and with no labels on this
    game there was no fair way to calibrate them. The dead-ball head already finds the restarts
    they would mark.
  - *Homography and video*: no GPU, and running a ball detector over 199k frames on CPU did not fit
    the time available.
  - *Broadcast*: not synchronised to the tactical timeline.
- **Tried and rejected:** a second-stage temporal stacking model. It looked better in CV, but
  the gain came from ball visibility, a cue that does not transfer to BAR-ATH (see §1.2). Once
  that cue was removed it was worse than stage 1 alone (0.498 vs 0.518).
- **Also tried and rejected** (judged on held-out Metrica games with their original labels):
  - *harmonising the IDSSE labels to controlled possession*, where a team spell shorter than 1 s is
    given back to the previous owner (`MTAH_HARM=25`): 0.5600 vs 0.5597;
  - *a minimum-duration Viterbi* with k = 6–38 frames (`viterbi_md`): 0.553–0.560.
  Neither beats the noise band, so the submitted configuration keeps both off.

## 4. Possession rules implemented

- **Dead ball** is a state, never a possession. The model learns it from the training labels
  (ball out, foul, goal, until the restart). The HMM makes short D flickers costly, so a D
  segment needs sustained evidence: tempo drop, players regrouping, ball outside the pitch.
- **Loose balls and challenges:** possession stays with the last team until the evidence for
  the other side outweighs the switch cost. This matches "a touch without control is not a
  change".
- **Kick-offs:** before the kickoff of a half the ball is not in play (Law 8). The second graded
  block opens about two minutes before its kickoff, while players walk back on, and the frame model
  can read that as play. The kickoff is found in the tracking alone (`possession/kickoff.py`): the
  first frame where each team stands in its own half for a second, the ball is within 5 m of the
  centre spot and starts to move. Every frame of the block before it is set to D. No frame number
  is hard-coded; in the first half the block opens at the kickoff and nothing changes.
- **Restarts:** the team head decides who restarts from the formation and the ball. No team is
  assigned to D, as the challenge requires.
- No `X` on graded frames. Ungraded frames (pre-match, half-time, post-match) are `X`.

## 5. Model selection

Every hyperparameter (`alpha`, `p_switch`, `p_dead`, model size) was chosen by **grouped
cross-validation on the 10 training games** using the official `spell_scores` from
`evaluation/possession_metrics.py`. The held-out games are masked exactly as in training. The
public 5 minutes were used only to check that frames were aligned and that labels meant what
we thought.

| version | held-out Metrica games, spellF1 | held-out IDSSE games, spellF1 |
|---|---|---|
| stage 1, trained on 2 Metrica games | 0.518 | — |
| + 7 IDSSE games | 0.551 | 0.489 |
| + larger boosting (600 iter, 63 leaves) — **submitted** | **0.560** | **0.505** |

On the public 5 minutes (check only): macroF1 0.91, spellF1 0.69; the published baseline on that
window scores 0.33.

## 6. Limitations

- Short possessions (< 2 s) are rarely recovered: recall at IoU 0.5 is about 5%. A transition
  needs several frames of evidence, and the ball is often missing exactly in duels.
- When the ball is missing, team identity depends on formation cues. Long ball-less stretches
  inside one possession can still flip.
- Semantics differ across label sources (Metrica events vs DFL feed vs Metrica corrected tracking).
- **With more time:** a ball detector on the tactical video to fill the 41% ball gaps; an
  explicit touch / reception detector to time transitions; calibrating the set-piece windows from
  Smart Tagging on other games.

## 7. Reproduction

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-possession.txt   # Windows
# external training data, next to the repository folder (or set MTAH_SAMPLE_DIR / MTAH_IDSSE_DIR):
git clone https://github.com/metrica-sports/sample-data ../extdata/sample-data
#   IDSSE: "Download all" at https://doi.org/10.6084/m9.figshare.28196177, unzip into ../extdata/idsse
# match data in data/ as described in DATA.md, then:
python -m possession.run --out submission/BecaData.csv
python evaluation/validate_submission.py submission/BecaData.csv
```

The submitted file was regenerated byte-for-byte from an empty `cache/` with Python 3.13.3 on
Windows 11 Home (10.0.22631), CPU only; `possession.run` from scratch took about 22 minutes.
Cross-validation: `python -m possession.cv3 all`. CPU only. A full run takes about 10 minutes plus
a one-off 2 minutes to parse the data. `cache/` holds parsed arrays and can be deleted at any time.
