# Pressing analysis: CSAS 2027 Data Challenge (NWSL / FAWSL)

Starter pipeline for evaluating how often, how well, and how riskily teams and
players press, using SkillCorner tracking (10 Hz) and dynamic event data.

It currently runs on SkillCorner's public **opendata** sample (20 A-League
2024/25 matches), which has the same format as the challenge data. When the
challenge credentials arrive, point it at the NWSL/FAWSL data with no code changes.

> **Data terms:** the challenge data must not be shared or published. `data/`,
> `cache/` and `outputs/` are git-ignored. Never commit data, extracts, or
> credentials.

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_opendata.py           # ~1.8 GB into data/opendata (use --no-tracking for events only)
python scripts/run_pipeline.py             # v0 descriptive tables + figures into outputs/
python scripts/run_value_model.py          # learned press value model into outputs/value/
```

Challenge data (once credentials arrive):

```bash
export SKILLCORNER_USERNAME=...  SKILLCORNER_PASSWORD=...
python scripts/fetch_skillcorner_api.py --list-editions
python scripts/fetch_skillcorner_api.py --edition <NWSL_2025_ID> --edition <FAWSL_2024_25_ID> --limit 2   # test first
python scripts/run_pipeline.py --data-dir data/challenge
python scripts/run_value_model.py --data-dir data/challenge
```

## What each file does

| File | Purpose |
|---|---|
| `press/config.py` | Paths (overridable with `PRESS_DATA_DIR`, `PRESS_CACHE_DIR`, `PRESS_OUTPUT_DIR`) and constants. |
| `press/io.py` | Load match metadata and dynamic events; team/player minutes out of possession. |
| `press/tracking.py` | Parse tracking jsonl into a long table (cached as parquet) and fast per-frame lookup. |
| `press/episodes.py` | On-ball engagements with outcome flags (regain, disruption, forced backward, danger reduced, beaten) and pressing-chain table. |
| `press/space.py` | Nearest-player space control on a 1 m grid, weighted toward the pressing team's goal; press support/cover counts. |
| `press/features.py` | Attaches space before/after (2 s) each engagement and support features. |
| `press/metrics.py` | Team, player, and pressing-chain tables; baseline press net score. |
| `press/value.py` | Possession value model V (shot for minus shot against in 10 s) and drift model, out-of-fold by match. |
| `press/press_value.py` | Press value per engagement, expected press value (xPV) model, team/player value tables. |
| `press/viz.py` | All figures (quadrant, press snapshot, value surface, calibration, decision vs. execution, xPV drivers, value by outcome). |
| `scripts/fetch_opendata.py` | Download the public opendata sample. |
| `scripts/fetch_skillcorner_api.py` | Download challenge data via the SkillCorner API (verify params on first run). |
| `scripts/run_pipeline.py` | Run the v0 descriptive pipeline end to end. |
| `scripts/run_value_model.py` | Train the value models and value every press. |

## Method (v0 baseline)

- **Press event:** a SkillCorner on-ball engagement of subtype `pressing`,
  `counter_press`, `recovery_press` or `pressure`. Event coordinates are flipped
  so the engaging (pressing) team attacks +x; tracking is transformed to match.
- **Reward:** success rate (regain, disruption, forced backward, or danger
  reduced) and regains over expected (regain minus SkillCorner xLoss at start).
- **Risk:** beaten rate and **space conceded**: change in the share of
  goal-weighted pitch value controlled by the team in possession, from
  engagement start to 2 s after it ends. Sanity check on the sample: when the
  presser is beaten, space conceded is about 4.5x larger than otherwise.
- **Coordination:** pressing chains (multi-player sequences): players involved,
  chain regain/disruption rate; defenders within 10 m and cover behind the press.
- **Press net score:** equal-weight z-score of reward minus risk. This is a
  placeholder for a learned press value model.

## Learned press value model (`run_value_model.py`)

Puts reward (regains, disruption) and risk (space and progression conceded) in
one unit: **net probability of a shot in the next 10 s**.

1. **Possession value V(state).** Two gradient-boosted classifiers on tracking
   features (ball location, goal angle, nearest defender, defenders goal-side,
   attackers ahead, gap to the defensive line, goal-weighted space control)
   predict P(team in possession shoots within 10 s) and P(opponent does).
   V = difference. States sampled every 1 s with known possession.
2. **Drift model.** How V normally changes over 5 s from a given state, pressed
   or not. Without this, presses are blamed for ordinary forward progress and
   credited for ordinary regression to the mean.
3. **Press value (PV).** For each press: change in V from the pressing team's
   view over 5 s from engagement start (own V if they regained the ball,
   minus opponent's V otherwise), minus the expected drift.
4. **Expected press value (xPV).** Gradient-boosted regressor predicting PV from
   the situation before the press (location, distance to carrier, support,
   cover, space, phase of play, press type, game state). Splits pressing into
   **decision quality** (mean xPV: pressing at good moments) and **execution**
   (mean PV - xPV: doing better than expected once pressing).

Every prediction is out-of-fold by match (GroupKFold), so no match is scored by
a model trained on it. Team values come with bootstrap 95% CIs.

**Validation on the sample (20 matches, ~13.6k presses):**
V model AUC 0.85 (shot for) / 0.88 (shot against), Brier better than base rate;
drift model R^2 0.24; xPV R^2 0.11 (most press outcome is execution and luck,
as expected). Mean PV by SkillCorner outcome label: forced backward +2.9,
regain +1.2, beaten -4.9 (x100), so the learned value agrees with independent
labels it was not trained on.

## Next steps

1. Velocity-aware pitch control instead of nearest-player control.
2. Add player velocities and passing-lane coverage to V and xPV.
3. Opponent adjustment (pressing strong vs. weak build-up teams).
4. Sensitivity to horizon (3/5/8 s) and press definition; player-level shrinkage.
5. Validate flagged presses against match video.
