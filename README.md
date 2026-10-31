# Pressing analysis: CSAS 2027 Data Challenge (NWSL / FAWSL)

Starter pipeline for evaluating how often, how well, and how riskily teams and
players press, using SkillCorner tracking (10 Hz) and dynamic event data.

It currently runs on SkillCorner's public **opendata** sample (10 A-League
2024/25 matches), which has the same format as the challenge data. When the
challenge credentials arrive, point it at the NWSL/FAWSL data with no code changes.

> **Data terms:** the challenge data must not be shared or published. `data/`,
> `cache/` and `outputs/` are git-ignored. Never commit data, extracts, or
> credentials.

## Quick start

```bash
pip install -r requirements.txt
python scripts/fetch_opendata.py           # ~1.8 GB into data/opendata (use --no-tracking for events only)
python scripts/run_pipeline.py             # tables + figures into outputs/
```

Challenge data (once credentials arrive):

```bash
export SKILLCORNER_USERNAME=...  SKILLCORNER_PASSWORD=...
python scripts/fetch_skillcorner_api.py --list-editions
python scripts/fetch_skillcorner_api.py --edition <NWSL_2025_ID> --edition <FAWSL_2024_25_ID> --limit 2   # test first
python scripts/run_pipeline.py --data-dir data/challenge
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
| `press/viz.py` | Team frequency-vs-quality quadrant and single-press snapshot figure. |
| `scripts/fetch_opendata.py` | Download the public opendata sample. |
| `scripts/fetch_skillcorner_api.py` | Download challenge data via the SkillCorner API (verify params on first run). |
| `scripts/run_pipeline.py` | Run everything end to end. |

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

## Next steps

1. Replace the equal-weight score with a learned value model (e.g. P(regain
   within 5 s) and EPV/xThreat change), so reward and risk share one unit.
2. Velocity-aware pitch control instead of nearest-player control.
3. Context adjustment (game state, opponent build-up quality, pitch zone).
4. Bootstrap confidence intervals on rankings; sensitivity to press definition.
5. Validate flagged presses against match video.
