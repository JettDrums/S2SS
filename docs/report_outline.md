# Report outline (working draft)

Working title: **Press Smarter, Not Harder: Separating Pressing Decisions from
Pressing Execution in the NWSL and WSL**

Budget: 3000 words max, no author names anywhere. Figures carry most of the
story; each section has one main figure.

Status key: [built] = implemented and runs on the opendata sample;
[full data] = needs the NWSL/FAWSL data to be conclusive.

## 1. Introduction (~300 words)
- Pressing is judged by outcomes (regains), which mixes good decisions with
  luck and ignores the space a press gives up.
- Contribution: one currency for reward and risk (net shot probability), a
  baseline for what would have happened anyway, and a split of every press into
  **decision quality** (was this a good moment to press?) and **execution**
  (did it go better than expected?).
- Headline question: which teams and players press *intelligently*, and is
  that a repeatable skill?

## 2. Data (~200 words)
- SkillCorner tracking (10 Hz) and dynamic events, NWSL 2025 and FAWSL 2024/25.
- Press = on-ball engagement of type pressing, counter-press, recovery press,
  pressure. Counts, matches, minutes out of possession.
- Do not reproduce raw data; report aggregates only.

## 3. Method (~800 words)
1. Possession value V from tracking: P(shot for in 10 s) - P(shot against). [built]
   Figure: value surface + calibration (appendix).
2. Drift baseline: expected change in V over 5 s from any state. Explain why
   it is needed (ordinary progression; regression to the mean). [built]
3. Press value PV = change in V from the pressing team's view minus drift.
   Validation: PV by independent SkillCorner outcome labels. [built]
   Figure: press_value_by_outcome.
4. xPV model from pre-press context -> decision quality; PV - xPV -> execution.
   Out-of-fold by match throughout. [built]
5. Reliability: ICC of match-level means with permutation tests; position
   adjustment for players; empirical-Bayes shrinkage for rankings. [built]

## 4. Results (~1200 words)
4.1 When is pressing a good decision? (decision_rules, decision_map) [built; confirm on full data]
   - Only pre-press information (no hindsight labels; see method).
   - Sample patterns to test: pressing within 3 s of a turnover is the
     costliest timing; value rises steadily with cover behind the press and
     falls steadily with teammates already near the ball (over-commitment);
     engaging from 9-11 m beats arriving late at 0-3 m.
   - Methods point worth a paragraph: SkillCorner's pressing/pressure split and
     phase labels use hindsight; including them doubles apparent decision R^2.
4.2 Teams: decision vs. execution quadrant + team profiles (style mix). [full data]
4.3 Is it a skill? Reliability table. Sample result: raw player decision
    quality looks highly repeatable (ICC 0.35) but almost all of it is
    position; position-adjusted ICC 0.06 (n.s.). Execution ICC ~0. [full data]
4.4 Player rankings within position, shrunk. [full data]
4.5 NWSL vs. WSL comparison of pressing decisions. [full data, to build]

## 5. Recommendations (~300 words)
- Team-specific: which situations each team over-/under-presses relative to
  the league decision rules.
- Player-specific: who to trust as a press trigger.

## 6. Limitations (~200 words)
- Nearest-player space model (no velocities); 5 s horizon choice; xLoss and
  phase labels are SkillCorner model outputs; observational, not causal;
  only pressed situations are observed (no "chose not to press" counterfactual yet).

## Figures shortlist
1. Press snapshot (concept) 2. Press value by outcome (validation)
3. Decision rules grid 4. Team decision vs. execution 5. Reliability table
6. Player ranking table (within position)
