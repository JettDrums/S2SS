# Press Smarter, Not Harder: Separating Pressing Decisions from Pressing Execution in the NWSL

<!-- Draft. Word budget 3000 (excluding tables, captions, references).
     No author names anywhere. Numbers marked [R] are filled from outputs/nwsl/
     after the final run; nothing here reproduces raw SkillCorner data. -->

## 1. Introduction

Pressing is usually judged by what happens next: did the team win the ball?
That standard has two problems. It ignores the cost of pressing, since a
defender who steps out to engage leaves space behind, and it mixes skill with
luck, since the same decision to press can end in a regain or a through ball
depending on a deflection. A team that presses at the wrong moments and gets
lucky looks better than a team that presses at the right moments and does not.

We separate those two things. Using SkillCorner broadcast tracking and dynamic
event data for the 2025 NWSL season, we value every press in a single
currency, the change it causes in the probability of a shot, net of how the
possession would normally have evolved from the same situation. We then split
that value into two parts:

- **Decision quality:** the value a press can be expected to add given only
  what was known when it started (where the ball was, how far away the
  presser was, how many teammates were nearby, how much cover was behind,
  how long ago possession changed hands).
- **Execution:** how much better or worse the press went than that expectation.

This split answers the challenge's central question directly. Intelligent
pressing is good decision-making, and we can test whether it is a repeatable
skill of teams and players or just noise.

## 2. Data

SkillCorner provided tracking data (10 Hz, all 22 players and the ball) and
dynamic events for the 2025 NWSL season. Of 189 matches, [R: 161] had both
tracking and processed dynamic events; the rest were unavailable from the
provider. A **press** is an on-ball engagement, SkillCorner's model-detected
event in which a defender actively engages the ball carrier, of type
pressing, pressure, counter-press or recovery press ([R: 98,675] presses).
We exclude the "other" type, which covers jockeying and contested duels
where the defender is not actively pressing.

## 3. Method

### 3.1 Possession value

We first need to know how dangerous any moment is. For every second of play
with a known team in possession ([R: ~530,000] states) we compute features
from tracking: ball location, distance and angle to goal, nearest defender,
defenders goal-side of the ball, attackers ahead of the ball, gap to the
defensive line, and the share of goal-weighted pitch area the attacking team
controls under a nearest-player model. Two gradient-boosted classifiers
predict whether the team in possession, and whether its opponent, takes a
shot in the next 10 seconds. Possession value V is the difference: a net
shot probability between -1 and 1.

All models are evaluated out of fold by match: each match is scored by a
model that never saw it. V predicts shots well (AUC [R: 0.87] for the team in
possession, [R: 0.89] for the opponent) and is well calibrated (Appendix).

### 3.2 Press value

For a press starting at time t, from the pressing team's point of view:

- before: minus the opponent's V at t;
- after (t + 5 s): the pressing team's own V if it has the ball, otherwise
  minus the opponent's V, plus one for a shot the pressing team takes inside
  the window and minus one for a shot it concedes.

Counting shots inside the window matters. Without it, an attack that ends in
a shot looks like a collapse in value, because after a shot the ball is dead
or with the keeper, so a press that failed to stop a shot would be credited.

Raw change in value is still misleading, because possessions drift on their
own: attacks progress, and dangerous moments regress toward average whether
or not anyone presses. We therefore fit a **drift model**: the expected
change over 5 seconds from any state, using the same definition, trained on
every state in the season. **Press value (PV)** is the observed change minus
the expected drift. Positive PV means the possession went worse for the
attackers than it normally would from the same situation.

As a check, PV agrees with SkillCorner's outcome labels, which it never
sees: presses that force a backward pass or win the ball score above average,
and presses where the defender is beaten score far below ([R: table]).

### 3.3 Decision quality and execution

**Expected press value (xPV)** is a gradient-boosted regression of PV on
information available when the press starts: location; distance to the
carrier; teammates within 5 and 10 m; cover goal-side within 20 m; space
controlled; the carrier's probability of losing the ball (SkillCorner xLoss);
team shape (block height, depth of the last line, team length); seconds since
the turnover; whether a pressing chain is already running; press kind; and
game state. Decision quality is xPV; execution is PV - xPV.

We deliberately exclude two SkillCorner labels that use hindsight. Per the
dynamic-events specification, an engagement is labelled "pressing" rather than
"pressure" only if the *next* possession is also pressed, and phase of play
is assigned at the end of the possession (a "chaotic" phase is one that
turned out short and contested). Including them roughly doubled the apparent
explanatory power of xPV on the development data, which is exactly the
leakage a decision metric must avoid.

### 3.4 Is it a skill?

A ranking is only useful if it measures something repeatable. For each team
and player we compute match-level averages of decision quality and execution
and estimate the intraclass correlation (ICC): the share of match-to-match
variation explained by who the team or player is. Significance comes from
permutation tests that shuffle identities across matches. Because
centre-backs and forwards press in very different situations, player decision
quality is also measured relative to others in the same position. Player
rankings use empirical-Bayes shrinkage so that small samples are pulled
toward the league average.

### 3.5 From metrics to recommendations

For each team we decompose the gap between its decision quality and the
league's into a **mix** term (how often it presses in situations that are
good or bad on average) and a **within** term (how good its presses are in
each situation). The situations a team over-uses that cost it the most are
its recommendations.

## 4. Results

<!-- Filled after the final NWSL run:
4.1 When is pressing a good decision? (decision rules figure)
4.2 Is it a skill? (reliability table)
4.3 Teams: decision vs. execution (quadrant) + decomposition
4.4 Players within position
4.5 Robustness: 3/5/8 s horizons -->

## 5. Recommendations

## 6. Limitations

- Observational: we see only presses that happened, so xPV compares presses
  with other presses in similar situations, not with choosing not to press.
- The average press scores below zero because defenders engage at dangerous
  moments; comparisons are made against the league average.
- Space control uses a nearest-player model without velocities.
- xLoss is itself a SkillCorner model output.
- [R: 28] matches were unavailable; FAWSL data was not yet provided.

## Appendix

- A1. Possession value surface and calibration.
- A2. Press value by SkillCorner outcome label.
- A3. Horizon robustness (3/5/8 s rank correlations).
