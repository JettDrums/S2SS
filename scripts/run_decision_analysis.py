"""Decision analysis on the output of run_value_model.py.

Writes to outputs/value/decisions/:
  decision_rules.csv            mean xPV / PV by situational factor
  reliability.csv               ICC of decision vs execution at team and player level
  team_profiles.csv             team decision quality, execution, and style mix
  player_rankings.csv           shrunk (empirical-Bayes) player decision/execution/value
  figures/decision_rules.png, figures/decision_map.png

Usage: python scripts/run_decision_analysis.py [--value-dir outputs/value]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from press import config, decisions, viz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-dir", default=str(config.OUTPUT_DIR / "value"))
    ap.add_argument("--pitch", type=float, nargs=2, default=(105.0, 68.0), metavar=("LENGTH", "WIDTH"))
    ap.add_argument("--min-player-presses", type=int, default=30)
    args = ap.parse_args()
    vdir = Path(args.value_dir)
    out = vdir / "decisions"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    p = decisions.valued_presses(pd.read_parquet(vdir / "engagements_valued.parquet"))
    rules = decisions.decision_rules(p)
    rel = decisions.reliability_table(p)
    prof = decisions.team_profiles(p)
    players = decisions.player_rankings(p, args.min_player_presses)
    rules.to_csv(out / "decision_rules.csv", index=False)
    rel.to_csv(out / "reliability.csv", index=False)
    prof.to_csv(out / "team_profiles.csv", index=False)
    players.to_csv(out / "player_rankings.csv", index=False)
    viz.decision_rules_grid(rules, out / "figures" / "decision_rules.png")
    viz.decision_map(p, *args.pitch, out / "figures" / "decision_map.png")

    pd.set_option("display.width", 220)
    print("Reliability (ICC of match-level means; perm_p = permutation p-value):")
    print(rel.round(3).to_string(index=False))
    print("\nDecision rules (x100):")
    r = rules.copy()
    r[["mean_xpv", "mean_pv"]] *= 100
    print(r.round(2).to_string(index=False))
    print("\nTeam profiles:")
    print(prof.round(4).to_string(index=False))
    print("\nTop 10 players (shrunk press value):")
    print(players.head(10).round(4).to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
