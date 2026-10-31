"""Train the possession value model, value every press, and fit expected press value.

Two passes so memory stays flat on a full season:
  1. per match: build value-model training states (cached in cache/)
  2. fit V out-of-fold by match; per match: press value for each engagement
Then fit xPV (decision quality) and write tables + figures to outputs/value/.

Usage:
    python scripts/run_value_model.py                         # opendata sample
    python scripts/run_value_model.py --data-dir data/challenge
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from press import config, episodes, features, io, press_value, tracking, value, viz  # noqa: E402


def load_context(data_dir, mid):
    match = io.load_match(data_dir, mid)
    events = io.load_events(data_dir, mid)
    trk = tracking.load_tracking(data_dir, mid, config.CACHE_DIR)
    frames = tracking.load_frames(data_dir, mid, config.CACHE_DIR)
    return match, events, trk, value.MatchContext(match, events, trk, frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "value"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-player-presses", type=int, default=30)
    args = ap.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    mids = [m for m in io.match_ids(data_dir)[: args.limit]
            if (data_dir / str(m) / f"{m}_tracking_extrapolated.jsonl").exists()]

    print("Pass 1: value-model training states")
    states = []
    for mid in mids:
        cache = config.CACHE_DIR / f"{mid}_states.parquet"
        if cache.exists():
            s = pd.read_parquet(cache)
        else:
            match, events, _, ctx = load_context(data_dir, mid)
            s = value.build_states(ctx, events)
            s.to_parquet(cache, index=False)
        print(f"  {mid}: {len(s)} states")
        states.append(s)
    states = pd.concat(states, ignore_index=True)

    print("Fitting possession value model (out-of-fold by match)")
    vm = value.ValueModel()
    states = vm.fit(states)
    vm.fit_drift(states)

    print("Pass 2: press value per engagement")
    all_eng, all_teams, all_players, pitch = [], [], [], None
    for mid in mids:
        match, events, trk, ctx = load_context(data_dir, mid)
        eng = features.add_spatial_features(episodes.engagements(events), trk, match)
        eng = press_value.press_values(ctx, eng, vm)
        print(f"  {mid}: press value for {eng.press_value.notna().mean():.0%} of engagements")
        all_eng.append(eng)
        all_teams.append(io.team_info(match))
        all_players.append(io.player_info(match))
        pitch = pitch or (match["pitch_length"], match["pitch_width"])
    eng = pd.concat(all_eng, ignore_index=True)
    teams, players = pd.concat(all_teams), pd.concat(all_players)

    print("Fitting expected press value (decision quality)")
    eng, imp = press_value.fit_expected(eng)

    team_tbl = press_value.team_value_table(eng, teams)
    player_tbl = press_value.player_value_table(eng, players, args.min_player_presses)
    team_tbl.to_csv(out / "team_press_value.csv", index=False)
    player_tbl.to_csv(out / "player_press_value.csv", index=False)
    imp.to_csv(out / "xpv_feature_importance.csv", index=False)
    eng.to_parquet(out / "engagements_valued.parquet", index=False)

    viz.value_surface(states, *pitch, out / "figures" / "value_surface.png")
    viz.calibration(states, out / "figures" / "value_calibration.png")
    viz.decision_execution(team_tbl, out / "figures" / "team_decision_vs_execution.png")
    viz.importance(imp, out / "figures" / "xpv_importance.png")
    viz.value_by_outcome(eng[eng.event_subtype.isin(press_value.PRESS_SUBTYPES)],
                         out / "figures" / "press_value_by_outcome.png")

    pd.set_option("display.width", 220)
    cols = ["team", "presses", "pv_per30_otip", "pv_per_press", "pv_per_press_ci_low",
            "pv_per_press_ci_high", "decision_xpv", "execution", "positive_xpv_share"]
    print("\nTeams (press value in net 10 s shot probability):\n", team_tbl[cols].round(4).to_string(index=False))
    print("\nTop players by value per press:\n",
          player_tbl[["player", "position_group", "presses", "pv_per_press", "decision_xpv", "execution"]]
          .head(10).round(4).to_string(index=False))
    print("\nxPV drivers:\n", imp.head(8).round(4).to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
