"""Run the pressing pipeline end to end and write tables + figures to outputs/.

Usage:
    python scripts/run_pipeline.py                       # opendata sample in data/opendata
    python scripts/run_pipeline.py --data-dir data/challenge
    python scripts/run_pipeline.py --no-tracking         # event-only (fast)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from press import config, episodes, features, io, metrics, tracking, viz  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--out", default=str(config.OUTPUT_DIR))
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N matches")
    ap.add_argument("--min-player-engagements", type=int, default=30)
    args = ap.parse_args()
    data_dir, out = Path(args.data_dir), Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    all_eng, all_teams, all_players, snapshot = [], [], [], None
    for mid in io.match_ids(data_dir)[: args.limit]:
        match = io.load_match(data_dir, mid)
        eng = episodes.engagements(io.load_events(data_dir, mid))
        trk_file = data_dir / str(mid) / f"{mid}_tracking_extrapolated.jsonl"
        if not args.no_tracking and trk_file.exists():
            trk = tracking.load_tracking(data_dir, mid, config.CACHE_DIR)
            eng = features.add_spatial_features(eng, trk, match)
            if snapshot is None:
                snapshot = (trk, match)
        else:
            eng["space_delta"] = float("nan")
        print(f"{mid}: {len(eng)} engagements, space computed for {eng.space_delta.notna().mean():.0%}")
        all_eng.append(eng)
        all_teams.append(io.team_info(match))
        all_players.append(io.player_info(match))

    eng = pd.concat(all_eng, ignore_index=True)
    teams, players = pd.concat(all_teams), pd.concat(all_players)
    ch = episodes.chains(eng)

    team_tbl = metrics.team_table(eng, teams)
    player_tbl = metrics.player_table(eng, players, args.min_player_engagements)
    chain_tbl = metrics.team_chain_table(ch)
    team_tbl.to_csv(out / "team_pressing.csv", index=False)
    player_tbl.to_csv(out / "player_pressing.csv", index=False)
    chain_tbl.to_csv(out / "team_pressing_chains.csv", index=False)
    eng.to_parquet(out / "engagements.parquet", index=False)

    viz.team_quadrant(team_tbl, out / "figures" / "team_quadrant.png")
    if snapshot is not None:
        trk, match = snapshot
        fi = tracking.FrameIndex(trk, {p["id"]: p["team_id"] for p in match["players"]})
        cand = eng[(eng.match_id == match["id"]) & eng.event_subtype.eq("pressing")
                   & eng.space_delta.notna()].sort_values("space_delta")
        if len(cand):
            viz.press_snapshot(fi, cand.iloc[-1], match, config.SPACE_LAG_FRAMES,
                               out / "figures" / "press_snapshot_most_space_conceded.png")

    pd.set_option("display.width", 200)
    cols = ["team", "engagements_per30_otip", "success_rate", "regains_over_expected",
            "beaten_rate", "mean_space_delta", "press_net_score"]
    print("\nTeams:\n", team_tbl[cols].round(3).to_string(index=False))
    print("\nChains:\n", chain_tbl.round(3).to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
