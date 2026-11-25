"""Pick presses for video review and draw a snapshot of each.

Groups (from the output of run_value_model.py):
  best_decisions   highest xPV: the model says pressing here was a good idea
  worst_decisions  lowest xPV: the model says pressing here was a bad idea
  danger_reduced   presses SkillCorner flags as reducing danger that the model
                   values very negatively (a known disagreement to explain)
At most one press per match per group, so the list covers many games.

Writes outputs/<run>/review/review_list.csv (match, date, teams, period, clock
time, players, model numbers) and one PNG per press. Find each press on video
by match, period and clock time.

Usage: python scripts/review_presses.py [--value-dir outputs/nwsl] [--data-dir data/challenge] [--n 8]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from press import config, io, tracking, viz  # noqa: E402
from press.metrics import PRESS_SUBTYPES  # noqa: E402


def pick(df, n, ascending):
    d = df.sort_values("xPV" if ascending is not None else "press_value", ascending=bool(ascending))
    return d.drop_duplicates("match_id").head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value-dir", default=str(config.OUTPUT_DIR / "nwsl"))
    ap.add_argument("--data-dir", default="data/challenge")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    vdir, data_dir = Path(args.value_dir), Path(args.data_dir)
    out = vdir / "review"
    out.mkdir(parents=True, exist_ok=True)

    e = pd.read_parquet(vdir / "engagements_valued.parquet")
    p = e[e.event_subtype.isin(PRESS_SUBTYPES) & e.xPV.notna()]
    groups = {
        "best_decisions": p.sort_values("xPV", ascending=False).drop_duplicates("match_id").head(args.n),
        "worst_decisions": p.sort_values("xPV").drop_duplicates("match_id").head(args.n),
        "danger_reduced": p[p.danger_reduced].sort_values("press_value").drop_duplicates("match_id").head(args.n),
    }

    rows = []
    for group, df in groups.items():
        for mid, g in df.groupby("match_id"):
            match = io.load_match(data_dir, mid)
            fi = tracking.FrameIndex(tracking.load_tracking(data_dir, mid, config.CACHE_DIR),
                                     {pl["id"]: pl["team_id"] for pl in match["players"]})
            for r in g.itertuples():
                clock = f"{int(r.minute_start):02d}:{int(r.second_start):02d}"
                name = f"{group}_{mid}_{int(r.frame_start)}.png"
                sub = (f"{match['home_team']['short_name']} v {match['away_team']['short_name']}, "
                       f"{match['date_time'][:10]}, period {int(r.period)}, {clock}  |  "
                       f"xPV {r.xPV * 100:+.1f}, PV {r.press_value * 100:+.1f} (x100)")
                viz.press_snapshot(fi, r, match, config.PRESS_HORIZON_FRAMES, out / name,
                                   from_start=True, subtitle=sub)
                rows.append({
                    "group": group, "match_id": mid, "date": match["date_time"][:10],
                    "home": match["home_team"]["short_name"], "away": match["away_team"]["short_name"],
                    "period": int(r.period), "clock": clock, "pressing_team": r.team_shortname,
                    "presser": r.player_name, "ball_carrier": r.player_in_possession_name,
                    "press_type": r.event_subtype, "secs_since_turnover": round(r.secs_since_turnover, 1),
                    "distance_to_carrier_m": r.interplayer_distance_start,
                    "teammates_within_10m": r.defenders_within_10m, "cover_behind_20m": r.cover_behind_20m,
                    "xPV_x100": round(r.xPV * 100, 2), "PV_x100": round(r.press_value * 100, 2),
                    "regain": r.regain, "beaten": r.beaten, "image": name,
                    "verdict": "", "notes": "",
                })
    pd.DataFrame(rows).to_csv(out / "review_list.csv", index=False)
    print(f"Wrote {len(rows)} presses to {out / 'review_list.csv'}")


if __name__ == "__main__":
    main()
