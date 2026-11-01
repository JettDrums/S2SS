"""Loaders for match metadata and dynamic events."""
import json
from pathlib import Path

import pandas as pd


def match_ids(data_dir: Path) -> list[int]:
    """All match ids in data_dir that have both a match json and dynamic events."""
    return sorted(int(p.name) for p in Path(data_dir).iterdir()
                  if p.is_dir() and (p / f"{p.name}_match.json").exists()
                  and (p / f"{p.name}_dynamic_events.csv").exists())


def load_match(data_dir: Path, match_id: int) -> dict:
    with open(Path(data_dir) / str(match_id) / f"{match_id}_match.json") as f:
        return json.load(f)


def load_events(data_dir: Path, match_id: int) -> pd.DataFrame:
    path = Path(data_dir) / str(match_id) / f"{match_id}_dynamic_events.csv"
    return pd.read_csv(path, low_memory=False)


def team_info(match: dict) -> pd.DataFrame:
    """One row per team: id, name, and minutes out of possession (for per-30 rates)."""
    rows = []
    for side in ("home", "away"):
        t = match[f"{side}_team"]
        pt = match[f"{side}_team_playing_time"]
        rows.append({"match_id": match["id"], "team_id": t["id"], "team": t["short_name"],
                     "side": side, "minutes_otip": pt["minutes_otip"], "minutes_tip": pt["minutes_tip"]})
    return pd.DataFrame(rows)


def player_info(match: dict) -> pd.DataFrame:
    """One row per player who appeared, with minutes out of possession."""
    rows = []
    for p in match["players"]:
        pt = p.get("playing_time") or {}
        tot = pt.get("total") or {}
        if not tot:
            continue
        rows.append({"match_id": match["id"], "player_id": p["id"], "player": p["short_name"],
                     "team_id": p["team_id"], "position_group": p["player_role"]["position_group"],
                     "minutes_played": tot.get("minutes_played") or 0.0,
                     "minutes_otip": tot.get("minutes_otip") or 0.0})
    return pd.DataFrame(rows)
