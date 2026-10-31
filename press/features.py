"""Attach tracking-based features to each engagement."""
import numpy as np
import pandas as pd

from . import config
from .space import frame_features
from .tracking import FrameIndex


def add_spatial_features(eng: pd.DataFrame, tracking: pd.DataFrame, match: dict) -> pd.DataFrame:
    """Space before/after each engagement and how many teammates supported it.

    space_delta > 0 means the team in possession controls more dangerous space
    2 s after the engagement than when it started: the "risk" side of pressing.
    """
    player_team = {p["id"]: p["team_id"] for p in match["players"]}
    fi = FrameIndex(tracking, player_team)
    L, W = match["pitch_length"], match["pitch_width"]
    rows = []
    for r in eng.itertuples():
        ltr = r.attacking_side == "left_to_right"
        f0 = frame_features(fi, r.frame_start, r.team_id, r.player_in_possession_id, ltr, L, W,
                            config.MIN_PLAYERS_TRACKED)
        f1 = frame_features(fi, r.frame_end + config.SPACE_LAG_FRAMES, r.team_id, r.player_in_possession_id,
                            ltr, L, W, config.MIN_PLAYERS_TRACKED)
        row = {"event_id": r.event_id}
        if f0:
            row.update({"space_start": f0["space"],
                        **{k: v for k, v in f0.items() if k != "space"}})
        if f1:
            row["space_after"] = f1["space"]
        rows.append(row)
    feats = pd.DataFrame(rows)
    out = eng.merge(feats, on="event_id", how="left")
    out["space_delta"] = out.get("space_after", np.nan) - out.get("space_start", np.nan)
    return out
