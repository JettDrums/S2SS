"""Tracking data: parse SkillCorner jsonl into arrays, cached as parquet."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BALL_ID = -1


def _parse_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            fr = json.loads(line)
            players = fr["player_data"]
            if not players:
                continue
            frame, period = fr["frame"], fr["period"]
            for p in players:
                rows.append((frame, period, p["player_id"], p["x"], p["y"], p["is_detected"]))
            b = fr["ball_data"]
            if b and b.get("x") is not None:
                rows.append((frame, period, BALL_ID, b["x"], b["y"], b["is_detected"]))
    return pd.DataFrame(rows, columns=["frame", "period", "object_id", "x", "y", "is_detected"])


def load_tracking(data_dir: Path, match_id: int, cache_dir: Path) -> pd.DataFrame:
    """Long-format tracking (one row per object per frame). Ball has object_id == -1."""
    cache = Path(cache_dir) / f"{match_id}_tracking.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = _parse_jsonl(Path(data_dir) / str(match_id) / f"{match_id}_tracking_extrapolated.jsonl")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


class FrameIndex:
    """Fast lookup of player positions at a frame, split by team."""

    def __init__(self, tracking: pd.DataFrame, player_team: dict[int, int]):
        t = tracking[tracking.object_id != BALL_ID].sort_values("frame")
        self.frames = t.frame.to_numpy()
        self.ids = t.object_id.to_numpy()
        self.xy = t[["x", "y"]].to_numpy(float)
        self.team = np.array([player_team.get(i, -1) for i in self.ids])
        self._starts = {}
        uniq, first = np.unique(self.frames, return_index=True)
        ends = np.append(first[1:], len(self.frames))
        self._starts = dict(zip(uniq, zip(first, ends)))
        ball = tracking[tracking.object_id == BALL_ID]
        self.ball = dict(zip(ball.frame, ball[["x", "y"]].to_numpy(float)))

    def get(self, frame: int):
        """(ids, xy, team_ids) at frame, or None if the frame has no player data."""
        se = self._starts.get(frame)
        if se is None:
            return None
        s, e = se
        return self.ids[s:e], self.xy[s:e], self.team[s:e]
