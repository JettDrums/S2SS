"""Spatial features from tracking: space conceded and press support.

Space control uses a nearest-player (Voronoi) model on a 1 m grid. This is a
deliberately simple baseline; a velocity-aware pitch-control model is the
obvious upgrade.
"""
import numpy as np
from scipy.spatial import cKDTree

from .tracking import FrameIndex

_GRIDS = {}


def _grid(length: float, width: float):
    key = (length, width)
    if key not in _GRIDS:
        xs = np.arange(-length / 2 + 0.5, length / 2, 1.0)
        ys = np.arange(-width / 2 + 0.5, width / 2, 1.0)
        gx, gy = np.meshgrid(xs, ys)
        cells = np.column_stack([gx.ravel(), gy.ravel()])
        # Value of a cell to the attackers: grows toward the pressing team's goal (x = -L/2).
        w = ((length / 2 - cells[:, 0]) / length) ** 2
        _GRIDS[key] = (cells, w)
    return _GRIDS[key]


def to_event_frame(xy: np.ndarray, pressing_team_attacks_ltr: bool) -> np.ndarray:
    """Tracking coords -> event coords (pressing team attacks +x)."""
    return xy if pressing_team_attacks_ltr else -xy


def attacker_space(xy: np.ndarray, is_attacker: np.ndarray, length: float, width: float) -> float:
    """Share of goal-weighted pitch value controlled by the team in possession (0..1)."""
    cells, w = _grid(length, width)
    _, owner = cKDTree(xy).query(cells)
    return float((w * is_attacker[owner]).sum() / w.sum())


def frame_features(fi: FrameIndex, frame: int, pressing_team: int, carrier_id, ltr: bool,
                   length: float, width: float, min_players: int):
    """Spatial features of one frame from the pressing team's point of view, or None."""
    got = fi.get(int(frame))
    if got is None:
        return None
    ids, xy, team = got
    known = team != -1
    if known.sum() < min_players:
        return None
    ids, xy, team = ids[known], to_event_frame(xy[known], ltr), team[known]
    is_att = team != pressing_team
    out = {"space": attacker_space(xy, is_att, length, width)}
    carrier = np.flatnonzero(ids == carrier_id)
    if carrier.size:
        c = xy[carrier[0]]
        d = np.hypot(*(xy[~is_att] - c).T)
        out["defenders_within_5m"] = int((d < 5).sum())
        out["defenders_within_10m"] = int((d < 10).sum())
        # Defenders goal-side of the carrier and within 20 m: cover behind the press.
        behind = xy[~is_att][:, 0] < c[0]
        out["cover_behind_20m"] = int((behind & (d < 20)).sum())
    return out
