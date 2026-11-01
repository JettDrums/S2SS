"""Possession value model V(state).

V = P(team in possession shoots within H s) - P(opponent shoots within H s),
estimated from the full tracking picture with gradient boosting. Press value
is then the change in V caused by a press, from the pressing team's view (see
press_value.py), so regains and conceded space share one unit: net shot
probability.

All predictions used downstream are out-of-fold by match (GroupKFold), so a
match is never scored by a model that saw it in training.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from . import config
from .space import attacker_space
from .tracking import FrameIndex

HORIZON_S = 10
GOAL_HALF_WIDTH = 3.66

STATE_FEATURES = ["ball_x", "ball_y_abs", "dist_goal", "goal_angle", "nearest_def_dist",
                  "def_within_10m", "def_goalside", "att_ahead", "def_line_gap", "att_space"]


def team_directions(events: pd.DataFrame) -> dict[tuple[int, int], bool]:
    """(team_id, period) -> True if that team attacks left-to-right in tracking coords."""
    d = events.dropna(subset=["attacking_side"]).groupby(["team_id", "period"]).attacking_side
    return {k: v.mode().iloc[0] == "left_to_right" for k, v in d}


def state_features(xy, is_att, ball, ltr, length, width) -> dict:
    """Features of a frame for the team in possession, in coords where it attacks +x."""
    s = 1.0 if ltr else -1.0
    xy, ball = xy * s, np.asarray(ball, float) * s
    gx = length / 2
    att, dfn = xy[is_att], xy[~is_att]
    dx, dy = gx - ball[0], ball[1]
    d_def = np.hypot(*(dfn - ball).T) if len(dfn) else np.array([np.inf])
    # Goal-side line: second-deepest defender (deepest is usually the goalkeeper).
    def_x = np.sort(dfn[:, 0])[::-1] if len(dfn) else np.array([gx])
    line_x = def_x[1] if len(def_x) > 1 else def_x[0]
    return {
        "ball_x": ball[0], "ball_y_abs": abs(ball[1]),
        "dist_goal": float(np.hypot(dx, dy)),
        "goal_angle": float(abs(np.arctan2(GOAL_HALF_WIDTH * 2 * dx, dx**2 + dy**2 - GOAL_HALF_WIDTH**2))),
        "nearest_def_dist": float(d_def.min()),
        "def_within_10m": int((d_def < 10).sum()),
        "def_goalside": int((dfn[:, 0] > ball[0]).sum()),
        "att_ahead": int((att[:, 0] > ball[0]).sum()),
        "def_line_gap": float(line_x - ball[0]),
        # attacker_space weights toward x = -L/2, so mirror to weight toward the goal attacked.
        "att_space": attacker_space(-xy, is_att, length, width),
    }


def possession_series(frames: pd.DataFrame, match: dict, max_gap_frames: int = 20) -> pd.Series:
    """frame -> team_id in possession; loose-ball gaps up to max_gap_frames are forward-filled."""
    gid = {"home team": match["home_team"]["id"], "away team": match["away_team"]["id"]}
    s = frames.set_index("frame").possession_group.map(gid)
    return s.ffill(limit=max_gap_frames)


def shot_frames(events: pd.DataFrame) -> pd.DataFrame:
    pp = events[(events.event_type == "player_possession") & (events.end_type == "shot")]
    return pp[["team_id", "period", "frame_end"]].rename(columns={"frame_end": "frame"})


class MatchContext:
    """Everything needed to evaluate V at any frame of one match."""

    def __init__(self, match, events, tracking, frames):
        self.match = match
        self.L, self.W = match["pitch_length"], match["pitch_width"]
        self.teams = (match["home_team"]["id"], match["away_team"]["id"])
        self.fi = FrameIndex(tracking, {p["id"]: p["team_id"] for p in match["players"]})
        self.dirs = team_directions(events)
        self.poss = possession_series(frames, match)
        self.period = frames.set_index("frame").period
        # Frames where possession changed hands, for "time since turnover".
        known = self.poss.dropna()
        change = known.ne(known.shift())
        self._change_frames = known.index[change.to_numpy()].to_numpy()

    def seconds_since_turnover(self, frame: int) -> float:
        i = np.searchsorted(self._change_frames, frame, side="right") - 1
        return np.nan if i < 0 else (frame - self._change_frames[i]) / config.FPS

    def features(self, frame: int, possessor: int):
        got = self.fi.get(int(frame))
        ball = self.fi.ball.get(int(frame))
        if got is None or ball is None or np.isnan(ball).any():
            return None
        ids, xy, team = got
        known = team != -1
        if known.sum() < config.MIN_PLAYERS_TRACKED:
            return None
        period = self.period.get(int(frame))
        ltr = self.dirs.get((possessor, period))
        if ltr is None:
            return None
        return state_features(xy[known], team[known] == possessor, ball, ltr, self.L, self.W)


def build_states(ctx: MatchContext, events: pd.DataFrame, step: int = config.FPS) -> pd.DataFrame:
    """Training rows: one per `step` frames with known possession, labelled with shots in the next H s."""
    shots = shot_frames(events)
    h = HORIZON_S * config.FPS
    rows = []
    poss = ctx.poss.dropna()
    # Sample on a fixed frame grid so a state's future (frame + horizon) is also a state.
    poss = poss[poss.index % step == 0]
    for frame, team in poss.items():
        f = ctx.features(frame, int(team))
        if f is None:
            continue
        period = ctx.period.get(frame)
        window = shots[(shots.period == period) & (shots.frame > frame) & (shots.frame <= frame + h)]
        f.update({"match_id": ctx.match["id"], "frame": frame, "team_id": int(team),
                  "shot_for": bool((window.team_id == team).any()),
                  "shot_against": bool((window.team_id != team).any())})
        rows.append(f)
    return pd.DataFrame(rows)


class ValueModel:
    """Per cross-validation fold: two classifiers (shot for / shot against) giving V, and a
    drift regressor giving the expected change in V over the press horizon from any state."""

    def __init__(self, n_folds: int = 5, seed: int = 0):
        self.n_folds, self.seed = n_folds, seed
        self.fold_of_match, self.models, self.drift_models = {}, [], []

    def _clf(self):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=100, l2_regularization=1.0,
                                              random_state=self.seed)

    def fit(self, states: pd.DataFrame) -> pd.DataFrame:
        """Fit per-fold models; return states with out-of-fold predictions and print diagnostics."""
        matches = states.match_id.unique()
        k = min(self.n_folds, len(matches))
        X, groups = states[STATE_FEATURES], states.match_id
        oof = states.copy()
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=k).split(X, groups=groups)):
            pair = {}
            for label in ("shot_for", "shot_against"):
                m = self._clf().fit(X.iloc[tr], states[label].iloc[tr])
                oof.loc[oof.index[te], f"p_{label}"] = m.predict_proba(X.iloc[te])[:, 1]
                pair[label] = m
            self.models.append(pair)
            for mid in groups.iloc[te].unique():
                self.fold_of_match[mid] = fold
        oof["V"] = oof.p_shot_for - oof.p_shot_against
        for label in ("shot_for", "shot_against"):
            y, p = oof[label], oof[f"p_{label}"]
            base = np.full(len(y), y.mean())
            print(f"  V model {label}: base rate {y.mean():.3f}, AUC {roc_auc_score(y, p):.3f}, "
                  f"Brier {brier_score_loss(y, p):.4f} (climatology {brier_score_loss(y, base):.4f})")
        return oof

    def fit_drift(self, states: pd.DataFrame, horizon: int = config.PRESS_HORIZON_FRAMES) -> pd.DataFrame:
        """Fit E[V(t + horizon) - V(t) | state at t], from the possessing team's view at t.

        This is how a possession normally evolves, pressed or not. Press value is
        measured against it, so presses are not penalised for ordinary forward
        progress or credited for ordinary regression to the mean.
        """
        fut = states[["match_id", "frame", "team_id", "V"]].copy()
        fut["frame"] -= horizon
        d = states.merge(fut, on=["match_id", "frame"], how="inner", suffixes=("", "_future"))
        same = d.team_id == d.team_id_future
        d["drift"] = np.where(same, d.V_future, -d.V_future) - d.V
        X, y = d[STATE_FEATURES], d.drift.to_numpy()
        oof = np.full(len(d), np.nan)
        for fold in range(len(self.models)):
            test = d.match_id.map(self.fold_of_match).eq(fold).to_numpy()
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=100, l2_regularization=1.0,
                                              random_state=self.seed).fit(X[~test], y[~test])
            oof[test] = m.predict(X[test])
            self.drift_models.append(m)
        print(f"  drift model: {len(d)} state pairs, mean drift {y.mean():+.4f}, out-of-fold R^2 {r2_score(y, oof):.3f}")
        return d.assign(drift_pred=oof)

    def predict_drift(self, match_id: int, feats: pd.DataFrame) -> np.ndarray:
        return self.drift_models[self.fold_of_match[match_id]].predict(feats[STATE_FEATURES])

    def predict(self, match_id: int, feats: pd.DataFrame) -> np.ndarray:
        """V for rows of one match, using the fold model that did not train on it."""
        pair = self.models[self.fold_of_match[match_id]]
        X = feats[STATE_FEATURES]
        return pair["shot_for"].predict_proba(X)[:, 1] - pair["shot_against"].predict_proba(X)[:, 1]


def nearest_possessor(ctx: MatchContext, frame: int, search: int = 10):
    """Team in possession at frame, or the nearest frame within +/- search frames."""
    for d in [0] + [x for k in range(1, search + 1) for x in (k, -k)]:
        t = ctx.poss.get(int(frame) + d)
        if t is not None and not pd.isna(t):
            return int(t)
    return None


