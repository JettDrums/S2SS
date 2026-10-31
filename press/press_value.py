"""Learned press value: what each press is worth, and how much of that was a good decision.

press_value_raw: change in V from the pressing team's view over a fixed 5 s from
engagement start. If the pressing team has the ball at the end (a regain), its
own V counts positively; otherwise the opponent's V counts negatively. Units:
net probability of a shot in the next 10 s.

press_value (PV): press_value_raw minus the change expected from that starting
state anyway (the drift model in value.py). Positive PV = the possession went
worse for the attackers than it normally would from the same situation.

expected press value (xPV): PV predicted from the situation *before* the press
(location, distances, support, cover, space, context). Out-of-fold by match.
    decision quality = xPV            (did they press at good moments?)
    execution        = PV - xPV       (did the press go better than expected?)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from . import config
from .metrics import PRESS_SUBTYPES
from .value import MatchContext, ValueModel, nearest_possessor

CONTEXT_NUM = ["x_start", "y_abs", "interplayer_distance_start", "defenders_within_5m",
               "defenders_within_10m", "cover_behind_20m", "space_start", "xloss", "V_before",
               "simultaneous", "index_in_chain"]
CONTEXT_CAT = ["event_subtype", "team_out_of_possession_phase_type", "game_state"]


def press_values(ctx: MatchContext, eng: pd.DataFrame, vm: ValueModel) -> pd.DataFrame:
    """Add V_before, V_after (attacking/possessing team view) and press_value to engagements."""
    horizon = config.PRESS_HORIZON_FRAMES
    rows0, rows1, keep = [], [], []
    for r in eng.itertuples():
        pressing = int(r.team_id)
        attackers = ctx.teams[1] if pressing == ctx.teams[0] else ctx.teams[0]
        f0 = ctx.features(r.frame_start, attackers)
        t1 = int(r.frame_start) + horizon
        owner1 = nearest_possessor(ctx, t1)
        f1 = ctx.features(t1, owner1) if owner1 is not None else None
        if f0 is None or f1 is None:
            continue
        rows0.append(f0)
        rows1.append(f1)
        keep.append((r.Index, owner1 == pressing))
    out = eng.copy()
    for c in ("V_before", "V_after", "press_value_raw", "expected_drift", "press_value", "regained_at_end"):
        out[c] = np.nan
    if not keep:
        return out
    idx = [k[0] for k in keep]
    regained = np.array([k[1] for k in keep])
    f0 = pd.DataFrame(rows0)
    v0 = vm.predict(ctx.match["id"], f0)
    v1 = vm.predict(ctx.match["id"], pd.DataFrame(rows1))
    # Attackers' expected drift from this state, flipped to the pressing team's view.
    expected = -vm.predict_drift(ctx.match["id"], f0)
    p0 = -v0                                  # pressing team's value before: opponent has the ball
    p1 = np.where(regained, v1, -v1)          # after: own value if regained, else minus opponent's
    out.loc[idx, "V_before"] = v0
    out.loc[idx, "V_after"] = v1
    out.loc[idx, "regained_at_end"] = regained.astype(float)
    out.loc[idx, "press_value_raw"] = p1 - p0
    out.loc[idx, "expected_drift"] = expected
    out.loc[idx, "press_value"] = (p1 - p0) - expected
    return out


def _design(eng: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=eng.index)
    X["x_start"] = eng.x_start
    X["y_abs"] = eng.y_start.abs()
    X["interplayer_distance_start"] = eng.interplayer_distance_start
    for c in ("defenders_within_5m", "defenders_within_10m", "cover_behind_20m", "space_start",
              "xloss", "V_before"):
        X[c] = eng[c]
    X["simultaneous"] = eng.simultaneous_defensive_engagement_same_target.fillna(False).astype(float)
    X["index_in_chain"] = eng.index_in_pressing_chain.fillna(0)
    for c in CONTEXT_CAT:
        X[c] = eng[c].astype("category")
    return X


def fit_expected(eng: pd.DataFrame, n_folds: int = 5, seed: int = 0):
    """Out-of-fold xPV for every press with a press_value; returns (eng, importance table)."""
    d = eng[eng.event_subtype.isin(PRESS_SUBTYPES) & eng.press_value.notna()].copy()
    X, y, groups = _design(d), d.press_value.to_numpy(), d.match_id
    k = min(n_folds, groups.nunique())
    oof = np.full(len(d), np.nan)
    imps = []
    for tr, te in GroupKFold(n_splits=k).split(X, groups=groups):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=80, l2_regularization=1.0,
                                          categorical_features="from_dtype", random_state=seed)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict(X.iloc[te])
        pi = permutation_importance(m, X.iloc[te], y[te], n_repeats=3, random_state=seed, scoring="r2")
        imps.append(pd.Series(pi.importances_mean, index=X.columns))
    d["xPV"] = oof
    d["execution"] = d.press_value - d.xPV
    print(f"  xPV model: out-of-fold R^2 {r2_score(y, oof):.3f} on {len(d)} presses")
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False).rename("r2_drop").reset_index()
    out = eng.merge(d[["match_id", "event_id", "xPV", "execution"]], on=["match_id", "event_id"], how="left")
    return out, imp.rename(columns={"index": "feature"})


def _bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def _value_summary(g: pd.DataFrame) -> pd.Series:
    pv = g.press_value.dropna().to_numpy()
    lo, hi = _bootstrap_ci(pv) if len(pv) > 1 else (np.nan, np.nan)
    return pd.Series({
        "presses": len(pv),
        "pv_total": pv.sum(),
        "pv_per_press": pv.mean() if len(pv) else np.nan,
        "pv_per_press_ci_low": lo, "pv_per_press_ci_high": hi,
        "decision_xpv": g.xPV.mean(),
        "execution": g.execution.mean(),
        "positive_xpv_share": (g.xPV.dropna() > 0).mean(),
    })


def team_value_table(eng: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    press = eng[eng.event_subtype.isin(PRESS_SUBTYPES) & eng.press_value.notna()]
    t = press.groupby("team_id").apply(_value_summary, include_groups=False)
    mins = teams.groupby("team_id").agg(team=("team", "first"), minutes_otip=("minutes_otip", "sum"))
    t = mins.join(t, how="inner")
    t["pv_per30_otip"] = t.pv_total / t.minutes_otip * 30
    return t.sort_values("pv_per30_otip", ascending=False).reset_index()


def player_value_table(eng: pd.DataFrame, players: pd.DataFrame, min_presses: int = 30) -> pd.DataFrame:
    press = eng[eng.event_subtype.isin(PRESS_SUBTYPES) & eng.press_value.notna()]
    p = press.groupby("player_id").apply(_value_summary, include_groups=False)
    meta = players.groupby("player_id").agg(player=("player", "first"), team_id=("team_id", "last"),
                                            position_group=("position_group", "first"),
                                            minutes_otip=("minutes_otip", "sum"))
    p = meta.join(p, how="inner")
    p = p[p.presses >= min_presses].copy()
    p["pv_per30_otip"] = p.pv_total / p.minutes_otip.where(p.minutes_otip > 0) * 30
    return p.sort_values("pv_per_press", ascending=False).reset_index()
