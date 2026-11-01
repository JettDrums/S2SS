"""Decision analysis on valued presses: when is pressing a good idea, is it a stable
skill, and who decides and executes best once small samples are accounted for.

Inputs are engagements with press_value (PV) and xPV from press_value.py.
"""
import numpy as np
import pandas as pd

from .metrics import PRESS_SUBTYPES
from .press_value import press_kind

# Situational factors a coach can act on, binned for readable decision rules.
# All factors are known when the press starts (no hindsight-based SkillCorner labels).
FACTORS = {
    "seconds_since_turnover": ("secs_since_turnover", [0, 3, 6, 10, 20, 1e6]),
    "block_height_m": ("block_height", [-60, -25, -15, -5, 60]),
    "third": ("third_start", None),
    "press_kind": ("press_kind", None),
    "distance_to_carrier_m": ("interplayer_distance_start", [0, 3, 5, 7, 9, 11]),
    "teammates_within_10m": ("defenders_within_10m", [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 99]),
    "cover_behind_20m": ("cover_behind_20m", [-0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 99]),
    "carrier_xloss": ("xloss", [0, 0.1, 0.2, 0.3, 0.5, 1.0]),
}


def valued_presses(eng: pd.DataFrame) -> pd.DataFrame:
    """Presses with xPV, plus decision quality relative to others in the same position.

    Raw xPV is dominated by role (centre-backs press in lower-value situations
    than forwards), so player decision quality is judged within position.
    """
    p = eng[eng.event_subtype.isin(PRESS_SUBTYPES) & eng.xPV.notna()].copy()
    p["xpv_vs_position"] = p.xPV - p.groupby("player_position").xPV.transform("mean")
    p["press_kind"] = press_kind(p.event_subtype)
    return p


def decision_rules(p: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    """Mean xPV / PV and share of good-decision presses for each level of each factor."""
    out = []
    for name, (col, bins) in FACTORS.items():
        key = pd.cut(p[col], bins) if bins else p[col]
        g = p.groupby(key, observed=True).agg(n=("xPV", "size"), mean_xpv=("xPV", "mean"),
                                             mean_pv=("press_value", "mean"),
                                             good_decision_share=("xPV", lambda s: (s > 0).mean()))
        g = g[g.n >= min_n].reset_index().rename(columns={col: "level"})
        g.insert(0, "factor", name)
        g["level"] = g.level.astype(str)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def icc(df: pd.DataFrame, unit: str, value: str, min_matches: int = 2, n_perm: int = 2000, seed: int = 0):
    """Intraclass correlation ICC(1) of unit-match means, with a permutation p-value.

    ICC = share of match-to-match variance explained by who the unit is. High
    ICC for decision quality means it is a repeatable trait, not noise.
    """
    um = df.groupby([unit, "match_id"])[value].mean().reset_index()
    counts = um.groupby(unit).match_id.transform("size")
    um = um[counts >= min_matches]
    if um[unit].nunique() < 3:
        return np.nan, np.nan, 0

    def _icc(labels, y):
        groups = pd.Series(y).groupby(labels)
        k = groups.size()
        n, a = len(y), len(k)
        k0 = (n - (k**2).sum() / n) / (a - 1)
        msb = (k * (groups.mean() - y.mean()) ** 2).sum() / (a - 1)
        msw = ((y - groups.transform("mean")) ** 2).sum() / (n - a)
        return (msb - msw) / (msb + (k0 - 1) * msw)

    y = um[value].to_numpy()
    labels = um[unit].to_numpy()
    obs = _icc(labels, y)
    rng = np.random.default_rng(seed)
    null = np.array([_icc(rng.permutation(labels), y) for _ in range(n_perm)])
    return float(obs), float((null >= obs).mean()), int(um[unit].nunique())


def reliability_table(p: pd.DataFrame, min_presses_per_match: int = 10) -> pd.DataFrame:
    rows = []
    for unit, label in (("team_id", "team"), ("player_id", "player")):
        d = p
        if unit == "player_id":
            n = d.groupby(["player_id", "match_id"]).xPV.transform("size")
            d = d[n >= min_presses_per_match]
        metrics = [("decision (xPV)", "xPV"), ("execution (PV - xPV)", "execution"),
                   ("press value (PV)", "press_value")]
        if unit == "player_id":
            metrics.insert(1, ("decision vs. position (xPV - position mean)", "xpv_vs_position"))
        for metric, col in metrics:
            r, pval, units = icc(d, unit, col)
            rows.append({"level": label, "metric": metric, "icc": r, "perm_p": pval, "units": units})
    return pd.DataFrame(rows)


def shrink(df: pd.DataFrame, group: str, value: str) -> pd.DataFrame:
    """Empirical-Bayes shrinkage of group means toward the overall mean.

    shrunk = (n * mean + k * grand) / (n + k), k = within variance / between variance.
    Groups with few presses are pulled toward average, so rankings reward
    sustained performance rather than small-sample luck.
    """
    g = df.groupby(group)[value].agg(["mean", "size", "var"])
    grand = df[value].mean()
    within = df.groupby(group)[value].var().mean()
    between = max(g["mean"].var() - (within / g["size"]).mean(), 1e-12)
    k = within / between
    return pd.DataFrame({f"{value}_mean": g["mean"], "n": g["size"],
                         f"{value}_shrunk": (g["size"] * g["mean"] + k * grand) / (g["size"] + k),
                         f"{value}_k": k})


def player_rankings(p: pd.DataFrame, min_presses: int = 30) -> pd.DataFrame:
    meta = p.groupby("player_id").agg(player=("player_name", "first"), team=("team_shortname", "last"),
                                      position=("player_position", "first"))
    out = meta
    for col in ("xPV", "xpv_vs_position", "execution", "press_value"):
        s = shrink(p, "player_id", col)
        out = out.join(s[[f"{col}_shrunk"]])
    out["presses"] = p.groupby("player_id").size()
    out["good_decision_share"] = p.groupby("player_id").xPV.apply(lambda s: (s > 0).mean())
    out = out[out.presses >= min_presses]
    return out.sort_values("press_value_shrunk", ascending=False).reset_index()


def team_profiles(p: pd.DataFrame) -> pd.DataFrame:
    """Where each team's decision quality comes from: phase mix and selectivity."""
    t = p.groupby("team_shortname").agg(
        presses=("xPV", "size"), decision_xpv=("xPV", "mean"), execution=("execution", "mean"),
        good_decision_share=("xPV", lambda s: (s > 0).mean()),
        high_press_share=("third_start", lambda s: s.eq("attacking_third").mean()),
        transition_press_share=("secs_since_turnover", lambda s: (s < 6).mean()),
        mean_block_height=("block_height", "mean"),
        mean_cover_behind=("cover_behind_20m", "mean"),
        mean_teammates_within_10m=("defenders_within_10m", "mean"),
    )
    return t.sort_values("decision_xpv", ascending=False).reset_index()
