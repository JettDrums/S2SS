"""Team and player pressing tables: frequency, effectiveness, risk."""
import pandas as pd

PRESS_SUBTYPES = {"pressing", "counter_press", "recovery_press", "pressure"}


def _summarise(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "engagements": len(g),
        "success_rate": g.success.mean(),
        "regain_rate": g.regain.mean(),
        "regains_over_expected": (g.regain - g.xloss).mean(),
        "disruption_rate": g.disruption.mean(),
        "forced_backward_rate": g.forced_backward.mean(),
        "beaten_rate": g.beaten.mean(),
        "foul_rate": g.foul.mean(),
        "high_press_share": g.third_start.eq("attacking_third").mean(),
        "chain_share": g.in_chain.mean(),
        "mean_space_delta": g.space_delta.mean(),
        "mean_defenders_within_10m": g.get("defenders_within_10m", pd.Series(dtype=float)).mean(),
        "mean_cover_behind_20m": g.get("cover_behind_20m", pd.Series(dtype=float)).mean(),
        "space_coverage": g.space_delta.notna().mean(),
    })


def press_net_score(df: pd.DataFrame) -> pd.Series:
    """Baseline risk-reward score (z-scored across rows): reward minus risk.

    Reward: success rate and regains over expected. Risk: being beaten and
    space conceded. Equal weights are a placeholder for a learned value model.
    """
    z = lambda s: (s - s.mean()) / s.std(ddof=0)  # noqa: E731
    return (z(df.success_rate) + z(df.regains_over_expected)
            - z(df.beaten_rate) - z(df.mean_space_delta.fillna(df.mean_space_delta.mean()))) / 4


def team_table(eng: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    press = eng[eng.event_subtype.isin(PRESS_SUBTYPES)]
    t = press.groupby("team_id").apply(_summarise, include_groups=False)
    mins = teams.groupby("team_id").agg(team=("team", "first"), matches=("match_id", "nunique"),
                                        minutes_otip=("minutes_otip", "sum"))
    t = mins.join(t, how="inner")
    t["engagements_per30_otip"] = t.engagements / t.minutes_otip * 30
    t["press_net_score"] = press_net_score(t)
    return t.sort_values("press_net_score", ascending=False).reset_index()


def player_table(eng: pd.DataFrame, players: pd.DataFrame, min_engagements: int = 30) -> pd.DataFrame:
    press = eng[eng.event_subtype.isin(PRESS_SUBTYPES)]
    p = press.groupby("player_id").apply(_summarise, include_groups=False)
    meta = players.groupby("player_id").agg(player=("player", "first"), team_id=("team_id", "last"),
                                            position_group=("position_group", "first"),
                                            minutes_otip=("minutes_otip", "sum"))
    p = meta.join(p, how="inner")
    p = p[p.engagements >= min_engagements].copy()
    p["engagements_per30_otip"] = p.engagements / p.minutes_otip.where(p.minutes_otip > 0) * 30
    p["press_net_score"] = press_net_score(p)
    return p.sort_values("press_net_score", ascending=False).reset_index()


def team_chain_table(ch: pd.DataFrame) -> pd.DataFrame:
    """Coordination: how often a team's multi-player presses end in a regain or disruption."""
    t = ch.groupby(["team_id", "team"]).agg(
        chains=("chain_key", "size"), mean_players=("n_players", "mean"),
        chain_regain_rate=("regain", "mean"), chain_disruption_rate=("disruption", "mean"),
        chain_beaten_rate=("any_beaten", "mean"),
    )
    t["chain_success_rate"] = t.chain_regain_rate + t.chain_disruption_rate
    return t.sort_values("chain_success_rate", ascending=False).reset_index()
