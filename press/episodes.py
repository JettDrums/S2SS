"""Press episodes from SkillCorner on-ball engagements.

Coordinate convention of dynamic events (verified against tracking): x/y are
flipped so that the event's team (here: the pressing team) attacks toward +x.
The pressing team's own goal is therefore at x = -pitch_length / 2.
"""
import pandas as pd

REGAIN = {"direct_regain", "indirect_regain"}
DISRUPTION = {"direct_disruption", "indirect_disruption"}


def engagements(events: pd.DataFrame) -> pd.DataFrame:
    """One row per on-ball engagement with boolean outcome flags."""
    e = events[events.event_type == "on_ball_engagement"].copy()
    b = lambda c: e[c].fillna(False).astype(bool)  # noqa: E731
    e["regain"] = e.end_type.isin(REGAIN)
    e["disruption"] = e.end_type.isin(DISRUPTION)
    e["foul"] = e.end_type.eq("foul_committed")
    e["forced_backward"] = b("force_backward")
    e["danger_reduced"] = b("stop_possession_danger") | b("reduce_possession_danger")
    e["beaten"] = b("beaten_by_possession") | b("beaten_by_movement")
    # Any positive defensive outcome for the engaging player.
    e["success"] = e.regain | e.disruption | e.forced_backward | e.danger_reduced
    # Chances the carrier would lose the ball anyway (SkillCorner xLoss): baseline for "regains over expected".
    e["xloss"] = e.xloss_player_possession_start
    e["in_chain"] = b("pressing_chain")
    e["chain_key"] = (e.match_id.astype(str) + "_" + e.team_id.astype(str) + "_"
                      + e.pressing_chain_index.astype("Int64").astype(str))
    e.loc[~e.in_chain, "chain_key"] = pd.NA
    return e


def chains(eng: pd.DataFrame) -> pd.DataFrame:
    """One row per pressing chain (a coordinated sequence of engagements by one team)."""
    c = eng[eng.in_chain]
    out = c.groupby("chain_key").agg(
        match_id=("match_id", "first"), team_id=("team_id", "first"), team=("team_shortname", "first"),
        n_engagements=("event_id", "size"), n_players=("player_id", "nunique"),
        frame_start=("frame_start", "min"), frame_end=("frame_end", "max"),
        x_start=("x_start", "first"), third=("third_start", "first"),
        end_type=("pressing_chain_end_type", lambda s: s.dropna().iloc[0] if s.notna().any() else "none"),
        any_beaten=("beaten", "any"),
    ).reset_index()
    out["duration_s"] = (out.frame_end - out.frame_start) / 10
    out["regain"] = out.end_type.eq("regain")
    out["disruption"] = out.end_type.eq("disruption")
    return out
