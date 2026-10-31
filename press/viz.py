"""Figures: team pressing map and a single-press snapshot."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mplsoccer import Pitch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from .space import to_event_frame  # noqa: E402


def team_quadrant(teams, path):
    """Frequency (x) vs. risk-reward (y): who presses often, and who presses well."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(teams.engagements_per30_otip, teams.press_net_score, s=60, color="#3b6ea5")
    for r in teams.itertuples():
        ax.annotate(r.team, (r.engagements_per30_otip, r.press_net_score), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(teams.engagements_per30_otip.median(), color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Pressing engagements per 30 min out of possession")
    ax.set_ylabel("Press net score (reward minus risk, z)")
    ax.set_title("How often vs. how well teams press")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def press_snapshot(fi, eng_row, match, frames_after, path):
    """Two panels: the frame an engagement starts and N frames after it ends.

    Shading shows space controlled by the team in possession (nearest-player model).
    """
    L, W = match["pitch_length"], match["pitch_width"]
    ltr = eng_row.attacking_side == "left_to_right"
    pitch = Pitch(pitch_type="skillcorner", pitch_length=L, pitch_width=W, line_color="#888")
    fig, axes = pitch.draw(nrows=1, ncols=2, figsize=(13, 4.8))
    xs, ys = np.meshgrid(np.arange(-L / 2, L / 2, 1.0) + 0.5, np.arange(-W / 2, W / 2, 1.0) + 0.5)
    cells = np.column_stack([xs.ravel(), ys.ravel()])
    for ax, frame, title in ((axes[0], eng_row.frame_start, "Engagement starts"),
                             (axes[1], eng_row.frame_end + frames_after, f"{frames_after / 10:.0f}s after")):
        got = fi.get(int(frame))
        ax.set_title(title)
        if got is None:
            continue
        ids, xy, team = got
        xy = to_event_frame(xy, ltr)
        is_def = team == eng_row.team_id
        _, owner = cKDTree(xy).query(cells)
        ax.imshow((~is_def[owner]).reshape(xs.shape), extent=(-L / 2, L / 2, -W / 2, W / 2),
                  origin="lower", cmap="Reds", alpha=0.25, vmin=0, vmax=1.6, zorder=0)
        pitch.scatter(*xy[is_def].T, ax=ax, color="#3b6ea5", s=60, edgecolors="k", zorder=3)
        pitch.scatter(*xy[~is_def].T, ax=ax, color="#c0392b", s=60, edgecolors="k", zorder=3)
        sel = ids == eng_row.player_id
        pitch.scatter(*xy[sel].T, ax=ax, color="gold", s=140, edgecolors="k", zorder=4)
    fig.suptitle(f"{eng_row.team_shortname} {eng_row.event_subtype}: {eng_row.player_name} "
                 f"on {eng_row.player_in_possession_name} (blue = pressing team, defends left goal; gold = presser; red shading = space held by team in possession)",
                 fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
