"""Figures: team pressing map and a single-press snapshot."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
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


def value_surface(states, length, width, path):
    """Average out-of-fold V by ball location, team in possession attacking right."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pitch = Pitch(pitch_type="skillcorner", pitch_length=length, pitch_width=width, line_color="#555")
    pitch.draw(ax=ax)
    # state features store |y|; mirror to show a symmetric surface
    s = states[["ball_x", "ball_y_abs", "V"]].dropna()
    xs = np.concatenate([s.ball_x, s.ball_x])
    ys = np.concatenate([s.ball_y_abs, -s.ball_y_abs])
    vs = np.concatenate([s.V, s.V])
    stat = pitch.bin_statistic(xs, ys, vs, statistic="mean", bins=(21, 14))
    hm = pitch.heatmap(stat, ax=ax, cmap="RdBu_r", vmin=-0.3, vmax=0.3, edgecolors="none")
    fig.colorbar(hm, ax=ax, shrink=0.8, label="V = P(shot for) - P(shot against), next 10 s")
    ax.set_title("Possession value by ball location (team in possession attacks right)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration(states, path, bins=10):
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, color in (("shot_for", "#3b6ea5"), ("shot_against", "#c0392b")):
        p, y = states[f"p_{label}"], states[label]
        q = pd.qcut(p, bins, duplicates="drop")
        g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).mean()
        ax.plot(g.p, g.y, "o-", color=color, label=label.replace("_", " "))
    m = max(states.p_shot_for.max(), states.p_shot_against.max())
    ax.plot([0, m], [0, m], "k--", lw=0.8)
    ax.set_xlabel("Predicted probability (out-of-fold)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Possession value model calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def decision_execution(teams, path):
    """Decision quality (xPV) vs. execution (PV - xPV) per team."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(teams.decision_xpv * 100, teams.execution * 100, s=teams.presses / 6,
                    c=teams.pv_per30_otip, cmap="RdBu", edgecolors="k")
    for r in teams.itertuples():
        ax.annotate(r.team, (r.decision_xpv * 100, r.execution * 100), xytext=(5, 5),
                    textcoords="offset points", fontsize=8)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(teams.decision_xpv.mean() * 100, color="grey", lw=0.8, ls="--")
    fig.colorbar(sc, ax=ax, label="Press value per 30 min out of possession")
    ax.set_xlabel("Decision quality: mean expected press value (xPV, x100)")
    ax.set_ylabel("Execution: mean (PV - xPV) (x100)")
    ax.margins(0.15)
    ax.set_title("Pressing decisions vs. execution (bubble size = presses)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance(imp, path, top=12):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(d.feature, d.r2_drop, color="#3b6ea5")
    ax.set_xlabel("Drop in out-of-fold R² when shuffled")
    ax.set_title("What drives expected press value (xPV)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def value_by_outcome(eng, path):
    """Validation: press value should be high for regains and low when the presser is beaten."""
    d = eng.dropna(subset=["press_value"])
    groups = {"Regain": d.regain, "Disruption": d.disruption, "Forced backward": d.forced_backward,
              "No listed outcome": ~(d.success | d.beaten), "Beaten": d.beaten}
    means = {k: d.press_value[m].mean() * 100 for k, m in groups.items()}
    ns = {k: int(m.sum()) for k, m in groups.items()}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["#3b6ea5" if v >= 0 else "#c0392b" for v in means.values()]
    ax.bar([f"{k}\n(n={ns[k]})" for k in means], list(means.values()), color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Mean press value (x100)")
    ax.set_title("Press value by SkillCorner outcome label")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def decision_rules_grid(rules, path):
    """Small multiples: mean xPV by level of each situational factor."""
    factors = list(dict.fromkeys(rules.factor))
    n = len(factors)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows))
    for ax, f in zip(axes.ravel(), factors):
        d = rules[rules.factor == f]
        vals = d.mean_xpv * 100
        labels = [f"{lv.replace('_', ' ')} (n={nn})" for lv, nn in zip(d.level, d.n)]
        ax.barh(labels, vals, color=["#3b6ea5" if v >= 0 else "#c0392b" for v in vals])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f.replace("_", " "))
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("When is pressing a good decision? Mean expected press value (xPV x100) by situation", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def decision_map(p, length, width, path):
    """Mean xPV by where the press happens (pressing team defends the left goal)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pitch = Pitch(pitch_type="skillcorner", pitch_length=length, pitch_width=width, line_color="#555")
    pitch.draw(ax=ax)
    stat = pitch.bin_statistic(p.x_start, p.y_start, p.xPV * 100, statistic="mean", bins=(12, 8))
    cnt = pitch.bin_statistic(p.x_start, p.y_start, statistic="count", bins=(12, 8))
    stat["statistic"] = np.where(cnt["statistic"] >= 40, stat["statistic"], np.nan)
    hm = pitch.heatmap(stat, ax=ax, cmap="RdBu", vmin=-3, vmax=3, edgecolors="white")
    fig.colorbar(hm, ax=ax, shrink=0.8, label="Mean xPV (x100); bins with < 40 presses hidden")
    ax.set_title("Where pressing is a good decision (pressing team defends the left goal)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
