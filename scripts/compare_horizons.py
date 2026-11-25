"""Robustness: are team and player rankings stable when presses are valued over
3, 5 or 8 seconds instead of 5?

Reads team_press_value.csv / player_press_value.csv from each run directory and
prints Spearman rank correlations between horizons for each metric.

Usage: python scripts/compare_horizons.py outputs/nwsl_h3 outputs/nwsl outputs/nwsl_h8
"""
import itertools
import sys
from pathlib import Path

import pandas as pd

METRICS = ["pv_per_press", "decision_xpv", "execution"]


def main(dirs):
    out = []
    for level, fname, key in (("team", "team_press_value.csv", "team"),
                              ("player", "player_press_value.csv", "player_id")):
        tables = {d: pd.read_csv(Path(d) / fname).set_index(key) for d in dirs}
        for a, b in itertools.combinations(dirs, 2):
            j = tables[a][METRICS].join(tables[b][METRICS], lsuffix="_a", rsuffix="_b", how="inner")
            for m in METRICS:
                out.append({"level": level, "run_a": Path(a).name, "run_b": Path(b).name, "metric": m,
                            "n": len(j), "spearman": j[f"{m}_a"].corr(j[f"{m}_b"], method="spearman")})
    res = pd.DataFrame(out)
    print(res.round(3).to_string(index=False))
    res.to_csv(Path(dirs[1]) / "horizon_robustness.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
