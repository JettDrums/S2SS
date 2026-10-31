"""Paths and constants. Override the data location with the PRESS_DATA_DIR env var."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Expected layout (same as SkillCorner opendata):
#   {DATA_DIR}/{match_id}/{match_id}_match.json
#   {DATA_DIR}/{match_id}/{match_id}_dynamic_events.csv
#   {DATA_DIR}/{match_id}/{match_id}_tracking_extrapolated.jsonl
DATA_DIR = Path(os.environ.get("PRESS_DATA_DIR", ROOT / "data" / "opendata"))
CACHE_DIR = Path(os.environ.get("PRESS_CACHE_DIR", ROOT / "cache"))
OUTPUT_DIR = Path(os.environ.get("PRESS_OUTPUT_DIR", ROOT / "outputs"))

FPS = 10
# How long after an engagement ends we measure the space it left behind.
SPACE_LAG_FRAMES = 2 * FPS
# Frames with fewer tracked players than this are skipped for spatial metrics.
MIN_PLAYERS_TRACKED = 16
