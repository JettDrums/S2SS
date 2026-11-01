"""Download the challenge data (NWSL 2025, FAWSL 2024/25) from the SkillCorner API.

Writes the same layout as the opendata sample, so the pipeline runs unchanged:
    data/challenge/{match_id}/{match_id}_match.json | _dynamic_events.csv | _tracking_extrapolated.jsonl

Credentials come from env vars; never hard-code or commit them:
    export SKILLCORNER_USERNAME=...  SKILLCORNER_PASSWORD=...

Usage:
    python scripts/fetch_skillcorner_api.py --list-editions
    python scripts/fetch_skillcorner_api.py --edition <id> [--edition <id>] [--limit 3] [--compact]

--compact parses each match's tracking into parquet caches (~4 MB vs ~140 MB raw)
and deletes the raw tracking file, so a full season fits on a laptop.

Verified against the challenge account (NWSL 2025 = competition edition 1093).
The API returns tracking as one JSON array; press.tracking reads both formats.
"""
import argparse
import os
import sys
from pathlib import Path

from skillcorner.client import SkillcornerClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from press import config, tracking  # noqa: E402


def results(resp):
    return resp["results"] if isinstance(resp, dict) and "results" in resp else resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/challenge")
    ap.add_argument("--edition", type=int, action="append", default=[])
    ap.add_argument("--list-editions", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max matches per edition (for testing)")
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--compact", action="store_true", help="cache tracking as parquet, delete raw file")
    args = ap.parse_args()

    client = SkillcornerClient(username=os.environ["SKILLCORNER_USERNAME"],
                               password=os.environ["SKILLCORNER_PASSWORD"])
    if args.list_editions:
        for e in results(client.get_competition_editions(params={"user": "true"})):
            print(e.get("id"), e.get("name"))
        return

    out = Path(args.out)
    for edition in args.edition:
        matches = results(client.get_matches(params={"competition_edition": edition, "user": "true"}))
        for m in matches[: args.limit]:
            mid = m["id"]
            d = out / str(mid)
            d.mkdir(parents=True, exist_ok=True)
            print(mid, flush=True)
            # Already compacted on an earlier run: raw file gone but caches present.
            cached = not tracking.raw_path(out, mid).exists() and tracking.has_tracking(out, mid, config.CACHE_DIR)
            targets = [
                (d / f"{mid}_match.json", lambda p: client.save_match(match_id=mid, filepath=str(p))),
                (d / f"{mid}_dynamic_events.csv",
                 lambda p: client.save_dynamic_events(match_id=mid, filepath=str(p), params={"file_format": "csv"})),
            ]
            if not args.no_tracking and not cached:
                targets.append((d / f"{mid}_tracking_extrapolated.jsonl",
                                lambda p: client.save_match_tracking_data(
                                    match_id=mid, filepath=str(p),
                                    params={"data_version": 3, "file_format": "jsonl"})))
            try:
                for path, save in targets:
                    if not path.exists():
                        save(path)
                if args.compact and tracking.raw_path(out, mid).exists():
                    tracking.build_caches(out, mid, config.CACHE_DIR)
                    tracking.raw_path(out, mid).unlink()
            except Exception as exc:  # keep going; rerunning the script resumes where it left off
                print(f"  failed: {exc!r}", flush=True)
                for path, _ in targets:
                    if path.exists() and path.stat().st_size == 0:
                        path.unlink()


if __name__ == "__main__":
    main()
