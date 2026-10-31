"""Download SkillCorner's public opendata sample (10 A-League matches) for development.

This is NOT the challenge data and is not covered by the challenge's data terms.
Usage: python scripts/fetch_opendata.py [--out data/opendata] [--no-tracking]
"""
import argparse
import json
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"  # tracking files are in Git LFS


def download(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/opendata")
    ap.add_argument("--no-tracking", action="store_true", help="skip ~90 MB/match tracking files")
    args = ap.parse_args()
    out = Path(args.out)
    download(f"{RAW}/matches.json", out / "matches.json")
    for m in json.loads((out / "matches.json").read_text()):
        mid = m["id"]
        print(mid)
        for name in (f"{mid}_match.json", f"{mid}_dynamic_events.csv", f"{mid}_phases_of_play.csv"):
            download(f"{RAW}/matches/{mid}/{name}", out / str(mid) / name)
        if not args.no_tracking:
            name = f"{mid}_tracking_extrapolated.jsonl"
            download(f"{LFS}/matches/{mid}/{name}", out / str(mid) / name)


if __name__ == "__main__":
    main()
