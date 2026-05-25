#!/usr/bin/env python3
"""
Download audio recordings from an Acoustic Workbench (api.acousticobservatory.org).

Usage:
    python download_audio.py                          # prompts for password
    python download_audio.py --token YOUR_AUTH_TOKEN  # skip login
    python download_audio.py --dry-run                # count only, no download
"""

import argparse
import getpass
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the same directory as this script (never committed to git)
load_dotenv(Path(__file__).parent / ".env")

WORKBENCH_URL = "https://api.acousticobservatory.org"
DEFAULT_USER   = os.getenv("ACOUSTIC_USER", "abhimanyuraj.singh@qcif.edu.au")

# Queensland is UTC+10 (no daylight saving).
AEST = timezone(timedelta(hours=10))
TIME_START_LOCAL = 15   # 3 pm AEST  (configurable via --time-start)
TIME_END_LOCAL   = 20   # 8 pm AEST  (configurable via --time-end)

# October 2024 and October 2025, status=ready, newest-first
OCTOBER_FILTER = {
    "projection": {
        "only": ["id", "recorded_date", "sites.name", "site_id", "canonical_file_name"]
    },
    "filter": {
        "status": {"eq": "ready"},
        "or": [
            {"and": [
                {"recorded_date": {"greater_than_or_equal": "2024-10-01T00:00:00.000Z"}},
                {"recorded_date": {"less_than":             "2024-11-01T00:00:00.000Z"}},
            ]},
            {"and": [
                {"recorded_date": {"greater_than_or_equal": "2025-10-01T00:00:00.000Z"}},
                {"recorded_date": {"less_than":             "2025-11-01T00:00:00.000Z"}},
            ]},
        ],
    },
    "sorting": {"order_by": "recorded_date", "direction": "desc"},
    "paging":  {"items": 100},
}

# Aug–Oct 2020, Robson Creek Dry A & Wet A, 15:00–20:00 AEST
ROBSON_2020_FILTER = {
    "projection": {
        "only": ["id", "recorded_date", "sites.name", "site_id", "canonical_file_name"]
    },
    "filter": {
        "and": [
            {"status": {"eq": "ready"}},
            {"recorded_date": {"greater_than_or_equal": "2020-08-01T00:00:00.000Z"}},
            {"recorded_date": {"less_than":             "2020-11-01T00:00:00.000Z"}},
            {"or": [
                {"sites.name": {"eq": "Robson Creek Dry A"}},
                {"sites.name": {"eq": "Robson Creek Wet A"}},
            ]},
        ],
    },
    "sorting": {"order_by": "recorded_date", "direction": "asc"},
    "paging":  {"items": 100},
}


def safe_folder_name(name: str) -> str:
    return re.sub(r"[^-_A-Za-z0-9]", "", name)


def in_evening_window(recorded_date_str: str) -> bool:
    """Return True if the recording falls in the configured TIME_START_LOCAL–TIME_END_LOCAL AEST window."""
    # API returns ISO 8601, e.g. "2024-10-15T07:30:00.000Z"
    dt_utc = datetime.fromisoformat(recorded_date_str.replace("Z", "+00:00"))
    dt_local = dt_utc.astimezone(AEST)
    return TIME_START_LOCAL <= dt_local.hour < TIME_END_LOCAL


def login(session: requests.Session, user: str, password: str) -> str:
    payload = {"email": user, "password": password} if "@" in user \
              else {"login": user, "password": password}
    r = session.post(f"{WORKBENCH_URL}/security", json=payload, timeout=30)
    r.raise_for_status()
    token = r.json()["data"]["auth_token"]
    print(f"Logged in as {user}")
    return token


def set_auth(session: requests.Session, token: str) -> None:
    session.headers.update({"Authorization": f"Bearer {token}"})


def iter_recordings(session: requests.Session, filter_body: dict):
    """Page through /audio_recordings/filter, yield recordings in the configured AEST window."""
    page = 0
    max_page = None
    total_seen = 0
    total_kept = 0

    while True:
        page += 1
        if max_page is not None and page > max_page:
            break

        print(f"  Fetching page {page}" + (f"/{max_page}" if max_page else "") + " …")
        r = session.post(
            f"{WORKBENCH_URL}/audio_recordings/filter",
            params={"page": page},
            json=filter_body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=60,
        )
        r.raise_for_status()
        body = r.json()
        max_page = body["meta"]["paging"]["max_page"]

        for rec in body["data"]:
            total_seen += 1
            if in_evening_window(rec["recorded_date"]):
                total_kept += 1
                yield rec

        if not body["data"]:
            break

    print(f"  {total_kept} recordings kept out of {total_seen} seen ({TIME_START_LOCAL:02d}:00–{TIME_END_LOCAL:02d}:00 AEST filter).")


def download_recording(session: requests.Session, rec: dict, target: Path) -> None:
    rec_id    = rec["id"]
    site_id   = rec["site_id"]
    filename  = rec["canonical_file_name"]
    site_name = safe_folder_name(rec.get("sites.name", str(site_id)))

    folder = target / f"{site_id}_{site_name}"
    folder.mkdir(parents=True, exist_ok=True)

    out_path = folder / filename
    if out_path.exists():
        print(f"  Skipping {filename} (already downloaded)")
        return

    print(f"  Downloading {rec_id} → {out_path.name}")
    with session.get(
        f"{WORKBENCH_URL}/audio_recordings/{rec_id}/original",
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"  Saved {filename}")


def main():
    parser = argparse.ArgumentParser(description="Download from Acoustic Workbench")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--token",    help="Auth token (skip login)")
    auth.add_argument("--user",     default=DEFAULT_USER, help="Email/username")
    parser.add_argument("--password",    help="Password (prompted if omitted)")
    parser.add_argument("--target",      default="./audio_data", help="Download directory")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Count matching recordings only — no download")
    parser.add_argument("--filter",      help="Custom filter JSON (overrides default)")
    parser.add_argument("--robson-2020", action="store_true",
                        help="Use Robson Creek Dry A & Wet A, Aug–Oct 2020, 15:00–20:00 AEST preset")
    parser.add_argument("--time-start",  type=int, default=None,
                        help="Local (AEST) hour to start keeping recordings (default: 15 for robson-2020, 17 otherwise)")
    parser.add_argument("--time-end",    type=int, default=None,
                        help="Local (AEST) hour to stop keeping recordings (default: 20)")
    args = parser.parse_args()

    # Resolve filter preset
    if args.filter:
        filter_body = json.loads(args.filter)
    elif args.robson_2020:
        filter_body = ROBSON_2020_FILTER
    else:
        filter_body = OCTOBER_FILTER

    # Apply time-window overrides
    global TIME_START_LOCAL, TIME_END_LOCAL
    if args.robson_2020 and args.time_start is None:
        TIME_START_LOCAL = 15
    elif args.time_start is not None:
        TIME_START_LOCAL = args.time_start
    if args.time_end is not None:
        TIME_END_LOCAL = args.time_end

    session = requests.Session()

    # Priority: --token flag → ACOUSTIC_TOKEN in .env → password login
    token = args.token or os.getenv("ACOUSTIC_TOKEN")
    if token:
        set_auth(session, token)
        print("Authenticated via token.")
    else:
        password = args.password or os.getenv("ACOUSTIC_PASSWORD") \
                   or getpass.getpass(f"Password for {args.user}: ")
        token = login(session, args.user, password)
        set_auth(session, token)

    if args.dry_run:
        print("\n--- DRY RUN (no files will be downloaded) ---")
        count = sum(1 for _ in iter_recordings(session, filter_body))
        print(f"\nTotal matching recordings: {count}")
        print(f"Estimated size: ~{count * 2.5:.0f} MB  ({count * 2.5 / 1024:.1f} GB)")
        return

    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to: {target}\n")

    count = 0
    for rec in iter_recordings(session, filter_body):
        download_recording(session, rec, target)
        count += 1

    print(f"\nDone — {count} recording(s) downloaded.")


if __name__ == "__main__":
    main()
