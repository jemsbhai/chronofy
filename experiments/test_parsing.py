"""Test timestamp parsing for Dial and Wiki subsets — NO API CALLS.

Inspects all 652 non-News rows to understand timestamp patterns,
builds parsers, and validates coverage before running experiments.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime

from experiments.reproducibility import TIME_LITE_REVISION

# ============================================================
# DIAL TIMESTAMP PATTERNS
# ============================================================

# Pattern: "Session 1 happened at 29.12.2023, 22:42:04."
DIAL_SESSION_PATTERN = re.compile(
    r"Session\s+\d+\s+happened\s+at\s+(\d{1,2})[./](\d{1,2})[./](\d{4}),?\s*(\d{1,2}):(\d{2}):?(\d{2})?"
)

# Pattern: "Session 1 happened at 3:21 pm on 22 July, 2022."
DIAL_SESSION_PATTERN2 = re.compile(
    r"Session\s+\d+\s+happened\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s*(\d{4})"
)

# Pattern: username (29.12.2023, 22:42:04):
DIAL_MSG_PATTERN = re.compile(
    r"\w+\s+\((\d{1,2})[./](\d{1,2})[./](\d{4}),?\s*(\d{1,2}):(\d{2}):?(\d{2})?\)"
)

# Pattern: username (3:21 pm on 22 July, 2022):  — less likely but check
DIAL_MSG_PATTERN2 = re.compile(r"\w+\s+\(.*?(\d{1,2})\s+(\w+),?\s*(\d{4}).*?\)")

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Sept": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_dial_timestamps(context: str) -> list[datetime]:
    """Parse all timestamps from a Dial context. Returns list of datetimes."""
    results = []

    # Try Session pattern 1: "Session 1 happened at 29.12.2023, 22:42:04."
    for match in DIAL_SESSION_PATTERN.finditer(context):
        day, month, year = match.group(1), match.group(2), match.group(3)
        hour, minute = match.group(4), match.group(5)
        second = match.group(6) or "0"
        try:
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            results.append(dt)
        except ValueError:
            pass

    # Try Session pattern 2: "Session 1 happened at 3:21 pm on 22 July, 2022."
    for match in DIAL_SESSION_PATTERN2.finditer(context):
        hour, minute, ampm = int(match.group(1)), int(match.group(2)), match.group(3)
        day, month_str, year = int(match.group(4)), match.group(5), int(match.group(6))
        month = MONTH_MAP.get(month_str)
        if month:
            if ampm.lower() == "pm" and hour != 12:
                hour += 12
            elif ampm.lower() == "am" and hour == 12:
                hour = 0
            try:
                dt = datetime(year, month, day, hour, minute)
                results.append(dt)
            except ValueError:
                pass

    # If no session headers found, try individual message timestamps
    if not results:
        for match in DIAL_MSG_PATTERN.finditer(context):
            day, month, year = match.group(1), match.group(2), match.group(3)
            hour, minute = match.group(4), match.group(5)
            second = match.group(6) or "0"
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                results.append(dt)
            except ValueError:
                pass

    return results


# ============================================================
# WIKI TIMESTAMP PATTERNS
# ============================================================

# Patterns found in Wiki prose:
# "from 2000 to February 2012"
# "joined Loyola Meralco Sparks F.C. in March 2012"
# "his career began in 2005"
# Various date formats in biographical text

# Full date: "January 15, 2017" or "15 January 2017"
WIKI_DATE_FULL1 = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),?\s+(\d{4})"
)
WIKI_DATE_FULL2 = re.compile(
    r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)"
    r",?\s+(\d{4})"
)

# Partial date: "in March 2012", "from 2000", "since 2015"
WIKI_MONTH_YEAR = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})"
)

# Just year in context: various patterns (handles **bold** markdown too)
WIKI_YEAR_RANGE = re.compile(r"\*{0,2}(\d{4})\*{0,2}\s*[-–]\s*\*{0,2}(\d{4})\*{0,2}")
WIKI_YEAR_SOLO = re.compile(
    r"(?:in|from|since|during|until|by|after|before)\s+\*{0,2}(\d{4})\*{0,2}"
)


def parse_wiki_timestamps(context: str) -> list[datetime]:
    """Parse timestamps from Wiki prose. Returns list of datetimes."""
    results = []
    seen_years = set()

    # Try full dates first (most precise)
    for match in WIKI_DATE_FULL1.finditer(context):
        month_str, day_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                dt = datetime(int(year_str), month, int(day_str))
                results.append(dt)
                seen_years.add(int(year_str))
            except ValueError:
                pass

    for match in WIKI_DATE_FULL2.finditer(context):
        day_str, month_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        if month:
            try:
                dt = datetime(int(year_str), month, int(day_str))
                results.append(dt)
                seen_years.add(int(year_str))
            except ValueError:
                pass

    # Month+Year
    for match in WIKI_MONTH_YEAR.finditer(context):
        month_str, year_str = match.groups()
        month = MONTH_MAP.get(month_str)
        year = int(year_str)
        if month and 1900 < year < 2030 and year not in seen_years:
            try:
                dt = datetime(year, month, 1)
                results.append(dt)
                seen_years.add(year)
            except ValueError:
                pass

    # Year ranges (take most recent)
    for match in WIKI_YEAR_RANGE.finditer(context):
        y1, y2 = int(match.group(1)), int(match.group(2))
        for y in [y1, y2]:
            if 1900 < y < 2030 and y not in seen_years:
                results.append(datetime(y, 6, 15))  # mid-year default
                seen_years.add(y)

    # Solo years with keyword prefix
    for match in WIKI_YEAR_SOLO.finditer(context):
        y = int(match.group(1))
        if 1900 < y < 2030 and y not in seen_years:
            results.append(datetime(y, 6, 15))
            seen_years.add(y)

    # Markdown bold years: **1975** (common in Wiki subset)
    for match in re.finditer(r"\*\*(\d{4})\*\*", context):
        y = int(match.group(1))
        if 1900 < y < 2030 and y not in seen_years:
            results.append(datetime(y, 6, 15))
            seen_years.add(y)

    return results


# ============================================================
# RUN COVERAGE TEST
# ============================================================


def main():
    from datasets import load_dataset

    ds = load_dataset("SylvainWei/TIME-Lite", revision=TIME_LITE_REVISION)
    all_rows = ds["train"]

    dial_rows = [r for r in all_rows if r["Dataset Name"] == "TIME-Lite-Dial"]
    wiki_rows = [r for r in all_rows if r["Dataset Name"] == "TIME-Lite-Wiki"]

    print(f"Dial rows: {len(dial_rows)}")
    print(f"Wiki rows: {len(wiki_rows)}")

    # ---- DIAL ----
    print(f"\n{'='*60}")
    print("DIAL SUBSET PARSING")
    print(f"{'='*60}")

    dial_success = 0
    dial_fail = 0
    dial_fail_examples = []

    for i, row in enumerate(dial_rows):
        timestamps = parse_dial_timestamps(row["Context"])
        if timestamps:
            dial_success += 1
        else:
            dial_fail += 1
            if len(dial_fail_examples) < 5:
                dial_fail_examples.append((i, row["Context"][:300]))

    print(f"Success: {dial_success}/{len(dial_rows)} ({100*dial_success/len(dial_rows):.1f}%)")
    print(f"Failed:  {dial_fail}/{len(dial_rows)}")

    if dial_fail_examples:
        print(f"\nFailed examples (first {len(dial_fail_examples)}):")
        for idx, ctx in dial_fail_examples:
            print(f"\n  [{idx}] {ctx}")

    # Show some successful parses
    print("\nSuccessful parse examples:")
    shown = 0
    for row in dial_rows:
        ts = parse_dial_timestamps(row["Context"])
        if ts and shown < 3:
            print(f"  Parsed {len(ts)} timestamps, earliest: {ts[0]}, latest: {ts[-1]}")
            print(f"    Context start: {row['Context'][:120]}")
            shown += 1

    # ---- WIKI ----
    print(f"\n{'='*60}")
    print("WIKI SUBSET PARSING")
    print(f"{'='*60}")

    wiki_success = 0
    wiki_fail = 0
    wiki_fail_examples = []
    wiki_ts_counts = []

    for i, row in enumerate(wiki_rows):
        timestamps = parse_wiki_timestamps(row["Context"])
        if timestamps:
            wiki_success += 1
            wiki_ts_counts.append(len(timestamps))
        else:
            wiki_fail += 1
            if len(wiki_fail_examples) < 5:
                wiki_fail_examples.append((i, row["Context"][:300]))

    print(f"Success: {wiki_success}/{len(wiki_rows)} ({100*wiki_success/len(wiki_rows):.1f}%)")
    print(f"Failed:  {wiki_fail}/{len(wiki_rows)}")

    if wiki_ts_counts:
        import numpy as np

        arr = np.array(wiki_ts_counts)
        print(
            f"Timestamps per row: mean={arr.mean():.1f}, median={np.median(arr):.0f}, "
            f"min={arr.min()}, max={arr.max()}"
        )

    if wiki_fail_examples:
        print(f"\nFailed examples (first {len(wiki_fail_examples)}):")
        for idx, ctx in wiki_fail_examples:
            print(f"\n  [{idx}] {ctx}")

    # Show some successful parses
    print("\nSuccessful parse examples:")
    shown = 0
    for row in wiki_rows:
        ts = parse_wiki_timestamps(row["Context"])
        if ts and shown < 3:
            print(f"  Parsed {len(ts)} timestamps, earliest: {min(ts)}, latest: {max(ts)}")
            print(f"    Context start: {row['Context'][:120]}")
            shown += 1

    # ---- TASK/SETTING DISTRIBUTION ----
    print(f"\n{'='*60}")
    print("TASK DISTRIBUTION")
    print(f"{'='*60}")
    for subset_name, rows in [("Dial", dial_rows), ("Wiki", wiki_rows)]:
        print(f"\n{subset_name}:")
        task_counts = Counter(r["Task"] for r in rows)
        for task, cnt in task_counts.most_common():
            print(f"  {task}: {cnt}")
        setting_counts = Counter(r["Setting"] for r in rows)
        print(f"  Settings: {dict(setting_counts)}")


if __name__ == "__main__":
    main()
