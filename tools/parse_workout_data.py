"""
tools/parse_workout_data.py

Load and validate normalized WorkoutRecords from the processed JSON file.
This is the primary data-access layer for all agents — nothing reads the
raw Strava or Garmin data directly after Phase 1.

WorkoutRecord schema
--------------------
activity_id        str    — 'strava_12345'
source             str    — 'strava'
date               str    — YYYY-MM-DD
name               str    — Activity name from Strava
distance_km        float
duration_min       float
avg_pace_min_km    float  — Average pace in decimal min/km (e.g. 4.233 = 4:14/km)
avg_hr             int    — None if not recorded
max_hr             int    — None if not recorded
elevation_m        float
suffer_score       int    — Strava only; None if not available
training_load      float  — Garmin only; None if not Garmin-enriched
aerobic_effect     float  — Garmin only (0–5); None if not Garmin-enriched
anaerobic_effect   float  — Garmin only (0–5); None if not Garmin-enriched
garmin_enriched    bool   — True if Garmin activity fields were backfilled
health_context     dict   — Daily snapshot: HRV, sleep, body battery, stress,
                            training readiness, resting HR (keyed by table_column)

Public API
----------
load_workouts(path, days)           -> list[dict]
get_workout_by_id(workouts, id)     -> dict | None
get_workouts_by_date(workouts, date)-> list[dict]
get_recent_workouts(workouts, n)    -> list[dict]
validate_workout(record)            -> ValidationResult
validate_all(workouts)              -> list[ValidationResult]
format_pace(decimal_pace)           -> str   e.g. 4.2333 -> '4:14/km'
pace_to_decimal(mm, ss)            -> float  e.g. (4, 14) -> 4.2333
summarise_workouts(workouts)        -> dict
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any


# ── Schema ────────────────────────────────────────────────────────────────────

REQUIRED_FIELDS: list[str] = [
    "activity_id",
    "source",
    "date",
    "name",
    "distance_km",
    "duration_min",
    "avg_pace_min_km",
    "avg_hr",
    "max_hr",
    "elevation_m",
    "suffer_score",
    "training_load",
    "aerobic_effect",
    "anaerobic_effect",
    "garmin_enriched",
    "health_context",
]

# Fields that must never be None for a record to be usable
CRITICAL_FIELDS: list[str] = [
    "activity_id",
    "source",
    "date",
    "distance_km",
    "duration_min",
    "avg_pace_min_km",
]

# Sanity bounds for numeric fields
FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "distance_km":     (0.1,  150.0),
    "duration_min":    (1.0,  600.0),
    "avg_pace_min_km": (2.0,  15.0),
    "avg_hr":          (50.0, 230.0),
    "max_hr":          (50.0, 230.0),
    "elevation_m":     (0.0,  5000.0),
    "aerobic_effect":  (0.0,  5.0),
    "anaerobic_effect":(0.0,  5.0),
}


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    activity_id: str
    date: str
    is_valid: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str]   = field(default_factory=list)

    def __str__(self) -> str:
        status = "✅ valid" if self.is_valid else "❌ invalid"
        lines  = [f"{status} | {self.activity_id} | {self.date}"]
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ── Pace utilities ────────────────────────────────────────────────────────────

def format_pace(decimal_pace: float | None) -> str:
    """
    Convert decimal min/km to mm:ss/km string.
    e.g. 4.2333 -> '4:14/km'
    Returns '--:--/km' if None or out of range.
    """
    if decimal_pace is None or not (2.0 <= decimal_pace <= 15.0):
        return "--:--/km"
    minutes = int(decimal_pace)
    seconds = round((decimal_pace - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/km"


def pace_to_decimal(minutes: int, seconds: int) -> float:
    """
    Convert mm:ss pace to decimal min/km.
    e.g. (4, 14) -> 4.2333
    """
    return minutes + seconds / 60


def pace_difference_seconds(pace_a: float, pace_b: float) -> float:
    """
    Return the difference between two decimal paces in seconds.
    Positive means pace_a is slower than pace_b.
    """
    return (pace_a - pace_b) * 60


# ── Loading ───────────────────────────────────────────────────────────────────

def load_workouts(
    path: str,
    days: int | None = None,
    since: str | None = None,
) -> list[dict]:
    """
    Load WorkoutRecords from the processed JSON file.

    Parameters
    ----------
    path  : Full path to workouts_normalized.json
    days  : If set, return only records from the last N days
    since : If set, return records on or after this date (YYYY-MM-DD)

    Returns list sorted most-recent first.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"WorkoutRecord file not found: {path}\n"
            "Run Phase 1 notebook to generate it."
        )

    with open(path, "r", encoding="utf-8") as f:
        workouts: list[dict] = json.load(f)

    # Sort most recent first
    workouts = sorted(
        workouts,
        key=lambda r: r.get("date") or "",
        reverse=True,
    )

    # Apply date filter
    if days is not None:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        workouts = [w for w in workouts if (w.get("date") or "") >= cutoff]

    if since is not None:
        workouts = [w for w in workouts if (w.get("date") or "") >= since]

    return workouts


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def get_workout_by_id(workouts: list[dict], activity_id: str) -> dict | None:
    """Return the WorkoutRecord matching the given activity_id, or None."""
    for w in workouts:
        if w.get("activity_id") == activity_id:
            return w
    return None


def get_workouts_by_date(workouts: list[dict], target_date: str) -> list[dict]:
    """Return all WorkoutRecords on a given date (YYYY-MM-DD)."""
    return [w for w in workouts if w.get("date") == target_date]


def get_recent_workouts(workouts: list[dict], n: int = 10) -> list[dict]:
    """Return the N most recent WorkoutRecords."""
    return workouts[:n]


def get_workouts_by_type(workouts: list[dict], keyword: str) -> list[dict]:
    """
    Filter WorkoutRecords whose name contains the keyword (case-insensitive).
    Useful for finding all 'tempo', 'long', 'interval' sessions etc.
    """
    keyword = keyword.lower()
    return [w for w in workouts if keyword in (w.get("name") or "").lower()]


# ── Validation ────────────────────────────────────────────────────────────────

def validate_workout(record: dict) -> ValidationResult:
    """
    Validate a single WorkoutRecord against the schema.

    Checks:
    - All required fields are present
    - Critical fields are not None
    - Numeric fields are within sanity bounds
    - Date is a valid ISO date string
    - avg_hr < max_hr when both are present
    - health_context is a dict
    """
    activity_id = record.get("activity_id", "UNKNOWN")
    rec_date    = record.get("date", "UNKNOWN")
    errors:   list[str] = []
    warnings: list[str] = []

    # 1. Required fields present
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        errors.append(f"Missing fields: {missing}")

    # 2. Critical fields not None
    for f in CRITICAL_FIELDS:
        if record.get(f) is None:
            errors.append(f"Critical field '{f}' is None")

    # 3. Date format
    if rec_date != "UNKNOWN":
        try:
            datetime.strptime(rec_date, "%Y-%m-%d")
        except ValueError:
            errors.append(f"Invalid date format: '{rec_date}' — expected YYYY-MM-DD")

    # 4. Numeric bounds
    for f, (lo, hi) in FIELD_BOUNDS.items():
        val = record.get(f)
        if val is not None:
            try:
                val = float(val)
                if not (lo <= val <= hi):
                    warnings.append(
                        f"'{f}' value {val} is outside expected range [{lo}, {hi}]"
                    )
            except (TypeError, ValueError):
                errors.append(f"'{f}' is not numeric: {val!r}")

    # 5. HR sanity
    avg_hr = record.get("avg_hr")
    max_hr = record.get("max_hr")
    if avg_hr is not None and max_hr is not None:
        try:
            if float(avg_hr) >= float(max_hr):
                warnings.append(
                    f"avg_hr ({avg_hr}) >= max_hr ({max_hr}) — check data"
                )
        except (TypeError, ValueError):
            pass

    # 6. health_context is a dict
    hc = record.get("health_context")
    if hc is not None and not isinstance(hc, dict):
        errors.append(f"'health_context' should be a dict, got {type(hc).__name__}")

    # 7. Warn if not Garmin-enriched (agents may have degraded output)
    if not record.get("garmin_enriched", False):
        warnings.append(
            "Not Garmin-enriched — training_load, aerobic_effect, "
            "anaerobic_effect will be None"
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        activity_id=activity_id,
        date=rec_date,
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
    )


def validate_all(
    workouts: list[dict],
    print_results: bool = False,
) -> list[ValidationResult]:
    """
    Validate all WorkoutRecords.

    Parameters
    ----------
    workouts      : List of WorkoutRecord dicts
    print_results : If True, print a summary to stdout

    Returns list of ValidationResult, one per record.
    """
    results = [validate_workout(w) for w in workouts]

    if print_results:
        invalid = [r for r in results if not r.is_valid]
        warned  = [r for r in results if r.is_valid and r.warnings]
        clean   = [r for r in results if r.is_valid and not r.warnings]

        print(f"Validation summary — {len(results)} records")
        print(f"  ✅ Clean:    {len(clean)}")
        print(f"  ⚠️  Warnings: {len(warned)}")
        print(f"  ❌ Invalid:  {len(invalid)}")

        if invalid:
            print("\nInvalid records:")
            for r in invalid:
                print(str(r))

        if warned:
            print("\nRecords with warnings:")
            for r in warned:
                print(str(r))

    return results


# ── Summary statistics ────────────────────────────────────────────────────────

def summarise_workouts(workouts: list[dict]) -> dict:
    """
    Return a summary statistics dict over a list of WorkoutRecords.
    Useful for agent context — gives a quick picture of recent training.

    Returns
    -------
    {
        count, total_distance_km, avg_distance_km,
        avg_pace_min_km, avg_pace_formatted,
        avg_hr, avg_suffer_score,
        garmin_enriched_count, date_range_start, date_range_end,
        avg_training_load, avg_aerobic_effect,
    }
    """
    if not workouts:
        return {}

    def safe_avg(values: list) -> float | None:
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    dates      = [w.get("date") for w in workouts if w.get("date")]
    distances  = [w.get("distance_km") for w in workouts]
    paces      = [w.get("avg_pace_min_km") for w in workouts]
    hrs        = [w.get("avg_hr") for w in workouts]
    suffer     = [w.get("suffer_score") for w in workouts]
    loads      = [w.get("training_load") for w in workouts]
    aerobic    = [w.get("aerobic_effect") for w in workouts]
    enriched   = sum(1 for w in workouts if w.get("garmin_enriched"))

    avg_pace = safe_avg(paces)

    return {
        "count":                  len(workouts),
        "total_distance_km":      round(sum(d for d in distances if d), 2),
        "avg_distance_km":        safe_avg(distances),
        "avg_pace_min_km":        avg_pace,
        "avg_pace_formatted":     format_pace(avg_pace),
        "avg_hr":                 safe_avg(hrs),
        "avg_suffer_score":       safe_avg(suffer),
        "avg_training_load":      safe_avg(loads),
        "avg_aerobic_effect":     safe_avg(aerobic),
        "garmin_enriched_count":  enriched,
        "date_range_start":       min(dates) if dates else None,
        "date_range_end":         max(dates) if dates else None,
    }


# ── Quick test (run as script) ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Accept path as CLI arg or fall back to default Drive path
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/content/drive/MyDrive/running_coach/data/processed/workouts_normalized.json"
    )

    print(f"Loading WorkoutRecords from:\n  {path}\n")
    workouts = load_workouts(path, days=90)
    print(f"Loaded {len(workouts)} records.\n")

    print("--- Validation ---")
    validate_all(workouts, print_results=True)

    print("\n--- Summary (last 90 days) ---")
    summary = summarise_workouts(workouts)
    for k, v in summary.items():
        print(f"  {k:<28} {v}")

    print("\n--- Most recent workout ---")
    if workouts:
        w = workouts[0]
        print(f"  {w['date']} | {w['name']}")
        print(f"  Distance:       {w['distance_km']} km")
        print(f"  Duration:       {w['duration_min']} min")
        print(f"  Avg pace:       {format_pace(w['avg_pace_min_km'])}")
        print(f"  Avg HR:         {w['avg_hr']} bpm")
        print(f"  Training load:  {w['training_load']}")
        print(f"  Aerobic TE:     {w['aerobic_effect']}")
        print(f"  Garmin enriched:{w['garmin_enriched']}")
        hc = w.get("health_context", {})
        if hc:
            print(f"  Health context keys: {list(hc.keys())[:6]}...")
