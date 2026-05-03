"""
tools/calculate_pace_zones.py

Classify WorkoutRecords by workout type and evaluate actual pace against
Will's coach-provided target paces.

Workout type detection
----------------------
1. Garmin-named structured workouts: activity name contains a YYYY-MM-DD
   pattern (e.g. "Halifax - 2026-04-15"). The embedded date is extracted
   and used to look up workout type from a known schedule or exception list.

2. Known exceptions: hardcoded one-off sessions that don't follow the
   date-name convention (e.g. April 8 "8 x 800m @ vo2 max").

3. Unknown / unstructured activities: easy runs, races, and anything else
   without a date in the name. The Feedback Agent should prompt the user
   to classify these before pace evaluation.

Target paces (from config)
--------------------------
easy        None        By feel — check below EASY_PACE_CEILING (5:30/km)
marathon    4:14/km
threshold   4:01/km
1hr         3:56/km
fartlek     3:49/km
8k          3:46/km
vo2max      3:40/km

Tolerance: ±5 seconds counts as on-target.

Public API
----------
detect_workout_type(workout)            -> WorkoutClassification
evaluate_pace(workout, workout_type)    -> PaceEvaluation
classify_and_evaluate(workout)          -> tuple[WorkoutClassification, PaceEvaluation | None]
classify_workouts(workouts)             -> list[dict]   (enriched WorkoutRecords)
format_pace(decimal_pace)               -> str
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# ── Config (mirrors config.py — keep in sync) ─────────────────────────────────

TRAINING_PACES: dict[str, float | None] = {
    "easy":      None,
    "marathon":  4 + 14 / 60,
    "threshold": 4 + 1  / 60,
    "1hr":       3 + 56 / 60,
    "fartlek":   3 + 49 / 60,
    "8k":        3 + 46 / 60,
    "vo2max":    3 + 40 / 60,
}

EASY_PACE_CEILING  = 4 + 45 / 60   # Faster than 4:45/km = too hard for easy run
PACE_TOLERANCE_SEC = 5              # ±5 seconds = on target


# ── Known exceptions ──────────────────────────────────────────────────────────
# Workouts that don't carry a YYYY-MM-DD name but have a known type.
# Key: YYYY-MM-DD date string. Value: workout type from TRAINING_PACES.

KNOWN_EXCEPTIONS: dict[str, str] = {
    "2026-04-08": "vo2max",   # 8 x 800m @ VO2 Max
}


# ── Date pattern for Garmin workout names ─────────────────────────────────────
# Matches YYYY-MM-DD anywhere in the activity name.
# e.g. "Halifax - 2026-04-15" → "2026-04-15"

_DATE_PATTERN = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class WorkoutClassification:
    """Result of classifying a WorkoutRecord by type."""
    activity_id:    str
    date:           str
    workout_type:   str | None      # From TRAINING_PACES keys, or None if unknown
    is_structured:  bool            # True if Garmin date-named or known exception
    needs_input:    bool            # True if agent should ask user for type
    detected_from:  str             # 'garmin_name' | 'known_exception' | 'unknown'
    notes:          str = ""

    @property
    def target_pace(self) -> float | None:
        """Return target pace in decimal min/km, or None if easy/unknown."""
        if self.workout_type is None:
            return None
        return TRAINING_PACES.get(self.workout_type)


@dataclass
class PaceEvaluation:
    """Result of evaluating actual vs target pace for a structured workout."""
    activity_id:        str
    date:               str
    workout_type:       str
    target_pace:        float | None
    target_pace_fmt:    str
    actual_pace:        float | None
    actual_pace_fmt:    str
    diff_seconds:       float | None    # Positive = slower than target
    on_target:          bool | None     # None if easy (no target)
    verdict:            str             # Human-readable assessment
    notes:              str = ""


# ── Pace utilities ────────────────────────────────────────────────────────────

def format_pace(decimal_pace: float | None) -> str:
    """Convert decimal min/km to mm:ss/km string. e.g. 4.2333 -> '4:14/km'"""
    if decimal_pace is None or not (2.0 <= decimal_pace <= 15.0):
        return "--:--/km"
    minutes = int(decimal_pace)
    seconds = round((decimal_pace - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}/km"


def _seconds_to_pace_str(seconds: float) -> str:
    """Format a pace difference in seconds as a +/- string. e.g. -8 -> '8s faster'"""
    if seconds > 0:
        return f"{abs(seconds):.0f}s slower than target"
    elif seconds < 0:
        return f"{abs(seconds):.0f}s faster than target"
    else:
        return "exactly on target"


# ── Classification ────────────────────────────────────────────────────────────

def detect_workout_type(workout: dict) -> WorkoutClassification:
    """
    Classify a WorkoutRecord by workout type.

    Detection order:
    1. Check KNOWN_EXCEPTIONS by date
    2. Look for YYYY-MM-DD in the Garmin activity name
       — if found, the date embedded in the name tells us it was a structured
         workout. Workout type must be resolved from a training plan lookup
         (not implemented here — Planner Agent responsibility). For now,
         structured workouts without a resolvable type are flagged as
         needs_input=False but workout_type=None (Planner Agent handles).
    3. Otherwise: unknown — Feedback Agent should ask the user.

    Note: Strava names are not used for classification since Will doesn't
    name his activities on Strava.
    """
    activity_id  = workout.get("activity_id", "unknown")
    date         = workout.get("date", "")
    # Use garmin_name if available, fall back to strava name
    garmin_name  = workout.get("garmin_name", "") or workout.get("name", "") or ""
    name_lower   = garmin_name.lower()

    # 1. Known exception by date
    if date in KNOWN_EXCEPTIONS:
        wtype = KNOWN_EXCEPTIONS[date]
        return WorkoutClassification(
            activity_id=activity_id,
            date=date,
            workout_type=wtype,
            is_structured=True,
            needs_input=False,
            detected_from="known_exception",
            notes=f"Matched known exception: {wtype}",
        )

    # 2. Auto-classify known easy run name patterns
    # "Halifax Running" → easy run
    # "Halifax - Easy w/ strides" → easy run (strides don't change overall classification)
    EASY_NAME_PATTERNS = [
        "halifax running",
        "easy",
        "morning run",
        "treadmill running",
        "running",   # catches "Saint James Running", "Antigonish County Running", etc.
    ]
    if any(p in name_lower for p in EASY_NAME_PATTERNS):
        note = "Easy run" if "easy" not in name_lower else "Easy run with strides"
        return WorkoutClassification(
            activity_id=activity_id,
            date=date,
            workout_type="easy",
            is_structured=False,
            needs_input=False,
            detected_from="name_pattern",
            notes=note,
        )

    # 3. Garmin date-named structured workout
    # Use the actual activity date, not the embedded date
    # (coach schedules by date but runs may happen the next day)
    match = _DATE_PATTERN.search(garmin_name)
    if match:
        embedded_date = match.group(1)
        try:
            datetime.strptime(embedded_date, "%Y-%m-%d")
            valid_date = True
        except ValueError:
            valid_date = False

        if valid_date:
            return WorkoutClassification(
                activity_id=activity_id,
                date=date,
                workout_type=None,   # Planner Agent resolves from training schedule
                is_structured=True,
                needs_input=False,
                detected_from="garmin_name",
                notes=(
                    f"Garmin date-named workout (scheduled {embedded_date}, "
                    f"done {date}). "
                    "Workout type to be resolved by Planner Agent."
                ),
            )

    # 4. Unknown — Feedback Agent should ask the user
    return WorkoutClassification(
        activity_id=activity_id,
        date=date,
        workout_type=None,
        is_structured=False,
        needs_input=True,
        detected_from="unknown",
        notes=(
            f"Unrecognised activity name: '{garmin_name}'. "
            "Feedback Agent should ask user to confirm workout type before pace evaluation."
        ),
    )


def resolve_workout_type(
    classification: WorkoutClassification,
    workout_type: str,
) -> WorkoutClassification:
    """
    Resolve a classification that needs_input=True with a user-provided type.
    Used by the Feedback Agent after prompting the user.

    Parameters
    ----------
    classification : The original WorkoutClassification with needs_input=True
    workout_type   : User-provided type string — must be a key in TRAINING_PACES

    Returns updated WorkoutClassification.
    Raises ValueError if workout_type is not recognised.
    """
    if workout_type not in TRAINING_PACES:
        valid = list(TRAINING_PACES.keys())
        raise ValueError(
            f"Unknown workout type '{workout_type}'. Valid types: {valid}"
        )

    return WorkoutClassification(
        activity_id=classification.activity_id,
        date=classification.date,
        workout_type=workout_type,
        is_structured=classification.is_structured,
        needs_input=False,
        detected_from="user_input",
        notes=f"Type confirmed by user: {workout_type}",
    )


# ── Pace evaluation ───────────────────────────────────────────────────────────

def evaluate_pace(
    workout: dict,
    classification: WorkoutClassification,
) -> PaceEvaluation:
    """
    Evaluate actual pace against the target for a classified workout.

    For easy runs (workout_type='easy' or None with no target):
      - Checks actual pace is above EASY_PACE_CEILING
      - No on_target bool — effort-based

    For structured workouts with a target:
      - on_target = True if within PACE_TOLERANCE_SEC seconds of target
      - diff_seconds = positive means slower than target

    Returns a PaceEvaluation with a human-readable verdict.
    """
    activity_id  = workout.get("activity_id", "unknown")
    date         = workout.get("date", "")
    actual_pace  = workout.get("avg_pace_min_km")
    workout_type = classification.workout_type
    target_pace  = classification.target_pace

    actual_fmt = format_pace(actual_pace)

    # ── Easy run ──────────────────────────────────────────────────────────────
    if workout_type == "easy" or (workout_type is None and not classification.is_structured):
        if actual_pace is None:
            verdict = "No pace data available."
        elif actual_pace >= EASY_PACE_CEILING:
            verdict = (
                f"Easy run pace of {actual_fmt} is within easy territory "
                f"(above {format_pace(EASY_PACE_CEILING)} ceiling). ✅"
            )
        else:
            diff = (EASY_PACE_CEILING - actual_pace) * 60
            verdict = (
                f"Easy run pace of {actual_fmt} is {diff:.0f}s/km faster than "
                f"the easy ceiling of {format_pace(EASY_PACE_CEILING)}. "
                f"Consider whether effort felt genuinely easy. ⚠️"
            )
        return PaceEvaluation(
            activity_id=activity_id,
            date=date,
            workout_type=workout_type or "easy",
            target_pace=None,
            target_pace_fmt="by feel",
            actual_pace=actual_pace,
            actual_pace_fmt=actual_fmt,
            diff_seconds=None,
            on_target=None,
            verdict=verdict,
        )

    # ── Structured workout with target ────────────────────────────────────────
    if target_pace is None or actual_pace is None:
        return PaceEvaluation(
            activity_id=activity_id,
            date=date,
            workout_type=workout_type or "unknown",
            target_pace=target_pace,
            target_pace_fmt=format_pace(target_pace),
            actual_pace=actual_pace,
            actual_pace_fmt=actual_fmt,
            diff_seconds=None,
            on_target=None,
            verdict="Cannot evaluate — target pace or actual pace is missing.",
        )

    diff_seconds = (actual_pace - target_pace) * 60
    on_target    = abs(diff_seconds) <= PACE_TOLERANCE_SEC
    target_fmt   = format_pace(target_pace)
    diff_str     = _seconds_to_pace_str(diff_seconds)

    if on_target:
        verdict = (
            f"{workout_type.capitalize()} target: {target_fmt}. "
            f"Actual: {actual_fmt} ({diff_str}). On target. ✅"
        )
    elif diff_seconds > 0:
        verdict = (
            f"{workout_type.capitalize()} target: {target_fmt}. "
            f"Actual: {actual_fmt} ({diff_str}). "
            f"Slightly under — check conditions, fatigue, or HR data. ⚠️"
        )
    else:
        verdict = (
            f"{workout_type.capitalize()} target: {target_fmt}. "
            f"Actual: {actual_fmt} ({diff_str}). "
            f"Faster than target — assess whether effort was controlled. ⚠️"
        )

    return PaceEvaluation(
        activity_id=activity_id,
        date=date,
        workout_type=workout_type,
        target_pace=target_pace,
        target_pace_fmt=target_fmt,
        actual_pace=actual_pace,
        actual_pace_fmt=actual_fmt,
        diff_seconds=round(diff_seconds, 1),
        on_target=on_target,
        verdict=verdict,
    )


# ── Batch classification ──────────────────────────────────────────────────────

def classify_workouts(workouts: list[dict]) -> list[dict]:
    """
    Run detect_workout_type and evaluate_pace over a list of WorkoutRecords.
    Returns enriched dicts with 'classification' and 'pace_evaluation' keys added.
    Does not modify the original dicts.
    """
    enriched = []
    for w in workouts:
        record = w.copy()
        classification = detect_workout_type(w)
        record["classification"] = {
            "workout_type":  classification.workout_type,
            "is_structured": classification.is_structured,
            "needs_input":   classification.needs_input,
            "detected_from": classification.detected_from,
            "notes":         classification.notes,
        }

        # Only evaluate pace if type is resolved
        if not classification.needs_input:
            evaluation = evaluate_pace(w, classification)
            record["pace_evaluation"] = {
                "workout_type":   evaluation.workout_type,
                "target_pace":    evaluation.target_pace,
                "target_pace_fmt":evaluation.target_pace_fmt,
                "actual_pace":    evaluation.actual_pace,
                "actual_pace_fmt":evaluation.actual_pace_fmt,
                "diff_seconds":   evaluation.diff_seconds,
                "on_target":      evaluation.on_target,
                "verdict":        evaluation.verdict,
            }
        else:
            record["pace_evaluation"] = None

        enriched.append(record)

    return enriched


# ── Quick test (run as script) ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from parse_workout_data import load_workouts

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/content/drive/MyDrive/running_coach/data/processed/workouts_normalized.json"
    )

    workouts = load_workouts(path, days=90)
    enriched = classify_workouts(workouts)

    structured   = [w for w in enriched if w["classification"]["is_structured"]]
    needs_input  = [w for w in enriched if w["classification"]["needs_input"]]
    exceptions   = [w for w in enriched if w["classification"]["detected_from"] == "known_exception"]

    print(f"Classification summary — {len(enriched)} workouts\n")
    print(f"  Structured (Garmin date-named): {len(structured)}")
    print(f"  Known exceptions:               {len(exceptions)}")
    print(f"  Needs user input:               {len(needs_input)}")

    print("\n--- Structured workouts (pace evaluation) ---")
    for w in structured[:5]:
        pe = w.get("pace_evaluation")
        if pe:
            print(f"\n  {w['date']} | {w['name']}")
            print(f"  {pe['verdict']}")

    print("\n--- Needs user input (sample) ---")
    for w in needs_input[:3]:
        print(f"  {w['date']} | {w['name']} | {format_pace(w.get('avg_pace_min_km'))}")
