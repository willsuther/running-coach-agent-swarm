"""
tools/query_garmin_db.py

Reusable query functions for the Garmin SQLite database.
Provides clean, typed access to the health and recovery tables
that the Recovery, Planner, and Feedback agents depend on.

All functions accept a sqlite3.Connection object so the connection
is managed by the caller (typically opened once per agent session).

Table coverage
--------------
hrv               — HRV status, weekly avg, last night value, baseline
sleep             — Sleep stages, SpO2, stress, feedback
body_battery      — Charge/drain, wake value, daily min/max
stress            — Daily avg/max stress, qualifier
training_readiness — Readiness score, level, factor breakdown
heart_rate        — Resting HR, daily min/max avg
race_predictions  — Predicted 5K/10K/HM/marathon times

Public API
----------
get_hrv(conn, days)                  -> list[HRVRecord]
get_sleep(conn, days)                -> list[SleepRecord]
get_body_battery(conn, days)         -> list[BodyBatteryRecord]
get_stress(conn, days)               -> list[StressRecord]
get_training_readiness(conn, days)   -> list[ReadinessRecord]
get_resting_hr(conn, days)           -> list[RestingHRRecord]
get_race_predictions(conn, n)        -> RacePredictions | None
get_daily_snapshot(conn, date)       -> DailySnapshot
get_recent_snapshot(conn, days)      -> list[DailySnapshot]
format_recovery_context(snapshots)   -> str   (for agent prompt injection)
format_race_predictions(pred)        -> str
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class HRVRecord:
    date:            str
    weekly_avg:      float | None
    last_night:      float | None
    status:          str | None       # e.g. 'BALANCED', 'UNBALANCED'
    baseline_low:    float | None
    baseline_upper:  float | None

    @property
    def within_baseline(self) -> bool | None:
        """True if last_night HRV is within the established baseline range."""
        if any(v is None for v in [self.last_night, self.baseline_low, self.baseline_upper]):
            return None  # Cannot determine — missing data
        return self.baseline_low <= self.last_night <= self.baseline_upper

    @property
    def status_label(self) -> str:
        if self.status == "BALANCED":
            return "Balanced ✅"
        elif self.status == "UNBALANCED":
            return "Unbalanced ⚠️"
        elif self.status == "LOW":
            return "Low ⚠️"
        return self.status or "Unknown"


@dataclass
class SleepRecord:
    date:                str
    total_sleep_hrs:     float | None
    deep_sleep_hrs:      float | None
    rem_sleep_hrs:       float | None
    light_sleep_hrs:     float | None
    awake_hrs:           float | None
    average_spo2:        float | None
    avg_sleep_stress:    float | None
    sleep_score_feedback: str | None

    @property
    def sleep_quality_label(self) -> str:
        """Rough quality label based on total sleep and deep/REM proportions."""
        if self.total_sleep_hrs is None:
            return "No data"
        if self.total_sleep_hrs >= 7.5:
            base = "Good duration"
        elif self.total_sleep_hrs >= 6.5:
            base = "Adequate duration"
        else:
            base = "Short sleep ⚠️"

        if self.deep_sleep_hrs is not None and self.total_sleep_hrs > 0:
            deep_pct = self.deep_sleep_hrs / self.total_sleep_hrs * 100
            if deep_pct < 10:
                return f"{base}, low deep sleep ⚠️"
        return base


@dataclass
class BodyBatteryRecord:
    date:        str
    charged:     float | None     # Total charged during day
    drained:     float | None     # Total drained during day
    highest:     float | None     # Peak value
    lowest:      float | None     # Lowest value
    at_wake:     float | None     # Value on waking — key recovery indicator

    @property
    def wake_label(self) -> str:
        if self.at_wake is None:
            return "No data"
        if self.at_wake >= 70:
            return "Well recovered ✅"
        elif self.at_wake >= 50:
            return "Adequately recovered"
        elif self.at_wake >= 30:
            return "Partially recovered ⚠️"
        return "Poorly recovered ⚠️"


@dataclass
class StressRecord:
    date:             str
    avg_stress:       float | None
    max_stress:       float | None
    stress_qualifier: str | None    # e.g. 'CALM', 'BALANCED', 'STRESSFUL'

    @property
    def stress_label(self) -> str:
        q = (self.stress_qualifier or "").upper()
        labels = {
            "CALM":       "Calm ✅",
            "BALANCED":   "Balanced",
            "STRESSFUL":  "Elevated stress ⚠️",
            "VERY_STRESSFUL": "High stress ⚠️",
        }
        return labels.get(q, self.stress_qualifier or "Unknown")


@dataclass
class ReadinessRecord:
    date:                          str
    score:                         float | None
    level:                         str | None      # e.g. 'HIGH', 'MODERATE', 'LOW'
    feedback_short:                str | None
    recovery_time:                 float | None    # Hours remaining recovery time
    hrv_factor_feedback:           str | None
    sleep_history_factor_feedback: str | None

    @property
    def readiness_label(self) -> str:
        if self.score is None:
            return "No data"
        if self.score >= 80:
            return "High ✅"
        elif self.score >= 60:
            return "Moderate"
        elif self.score >= 40:
            return "Low ⚠️"
        return "Very low ⚠️"


@dataclass
class RestingHRRecord:
    date:       str
    resting_hr: float | None
    min_hr:     float | None
    max_hr:     float | None
    avg_hr:     float | None


@dataclass
class RacePredictions:
    date:               str
    time_5k:            str | None
    time_10k:           str | None
    time_half_marathon: str | None
    time_marathon:      str | None


@dataclass
class DailySnapshot:
    """
    Full daily health and recovery context for a single date.
    This is the primary object injected into agent prompts.
    """
    date:         str
    hrv:          HRVRecord | None          = None
    sleep:        SleepRecord | None        = None
    body_battery: BodyBatteryRecord | None  = None
    stress:       StressRecord | None       = None
    readiness:    ReadinessRecord | None    = None
    resting_hr:   RestingHRRecord | None    = None

    @property
    def recovery_score(self) -> float | None:
        """
        Simple composite recovery score (0-100) from available signals.
        Weights: readiness 40%, body battery at wake 30%, HRV status 30%.
        Returns None if insufficient data.
        """
        scores = []
        weights = []

        if self.readiness and self.readiness.score is not None:
            scores.append(self.readiness.score)
            weights.append(0.40)

        if self.body_battery and self.body_battery.at_wake is not None:
            scores.append(self.body_battery.at_wake)
            weights.append(0.30)

        if self.hrv:
            if self.hrv.status == "BALANCED":
                scores.append(80.0)
                weights.append(0.30)
            elif self.hrv.status in ("UNBALANCED", "LOW"):
                scores.append(40.0)
                weights.append(0.30)

        if not scores:
            return None

        total_weight = sum(weights)
        return round(sum(s * w for s, w in zip(scores, weights)) / total_weight, 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_range(days: int) -> tuple[str, str]:
    """Return (start_date, end_date) strings for a trailing N-day window."""
    end   = date.today()
    start = end - timedelta(days=days - 1)
    return start.isoformat(), end.isoformat()


def _safe_div(seconds: float | None, divisor: float) -> float | None:
    """Safely divide a value, returning None if input is None."""
    if seconds is None:
        return None
    return round(seconds / divisor, 2)


def _fmt_race_time(seconds: float | None) -> str:
    """Convert seconds to H:MM:SS race time string. e.g. 5016 -> '1:23:36'"""
    if seconds is None:
        return "N/A"
    try:
        s = int(float(seconds))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h > 0:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"
    except (TypeError, ValueError):
        return str(seconds)


def _cap_recovery_time(hours: float | None, max_hours: float = 72.0) -> float | None:
    """Cap recovery time at max_hours — values above this are Garmin data artefacts."""
    if hours is None:
        return None
    return hours if hours <= max_hours else None
    """Return (start_date, end_date) strings for a trailing N-day window."""
    end   = date.today()
    start = end - timedelta(days=days - 1)
    return start.isoformat(), end.isoformat()


# ── Query functions ───────────────────────────────────────────────────────────

def get_hrv(conn: sqlite3.Connection, days: int = 14) -> list[HRVRecord]:
    """Return HRV records for the last N days, most recent first.
    Parses lastNightAvg from raw_json since the last_night column is unpopulated."""
    import json as _json
    start, end = _date_range(days)
    rows = conn.execute("""
        SELECT calendar_date, weekly_avg, last_night, status,
               baseline_low, baseline_upper, raw_json
        FROM hrv
        WHERE calendar_date BETWEEN ? AND ?
        ORDER BY calendar_date DESC
    """, (start, end)).fetchall()

    records = []
    for r in rows:
        last_night = r[2]
        # Parse lastNightAvg from raw_json if last_night column is null
        if last_night is None and r[6]:
            try:
                raw = _json.loads(r[6])
                last_night = raw.get('hrvSummary', {}).get('lastNightAvg')
            except Exception:
                pass

        # Parse baseline from raw_json if columns are null
        baseline_low   = r[4]
        baseline_upper = r[5]
        if (baseline_low is None or baseline_upper is None) and r[6]:
            try:
                raw = _json.loads(r[6]) if not isinstance(r[6], dict) else r[6]
                baseline = raw.get('hrvSummary', {}).get('baseline', {})
                baseline_low   = baseline_low   or baseline.get('balancedLow')
                baseline_upper = baseline_upper or baseline.get('balancedUpper')
            except Exception:
                pass

        records.append(HRVRecord(
            date=r[0], weekly_avg=r[1], last_night=last_night,
            status=r[3], baseline_low=baseline_low, baseline_upper=baseline_upper,
        ))
    return records


def get_sleep(conn: sqlite3.Connection, days: int = 14) -> list[SleepRecord]:
    """Return sleep records for the last N days, most recent first."""
    start, end = _date_range(days)
    rows = conn.execute("""
        SELECT calendar_date,
               sleep_time_seconds, deep_sleep_seconds, rem_sleep_seconds,
               light_sleep_seconds, awake_sleep_seconds,
               average_spo2, avg_sleep_stress, sleep_score_feedback
        FROM sleep
        WHERE calendar_date BETWEEN ? AND ?
        ORDER BY calendar_date DESC
    """, (start, end)).fetchall()

    return [
        SleepRecord(
            date=r[0],
            total_sleep_hrs=_safe_div(r[1], 3600),
            deep_sleep_hrs=_safe_div(r[2], 3600),
            rem_sleep_hrs=_safe_div(r[3], 3600),
            light_sleep_hrs=_safe_div(r[4], 3600),
            awake_hrs=_safe_div(r[5], 3600),
            average_spo2=r[6],
            avg_sleep_stress=r[7],
            sleep_score_feedback=r[8],
        )
        for r in rows
    ]


def get_body_battery(conn: sqlite3.Connection, days: int = 14) -> list[BodyBatteryRecord]:
    """Return body battery records for the last N days, most recent first.
    Parses highest, lowest, and at_wake from raw_json since structured columns
    are unpopulated in garmin-givemydata output."""
    import json as _json
    start, end = _date_range(days)
    rows = conn.execute("""
        SELECT calendar_date, charged, drained, highest, lowest, at_wake, raw_json
        FROM body_battery
        WHERE calendar_date BETWEEN ? AND ?
        ORDER BY calendar_date DESC
    """, (start, end)).fetchall()

    records = []
    for r in rows:
        highest, lowest, at_wake = r[3], r[4], r[5]

        # Parse from raw_json if structured columns are null
        if any(v is None for v in [highest, lowest, at_wake]) and r[6]:
            try:
                raw = _json.loads(r[6])
                bb_data = raw.get('bodyBattery', {}).get('data', [])
                if bb_data:
                    levels = [entry[1] for entry in bb_data if len(entry) > 1]
                    if levels:
                        highest = highest or max(levels)
                        lowest  = lowest  or min(levels)
                        at_wake = at_wake or levels[0]
            except Exception:
                pass

        # Fallback: pull body battery from stress table raw_json (bodyBatteryValuesArray)
        if any(v is None for v in [highest, lowest, at_wake]):
            try:
                stress_row = conn.execute("""
                    SELECT raw_json FROM stress
                    WHERE calendar_date = ?
                """, (r[0],)).fetchone()
                if stress_row and stress_row[0]:
                    raw = _json.loads(stress_row[0])
                    bb_array = raw.get('bodyBatteryValuesArray', [])
                    if bb_array:
                        # Each entry: [timestamp_ms, status, level, version]
                        levels = [entry[2] for entry in bb_array
                                  if len(entry) > 2 and entry[1] == 'MEASURED']
                        if levels:
                            highest = highest or max(levels)
                            lowest  = lowest  or min(levels)
                            at_wake = at_wake or levels[0]
            except Exception:
                pass

        records.append(BodyBatteryRecord(
            date=r[0], charged=r[1], drained=r[2],
            highest=highest, lowest=lowest, at_wake=at_wake,
        ))
    return records


def get_stress(conn: sqlite3.Connection, days: int = 14) -> list[StressRecord]:
    """Return stress records for the last N days, most recent first.
    Parses avgStressLevel from raw_json since avg_stress column is unpopulated.
    Note: the stress table raw_json also contains body battery time series data
    which is used as a fallback by get_body_battery."""
    import json as _json
    start, end = _date_range(days)
    rows = conn.execute("""
        SELECT calendar_date, avg_stress, max_stress, stress_qualifier, raw_json
        FROM stress
        WHERE calendar_date BETWEEN ? AND ?
        ORDER BY calendar_date DESC
    """, (start, end)).fetchall()

    records = []
    for r in rows:
        avg_stress       = r[1]
        stress_qualifier = r[3]

        # Parse avgStressLevel from raw_json if column is null
        if avg_stress is None and r[4]:
            try:
                raw = _json.loads(r[4])
                avg_stress = raw.get('avgStressLevel')
            except Exception:
                pass

        records.append(StressRecord(
            date=r[0], avg_stress=avg_stress,
            max_stress=r[2], stress_qualifier=stress_qualifier,
        ))
    return records


def get_training_readiness(
    conn: sqlite3.Connection, days: int = 14
) -> list[ReadinessRecord]:
    """Return training readiness records for the last N days, most recent first."""
    start, end = _date_range(days)
    rows = conn.execute("""
        SELECT calendar_date, score, level, feedback_short,
               recovery_time, hrv_factor_feedback,
               sleep_history_factor_feedback
        FROM training_readiness
        WHERE calendar_date BETWEEN ? AND ?
        ORDER BY calendar_date DESC
    """, (start, end)).fetchall()

    return [
        ReadinessRecord(
            date=r[0], score=r[1], level=r[2], feedback_short=r[3],
            recovery_time=_cap_recovery_time(r[4]),
            hrv_factor_feedback=r[5],
            sleep_history_factor_feedback=r[6],
        )
        for r in rows
    ]


def get_resting_hr(conn: sqlite3.Connection, days: int = 14) -> list[RestingHRRecord]:
    """Return resting HR records for the last N days, most recent first."""
    start, end = _date_range(days)
    rows = conn.execute("""
        SELECT calendar_date, resting_hr, min_hr, max_hr, avg_hr
        FROM heart_rate
        WHERE calendar_date BETWEEN ? AND ?
        ORDER BY calendar_date DESC
    """, (start, end)).fetchall()

    return [
        RestingHRRecord(
            date=r[0], resting_hr=r[1],
            min_hr=r[2], max_hr=r[3], avg_hr=r[4],
        )
        for r in rows
    ]


def get_activity_splits(
    conn: sqlite3.Connection,
    activity_id: int | str,
    min_distance_m: float = 50.0,
) -> list[dict]:
    """
    Return lap splits for a Garmin activity, formatted for agent context.

    Filters out micro-splits below min_distance_m (transition laps, pauses).
    Pace is formatted as mm:ss/km. Returns list of dicts sorted by split_number.

    Parameters
    ----------
    activity_id     : Garmin activity ID (integer or string)
    min_distance_m  : Minimum split distance to include (default 50m)
    """
    rows = conn.execute("""
        SELECT split_number, distance_meters, duration_seconds,
               average_hr, max_hr, elevation_gain
        FROM activity_splits
        WHERE activity_id = ?
          AND distance_meters >= ?
        ORDER BY split_number
    """, (str(activity_id), min_distance_m)).fetchall()

    splits = []
    for r in rows:
        split_num, dist_m, dur_s, avg_hr, max_hr, elev = r
        dist_m  = dist_m  or 0.0
        dur_s   = dur_s   or 0.0
        dist_km = dist_m / 1000.0

        if dist_km > 0 and dur_s > 0:
            pace_dec = (dur_s / 60.0) / dist_km
            mins = int(pace_dec)
            secs = round((pace_dec - mins) * 60)
            if secs == 60:
                mins += 1
                secs = 0
            pace_fmt = f"{mins}:{secs:02d}/km"
        else:
            pace_fmt = "--:--/km"

        splits.append({
            'split_number': split_num,
            'distance_m':   round(dist_m, 0),
            'duration_s':   round(dur_s, 1),
            'pace_fmt':     pace_fmt,
            'avg_hr':       avg_hr,
            'max_hr':       max_hr,
            'elevation_m':  elev,
        })

    return splits


def format_splits(splits: list[dict]) -> str:
    """
    Format activity splits as a plain-text table for agent prompt injection.

    Automatically detects workout structure by flagging splits that are
    significantly faster than the session average (likely hard efforts).
    """
    if not splits:
        return "No split data available."

    # Compute session average pace from full-km splits only
    full_km = [s for s in splits if s['distance_m'] >= 900]
    if full_km:
        avg_dur_per_km = sum(
            s['duration_s'] / (s['distance_m'] / 1000) for s in full_km
        ) / len(full_km)
        avg_pace_s_per_km = avg_dur_per_km
    else:
        avg_pace_s_per_km = None

    lines = ["Split  Distance    Duration  Pace       Avg HR  Max HR  Elev   Note"]
    lines.append("-" * 72)

    for s in splits:
        dist_m  = s['distance_m']
        dur_s   = s['duration_s']
        dist_km = dist_m / 1000.0

        # Flag hard vs easy efforts vs the session average
        note = ""
        if avg_pace_s_per_km and dist_km > 0 and dur_s > 0:
            split_s_per_km = dur_s / dist_km
            diff = avg_pace_s_per_km - split_s_per_km  # positive = faster than avg
            if diff > 15:
                note = "⚡ hard"
            elif diff < -15:
                note = "🔄 easy"

        dist_str = f"{dist_m:.0f}m" if dist_m < 1000 else f"{dist_m/1000:.2f}km"
        dur_str  = f"{int(dur_s//60)}:{int(dur_s%60):02d}"

        lines.append(
            f"  {s['split_number']:<5} {dist_str:<10}  {dur_str:<8}  "
            f"{s['pace_fmt']:<10} {str(s['avg_hr']):<7} {str(s['max_hr']):<7} "
            f"{str(s['elevation_m'] or 0):<6} {note}"
        )

    return "\n".join(lines)


def get_race_predictions(
    conn: sqlite3.Connection, n: int = 1
) -> RacePredictions | None:
    """Return the most recent race prediction record."""
    rows = conn.execute("""
        SELECT calendar_date, time_5k, time_10k,
               time_half_marathon, time_marathon
        FROM race_predictions
        ORDER BY calendar_date DESC
        LIMIT ?
    """, (n,)).fetchall()

    if not rows:
        return None

    r = rows[0]
    return RacePredictions(
        date=r[0], time_5k=r[1], time_10k=r[2],
        time_half_marathon=r[3], time_marathon=r[4],
    )


# ── Snapshot assembly ─────────────────────────────────────────────────────────

def get_daily_snapshot(
    conn: sqlite3.Connection,
    target_date: str,
) -> DailySnapshot:
    """
    Assemble a full DailySnapshot for a single date by querying
    all health tables. Returns a DailySnapshot with None fields
    where data is not available for that date.
    """
    def first_or_none(records: list, target: str):
        for r in records:
            if r.date == target:
                return r
        return None

    # Query a small window around the target date
    hrv_records   = get_hrv(conn, days=3)
    sleep_records = get_sleep(conn, days=3)
    bb_records    = get_body_battery(conn, days=3)
    stress_records = get_stress(conn, days=3)
    ready_records  = get_training_readiness(conn, days=3)
    hr_records     = get_resting_hr(conn, days=3)

    return DailySnapshot(
        date=target_date,
        hrv=first_or_none(hrv_records, target_date),
        sleep=first_or_none(sleep_records, target_date),
        body_battery=first_or_none(bb_records, target_date),
        stress=first_or_none(stress_records, target_date),
        readiness=first_or_none(ready_records, target_date),
        resting_hr=first_or_none(hr_records, target_date),
    )


def get_recent_snapshots(
    conn: sqlite3.Connection,
    days: int = 7,
) -> list[DailySnapshot]:
    """
    Return DailySnapshots for each of the last N days.
    Most recent first. Days with missing data have None fields.
    """
    hrv_records    = get_hrv(conn, days=days)
    sleep_records  = get_sleep(conn, days=days)
    bb_records     = get_body_battery(conn, days=days)
    stress_records = get_stress(conn, days=days)
    ready_records  = get_training_readiness(conn, days=days)
    hr_records     = get_resting_hr(conn, days=days)

    # Index each by date
    def by_date(records):
        return {r.date: r for r in records}

    hrv_map     = by_date(hrv_records)
    sleep_map   = by_date(sleep_records)
    bb_map      = by_date(bb_records)
    stress_map  = by_date(stress_records)
    ready_map   = by_date(ready_records)
    hr_map      = by_date(hr_records)

    snapshots = []
    for i in range(days):
        d = (date.today() - timedelta(days=i)).isoformat()
        snapshots.append(DailySnapshot(
            date=d,
            hrv=hrv_map.get(d),
            sleep=sleep_map.get(d),
            body_battery=bb_map.get(d),
            stress=stress_map.get(d),
            readiness=ready_map.get(d),
            resting_hr=hr_map.get(d),
        ))

    return snapshots


# ── Agent-ready formatting ────────────────────────────────────────────────────

def format_recovery_context(
    snapshots: list[DailySnapshot],
    include_days: int = 3,
) -> str:
    """
    Format recent DailySnapshots into a concise plain-text block
    for injection into Recovery Agent or Coordinator prompts.

    Only includes the most recent `include_days` snapshots.
    """
    lines = ["=== Recovery Context ==="]

    for snap in snapshots[:include_days]:
        lines.append(f"\n--- {snap.date} ---")

        # Composite score
        score = snap.recovery_score
        if score is not None:
            lines.append(f"  Recovery score (composite): {score:.0f}/100")

        # Readiness
        if snap.readiness:
            r = snap.readiness
            lines.append(
                f"  Training readiness: {r.score} — {r.readiness_label}"
            )
            if r.feedback_short:
                lines.append(f"    Feedback: {r.feedback_short}")
            if r.recovery_time:
                lines.append(f"    Recovery time remaining: {r.recovery_time:.0f}h")
            if r.hrv_factor_feedback:
                lines.append(f"    HRV factor: {r.hrv_factor_feedback}")
            if r.sleep_history_factor_feedback:
                lines.append(f"    Sleep history: {r.sleep_history_factor_feedback}")

        # HRV
        if snap.hrv:
            h = snap.hrv
            if h.within_baseline is True:
                baseline_str = "within baseline ✅"
            elif h.within_baseline is False:
                baseline_str = "outside baseline ⚠️"
            else:
                baseline_str = "baseline N/A"
            lines.append(
                f"  HRV: {h.last_night} (weekly avg: {h.weekly_avg}) "
                f"— {h.status_label}, {baseline_str}"
            )

        # Sleep
        if snap.sleep:
            s = snap.sleep
            lines.append(
                f"  Sleep: {s.total_sleep_hrs}h total "
                f"(deep: {s.deep_sleep_hrs}h, REM: {s.rem_sleep_hrs}h) "
                f"— {s.sleep_quality_label}"
            )
            if s.sleep_score_feedback:
                lines.append(f"    Feedback: {s.sleep_score_feedback}")
            if s.average_spo2:
                lines.append(f"    SpO2: {s.average_spo2}%")

        # Body battery
        if snap.body_battery:
            b = snap.body_battery
            lines.append(
                f"  Body battery at wake: {b.at_wake} — {b.wake_label} "
                f"(peak: {b.highest}, low: {b.lowest})"
            )

        # Stress
        if snap.stress:
            st = snap.stress
            lines.append(
                f"  Stress: avg {st.avg_stress}, max {st.max_stress} "
                f"— {st.stress_label}"
            )

        # Resting HR
        if snap.resting_hr:
            lines.append(f"  Resting HR: {snap.resting_hr.resting_hr} bpm")

    return "\n".join(lines)


def format_race_predictions(pred: RacePredictions) -> str:
    """Format race predictions for agent context injection."""
    if pred is None:
        return "No race predictions available."

    return (
        f"=== Garmin Race Predictions ({pred.date}) ===\n"
        f"  5K:             {_fmt_race_time(pred.time_5k)}\n"
        f"  10K:            {_fmt_race_time(pred.time_10k)}\n"
        f"  Half Marathon:  {_fmt_race_time(pred.time_half_marathon)}\n"
        f"  Marathon:       {_fmt_race_time(pred.time_marathon)}"
    )


# ── Quick test (run as script) ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/content/drive/MyDrive/running_coach/data/raw/garmin/garmin.db"
    )

    conn = sqlite3.connect(db_path)
    print(f"Connected to {db_path}\n")

    print("--- Recent snapshots (last 3 days) ---")
    snapshots = get_recent_snapshots(conn, days=3)
    print(format_recovery_context(snapshots, include_days=3))

    print("\n--- Race predictions ---")
    pred = get_race_predictions(conn)
    print(format_race_predictions(pred))

    print("\n--- HRV trend (14 days) ---")
    hrv = get_hrv(conn, days=14)
    print(f"  {'Date':12} {'Last Night':>12} {'Weekly Avg':>12} {'Status':>15} {'In Baseline':>12}")
    print("  " + "-" * 65)
    for h in hrv:
        within = "Yes" if h.within_baseline else "No" if h.within_baseline is False else "N/A"
        print(
            f"  {h.date:12} {str(h.last_night):>12} {str(h.weekly_avg):>12} "
            f"{str(h.status):>15} {within:>12}"
        )

    print("\n--- Sleep trend (7 days) ---")
    sleep = get_sleep(conn, days=7)
    print(f"  {'Date':12} {'Total':>7} {'Deep':>6} {'REM':>6} {'SpO2':>6} {'Quality'}")
    print("  " + "-" * 60)
    for s in sleep:
        print(
            f"  {s.date:12} {str(s.total_sleep_hrs):>7} {str(s.deep_sleep_hrs):>6} "
            f"{str(s.rem_sleep_hrs):>6} {str(s.average_spo2):>6} {s.sleep_quality_label}"
        )

    print("\n--- Body battery (7 days) ---")
    bb = get_body_battery(conn, days=7)
    print(f"  {'Date':12} {'At Wake':>8} {'Highest':>8} {'Lowest':>8} {'Status'}")
    print("  " + "-" * 55)
    for b in bb:
        print(
            f"  {b.date:12} {str(b.at_wake):>8} {str(b.highest):>8} "
            f"{str(b.lowest):>8} {b.wake_label}"
        )

    conn.close()
    print("\n✅ query_garmin_db smoke test complete.")
