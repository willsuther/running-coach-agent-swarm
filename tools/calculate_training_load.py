"""
tools/calculate_training_load.py

Compute ATL, CTL, and TSB (the Performance Management Chart model) from
WorkoutRecord training_load values sourced from Garmin.

Model
-----
ATL (Acute Training Load)   — 7-day exponentially weighted average of daily load.
                              Represents short-term fatigue.
CTL (Chronic Training Load) — 42-day exponentially weighted average of daily load.
                              Represents long-term fitness.
TSB (Training Stress Balance) — CTL minus ATL.
                              Positive = fresh/recovered. Negative = fatigued.
                              Alert threshold: TSB < -30 (config: TSB_ALERT_THRESHOLD)

Input
-----
Uses Garmin's `training_load` field from WorkoutRecords (populated during Phase 1
enrichment from the Garmin `activity` table). This is Garmin's proprietary per-session
load score based on HR, duration, and Training Effect — more accurate than a proxy.

On days with no recorded load (rest days, or sessions without Garmin enrichment),
load is treated as 0.

Decay formula
-------------
ATL_today = ATL_yesterday * exp(-1 / ATL_DAYS) + load_today * (1 - exp(-1 / ATL_DAYS))
CTL_today = CTL_yesterday * exp(-1 / CTL_DAYS) + load_today * (1 - exp(-1 / CTL_DAYS))

Public API
----------
compute_daily_load(workouts)                -> dict[str, float]   date -> load
compute_atl_ctl_tsb(daily_load, start_date, end_date)
                                            -> list[DailyMetrics]
get_current_metrics(workouts)               -> DailyMetrics
get_metrics_for_date(workouts, date)        -> DailyMetrics
get_recent_trend(workouts, days)            -> TrendSummary
check_recovery_alert(metrics)               -> RecoveryAlert | None
weekly_load_summary(workouts, weeks)        -> list[WeekSummary]
format_metrics_report(metrics, trend)       -> str   (for agent context)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any


# ── Config (mirrors config.py — keep in sync) ─────────────────────────────────

ATL_DAYS             = 7     # Acute Training Load decay constant
CTL_DAYS             = 42    # Chronic Training Load decay constant
TSB_ALERT_THRESHOLD  = -30   # [RECOVERY ALERT] when TSB drops below this
MILEAGE_RULE_PCT     = 0.10  # 10% weekly mileage increase cap

# Exponential decay factors (pre-computed)
ATL_DECAY = math.exp(-1 / ATL_DAYS)
CTL_DECAY = math.exp(-1 / CTL_DAYS)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DailyMetrics:
    """ATL, CTL, TSB for a single day."""
    date:        str
    load:        float          # Training load for this day (0 = rest)
    atl:         float          # Acute Training Load (fatigue)
    ctl:         float          # Chronic Training Load (fitness)
    tsb:         float          # Training Stress Balance (form)
    in_alert:    bool           # True if TSB < TSB_ALERT_THRESHOLD

    @property
    def form_label(self) -> str:
        """Human-readable form description based on TSB."""
        if self.tsb >= 15:
            return "Very fresh — peak form"
        elif self.tsb >= 5:
            return "Fresh — good form"
        elif self.tsb >= -5:
            return "Neutral — balanced"
        elif self.tsb >= -15:
            return "Slightly fatigued — normal training stress"
        elif self.tsb >= -30:
            return "Fatigued — monitor recovery"
        else:
            return "Highly fatigued — recovery priority ⚠️"

    @property
    def fitness_label(self) -> str:
        """Human-readable fitness level based on CTL."""
        if self.ctl >= 80:
            return "High fitness"
        elif self.ctl >= 60:
            return "Good fitness"
        elif self.ctl >= 40:
            return "Moderate fitness"
        elif self.ctl >= 20:
            return "Building fitness"
        else:
            return "Early base"


@dataclass
class TrendSummary:
    """Trend in ATL/CTL/TSB over a recent window."""
    days:            int
    start_date:      str
    end_date:        str
    atl_start:       float
    atl_end:         float
    ctl_start:       float
    ctl_end:         float
    tsb_start:       float
    tsb_end:         float
    atl_trend:       str        # 'rising' | 'falling' | 'stable'
    ctl_trend:       str
    tsb_trend:       str
    total_load:      float
    avg_daily_load:  float
    peak_load:       float
    rest_days:       int


@dataclass
class RecoveryAlert:
    """Triggered when TSB drops below TSB_ALERT_THRESHOLD."""
    date:            str
    tsb:             float
    atl:             float
    ctl:             float
    consecutive_days: int       # How many days TSB has been below threshold
    message:         str


@dataclass
class WeekSummary:
    """Training load summary for a single week."""
    week_start:      str        # Monday date
    week_end:        str        # Sunday date
    total_load:      float
    session_count:   int
    rest_days:       int
    avg_atl:         float
    avg_ctl:         float
    avg_tsb:         float
    pct_change_load: float | None   # vs previous week; None for first week


# ── Module-level flag — suppresses repeated skip messages ─────────────────────
_skip_message_shown = False


# ── Daily load computation ────────────────────────────────────────────────────

def compute_daily_load(workouts: list[dict]) -> dict[str, float]:
    """
    Aggregate WorkoutRecord training_load values by date.

    Days with multiple sessions have their loads summed.
    Days with no sessions or missing training_load are not included
    (treated as 0 in the ATL/CTL/TSB calculation).

    Only uses Garmin-enriched records for load — records without
    training_load are skipped with a warning.

    Returns dict of {date_str: total_load}.
    """
    daily: dict[str, float] = {}
    skipped = 0

    for w in workouts:
        date_str = w.get("date")
        load     = w.get("training_load")

        if not date_str:
            continue

        if load is None:
            skipped += 1
            continue

        try:
            load = float(load)
        except (TypeError, ValueError):
            skipped += 1
            continue

        daily[date_str] = daily.get(date_str, 0.0) + load

    global _skip_message_shown
    if skipped > 0 and not _skip_message_shown:
        print(
            f"  ℹ️  {skipped} record(s) skipped (no training_load — not Garmin-enriched). "
            "Treated as 0 load."
        )
        _skip_message_shown = True

    return daily


# ── ATL / CTL / TSB computation ───────────────────────────────────────────────

def compute_atl_ctl_tsb(
    daily_load: dict[str, float],
    start_date: str,
    end_date:   str,
    seed_atl:   float = 0.0,
    seed_ctl:   float = 0.0,
) -> list[DailyMetrics]:
    """
    Compute ATL, CTL, TSB for every day in [start_date, end_date].

    Parameters
    ----------
    daily_load  : Output of compute_daily_load()
    start_date  : YYYY-MM-DD — first day of computation window
    end_date    : YYYY-MM-DD — last day (inclusive)
    seed_atl    : Starting ATL value (default 0 — use longer history for accuracy)
    seed_ctl    : Starting CTL value (default 0)

    Returns list of DailyMetrics sorted chronologically.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date()

    atl = seed_atl
    ctl = seed_ctl
    results: list[DailyMetrics] = []

    current = start
    while current <= end:
        date_str = current.isoformat()
        load     = daily_load.get(date_str, 0.0)

        # Exponential decay update
        atl = atl * ATL_DECAY + load * (1 - ATL_DECAY)
        ctl = ctl * CTL_DECAY + load * (1 - CTL_DECAY)
        tsb = ctl - atl

        results.append(DailyMetrics(
            date=date_str,
            load=load,
            atl=round(atl, 2),
            ctl=round(ctl, 2),
            tsb=round(tsb, 2),
            in_alert=tsb < TSB_ALERT_THRESHOLD,
        ))

        current += timedelta(days=1)

    return results


def _build_metrics(workouts: list[dict], history_days: int = 180) -> list[DailyMetrics]:
    """
    Internal helper — builds full ATL/CTL/TSB series from workout history.
    Uses history_days of data with a 42-day warm-up buffer for CTL stability.
    """
    if not workouts:
        return []

    daily_load = compute_daily_load(workouts)

    # Extend start back by CTL_DAYS as a warm-up buffer
    end_date   = date.today()
    start_date = end_date - timedelta(days=history_days + CTL_DAYS)

    return compute_atl_ctl_tsb(
        daily_load,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )


# ── Public query functions ────────────────────────────────────────────────────

def get_current_metrics(workouts: list[dict]) -> DailyMetrics | None:
    """Return ATL/CTL/TSB for today."""
    series = _build_metrics(workouts)
    if not series:
        return None
    return series[-1]


def get_metrics_for_date(workouts: list[dict], target_date: str) -> DailyMetrics | None:
    """Return ATL/CTL/TSB for a specific date."""
    series = _build_metrics(workouts)
    for m in series:
        if m.date == target_date:
            return m
    return None


def get_recent_trend(workouts: list[dict], days: int = 14) -> TrendSummary | None:
    """
    Summarise ATL/CTL/TSB trend over the last N days.
    Returns a TrendSummary with direction labels and load statistics.
    """
    series = _build_metrics(workouts)
    if len(series) < days:
        return None

    window = series[-days:]
    first  = window[0]
    last   = window[-1]

    def trend_label(start: float, end: float, threshold: float = 1.0) -> str:
        diff = end - start
        if abs(diff) < threshold:
            return "stable"
        return "rising" if diff > 0 else "falling"

    daily_load  = compute_daily_load(workouts)
    window_load = [daily_load.get(m.date, 0.0) for m in window]
    rest_days   = sum(1 for l in window_load if l == 0.0)

    return TrendSummary(
        days=days,
        start_date=first.date,
        end_date=last.date,
        atl_start=first.atl,
        atl_end=last.atl,
        ctl_start=first.ctl,
        ctl_end=last.ctl,
        tsb_start=first.tsb,
        tsb_end=last.tsb,
        atl_trend=trend_label(first.atl, last.atl),
        ctl_trend=trend_label(first.ctl, last.ctl),
        tsb_trend=trend_label(first.tsb, last.tsb),
        total_load=round(sum(window_load), 1),
        avg_daily_load=round(sum(window_load) / days, 1),
        peak_load=round(max(window_load), 1),
        rest_days=rest_days,
    )


def check_recovery_alert(workouts: list[dict]) -> RecoveryAlert | None:
    """
    Check whether current TSB is below TSB_ALERT_THRESHOLD.
    Returns a RecoveryAlert if triggered, None otherwise.
    Includes how many consecutive days TSB has been in alert territory.
    """
    series = _build_metrics(workouts)
    if not series:
        return None

    current = series[-1]
    if not current.in_alert:
        return None

    # Count consecutive alert days
    consecutive = 0
    for m in reversed(series):
        if m.in_alert:
            consecutive += 1
        else:
            break

    return RecoveryAlert(
        date=current.date,
        tsb=current.tsb,
        atl=current.atl,
        ctl=current.ctl,
        consecutive_days=consecutive,
        message=(
            f"[RECOVERY ALERT] TSB is {current.tsb:.1f} "
            f"(threshold: {TSB_ALERT_THRESHOLD}). "
            f"TSB has been below threshold for {consecutive} consecutive day(s). "
            f"Current fatigue (ATL): {current.atl:.1f}. "
            f"Fitness (CTL): {current.ctl:.1f}. "
            f"Recommend reduced load or rest day before next hard session."
        ),
    )


def weekly_load_summary(
    workouts: list[dict],
    weeks: int = 8,
) -> list[WeekSummary]:
    """
    Summarise training load week by week for the last N weeks.
    Weeks run Monday to Sunday. Returns list sorted most recent first.
    """
    series     = _build_metrics(workouts)
    daily_load = compute_daily_load(workouts)

    if not series:
        return []

    # Find most recent Monday
    today      = date.today()
    days_since = today.weekday()   # 0 = Monday
    this_monday = today - timedelta(days=days_since)

    summaries: list[WeekSummary] = []
    prev_total: float | None = None

    for i in range(weeks):
        week_start = this_monday - timedelta(weeks=i)
        week_end   = week_start + timedelta(days=6)

        # Gather daily metrics for this week
        week_metrics = [
            m for m in series
            if week_start.isoformat() <= m.date <= week_end.isoformat()
        ]
        week_loads = [
            daily_load.get(m.date, 0.0) for m in week_metrics
        ]

        if not week_metrics:
            continue

        total_load    = round(sum(week_loads), 1)
        session_count = sum(1 for l in week_loads if l > 0)
        rest_days     = sum(1 for l in week_loads if l == 0)
        avg_atl       = round(sum(m.atl for m in week_metrics) / len(week_metrics), 1)
        avg_ctl       = round(sum(m.ctl for m in week_metrics) / len(week_metrics), 1)
        avg_tsb       = round(sum(m.tsb for m in week_metrics) / len(week_metrics), 1)

        pct_change = None
        if prev_total is not None and prev_total > 0:
            pct_change = round((total_load - prev_total) / prev_total * 100, 1)

        summaries.append(WeekSummary(
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            total_load=total_load,
            session_count=session_count,
            rest_days=rest_days,
            avg_atl=avg_atl,
            avg_ctl=avg_ctl,
            avg_tsb=avg_tsb,
            pct_change_load=pct_change,
        ))

        prev_total = total_load

    return summaries   # Most recent first


def check_mileage_rule(workouts: list[dict]) -> dict:
    """
    Check whether this week's projected load increase vs last week exceeds
    the 10% mileage rule.

    If the current week is incomplete (fewer than 7 days elapsed since Monday),
    the current week's load is projected forward based on sessions so far
    rather than comparing a partial week to a full one.

    Returns a dict with the verdict and details.
    """
    weeks = weekly_load_summary(workouts, weeks=2)
    if len(weeks) < 2:
        return {"ok": True, "message": "Insufficient history to check mileage rule."}

    this_week = weeks[0]
    last_week = weeks[1]

    if last_week.total_load == 0:
        return {"ok": True, "message": "No load last week — mileage rule not applicable."}

    # Check how many days have elapsed in the current week (0 = Monday only)
    today        = date.today()
    days_elapsed = today.weekday() + 1   # 1 on Monday, 7 on Sunday

    # If fewer than 4 days elapsed, project the week forward
    if days_elapsed < 4 and this_week.session_count == 0:
        return {
            "ok": True,
            "this_week_load":    this_week.total_load,
            "last_week_load":    last_week.total_load,
            "days_elapsed":      days_elapsed,
            "message": (
                f"ℹ️  Week just started (day {days_elapsed}/7) — "
                f"no sessions logged yet. "
                f"Last week total load: {last_week.total_load:.0f}. "
                f"10% rule target for this week: ≤ {last_week.total_load * 1.10:.0f}."
            ),
        }

    # Project weekly total if week is incomplete
    if days_elapsed < 7:
        projected_load = (this_week.total_load / days_elapsed) * 7
        projection_note = (
            f" (projected from {days_elapsed} days: "
            f"{this_week.total_load:.0f} actual → {projected_load:.0f} projected)"
        )
    else:
        projected_load = this_week.total_load
        projection_note = ""

    pct_change = (projected_load - last_week.total_load) / last_week.total_load

    if pct_change > MILEAGE_RULE_PCT:
        return {
            "ok": False,
            "this_week_load":  this_week.total_load,
            "projected_load":  round(projected_load, 1),
            "last_week_load":  last_week.total_load,
            "pct_increase":    round(pct_change * 100, 1),
            "days_elapsed":    days_elapsed,
            "message": (
                f"⚠️  Weekly load on track to increase by {pct_change * 100:.1f}%"
                f"{projection_note}, "
                f"exceeding the 10% rule "
                f"(last week: {last_week.total_load:.0f}). "
                f"Consider scaling back a session."
            ),
        }

    return {
        "ok": True,
        "this_week_load":  this_week.total_load,
        "projected_load":  round(projected_load, 1),
        "last_week_load":  last_week.total_load,
        "pct_increase":    round(pct_change * 100, 1),
        "days_elapsed":    days_elapsed,
        "message": (
            f"✅ Weekly load on track: {pct_change * 100:+.1f}%"
            f"{projection_note} vs last week "
            f"({last_week.total_load:.0f}). Within 10% rule."
        ),
    }


# ── Agent-ready report ────────────────────────────────────────────────────────

def format_metrics_report(
    metrics: DailyMetrics,
    trend:   TrendSummary | None = None,
    alert:   RecoveryAlert | None = None,
    mileage: dict | None = None,
) -> str:
    """
    Format a concise training load report for injection into agent context.
    Returns a plain-text string the Recovery Agent can include in its response.
    """
    lines = [
        f"=== Training Load Report — {metrics.date} ===",
        f"  ATL (fatigue):   {metrics.atl:.1f}",
        f"  CTL (fitness):   {metrics.ctl:.1f}",
        f"  TSB (form):      {metrics.tsb:.1f}  →  {metrics.form_label}",
        f"  Fitness level:   {metrics.fitness_label}",
    ]

    if alert:
        lines.append(f"\n  {alert.message}")

    if trend:
        lines += [
            f"\n  14-day trend:",
            f"    ATL:  {trend.atl_start:.1f} → {trend.atl_end:.1f}  ({trend.atl_trend})",
            f"    CTL:  {trend.ctl_start:.1f} → {trend.ctl_end:.1f}  ({trend.ctl_trend})",
            f"    TSB:  {trend.tsb_start:.1f} → {trend.tsb_end:.1f}  ({trend.tsb_trend})",
            f"    Total load:     {trend.total_load:.0f}",
            f"    Avg daily load: {trend.avg_daily_load:.1f}",
            f"    Peak load:      {trend.peak_load:.1f}",
            f"    Rest days:      {trend.rest_days} / {trend.days}",
        ]

    if mileage:
        lines.append(f"\n  Mileage rule: {mileage['message']}")

    return "\n".join(lines)


# ── Quick test (run as script) ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from parse_workout_data import load_workouts

    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/content/drive/MyDrive/running_coach/data/processed/workouts_normalized.json"
    )

    workouts = load_workouts(path, days=180)
    print(f"Loaded {len(workouts)} workouts (180 days).\n")

    print("--- Current metrics ---")
    metrics = get_current_metrics(workouts)
    if metrics:
        print(f"  Date:   {metrics.date}")
        print(f"  ATL:    {metrics.atl}")
        print(f"  CTL:    {metrics.ctl}")
        print(f"  TSB:    {metrics.tsb}  ({metrics.form_label})")
        print(f"  Alert:  {metrics.in_alert}")

    print("\n--- 14-day trend ---")
    trend = get_recent_trend(workouts, days=14)
    if trend:
        print(f"  ATL: {trend.atl_start:.1f} → {trend.atl_end:.1f} ({trend.atl_trend})")
        print(f"  CTL: {trend.ctl_start:.1f} → {trend.ctl_end:.1f} ({trend.ctl_trend})")
        print(f"  TSB: {trend.tsb_start:.1f} → {trend.tsb_end:.1f} ({trend.tsb_trend})")
        print(f"  Total load: {trend.total_load} | Rest days: {trend.rest_days}/{trend.days}")

    print("\n--- Recovery alert check ---")
    alert = check_recovery_alert(workouts)
    if alert:
        print(f"  {alert.message}")
    else:
        print("  No alert — TSB within acceptable range.")

    print("\n--- Mileage rule ---")
    mileage = check_mileage_rule(workouts)
    print(f"  {mileage['message']}")

    print("\n--- Weekly load (last 8 weeks) ---")
    weeks = weekly_load_summary(workouts, weeks=8)
    print(f"  {'Week':12} {'Load':>8} {'Sessions':>9} {'Rest':>5} {'ATL':>6} {'CTL':>6} {'TSB':>6} {'Change':>8}")
    print("  " + "-" * 65)
    for w in weeks:
        change = f"{w.pct_change_load:+.1f}%" if w.pct_change_load is not None else "  base"
        print(
            f"  {w.week_start:12} {w.total_load:>8.1f} {w.session_count:>9} "
            f"{w.rest_days:>5} {w.avg_atl:>6.1f} {w.avg_ctl:>6.1f} {w.avg_tsb:>6.1f} {change:>8}"
        )

    print("\n--- Full agent report ---")
    print(format_metrics_report(metrics, trend, alert, mileage))
