"""
agents/planner_agent.py

Planner Agent for the Personal Running Coach Agent Swarm.

Responsibilities
----------------
1. Generate a proposed week of sessions based on:
   - Days to race (taper vs build vs peak logic)
   - Current CTL, ATL, TSB
   - Recovery signals (HRV, readiness, body battery)
   - 10% mileage rule
   - Weekly structure (Tu/W/Th/Sa/Su)
2. Review a planned week and flag sessions to modify given current recovery
3. Enforce taper logic in final 2 weeks before race
4. Injury disclaimer intercept

Taper logic
-----------
- 8+ days to race:  normal training week, follow structure
- 4-7 days to race: reduced volume (70%), keep one quality session,
                    rest legs for race
- 1-3 days to race: very light (2-3 easy km max), no quality sessions
- Race week specific: Wednesday short quality, Thursday/Friday easy or rest,
                      Saturday/Sunday rest or very short shakeout

Usage
-----
    from agents.planner_agent import PlannerAgent
    agent = PlannerAgent(api_key=GEMINI_API_KEY)
    agent.run()
"""

from __future__ import annotations

import os
import sys
import sqlite3
from datetime import date, timedelta

import google.generativeai as genai

BASE_DIR = "/content/drive/MyDrive/running_coach"
sys.path.insert(0, f"{BASE_DIR}/tools")

from parse_workout_data      import load_workouts, format_pace, summarise_workouts
from calculate_training_load import (
    get_current_metrics, get_recent_trend,
    check_recovery_alert, weekly_load_summary,
    check_mileage_rule
)
from query_garmin_db import (
    get_recent_snapshots, get_race_predictions,
    format_recovery_context, format_race_predictions
)
from memory_retrieval import get_client, retrieve_profile, retrieve_workouts as mem_retrieve_workouts


# ── Constants ─────────────────────────────────────────────────────────────────

WORKOUTS_PATH  = f"{BASE_DIR}/data/processed/workouts_normalized.json"
GARMIN_DB_PATH = f"{BASE_DIR}/data/raw/garmin/garmin.db"
CHROMA_DIR     = f"{BASE_DIR}/memory/chroma"
GEMINI_MODEL   = "gemini-2.5-flash"

RACE_DATE      = date(2026, 5, 10)   # Fredericton Half Marathon
RACE_NAME      = "Fredericton Half Marathon"

# Secondary race — can swap target when Fredericton is done
SECONDARY_RACE_DATE = date(2026, 10, 18)  # Toronto Half Marathon (placeholder)
SECONDARY_RACE_NAME = "Toronto Half Marathon"

INJURY_KEYWORDS = [
    "pain","hurt","injury","injured","sore","ache","strain",
    "sprain","tight","tightness","swollen","knee","shin",
    "hamstring","calf","achilles","plantar","niggle","pulled","tear"
]
INJURY_DISCLAIMER = (
    "\n⚠️  INJURY DISCLAIMER: You've mentioned something that sounds like "
    "a physical symptom or injury. Please consult your coach or a "
    "physiotherapist before continuing training."
)

TRAINING_PACES = {
    "easy":      "by feel, slower than 4:45/km",
    "marathon":  "4:14/km",
    "threshold": "4:01/km",
    "1hr":       "3:56/km",
    "fartlek":   "3:49/km",
    "8k":        "3:46/km",
    "vo2max":    "3:40/km",
    "race":      "~3:59/km (1:24 target)",
    "rest":      "full rest",
    "shakeout":  "very easy 2-3km, just to loosen up",
}

WEEKLY_STRUCTURE = {
    "Monday":    "rest",
    "Tuesday":   "easy or fartlek",
    "Wednesday": "key speed session",
    "Thursday":  "easy",
    "Friday":    "rest",
    "Saturday":  "long run with structure",
    "Sunday":    "easy",
}

SYSTEM_PROMPT_GENERATE = """You are a personal running coach for Will Sutherland,
targeting sub-1:24 at the Fredericton Half Marathon on May 10 2026.

Generate a proposed training week in this exact format:

**Week Plan — [date range]**
**Phase:** [Taper / Race Week / Build / Peak]
**Focus:** [1 sentence on the week's objective]

| Day | Session | Details | Notes |
|-----|---------|---------|-------|
| Monday | Rest | — | — |
| Tuesday | [type] | [distance, pace target] | [any flag] |
| Wednesday | [type] | [distance, pace target, reps if interval] | [key session] |
| Thursday | [type] | [distance, pace target] | [any flag] |
| Friday | [type or Rest] | [details] | — |
| Saturday | [type] | [distance, structure] | [key session] |
| Sunday | [type] | [distance, pace target] | [any flag] |

**Weekly volume:** [total km]
**10% rule check:** [compliant / flag]
**Load target:** [ATL/CTL/TSB targets for end of week]

**Rationale**
[2-3 sentences explaining the week's design given current fitness and recovery]

Rules:
- Enforce taper logic strictly in final 7 days before race
- Apply 10% mileage rule — flag if volume would breach it
- Use coach-prescribed paces exactly
- If recovery signals are poor, reduce intensity before volume
- Never recommend a hard session on consecutive days
- [RECOVERY ALERT] if TSB < -30
"""

SYSTEM_PROMPT_REVIEW = """You are a personal running coach for Will Sutherland,
targeting sub-1:24 at the Fredericton Half Marathon on May 10 2026.

Review the planned sessions for this week and flag any that should be modified
given current recovery signals. Format:

**Week Review — [date range]**

| Day | Planned | Status | Recommended change |
|-----|---------|--------|--------------------|
| [day] | [session] | ✅ GO / ⚠️ MODIFY / 🛑 BACK OFF | [what to change, or "proceed as planned"] |

**Priority flags**
[bullet list of the most important modifications, if any]

**Overall assessment**
[2 sentences — is this a good week given where Will is right now?]

Rules:
- ✅ GO = recovery signals support this session
- ⚠️ MODIFY = proceed but adjust intensity or volume
- 🛑 BACK OFF = swap for easy or rest given signals
- Always give a specific alternative when flagging MODIFY or BACK OFF
- [RECOVERY ALERT] if TSB < -30
"""


class PlannerAgent:
    """
    Planner Agent — weekly schedule generation and review.
    """

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            try:
                from google.colab import userdata
                api_key = userdata.get('key')
            except Exception:
                raise ValueError("API key not found.")

        genai.configure(api_key=api_key)
        self.model      = genai.GenerativeModel(GEMINI_MODEL)
        self.mem_client, self.mem_ef = get_client(CHROMA_DIR, api_key)
        self.workouts   = None

        print("✅ Planner Agent initialized.")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self):
        if self.workouts is None:
            self.workouts = load_workouts(WORKOUTS_PATH, days=180)
            print(f"   Loaded {len(self.workouts)} workouts.")

    # ── Race context ──────────────────────────────────────────────────────────

    def _race_context(self) -> dict:
        """Return days to race, phase, and taper status."""
        today     = date.today()
        days_left = (RACE_DATE - today).days

        if days_left < 0:
            # Fredericton is done — switch to Toronto
            days_left  = (SECONDARY_RACE_DATE - today).days
            race_name  = SECONDARY_RACE_NAME
            race_date  = SECONDARY_RACE_DATE
        else:
            race_name  = RACE_NAME
            race_date  = RACE_DATE

        if days_left <= 3:
            phase = "Race Week — final prep"
        elif days_left <= 7:
            phase = "Taper — final week"
        elif days_left <= 14:
            phase = "Taper — week 2"
        elif days_left <= 28:
            phase = "Peak / sharpening"
        else:
            phase = "Build"

        return {
            "race_name":  race_name,
            "race_date":  str(race_date),
            "days_left":  days_left,
            "phase":      phase,
            "is_taper":   days_left <= 14,
            "is_race_week": days_left <= 7,
        }

    # ── Mode selection ────────────────────────────────────────────────────────

    def _select_mode(self) -> str:
        """Present mode menu. Returns 'generate', 'review', or 'quit'."""
        rc = self._race_context()
        print("\n" + "=" * 60)
        print("PLANNER AGENT")
        print("=" * 60)
        print(f"\n  Race: {rc['race_name']} — {rc['race_date']}")
        print(f"  Days to race: {rc['days_left']}  |  Phase: {rc['phase']}")
        print()
        print("  [1] Generate next week's training plan")
        print("  [2] Review this week's plan given current recovery")
        print("  [Q] Quit")

        while True:
            choice = input("\nYour choice: ").strip().upper()
            if choice == 'Q':
                return 'quit'
            if choice == '1':
                return 'generate'
            if choice == '2':
                return 'review'
            print("  Invalid — enter 1, 2, or Q.")

    # ── Context assembly ──────────────────────────────────────────────────────

    def _build_context(self, mode: str, planned_sessions: dict | None = None) -> str:
        """Assemble full planning context for Gemini prompt."""
        lines = []
        rc    = self._race_context()

        # ── Race context ───────────────────────────────────────────────────────
        lines += [
            "=== Race Context ===",
            f"Race: {rc['race_name']} on {rc['race_date']}",
            f"Days to race: {rc['days_left']}",
            f"Phase: {rc['phase']}",
            f"Taper: {'yes' if rc['is_taper'] else 'no'}",
            f"Race week: {'yes' if rc['is_race_week'] else 'no'}",
        ]

        # ── Week dates ─────────────────────────────────────────────────────────
        today      = date.today()
        monday     = today - timedelta(days=today.weekday())
        next_monday = monday + timedelta(weeks=1)
        if mode == 'generate':
            week_start = next_monday
        else:
            week_start = monday

        lines += [
            "",
            f"=== {'Next' if mode == 'generate' else 'Current'} Week ===",
            f"Week: {week_start} to {week_start + timedelta(days=6)}",
            f"Today: {today} ({today.strftime('%A')})",
        ]

        # ── Training load ──────────────────────────────────────────────────────
        try:
            metrics = get_current_metrics(self.workouts)
            trend   = get_recent_trend(self.workouts, days=14)
            alert   = check_recovery_alert(self.workouts)
            mileage = check_mileage_rule(self.workouts)
            weeks   = weekly_load_summary(self.workouts, weeks=4)

            lines += [
                "",
                "=== Training Load ===",
                f"ATL (fatigue): {metrics.atl:.1f}",
                f"CTL (fitness): {metrics.ctl:.1f}",
                f"TSB (form):    {metrics.tsb:.1f} ({metrics.form_label})",
                f"Fitness:       {metrics.fitness_label}",
            ]
            if alert:
                lines.append(f"[RECOVERY ALERT] TSB = {alert.tsb:.1f} for {alert.consecutive_days} day(s)")

            lines.append(f"Mileage rule: {mileage['message']}")

            if weeks:
                lines.append("\nRecent weekly load:")
                for w in weeks:
                    lines.append(
                        f"  {w.week_start}: load {w.total_load:.0f}, "
                        f"{w.session_count} sessions, TSB avg {w.avg_tsb:.1f}"
                    )

            if trend:
                lines += [
                    f"\n14-day trend: ATL {trend.atl_trend}, CTL {trend.ctl_trend}, TSB {trend.tsb_trend}",
                    f"Total load 14d: {trend.total_load:.0f} | Rest days: {trend.rest_days}/{trend.days}",
                ]
        except Exception as e:
            lines.append(f"\n(Load unavailable: {e})")

        # ── Recovery signals ───────────────────────────────────────────────────
        try:
            db_conn   = sqlite3.connect(GARMIN_DB_PATH)
            snapshots = get_recent_snapshots(db_conn, days=3)
            db_conn.close()
            lines.append("")
            lines.append(format_recovery_context(snapshots, include_days=2))
        except Exception as e:
            lines.append(f"\n(Garmin unavailable: {e})")

        # ── Weekly structure and paces ─────────────────────────────────────────
        lines += [
            "",
            "=== Training Structure ===",
            "Training days: Tuesday, Wednesday, Thursday, Saturday, Sunday",
            "Rest days: Monday, Friday",
            "Wednesday: key speed session",
            "Saturday: long run with structure (WU/CD at MP+10s, middle at HM pace)",
            "",
            "Coach-prescribed paces:",
        ]
        for k, v in TRAINING_PACES.items():
            lines.append(f"  {k:<12} {v}")

        # ── Garmin race prediction ─────────────────────────────────────────────
        try:
            db_conn = sqlite3.connect(GARMIN_DB_PATH)
            pred    = get_race_predictions(db_conn)
            db_conn.close()
            if pred:
                lines += ["", format_race_predictions(pred)]
        except Exception:
            pass

        # ── Planned sessions (for review mode) ────────────────────────────────
        if mode == 'review' and planned_sessions:
            lines += ["", "=== This Week's Planned Sessions ==="]
            for day, session in planned_sessions.items():
                lines.append(f"  {day}: {session}")

        # ── Recent training history from memory ────────────────────────────────
        try:
            recent = mem_retrieve_workouts(
                self.mem_client, self.mem_ef,
                "recent structured workouts threshold long run", n=4
            )
            if recent:
                lines += ["", "=== Recent Structured Sessions ==="]
                for r in recent:
                    lines.append(r['document'].split('\n')[0])
        except Exception:
            pass

        # ── Athlete profile ────────────────────────────────────────────────────
        try:
            profile = retrieve_profile(
                self.mem_client, self.mem_ef,
                "training plan weekly structure goals race", n=3
            )
            if profile:
                lines += ["", "=== Athlete Context ===", profile]
        except Exception:
            pass

        return "\n".join(lines)

    # ── Planned sessions input (review mode) ──────────────────────────────────

    def _get_planned_sessions(self) -> dict:
        """
        Prompt user to enter this week's planned sessions for review.
        Returns dict of {day: session_description}.
        """
        print("\nEnter this week's planned sessions.")
        print("Press Enter to skip a day (will default to rest/easy).")
        print()

        sessions = {}
        for day in ["Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"]:
            val = input(f"  {day}: ").strip()
            if val:
                sessions[day] = val

        return sessions

    # ── Injury check ──────────────────────────────────────────────────────────

    def _check_injury(self, text: str) -> bool:
        return any(kw in text.lower() for kw in INJURY_KEYWORDS)

    # ── Response generation ───────────────────────────────────────────────────

    def _generate_response(self, context: str, mode: str) -> str:
        system = SYSTEM_PROMPT_GENERATE if mode == 'generate' else SYSTEM_PROMPT_REVIEW
        prompt = f"{system}\n\nPlanning context:\n{context}"
        return self.model.generate_content(prompt).text

    # ── Follow-up loop ────────────────────────────────────────────────────────

    def _followup_loop(self, context: str, initial_response: str):
        history = [
            {"role": "user",  "parts": [f"Planning context:\n{context}"]},
            {"role": "model", "parts": [initial_response]},
        ]
        chat = self.model.start_chat(history=history)

        print("\n" + "-" * 60)
        print("Ask a follow-up question, or type 'done' to exit.")
        print("-" * 60)

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ('done', 'exit', 'quit', 'q'):
                print("Closing Planner Agent. Good luck with your training! 📋")
                break

            if self._check_injury(user_input):
                print(INJURY_DISCLAIMER)
                continue

            try:
                response = chat.send_message(user_input)
                print(f"\nCoach: {response.text}")
            except Exception as e:
                print(f"⚠️  Error: {e}")

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, mode: str | None = None):
        """
        Run the Planner Agent.

        Parameters
        ----------
        mode : 'generate' | 'review' | None (shows menu if None)
        """
        self._load()

        # ── Mode selection ─────────────────────────────────────────────────────
        if mode is None:
            mode = self._select_mode()
        if mode == 'quit':
            return

        # ── Planned sessions input (review mode only) ──────────────────────────
        planned_sessions = None
        if mode == 'review':
            planned_sessions = self._get_planned_sessions()

        # ── Build context & generate ───────────────────────────────────────────
        rc = self._race_context()
        print(f"\n📋 Generating {'week plan' if mode == 'generate' else 'week review'}...")
        print(f"   Phase: {rc['phase']} | Days to race: {rc['days_left']}")

        context  = self._build_context(mode, planned_sessions)
        response = self._generate_response(context, mode)

        label = "WEEK PLAN" if mode == 'generate' else "WEEK REVIEW"
        print("\n" + "=" * 60)
        print(f"{label} — {rc['phase'].upper()}")
        print("=" * 60)
        print(response)

        self._followup_loop(context, response)


# ── Convenience runner ────────────────────────────────────────────────────────

def run_planner_agent(mode: str | None = None):
    """Convenience function for running the agent from a notebook cell."""
    from google.colab import userdata
    api_key = userdata.get('key')
    agent   = PlannerAgent(api_key=api_key)
    agent.run(mode=mode)
