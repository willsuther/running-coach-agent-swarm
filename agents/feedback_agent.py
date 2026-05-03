"""
agents/feedback_agent.py

Workout Feedback Agent for the Personal Running Coach Agent Swarm.

Responsibilities
----------------
1. Present the user with workout selection options:
   - Most recent run (any type)
   - Most recent easy run
   - A specific date (user-provided)
2. Load and classify the selected workout
3. If workout type is unknown → ask user to confirm before evaluating
4. Evaluate actual pace against coach-prescribed targets
5. Retrieve relevant memory (similar sessions, athlete profile, session notes)
6. Generate a coaching-style feedback response:
   - Brief numbers summary (pace, HR, load, recovery context)
   - Commentary on performance, patterns, and next steps
7. Injury disclaimer intercept — redirect any pain/injury mentions to coach/physio

Usage
-----
Run interactively in a Colab cell:

    from agents.feedback_agent import FeedbackAgent
    agent = FeedbackAgent()
    agent.run()

Or query a specific date:

    agent.run(date="2026-04-22")
"""

from __future__ import annotations

import os
import sys
import json
from datetime import date, timedelta

import google.generativeai as genai

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = "/content/drive/MyDrive/running_coach"
sys.path.insert(0, f"{BASE_DIR}/tools")

from parse_workout_data   import (
    load_workouts, get_workouts_by_date, get_recent_workouts,
    get_workouts_by_type, format_pace, summarise_workouts
)
from calculate_pace_zones import (
    classify_workouts, detect_workout_type,
    evaluate_pace, resolve_workout_type
)
from calculate_training_load import (
    get_current_metrics, get_metrics_for_date, format_metrics_report
)
import sqlite3
from query_garmin_db import get_activity_splits, format_splits
from memory_retrieval import (
    get_client, retrieve_workouts, retrieve_profile,
    retrieve_notes, format_memory_context
)


# ── Constants ─────────────────────────────────────────────────────────────────
WORKOUTS_PATH = f"{BASE_DIR}/data/processed/workouts_normalized.json"
GARMIN_DB_PATH = f"{BASE_DIR}/data/raw/garmin/garmin.db"
CHROMA_DIR    = f"{BASE_DIR}/memory/chroma"
GEMINI_MODEL  = "gemini-2.5-flash"

INJURY_KEYWORDS = [
    "pain", "hurt", "injury", "injured", "sore", "ache", "aching",
    "strain", "sprain", "tight", "tightness", "swollen", "swelling",
    "knee", "shin", "hamstring", "calf", "achilles", "plantar",
    "blister", "niggle", "pulled", "tear",
]

INJURY_DISCLAIMER = (
    "\n⚠️  INJURY DISCLAIMER: You've mentioned something that sounds like "
    "a physical symptom or injury. I'm not a medical professional and cannot "
    "provide injury advice. Please consult your coach or a physiotherapist "
    "before continuing training."
)

SYSTEM_PROMPT = """You are a personal running coach assistant for Will Sutherland,
a competitive half-marathon runner targeting sub-1:24 at the Fredericton Half Marathon.

Your role is to provide concise, honest workout feedback in this exact format:

**Numbers**
- [key metrics: pace, target, difference, HR, training load, aerobic effect, TSB on day]

**How it went**
- [2-3 sentences: did they hit the target, what do the numbers suggest about effort/fatigue]

**Patterns**
- [1-2 sentences: how does this compare to similar recent sessions from memory]

**Next session consideration**
- [1 sentence: one practical takeaway for the next session]

Rules:
- Be direct and specific — use actual numbers, not vague language
- If TSB was very negative on the day, acknowledge the context
- If HRV or readiness was low, factor that into your assessment
- Never diagnose injuries — redirect to coach or physio
- Keep the total response under 200 words
- Tone: like a knowledgeable training partner, not a corporate chatbot
"""


class FeedbackAgent:
    """
    Workout Feedback Agent.

    Presents workout selection, classifies the session, evaluates pace,
    retrieves memory context, and generates Gemini-powered coaching feedback.
    """

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            try:
                from google.colab import userdata
                api_key = userdata.get('GEMINI_API_KEY')
            except Exception:
                raise ValueError(
                    "GEMINI_API_KEY not found. Pass it explicitly or set in Colab Secrets."
                )

        genai.configure(api_key=api_key)
        self.model      = genai.GenerativeModel(GEMINI_MODEL)
        self.mem_client, self.mem_ef = get_client(CHROMA_DIR, api_key)
        self.workouts   = None
        self.enriched   = None

        print("✅ Feedback Agent initialized.")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self):
        """Load and classify workouts (cached for session)."""
        if self.workouts is None:
            self.workouts = load_workouts(WORKOUTS_PATH, days=180)
            self.enriched = classify_workouts(self.workouts)
            print(f"   Loaded {len(self.workouts)} workouts.")

    # ── Workout selection ─────────────────────────────────────────────────────

    def _get_most_recent(self) -> dict | None:
        """Return the most recent workout of any type."""
        return self.enriched[0] if self.enriched else None

    def _get_most_recent_easy(self) -> dict | None:
        """Return the most recent easy run."""
        for w in self.enriched:
            if w['classification'].get('workout_type') == 'easy':
                return w
        return None

    def _get_by_date(self, target_date: str) -> dict | None:
        """Return the workout on a specific date (most recent if multiple)."""
        matches = get_workouts_by_date(self.enriched, target_date)
        return matches[0] if matches else None

    def _present_options(self) -> dict | None:
        """
        Present workout selection menu to user.
        Returns the selected WorkoutRecord or None if cancelled.
        """
        recent = self._get_most_recent()
        easy   = self._get_most_recent_easy()

        print("\n" + "=" * 60)
        print("FEEDBACK AGENT — Select a workout to analyse")
        print("=" * 60)

        options = {}

        if recent:
            r_type = recent['classification'].get('workout_type') or 'structured'
            r_pace = format_pace(recent.get('avg_pace_min_km'))
            print(f"\n  [1] Most recent run")
            print(f"      {recent['date']} | {r_type} | {recent.get('distance_km')}km | {r_pace}")
            options['1'] = recent

        if easy and (not recent or easy['activity_id'] != recent['activity_id']):
            e_pace = format_pace(easy.get('avg_pace_min_km'))
            print(f"\n  [2] Most recent easy run")
            print(f"      {easy['date']} | easy | {easy.get('distance_km')}km | {e_pace}")
            options['2'] = easy

        print(f"\n  [D] Specific date (enter as YYYY-MM-DD)")
        print(f"  [Q] Quit")
        print()

        while True:
            choice = input("Your choice: ").strip().upper()

            if choice == 'Q':
                print("Exiting Feedback Agent.")
                return None

            if choice in options:
                return options[choice]

            if choice == 'D':
                date_str = input("Enter date (YYYY-MM-DD): ").strip()
                workout  = self._get_by_date(date_str)
                if workout:
                    return workout
                else:
                    print(f"  ⚠️  No workout found for {date_str}. Try another date.")
                    continue

            print("  Invalid choice — enter 1, 2, D, or Q.")

    # ── Workout type resolution ───────────────────────────────────────────────

    def _resolve_type(self, workout: dict) -> dict:
        """
        If the workout type is pending (Garmin date-named but unresolved),
        ask the user to confirm the type before evaluating pace.
        Returns the workout with classification updated.
        """
        cl = workout['classification']

        if cl.get('needs_input'):
            print(f"\n⚠️  Workout type unknown for {workout['date']} ('{workout.get('garmin_name') or workout.get('name')}')")
            print("   What type of session was this?")
            print("   Options: easy / marathon / threshold / 1hr / fartlek / 8k / vo2max")
            while True:
                user_type = input("   Workout type: ").strip().lower()
                try:
                    from calculate_pace_zones import resolve_workout_type, WorkoutClassification
                    resolved_cl = resolve_workout_type(
                        WorkoutClassification(
                            activity_id=cl.get('activity_id', ''),
                            date=workout['date'],
                            workout_type=None,
                            is_structured=cl.get('is_structured', False),
                            needs_input=True,
                            detected_from=cl.get('detected_from', 'unknown'),
                        ),
                        user_type
                    )
                    workout = workout.copy()
                    workout['classification'] = {
                        'workout_type':  resolved_cl.workout_type,
                        'is_structured': resolved_cl.is_structured,
                        'needs_input':   resolved_cl.needs_input,
                        'detected_from': resolved_cl.detected_from,
                        'notes':         resolved_cl.notes,
                    }
                    # Re-evaluate pace with resolved type
                    from calculate_pace_zones import evaluate_pace, WorkoutClassification as WC
                    pe = evaluate_pace(workout, resolved_cl)
                    workout['pace_evaluation'] = {
                        'workout_type':    pe.workout_type,
                        'target_pace':     pe.target_pace,
                        'target_pace_fmt': pe.target_pace_fmt,
                        'actual_pace':     pe.actual_pace,
                        'actual_pace_fmt': pe.actual_pace_fmt,
                        'diff_seconds':    pe.diff_seconds,
                        'on_target':       pe.on_target,
                        'verdict':         pe.verdict,
                    }
                    return workout
                except ValueError as e:
                    print(f"   ⚠️  {e}")

        return workout

    # ── Context assembly ──────────────────────────────────────────────────────

    def _build_context(self, workout: dict) -> str:
        """
        Assemble full context block for the Gemini prompt:
        - Workout data
        - Pace evaluation
        - Training load on the day
        - Memory: similar sessions, profile, notes
        """
        cl = workout['classification']
        pe = workout.get('pace_evaluation', {})
        hc = workout.get('health_context', {})
        workout_type = cl.get('workout_type') or 'unclassified'

        # ── Workout summary ────────────────────────────────────────────────────
        lines = [
            "=== Workout Data ===",
            f"Date:          {workout['date']}",
            f"Type:          {workout_type}",
            f"Distance:      {workout.get('distance_km')} km",
            f"Duration:      {workout.get('duration_min')} min",
            f"Avg pace:      {format_pace(workout.get('avg_pace_min_km'))}",
            f"Avg HR:        {workout.get('avg_hr')} bpm",
            f"Max HR:        {workout.get('max_hr')} bpm",
            f"Elevation:     {workout.get('elevation_m')} m",
            f"Training load: {workout.get('training_load')}",
            f"Aerobic TE:    {workout.get('aerobic_effect')}",
            f"Anaerobic TE:  {workout.get('anaerobic_effect')}",
            f"Suffer score:  {workout.get('suffer_score')}",
            f"Garmin data:   {'yes' if workout.get('garmin_enriched') else 'no'}",
        ]

        # ── Pace evaluation ────────────────────────────────────────────────────
        if pe:
            lines += [
                "",
                "=== Pace Evaluation ===",
                f"Target:        {pe.get('target_pace_fmt', 'N/A')}",
                f"Actual:        {pe.get('actual_pace_fmt', 'N/A')}",
                f"Difference:    {pe.get('diff_seconds')}s" if pe.get('diff_seconds') is not None else "Difference:    N/A",
                f"On target:     {pe.get('on_target')}",
                f"Verdict:       {pe.get('verdict', '')}",
            ]

        # ── Health context on the day ──────────────────────────────────────────
        if hc:
            lines.append("")
            lines.append("=== Recovery Context on Day ===")
            hc_map = {
                'hrv_status':                   'HRV status',
                'hrv_last_night':               'HRV last night',
                'hrv_weekly_avg':               'HRV weekly avg',
                'sleep_sleep_time_seconds':     'Sleep (hrs)',
                'body_battery_at_wake':         'Body battery at wake',
                'training_readiness_score':     'Readiness score',
                'training_readiness_level':     'Readiness level',
                'training_readiness_feedback_short': 'Readiness feedback',
                'heart_rate_resting_hr':        'Resting HR',
                'stress_avg_stress':            'Avg stress',
            }
            for key, label in hc_map.items():
                val = hc.get(key)
                if val is not None:
                    if key == 'sleep_sleep_time_seconds':
                        val = f"{round(val / 3600, 1)}h"
                    lines.append(f"{label}: {val}")

        # ── Training load on the day ───────────────────────────────────────────
        try:
            metrics = get_metrics_for_date(self.workouts, workout['date'])
            if metrics:
                lines += [
                    "",
                    "=== Training Load on Day ===",
                    f"ATL (fatigue):  {metrics.atl}",
                    f"CTL (fitness):  {metrics.ctl}",
                    f"TSB (form):     {metrics.tsb}  ({metrics.form_label})",
                ]
        except Exception:
            pass

        # ── Activity splits from Garmin DB ────────────────────────────────────
        try:
            db_conn = sqlite3.connect(GARMIN_DB_PATH)
            row = db_conn.execute("""
                SELECT activity_id FROM activity
                WHERE DATE(start_time_local) = ?
                  AND LOWER(activity_type) LIKE '%run%'
                ORDER BY start_time_local DESC
                LIMIT 1
            """, (workout['date'],)).fetchone()
            if row:
                splits = get_activity_splits(db_conn, row[0])
                if splits:
                    lines += ["", "=== Lap Splits ===", format_splits(splits)]
            db_conn.close()
        except Exception as e:
            lines.append(f"\n(Splits unavailable: {e})")

        # ── Memory context ────────────────────────────────────────────────────
        query = f"{workout_type} session {workout['date']} pace {format_pace(workout.get('avg_pace_min_km'))}"
        try:
            similar = retrieve_workouts(
                self.mem_client, self.mem_ef, query, n=3,
                filters={"workout_type": workout_type} if workout_type not in ('unclassified', 'easy') else None
            )
            profile = retrieve_profile(self.mem_client, self.mem_ef, query, n=2)
            notes   = retrieve_notes(self.mem_client, self.mem_ef, query, n=2)

            if similar:
                lines.append("\n=== Similar Past Sessions ===")
                for r in similar:
                    if r['metadata'].get('date') != workout['date']:
                        lines.append(r['document'].split('\n')[0])

            if profile:
                lines += ["\n=== Athlete Context ===", profile]

            if notes:
                lines.append("\n=== Session Notes ===")
                for n in notes:
                    lines.append(f"[{n['metadata'].get('date', '')}] {n['document']}")
        except Exception as e:
            lines.append(f"\n(Memory retrieval unavailable: {e})")

        return "\n".join(lines)

    # ── Injury check ──────────────────────────────────────────────────────────

    def _check_injury(self, text: str) -> bool:
        """Return True if the text contains injury-related keywords."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in INJURY_KEYWORDS)

    # ── Main response generation ──────────────────────────────────────────────

    def _generate_feedback(self, workout: dict, context: str) -> str:
        """Call Gemini to generate coaching feedback."""
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Here is the workout data and context:\n\n"
            f"{context}\n\n"
            f"Please provide feedback on this {workout['classification'].get('workout_type') or 'workout'} session."
        )
        response = self.model.generate_content(prompt)
        return response.text

    # ── Follow-up conversation ────────────────────────────────────────────────

    def _followup_loop(self, workout: dict, context: str, initial_feedback: str):
        """Allow the user to ask follow-up questions about the workout."""
        history = [
            {"role": "user", "parts": [f"Analyse this workout:\n\n{context}"]},
            {"role": "model", "parts": [initial_feedback]},
        ]
        chat = self.model.start_chat(history=history)

        print("\n" + "-" * 60)
        print("Ask a follow-up question, or type 'done' to exit.")
        print("-" * 60)

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ('done', 'exit', 'quit', 'q'):
                print("Closing Feedback Agent. Good luck with your next session! 🏃")
                break

            if self._check_injury(user_input):
                print(INJURY_DISCLAIMER)
                continue

            try:
                response = chat.send_message(user_input)
                print(f"\nCoach: {response.text}")
            except Exception as e:
                print(f"⚠️  Error generating response: {e}")

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, date: str | None = None):
        """
        Run the Feedback Agent.

        Parameters
        ----------
        date : Optional YYYY-MM-DD string. If provided, skips the selection
               menu and goes straight to analysing that date's workout.
        """
        self._load()

        # ── Workout selection ──────────────────────────────────────────────────
        if date:
            workout = self._get_by_date(date)
            if not workout:
                print(f"❌ No workout found for {date}.")
                return
            print(f"\n✅ Loading workout for {date}...")
        else:
            workout = self._present_options()
            if workout is None:
                return

        print(f"\n📊 Analysing: {workout['date']} | "
              f"{workout['classification'].get('workout_type') or 'unclassified'} | "
              f"{workout.get('distance_km')}km | "
              f"{format_pace(workout.get('avg_pace_min_km'))}")

        # ── Type resolution ────────────────────────────────────────────────────
        workout = self._resolve_type(workout)

        # ── Injury check on session notes ──────────────────────────────────────
        existing_note = ""
        try:
            from memory_retrieval import retrieve_notes
            notes = retrieve_notes(
                self.mem_client, self.mem_ef,
                f"session note {workout['date']}", n=1
            )
            if notes and notes[0]['metadata'].get('date') == workout['date']:
                existing_note = notes[0]['document']
                if self._check_injury(existing_note):
                    print(INJURY_DISCLAIMER)
        except Exception:
            pass

        # ── Build context & generate feedback ─────────────────────────────────
        print("\n🔍 Retrieving context and generating feedback...")
        context  = self._build_context(workout)
        feedback = self._generate_feedback(workout, context)

        print("\n" + "=" * 60)
        print(f"FEEDBACK — {workout['date']}")
        print("=" * 60)
        print(feedback)

        # ── Follow-up loop ─────────────────────────────────────────────────────
        self._followup_loop(workout, context, feedback)


# ── Convenience runner ────────────────────────────────────────────────────────

def run_feedback_agent(date: str | None = None):
    """Convenience function for running the agent from a notebook cell."""
    agent = FeedbackAgent()
    agent.run(date=date)


if __name__ == "__main__":
    import sys
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_feedback_agent(date=date_arg)
