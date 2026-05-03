"""
agents/recovery_agent.py

Recovery Agent for the Personal Running Coach Agent Swarm.

Responsibilities
----------------
1. Daily check-in mode — assess current recovery status across all signals:
   HRV, sleep, body battery, stress, training readiness, resting HR, TSB
2. Pre-workout mode — given today's planned session, assess whether to
   proceed as planned, modify, or flag for rest
3. Flags [RECOVERY ALERT] when TSB < -30
4. Suggests session modifications when readiness is low
5. Injury disclaimer intercept on any pain/symptom mentions
6. Follow-up conversation loop for deeper questions

Usage
-----
    from agents.recovery_agent import RecoveryAgent
    agent = RecoveryAgent(api_key=GEMINI_API_KEY)

    # Daily check-in
    agent.run()

    # Pre-workout check for a specific session type
    agent.run(planned_session="threshold")
"""

from __future__ import annotations

import os
import sys
import sqlite3

import google.generativeai as genai

BASE_DIR = "/content/drive/MyDrive/running_coach"
sys.path.insert(0, f"{BASE_DIR}/tools")

from parse_workout_data      import load_workouts, format_pace
from calculate_training_load import (
    get_current_metrics, get_recent_trend,
    check_recovery_alert, weekly_load_summary,
    check_mileage_rule, format_metrics_report
)
from query_garmin_db import (
    get_recent_snapshots, get_race_predictions,
    format_recovery_context, format_race_predictions
)
from memory_retrieval import get_client, retrieve_profile, retrieve_notes


# ── Constants ─────────────────────────────────────────────────────────────────

WORKOUTS_PATH  = f"{BASE_DIR}/data/processed/workouts_normalized.json"
GARMIN_DB_PATH = f"{BASE_DIR}/data/raw/garmin/garmin.db"
CHROMA_DIR     = f"{BASE_DIR}/memory/chroma"
GEMINI_MODEL   = "gemini-2.5-flash"

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

# Session modification thresholds
READINESS_LOW        = 50    # Below this → suggest modification
READINESS_VERY_LOW   = 35    # Below this → suggest rest or easy only
TSB_ALERT            = -30   # Below this → [RECOVERY ALERT]
BODY_BATTERY_LOW     = 30    # Below this → flag low energy reserves
HRV_CONCERN          = "UNBALANCED"  # HRV status that warrants attention

TRAINING_PACES = {
    "easy":      "by feel, slower than 4:45/km",
    "marathon":  "4:14/km",
    "threshold": "4:01/km",
    "1hr":       "3:56/km",
    "fartlek":   "3:49/km",
    "8k":        "3:46/km",
    "vo2max":    "3:40/km",
}

SYSTEM_PROMPT_CHECKIN = """You are a personal running coach assistant for Will Sutherland,
a competitive half-marathon runner targeting sub-1:24 at the Fredericton Half Marathon.

Your role is to assess his current recovery status and provide a concise daily check-in
in this exact format:

**Recovery Status: [GOOD / MODERATE / LOW / ALERT]**

**Key signals**
- [HRV: value, status, vs baseline]
- [Sleep: hours, quality label]
- [Body battery at wake: value, label]
- [Training readiness: score, level]
- [TSB (form): value, label]
- [Resting HR: value if available]

**Assessment**
[2-3 sentences: overall recovery picture, what the signals are saying together,
any conflicting signals worth noting]

**Recommendation**
[1-2 sentences: what today should look like from a recovery standpoint —
train as planned / modify / take easy / rest]

Rules:
- Be direct — use actual numbers
- If multiple signals point the same way, say so clearly
- If signals conflict (e.g. good HRV but low body battery), acknowledge the nuance
- [RECOVERY ALERT] must appear prominently if TSB < -30
- Never diagnose injuries — redirect to coach or physio
- Keep total response under 250 words
- Tone: knowledgeable training partner, not corporate chatbot
"""

SYSTEM_PROMPT_PREWORKOUT = """You are a personal running coach assistant for Will Sutherland,
a competitive half-marathon runner targeting sub-1:24 at the Fredericton Half Marathon.

Your role is to assess whether Will should proceed with his planned session, modify it,
or back off, based on his current recovery signals.

Respond in this exact format:

**Session Check: [PROCEED / MODIFY / BACK OFF]**

**Planned session:** [session type and target pace]

**Recovery signals right now**
- [key signals: HRV, sleep, body battery, readiness, TSB — numbers only]

**Verdict**
[2-3 sentences: should he do it as planned? If modifying, what specifically —
reduce intensity, reduce volume, swap for easy, or rest entirely?]

**If modifying:** [specific modified session description]

Rules:
- PROCEED = all signals green, go as planned
- MODIFY = 1-2 concerning signals, adjust intensity or volume but still train
- BACK OFF = multiple red signals or [RECOVERY ALERT] — swap for easy or rest
- Always give a concrete modified session if recommending MODIFY or BACK OFF
- [RECOVERY ALERT] must appear if TSB < -30
- Never diagnose injuries
- Keep total response under 200 words
"""


class RecoveryAgent:
    """
    Recovery Agent — daily check-in and pre-workout readiness assessment.
    """

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            try:
                from google.colab import userdata
                api_key = userdata.get('GEMINI_API_KEY')
            except Exception:
                raise ValueError("GEMINI_API_KEY not found.")

        genai.configure(api_key=api_key)
        self.model      = genai.GenerativeModel(GEMINI_MODEL)
        self.mem_client, self.mem_ef = get_client(CHROMA_DIR, api_key)
        self.workouts   = None
        self._api_key   = api_key

        print("✅ Recovery Agent initialized.")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self):
        if self.workouts is None:
            self.workouts = load_workouts(WORKOUTS_PATH, days=180)
            print(f"   Loaded {len(self.workouts)} workouts.")

    # ── Mode selection ────────────────────────────────────────────────────────

    def _select_mode(self) -> tuple[str, str | None]:
        """
        Present mode selection to the user.
        Returns (mode, planned_session) where mode is 'checkin' or 'preworkout'.
        """
        print("\n" + "=" * 60)
        print("RECOVERY AGENT — What would you like to check?")
        print("=" * 60)
        print("\n  [1] Daily check-in — how am I recovering?")
        print("  [2] Pre-workout check — should I do today's session as planned?")
        print("  [Q] Quit")

        while True:
            choice = input("\nYour choice: ").strip().upper()

            if choice == 'Q':
                return 'quit', None

            if choice == '1':
                return 'checkin', None

            if choice == '2':
                print("\n  What session is planned? Options:")
                for k, v in TRAINING_PACES.items():
                    print(f"    {k:<12} {v}")
                session = input("\n  Session type: ").strip().lower()
                if session not in TRAINING_PACES:
                    print(f"  ⚠️  Unknown type. Valid: {list(TRAINING_PACES.keys())}")
                    continue
                return 'preworkout', session

            print("  Invalid — enter 1, 2, or Q.")

    # ── Context assembly ──────────────────────────────────────────────────────

    def _build_recovery_context(self, planned_session: str | None = None) -> str:
        """
        Assemble full recovery context for the Gemini prompt.
        Pulls: snapshots (HRV, sleep, BB, stress, readiness, HR),
               training load metrics, weekly summary, race predictions,
               athlete profile from memory.
        """
        lines = []

        # ── Garmin health snapshots (last 3 days) ─────────────────────────────
        try:
            db_conn  = sqlite3.connect(GARMIN_DB_PATH)
            snapshots = get_recent_snapshots(db_conn, days=3)
            lines.append(format_recovery_context(snapshots, include_days=3))

            # Race predictions
            pred = get_race_predictions(db_conn)
            if pred:
                lines += ["", format_race_predictions(pred)]
            db_conn.close()
        except Exception as e:
            lines.append(f"(Garmin DB unavailable: {e})")

        # ── Training load ─────────────────────────────────────────────────────
        try:
            metrics = get_current_metrics(self.workouts)
            trend   = get_recent_trend(self.workouts, days=14)
            alert   = check_recovery_alert(self.workouts)
            mileage = check_mileage_rule(self.workouts)
            weeks   = weekly_load_summary(self.workouts, weeks=4)

            lines.append("")
            lines.append(format_metrics_report(metrics, trend, alert, mileage))

            if weeks:
                lines.append("\n=== Weekly Load (last 4 weeks) ===")
                lines.append(f"  {'Week':<12} {'Load':>8} {'Sessions':>9} {'TSB avg':>8}")
                lines.append("  " + "-" * 42)
                for w in weeks:
                    lines.append(
                        f"  {w.week_start:<12} {w.total_load:>8.1f} "
                        f"{w.session_count:>9} {w.avg_tsb:>8.1f}"
                    )
        except Exception as e:
            lines.append(f"\n(Training load unavailable: {e})")

        # ── Planned session context ───────────────────────────────────────────
        if planned_session:
            pace = TRAINING_PACES.get(planned_session, "unknown")
            lines += [
                "",
                "=== Planned Session ===",
                f"Type:   {planned_session}",
                f"Target: {pace}",
            ]

        # ── Athlete profile from memory ───────────────────────────────────────
        try:
            profile = retrieve_profile(
                self.mem_client, self.mem_ef,
                query="recovery training load readiness race goals",
                n=3
            )
            if profile:
                lines += ["", "=== Athlete Context ===", profile]
        except Exception:
            pass

        # ── Recent session notes ──────────────────────────────────────────────
        try:
            notes = retrieve_notes(
                self.mem_client, self.mem_ef,
                query="fatigue tired sore recovery energy",
                n=2
            )
            if notes:
                lines.append("\n=== Recent Session Notes ===")
                for n in notes:
                    lines.append(f"[{n['metadata'].get('date', '')}] {n['document']}")
        except Exception:
            pass

        return "\n".join(lines)

    # ── Modification logic ────────────────────────────────────────────────────

    def _assess_modification(
        self,
        snapshots,
        metrics,
        alert,
    ) -> tuple[str, list[str]]:
        """
        Determine if a session modification is warranted based on signals.
        Returns (recommendation_level, list_of_flags).
        recommendation_level: 'PROCEED' | 'MODIFY' | 'BACK OFF'
        """
        flags = []

        # TSB alert
        if alert:
            flags.append(f"[RECOVERY ALERT] TSB = {alert.tsb:.1f} (threshold: {TSB_ALERT})")

        if metrics:
            if metrics.tsb < TSB_ALERT:
                flags.append(f"TSB critically low: {metrics.tsb:.1f}")
            elif metrics.tsb < -20:
                flags.append(f"TSB elevated fatigue: {metrics.tsb:.1f}")

        # Snapshot signals from most recent day
        if snapshots:
            today = snapshots[0]

            if today.readiness:
                score = today.readiness.score or 0
                if score < READINESS_VERY_LOW:
                    flags.append(f"Readiness very low: {score}/100")
                elif score < READINESS_LOW:
                    flags.append(f"Readiness low: {score}/100")

            if today.hrv and today.hrv.status == HRV_CONCERN:
                flags.append(f"HRV unbalanced (last night: {today.hrv.last_night})")

            if today.body_battery and today.body_battery.at_wake is not None:
                if today.body_battery.at_wake < BODY_BATTERY_LOW:
                    flags.append(f"Body battery low at wake: {today.body_battery.at_wake}")

            if today.sleep and today.sleep.total_sleep_hrs is not None:
                if today.sleep.total_sleep_hrs < 6.0:
                    flags.append(f"Short sleep: {today.sleep.total_sleep_hrs}h")

        # Determine level
        critical = sum(1 for f in flags if any(
            kw in f for kw in ['ALERT', 'critically', 'very low']
        ))
        concerning = len(flags)

        if critical >= 1 or concerning >= 3:
            return 'BACK OFF', flags
        elif concerning >= 1:
            return 'MODIFY', flags
        return 'PROCEED', flags

    # ── Injury check ──────────────────────────────────────────────────────────

    def _check_injury(self, text: str) -> bool:
        return any(kw in text.lower() for kw in INJURY_KEYWORDS)

    # ── Response generation ───────────────────────────────────────────────────

    def _generate_response(self, context: str, mode: str) -> str:
        system = SYSTEM_PROMPT_CHECKIN if mode == 'checkin' else SYSTEM_PROMPT_PREWORKOUT
        prompt = f"{system}\n\nHere is the recovery data:\n\n{context}"
        response = self.model.generate_content(prompt)
        return response.text

    # ── Follow-up loop ────────────────────────────────────────────────────────

    def _followup_loop(self, context: str, initial_response: str):
        history = [
            {"role": "user",  "parts": [f"Assess my recovery:\n\n{context}"]},
            {"role": "model", "parts": [initial_response]},
        ]
        chat = self.model.start_chat(history=history)

        print("\n" + "-" * 60)
        print("Ask a follow-up question, or type 'done' to exit.")
        print("-" * 60)

        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ('done', 'exit', 'quit', 'q'):
                print("Closing Recovery Agent. Train smart! 💪")
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

    def run(self, planned_session: str | None = None):
        """
        Run the Recovery Agent.

        Parameters
        ----------
        planned_session : Optional session type string (e.g. 'threshold').
                          If provided, skips the mode menu and goes straight
                          to pre-workout check for that session type.
        """
        self._load()

        # ── Mode selection ─────────────────────────────────────────────────────
        if planned_session:
            if planned_session not in TRAINING_PACES:
                print(f"❌ Unknown session type '{planned_session}'. "
                      f"Valid: {list(TRAINING_PACES.keys())}")
                return
            mode = 'preworkout'
            print(f"\n🔍 Pre-workout check for: {planned_session} ({TRAINING_PACES[planned_session]})")
        else:
            mode, planned_session = self._select_mode()
            if mode == 'quit':
                return

        # ── Build context ──────────────────────────────────────────────────────
        print("\n📊 Pulling recovery data...")

        # Pre-compute signals for modification logic
        try:
            db_conn   = sqlite3.connect(GARMIN_DB_PATH)
            snapshots = get_recent_snapshots(db_conn, days=1)
            db_conn.close()
        except Exception:
            snapshots = []

        try:
            metrics = get_current_metrics(self.workouts)
            alert   = check_recovery_alert(self.workouts)
        except Exception:
            metrics = None
            alert   = None

        # Assess modification need (used for pre-workout mode)
        recommendation, flags = self._assess_modification(snapshots, metrics, alert)

        if mode == 'preworkout' and flags:
            print(f"\n⚠️  Signals flagged before generating response:")
            for f in flags:
                print(f"   • {f}")

        context  = self._build_recovery_context(planned_session)
        response = self._generate_response(context, mode)

        # ── Print response ─────────────────────────────────────────────────────
        label = "RECOVERY CHECK-IN" if mode == 'checkin' else f"PRE-WORKOUT CHECK — {(planned_session or '').upper()}"
        print("\n" + "=" * 60)
        print(label)
        print("=" * 60)
        print(response)

        # ── Follow-up ──────────────────────────────────────────────────────────
        self._followup_loop(context, response)


# ── Convenience runner ────────────────────────────────────────────────────────

def run_recovery_agent(planned_session: str | None = None):
    """Convenience function for running the agent from a notebook cell."""
    from google.colab import userdata
    api_key = userdata.get('GEMINI_API_KEY')
    agent = RecoveryAgent(api_key=api_key)
    agent.run(planned_session=planned_session)
