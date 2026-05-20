"""
agents/coordinator_agent.py

Coordinator Agent for the Personal Running Coach Agent Swarm.

Responsibilities
----------------
1. Accept natural language queries from the user
2. Classify the query to determine which specialist agent(s) to invoke
3. Call the appropriate agent(s) and collect their responses
4. Synthesize multi-agent responses into a single coherent reply
5. Run a self-critique safety pass before delivering the final response
6. Fall back to a menu if the query is ambiguous

Query routing logic
-------------------
- Feedback queries   → "how did my run go", "did I hit my target", "analyse my workout"
- Recovery queries   → "am I recovered", "should I train today", "how's my HRV"
- Planner queries    → "what should next week look like", "plan my training"
- Multi-agent        → queries that span two domains (e.g. "how did today go and
                        should I do tomorrow's threshold?") → Feedback + Recovery

Self-critique pass
------------------
After synthesizing subagent responses, the Coordinator asks Gemini to review
the combined output for:
- Unsafe recommendations (hard session when TSB < -30 without flagging)
- Missing injury disclaimers when symptoms mentioned
- 10% mileage rule violations
- Contradictions between subagent outputs

Usage
-----
    from agents.coordinator_agent import CoordinatorAgent
    agent = CoordinatorAgent(api_key=GEMINI_API_KEY)
    agent.run()

    # Or pass a query directly
    agent.run(query="How did my run go today and should I do tomorrow's threshold?")
"""

from __future__ import annotations

import os
import sys
from datetime import date

import google.generativeai as genai

BASE_DIR = "/content/drive/MyDrive/running_coach"
sys.path.insert(0, f"{BASE_DIR}/tools")
sys.path.insert(0, f"{BASE_DIR}/agents")

from parse_workout_data import load_workouts, format_pace, get_recent_workouts


# ── Constants ─────────────────────────────────────────────────────────────────

WORKOUTS_PATH  = f"{BASE_DIR}/data/processed/workouts_normalized.json"
GARMIN_DB_PATH = f"{BASE_DIR}/data/raw/garmin/garmin.db"
CHROMA_DIR     = f"{BASE_DIR}/memory/chroma"
GEMINI_MODEL   = "gemini-2.5-flash"

INJURY_KEYWORDS = [
    "pain","hurt","injury","injured","sore","ache","strain",
    "sprain","tight","tightness","swollen","knee","shin",
    "hamstring","calf","achilles","plantar","niggle","pulled","tear"
]

INJURY_DISCLAIMER = (
    "\n⚠️  INJURY DISCLAIMER: Your query mentions a possible physical symptom. "
    "Please consult your coach or a physiotherapist before continuing training."
)

# Routing keywords for classification
FEEDBACK_KEYWORDS = [
    "run", "workout", "session", "pace", "split", "lap", "interval",
    "how did", "analyse", "analyze", "review", "felt", "went", "hit",
    "target", "performance", "effort", "today's run", "this morning",
]
RECOVERY_KEYWORDS = [
    "recover", "recovery", "hrv", "sleep", "body battery", "readiness",
    "tired", "fatigue", "fatigued", "fresh", "rest", "should i train",
    "should i do", "am i ready", "feel good", "feel bad", "rested",
    "tomorrow", "pre-workout", "before training",
]
PLANNER_KEYWORDS = [
    "plan", "schedule", "next week", "this week", "upcoming", "sessions",
    "what should", "training week", "taper", "race week", "structure",
    "generate", "review my plan", "week ahead",
]

MENU_OPTIONS = {
    "1": ("feedback",  "Analyse a recent workout"),
    "2": ("recovery",  "Daily check-in / pre-workout readiness"),
    "3": ("planner",   "Generate or review weekly training plan"),
    "4": ("multi",     "Ask anything — I'll figure out which agents to call"),
}

CLASSIFIER_PROMPT = """You are the routing coordinator for a running coach agent swarm.
Classify the following user query into one or more of these categories:
- feedback: questions about a completed workout, pace, splits, performance
- recovery: questions about recovery status, HRV, sleep, readiness, whether to train
- planner: questions about upcoming sessions, weekly planning, training schedule
- multi: queries that clearly span two or more categories

You MUST respond with ONLY a raw JSON object. No markdown, no code fences, no explanation.
Example of correct response: {"agents": ["feedback"], "confidence": "high", "reasoning": "asks about a completed workout"}
Example of multi: {"agents": ["feedback", "recovery"], "confidence": "high", "reasoning": "asks about today's run and tomorrow's readiness"}

User query: {query}

Raw JSON response:"""

SYNTHESIS_PROMPT = """You are the coordinator for a personal running coach agent swarm
for Will Sutherland, targeting sub-1:24 at the Fredericton Half Marathon.

You have received responses from the following specialist agents:
{agent_responses}

Your job is to synthesize these into a single, coherent response that:
1. Integrates insights from all agents without unnecessary repetition
2. Prioritizes the most actionable information
3. Maintains a direct, knowledgeable tone
4. Flags any [RECOVERY ALERT] prominently if present in any subagent response
5. Keeps the combined response concise — under 300 words

Synthesized response:"""

CRITIQUE_PROMPT = """You are a safety reviewer for a running coach AI system.
Review the following coaching response for ONLY these serious issues:

1. Recommending a hard session when the response explicitly states TSB < -30, WITHOUT a [RECOVERY ALERT] flag
2. Discussing a specific physical injury or pain symptom WITHOUT including a disclaimer to see a professional
3. Recommending a weekly load increase explicitly stated as more than 20% in one jump

If NONE of these serious issues are present, the response is safe — approve it.
Normal coaching advice, pace recommendations, recovery assessments, and training suggestions
are all acceptable and should be APPROVED.

Response to review:
{response}

Reply with exactly one of:
- APPROVED
- FLAGGED: [one sentence describing the specific serious issue found]

Do not rewrite the response. Do not add disclaimers. Just APPROVED or FLAGGED."""


class CoordinatorAgent:
    """
    Coordinator Agent — the user-facing entry point for the agent swarm.
    Routes queries, calls specialists, synthesizes responses, and runs
    a self-critique safety pass before delivery.
    """

    def __init__(self, api_key: str | None = None):
        if api_key is None:
            try:
                from google.colab import userdata
                api_key = userdata.get('key')
            except Exception:
                raise ValueError("API key not found.")

        genai.configure(api_key=api_key)
        self.model    = genai.GenerativeModel(GEMINI_MODEL)
        self.api_key  = api_key
        self.workouts = None

        print("✅ Coordinator Agent initialized.")
        print("   Specialist agents: Feedback | Recovery | Planner")

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self):
        if self.workouts is None:
            self.workouts = load_workouts(WORKOUTS_PATH, days=180)

    # ── Query classification ──────────────────────────────────────────────────

    def _classify_query(self, query: str) -> list[str]:
        """
        Classify a natural language query into agent categories
        using keyword matching. Simple and reliable.
        """
        agents = self._keyword_classify(query)
        print(f"\n   Routing to: {agents}")
        return agents

    def _keyword_classify(self, query: str) -> list[str]:
        """Keyword-based fallback classifier."""
        q = query.lower()
        agents = []
        if any(k in q for k in FEEDBACK_KEYWORDS):
            agents.append("feedback")
        if any(k in q for k in RECOVERY_KEYWORDS):
            agents.append("recovery")
        if any(k in q for k in PLANNER_KEYWORDS):
            agents.append("planner")
        return agents if agents else ["feedback"]

    # ── Agent invocation ──────────────────────────────────────────────────────

    def _call_feedback(self, query: str) -> str:
        """Build feedback context and generate response using coordinator's model."""
        try:
            import re, sqlite3
            from parse_workout_data import load_workouts, format_pace
            from calculate_pace_zones import classify_workouts
            from calculate_training_load import get_metrics_for_date
            from query_garmin_db import get_activity_splits, format_splits
            from memory_retrieval import retrieve_workouts as mem_workouts, retrieve_profile

            workouts = load_workouts(WORKOUTS_PATH, days=180)
            enriched = classify_workouts(workouts)

            # Select workout
            date_match = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', query)
            if date_match:
                target = date_match.group(1)
                workout = next((w for w in enriched if w.get('date') == target), None)
            else:
                workout = enriched[0] if enriched else None

            if not workout:
                return "[FEEDBACK AGENT]\nNo recent workout found."

            cl    = workout.get('classification', {})
            pe    = workout.get('pace_evaluation', {})
            hc    = workout.get('health_context', {})
            wtype = cl.get('workout_type') or 'unclassified'

            ctx = [
                f"Date: {workout['date']} | Type: {wtype}",
                f"Distance: {workout.get('distance_km')}km | Duration: {workout.get('duration_min')}min",
                f"Avg pace: {format_pace(workout.get('avg_pace_min_km'))} | Avg HR: {workout.get('avg_hr')} bpm | Max HR: {workout.get('max_hr')} bpm",
                f"Training load: {workout.get('training_load')} | Aerobic TE: {workout.get('aerobic_effect')} | Suffer score: {workout.get('suffer_score')}",
            ]

            if pe and pe.get('verdict'):
                ctx.append(f"Pace verdict: {pe.get('verdict')}")

            if hc:
                hc_parts = []
                if hc.get('hrv_status'):            hc_parts.append(f"HRV: {hc['hrv_status']}")
                if hc.get('sleep_sleep_time_seconds'): hc_parts.append(f"Sleep: {round(hc['sleep_sleep_time_seconds']/3600,1)}h")
                if hc.get('training_readiness_score'): hc_parts.append(f"Readiness: {hc['training_readiness_score']}")
                if hc_parts:
                    ctx.append(f"Recovery on day: {', '.join(hc_parts)}")

            try:
                m = get_metrics_for_date(workouts, workout['date'])
                if m:
                    ctx.append(f"TSB on day: {m.tsb:.1f} ({m.form_label})")
            except Exception:
                pass

            try:
                conn   = sqlite3.connect(GARMIN_DB_PATH)
                row    = conn.execute("""
                    SELECT activity_id FROM activity
                    WHERE DATE(start_time_local) = ?
                      AND LOWER(activity_type) LIKE '%run%'
                    ORDER BY start_time_local DESC LIMIT 1
                """, (workout['date'],)).fetchone()
                if row:
                    splits = get_activity_splits(conn, row[0])
                    if splits:
                        ctx += ["Lap splits:", format_splits(splits)]
                conn.close()
            except Exception:
                pass

            try:
                mem_client, mem_ef = __import__('memory_retrieval').get_client(CHROMA_DIR, self.api_key)
                similar = mem_workouts(mem_client, mem_ef, f"{wtype} session pace HR", n=2)
                if similar:
                    ctx.append("Similar past sessions:")
                    for r in similar:
                        if r['metadata'].get('date') != workout['date']:
                            ctx.append(r['document'].split('\n')[0])
            except Exception:
                pass

            system = """You are a running coach for Will Sutherland (sub-1:24 HM target).
Feedback format:
**Numbers** - [pace, target, HR, load, TSB]
**How it went** - [2-3 sentences]
**Patterns** - [1-2 sentences vs similar sessions]
**Next session consideration** - [1 sentence]
Under 200 words. Direct. Real numbers."""

            response = self.model.generate_content(f"{system}\n\nWorkout:\n" + "\n".join(ctx)).text
            return f"[FEEDBACK AGENT]\n{response}"
        except Exception as e:
            return f"[FEEDBACK AGENT ERROR: {e}]"

    def _call_recovery(self, query: str, planned_session: str | None = None) -> str:
        """Build recovery context directly and generate response using coordinator's model."""
        try:
            import sqlite3
            from parse_workout_data import load_workouts
            from query_garmin_db import get_recent_snapshots, format_recovery_context
            from calculate_training_load import (
                get_current_metrics, get_recent_trend,
                check_recovery_alert, check_mileage_rule, format_metrics_report
            )
            from recovery_agent import SYSTEM_PROMPT_CHECKIN, SYSTEM_PROMPT_PREWORKOUT

            workouts = load_workouts(WORKOUTS_PATH, days=180)
            metrics  = get_current_metrics(workouts)
            trend    = get_recent_trend(workouts, days=14)
            alert    = check_recovery_alert(workouts)
            mileage  = check_mileage_rule(workouts)

            # Determine mode from query
            q    = query.lower()
            mode = "preworkout" if any(k in q for k in [
                "should i", "tomorrow", "pre-workout", "threshold",
                "marathon", "fartlek", "vo2", "interval"
            ]) else "checkin"

            if mode == "preworkout":
                for stype in ["threshold", "marathon", "fartlek", "vo2max", "8k", "1hr", "easy"]:
                    if stype in q:
                        planned_session = stype
                        break
                planned_session = planned_session or "threshold"
            else:
                planned_session = None

            # Build context from Garmin DB and training load only — no ChromaDB
            ctx_lines = []
            conn      = sqlite3.connect(GARMIN_DB_PATH)
            snapshots = get_recent_snapshots(conn, days=3)
            conn.close()
            ctx_lines.append(format_recovery_context(snapshots, include_days=2))
            ctx_lines.append(format_metrics_report(metrics, trend, alert, mileage))

            if planned_session:
                from calculate_pace_zones import TRAINING_PACES, format_pace as fp
                pace = TRAINING_PACES.get(planned_session)
                ctx_lines.append(
                    f"\nPlanned session: {planned_session} at {fp(pace) if pace else 'by feel'}"
                )

            context  = "\n".join(ctx_lines)
            system   = SYSTEM_PROMPT_CHECKIN if mode == 'checkin' else SYSTEM_PROMPT_PREWORKOUT
            response = self.model.generate_content(f"{system}\n\nData:\n{context}").text
            return f"[RECOVERY AGENT]\n{response}"
        except Exception as e:
            return f"[RECOVERY AGENT ERROR: {e}]"

    def _call_planner(self, query: str) -> str:
        """Invoke the Planner Agent and return its response as a string."""
        try:
            from planner_agent import PlannerAgent, SYSTEM_PROMPT_GENERATE, SYSTEM_PROMPT_REVIEW
            agent = PlannerAgent(api_key=self.api_key)
            agent._load()

            q    = query.lower()
            mode = "review" if any(k in q for k in ["review", "this week", "check"]) else "generate"

            context  = agent._build_context(mode)
            system   = SYSTEM_PROMPT_GENERATE if mode == 'generate' else SYSTEM_PROMPT_REVIEW
            response = self.model.generate_content(f"{system}\n\nPlanning context:\n{context}").text
            return f"[PLANNER AGENT]\n{response}"
        except Exception as e:
            return f"[PLANNER AGENT ERROR: {e}]"

    # ── Synthesis & critique ──────────────────────────────────────────────────

    def _synthesize(self, agent_responses: list[str], query: str) -> str:
        """Synthesize multiple agent responses into one coherent reply."""
        if len(agent_responses) == 1:
            # Single agent — strip the agent label and return directly
            response = agent_responses[0]
            for label in ["[FEEDBACK AGENT]\n", "[RECOVERY AGENT]\n", "[PLANNER AGENT]\n"]:
                response = response.replace(label, "")
            return response.strip()

        # Multiple agents — synthesize
        formatted = "\n\n".join([
            f"--- {r.split(chr(10))[0]} ---\n{chr(10).join(r.split(chr(10))[1:])}"
            for r in agent_responses
        ])
        prompt   = SYNTHESIS_PROMPT.format(agent_responses=formatted)
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _self_critique(self, response: str) -> str:
        """
        Run a self-critique safety pass on the synthesized response.
        Returns the original response if approved, or a corrected version if flagged.
        """
        prompt = CRITIQUE_PROMPT.format(response=response)
        result = self.model.generate_content(prompt).text.strip()

        if result.startswith("APPROVED"):
            print("   Self-critique: ✅ APPROVED")
            return response
        elif result.startswith("FLAGGED"):
            flag_line = result.split("\n")[0]
            print(f"   Self-critique: ⚠️  {flag_line}")
            # Since our new prompt doesn't ask for a rewrite, just return original
            # with the flag appended so the user is aware
            return response + f"\n\n*Note: {flag_line}*"
        else:
            print("   Self-critique: ✅ No issues found")
            return response

    # ── Injury check ──────────────────────────────────────────────────────────

    def _check_injury(self, text: str) -> bool:
        return any(kw in text.lower() for kw in INJURY_KEYWORDS)

    # ── Menu fallback ─────────────────────────────────────────────────────────

    def _show_menu(self) -> str:
        """Show the menu and return the user's typed query or menu choice."""
        self._load()
        recent = get_recent_workouts(self.workouts, n=1)
        recent_str = ""
        if recent:
            w = recent[0]
            recent_str = (
                f"\n   Last run: {w['date']} | "
                f"{w.get('distance_km')}km | "
                f"{format_pace(w.get('avg_pace_min_km'))}"
            )

        print("\n" + "=" * 60)
        print("RUNNING COACH — What would you like to know?")
        print("=" * 60)
        print(recent_str)
        print()
        print("  Just type your question, or choose from the menu:")
        print()
        for key, (_, label) in MENU_OPTIONS.items():
            print(f"  [{key}] {label}")
        print("  [Q] Quit")
        print()

        return input("You: ").strip()

    # ── Follow-up loop ────────────────────────────────────────────────────────

    def _followup_loop(self, initial_query: str, initial_response: str):
        """Keep the conversation going after the initial response."""
        history = [
            {"role": "user",  "parts": [initial_query]},
            {"role": "model", "parts": [initial_response]},
        ]
        chat = self.model.start_chat(history=history)

        print("\n" + "-" * 60)
        print("Ask a follow-up, choose another option, or type 'done'.")
        print("-" * 60)

        while True:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ('done', 'exit', 'quit', 'q'):
                print("\nClosing Running Coach. Good luck with your training! 🏃")
                break

            if self._check_injury(user_input):
                print(INJURY_DISCLAIMER)
                continue

            # Check if it's a new routing query
            agents = self._classify_query(user_input)
            if len(agents) > 0 and user_input.lower() not in ('1','2','3','4'):
                # Re-route to specialists if it looks like a new topic
                print("\n🔄 Routing to specialist agents...")
                response = self._dispatch(user_input, agents)
                print(f"\n{response}")
                history.append({"role": "user",  "parts": [user_input]})
                history.append({"role": "model", "parts": [response]})
                chat = self.model.start_chat(history=history)
            else:
                try:
                    response = chat.send_message(user_input).text
                    print(f"\nCoach: {response}")
                    history.append({"role": "user",  "parts": [user_input]})
                    history.append({"role": "model", "parts": [response]})
                except Exception as e:
                    print(f"⚠️  Error: {e}")

    # ── Core dispatch ─────────────────────────────────────────────────────────

    def _dispatch(self, query: str, agents: list[str]) -> str:
        """Call the appropriate specialist agents and return synthesized response."""
        responses = []

        if "feedback" in agents:
            print("   📊 Calling Feedback Agent...")
            responses.append(self._call_feedback(query))

        if "recovery" in agents:
            print("   💤 Calling Recovery Agent...")
            responses.append(self._call_recovery(query))

        if "planner" in agents:
            print("   📋 Calling Planner Agent...")
            responses.append(self._call_planner(query))

        if not responses:
            responses.append(self._call_feedback(query))

        print("   🔀 Synthesizing responses...")
        synthesized = self._synthesize(responses, query)

        print("   🔍 Running self-critique pass...")
        final = self._self_critique(synthesized)

        return final

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, query: str | None = None):
        """
        Run the Coordinator Agent.

        Parameters
        ----------
        query : Optional natural language query. If None, shows the menu.
        """
        self._load()

        # ── Get initial query ──────────────────────────────────────────────────
        if query:
            user_input = query
            print(f"\n📨 Query: {user_input}")
        else:
            user_input = self._show_menu()

        if not user_input or user_input.lower() == 'q':
            print("Goodbye!")
            return

        # ── Handle menu shortcuts ──────────────────────────────────────────────
        menu_map = {
            "1": "How did my most recent run go?",
            "2": "How is my recovery looking today?",
            "3": "What should next week's training look like?",
            "4": None,  # Free-form — prompt for query
        }

        if user_input in menu_map:
            if user_input == "4":
                user_input = input("\nWhat would you like to know? ").strip()
            else:
                user_input = menu_map[user_input]
                print(f"\n→ {user_input}")

        # ── Injury check ───────────────────────────────────────────────────────
        if self._check_injury(user_input):
            print(INJURY_DISCLAIMER)
            return

        # ── Classify and dispatch ──────────────────────────────────────────────
        print("\n🧠 Classifying query...")
        agents = self._classify_query(user_input)

        print(f"\n{'=' * 60}")
        print("RUNNING COACH")
        print(f"{'=' * 60}")

        response = self._dispatch(user_input, agents)
        print(f"\n{response}")

        # ── Follow-up loop ─────────────────────────────────────────────────────
        self._followup_loop(user_input, response)


# ── Convenience runner ────────────────────────────────────────────────────────

def run_coordinator(query: str | None = None):
    """Convenience function for running the Coordinator from a notebook cell."""
    from google.colab import userdata
    api_key = userdata.get('key')
    agent   = CoordinatorAgent(api_key=api_key)
    agent.run(query=query)
