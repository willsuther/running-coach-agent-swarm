"""
ui/coach_ui.py

Gradio interface for the Personal Running Coach Agent Swarm.

Tabs:
  🏃 Feedback    — analyse a recent workout
  💤 Recovery    — daily check-in or pre-workout readiness
  📊 Dashboard   — live training load, recent splits, race countdown

Run in Colab:
    from ui.coach_ui import launch
    launch()
"""

from __future__ import annotations
import os, sys, sqlite3
from datetime import date, timedelta

BASE_DIR = "/content/drive/MyDrive/running_coach"
sys.path.insert(0, f"{BASE_DIR}/tools")
sys.path.insert(0, f"{BASE_DIR}/agents")

import gradio as gr
import google.generativeai as genai
from google.colab import userdata

from parse_workout_data      import load_workouts, format_pace, get_recent_workouts
from calculate_training_load import (
    get_current_metrics, get_recent_trend,
    check_recovery_alert, weekly_load_summary, check_mileage_rule
)
from query_garmin_db import (
    get_recent_snapshots, get_race_predictions,
    format_race_predictions, get_activity_splits, format_splits
)
from memory_retrieval import get_client, retrieve_profile, retrieve_notes

# ── Shared state ──────────────────────────────────────────────────────────────

API_KEY    = userdata.get('key')
genai.configure(api_key=API_KEY)
MODEL      = genai.GenerativeModel("gemini-2.5-flash")
MEM_CLIENT, MEM_EF = get_client(f"{BASE_DIR}/memory/chroma", API_KEY)

WORKOUTS_PATH  = f"{BASE_DIR}/data/processed/workouts_normalized.json"
GARMIN_DB_PATH = f"{BASE_DIR}/data/raw/garmin/garmin.db"

TRAINING_PACES = {
    "easy":      None,
    "marathon":  4 + 14/60,
    "threshold": 4 + 1/60,
    "1hr":       3 + 56/60,
    "fartlek":   3 + 49/60,
    "8k":        3 + 46/60,
    "vo2max":    3 + 40/60,
}

INJURY_KEYWORDS = [
    "pain","hurt","injury","injured","sore","ache","strain",
    "sprain","tight","swollen","knee","shin","hamstring",
    "calf","achilles","plantar","niggle","pulled","tear"
]
INJURY_DISCLAIMER = (
    "⚠️ **Injury disclaimer:** You've mentioned something that sounds like a "
    "physical symptom. Please consult your coach or a physiotherapist before "
    "continuing training — I can't provide injury advice."
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _workouts():
    return load_workouts(WORKOUTS_PATH, days=180)

def _check_injury(text: str) -> bool:
    return any(k in text.lower() for k in INJURY_KEYWORDS)

def _garmin_conn():
    return sqlite3.connect(GARMIN_DB_PATH)

# ── Tab 1 — Feedback ──────────────────────────────────────────────────────────

def get_workout_options() -> list[str]:
    """Build the dropdown options for workout selection."""
    workouts = _workouts()
    options  = []
    seen     = set()

    # Most recent run
    if workouts:
        w = workouts[0]
        label = f"Most recent — {w['date']} | {w.get('distance_km')}km | {format_pace(w.get('avg_pace_min_km'))}"
        options.append(label)
        seen.add(w['activity_id'])

    # Most recent easy
    for w in workouts:
        cl = w.get('classification', {})
        if cl.get('workout_type') == 'easy' and w['activity_id'] not in seen:
            label = f"Most recent easy — {w['date']} | {w.get('distance_km')}km | {format_pace(w.get('avg_pace_min_km'))}"
            options.append(label)
            seen.add(w['activity_id'])
            break

    # Last 10 runs as selectable options
    for w in workouts[:10]:
        if w['activity_id'] not in seen:
            cl    = w.get('classification', {})
            wtype = cl.get('workout_type') or 'unclassified'
            label = f"{w['date']} | {wtype} | {w.get('distance_km')}km | {format_pace(w.get('avg_pace_min_km'))}"
            options.append(label)
            seen.add(w['activity_id'])

    return options


def analyse_workout(
    selection: str,
    custom_date: str,
    workout_type_override: str,
    chat_history: list,
) -> tuple[str, list, str]:
    """
    Core feedback function. Returns (feedback_text, chat_history, splits_table).
    """
    workouts = _workouts()

    # ── Resolve workout ────────────────────────────────────────────────────────
    workout = None
    if custom_date and custom_date.strip():
        target = custom_date.strip()
        for w in workouts:
            if w.get('date') == target:
                workout = w
                break
        if not workout:
            return f"No workout found for {target}.", chat_history, ""
    elif selection:
        # Extract date from label string (format: "... — YYYY-MM-DD | ...")
        parts = selection.replace("Most recent — ", "").replace("Most recent easy — ", "")
        date_str = parts.split(" | ")[0].strip()
        for w in workouts:
            if w.get('date') == date_str:
                workout = w
                break

    if not workout:
        return "Please select a workout or enter a date.", chat_history, ""

    # ── Apply workout type override if provided ────────────────────────────────
    if workout_type_override and workout_type_override != "auto-detect":
        from calculate_pace_zones import WorkoutClassification, evaluate_pace
        cl = WorkoutClassification(
            activity_id=workout.get('activity_id',''),
            date=workout.get('date',''),
            workout_type=workout_type_override,
            is_structured=True,
            needs_input=False,
            detected_from='user_input',
        )
        pe = evaluate_pace(workout, cl)
        workout = workout.copy()
        workout['classification'] = {
            'workout_type':  cl.workout_type,
            'is_structured': cl.is_structured,
            'needs_input':   cl.needs_input,
            'detected_from': cl.detected_from,
            'notes':         cl.notes,
        }
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

    # ── Build splits table ─────────────────────────────────────────────────────
    splits_text = ""
    try:
        conn = _garmin_conn()
        row  = conn.execute("""
            SELECT activity_id FROM activity
            WHERE DATE(start_time_local) = ?
              AND LOWER(activity_type) LIKE '%run%'
            ORDER BY start_time_local DESC LIMIT 1
        """, (workout['date'],)).fetchone()
        if row:
            splits = get_activity_splits(conn, row[0])
            splits_text = format_splits(splits) if splits else "No split data available."
        conn.close()
    except Exception as e:
        splits_text = f"Splits unavailable: {e}"

    # ── Build context ──────────────────────────────────────────────────────────
    cl  = workout.get('classification', {})
    pe  = workout.get('pace_evaluation', {})
    hc  = workout.get('health_context', {})
    wtype = cl.get('workout_type') or 'unclassified'

    ctx_lines = [
        f"Date: {workout['date']} | Type: {wtype}",
        f"Distance: {workout.get('distance_km')}km | Duration: {workout.get('duration_min')}min",
        f"Avg pace: {format_pace(workout.get('avg_pace_min_km'))} | Avg HR: {workout.get('avg_hr')} bpm | Max HR: {workout.get('max_hr')} bpm",
        f"Training load: {workout.get('training_load')} | Aerobic TE: {workout.get('aerobic_effect')} | Suffer score: {workout.get('suffer_score')}",
    ]

    if pe:
        ctx_lines += [
            f"\nPace target: {pe.get('target_pace_fmt','N/A')} | Actual: {pe.get('actual_pace_fmt','N/A')} | Diff: {pe.get('diff_seconds')}s | On target: {pe.get('on_target')}",
            f"Verdict: {pe.get('verdict','')}",
        ]

    if hc:
        hc_parts = []
        if hc.get('hrv_status'):            hc_parts.append(f"HRV: {hc['hrv_status']}")
        if hc.get('hrv_last_night'):        hc_parts.append(f"HRV last night: {hc['hrv_last_night']}")
        if hc.get('sleep_sleep_time_seconds'): hc_parts.append(f"Sleep: {round(hc['sleep_sleep_time_seconds']/3600,1)}h")
        if hc.get('body_battery_at_wake'):  hc_parts.append(f"BB at wake: {hc['body_battery_at_wake']}")
        if hc.get('training_readiness_score'): hc_parts.append(f"Readiness: {hc['training_readiness_score']}")
        if hc_parts:
            ctx_lines.append(f"\nRecovery on day: {', '.join(hc_parts)}")

    try:
        from calculate_training_load import get_metrics_for_date
        m = get_metrics_for_date(workouts, workout['date'])
        if m:
            ctx_lines.append(f"TSB on day: {m.tsb:.1f} ({m.form_label}) | ATL: {m.atl:.1f} | CTL: {m.ctl:.1f}")
    except Exception:
        pass

    if splits_text and splits_text != "No split data available.":
        ctx_lines += ["\nLap splits:", splits_text]

    try:
        profile = retrieve_profile(MEM_CLIENT, MEM_EF, f"{wtype} session pace HR", n=2)
        if profile:
            ctx_lines += ["\nAthlete context:", profile]
    except Exception:
        pass

    context = "\n".join(ctx_lines)

    # ── Generate feedback ──────────────────────────────────────────────────────
    system = """You are a personal running coach for Will Sutherland, targeting sub-1:24 half marathon.

Provide feedback in this format:

**Numbers**
- [pace, target, diff, HR, load, TE, TSB]

**How it went**
[2-3 sentences — hit the target? effort vs numbers?]

**Patterns**
[1-2 sentences — vs similar past sessions]

**Next session consideration**
[1 sentence — one concrete takeaway]

Be direct, use real numbers, under 200 words."""

    prompt   = f"{system}\n\nWorkout data:\n{context}"
    response = MODEL.generate_content(prompt)
    feedback = response.text

    # ── Update chat history ────────────────────────────────────────────────────
    chat_history = [{"role": "assistant", "content": feedback}]

    return feedback, chat_history, splits_text


def feedback_chat(message: str, history: list, context_state: str) -> tuple[str, list]:
    """Handle follow-up chat messages in the Feedback tab."""
    if not message.strip():
        return "", history

    if _check_injury(message):
        reply = INJURY_DISCLAIMER
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return "", history

    # Reconstruct chat for Gemini
    gemini_history = []
    if context_state:
        gemini_history.append({"role": "user", "parts": [f"Workout context:\n{context_state}"]})
        if history:
            gemini_history.append({"role": "model", "parts": [history[0]['content']]})

    for h in history[1:]:
        role = "user" if h['role'] == "user" else "model"
        gemini_history.append({"role": role, "parts": [h['content']]})

    chat    = MODEL.start_chat(history=gemini_history)
    reply   = chat.send_message(message).text
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history


# ── Tab 2 — Recovery ──────────────────────────────────────────────────────────

def run_recovery_checkin(planned_session: str) -> tuple[str, str, list]:
    """
    Run a recovery check. Returns (response, context, chat_history).
    planned_session: 'Daily check-in' or a session type string.
    """
    workouts = _workouts()
    mode     = 'checkin' if planned_session == 'Daily check-in' else 'preworkout'
    session  = None if mode == 'checkin' else planned_session.lower()

    # ── Pull all signals ───────────────────────────────────────────────────────
    ctx_lines = []
    try:
        conn      = _garmin_conn()
        snapshots = get_recent_snapshots(conn, days=3)
        pred      = get_race_predictions(conn)
        conn.close()

        for snap in snapshots[:3]:
            ctx_lines.append(f"\n--- {snap.date} ---")
            if snap.readiness:
                ctx_lines.append(f"Readiness: {snap.readiness.score} ({snap.readiness.level}) — {snap.readiness.feedback_short}")
                if snap.readiness.recovery_time:
                    ctx_lines.append(f"Recovery time remaining: {snap.readiness.recovery_time:.0f}h")
            if snap.hrv:
                ctx_lines.append(f"HRV: {snap.hrv.last_night} (weekly avg {snap.hrv.weekly_avg}) — {snap.hrv.status}")
            if snap.sleep:
                ctx_lines.append(f"Sleep: {snap.sleep.total_sleep_hrs}h (deep {snap.sleep.deep_sleep_hrs}h, REM {snap.sleep.rem_sleep_hrs}h) — {snap.sleep.sleep_quality_label}")
            if snap.body_battery:
                ctx_lines.append(f"Body battery at wake: {snap.body_battery.at_wake} (peak {snap.body_battery.highest})")
            if snap.stress:
                ctx_lines.append(f"Stress: avg {snap.stress.avg_stress}, max {snap.stress.max_stress}")
            if snap.resting_hr:
                ctx_lines.append(f"Resting HR: {snap.resting_hr.resting_hr} bpm")

        if pred:
            ctx_lines.append(f"\n{format_race_predictions(pred)}")
    except Exception as e:
        ctx_lines.append(f"(Garmin unavailable: {e})")

    try:
        metrics = get_current_metrics(workouts)
        trend   = get_recent_trend(workouts, days=14)
        alert   = check_recovery_alert(workouts)
        mileage = check_mileage_rule(workouts)
        weeks   = weekly_load_summary(workouts, weeks=4)

        ctx_lines.append(f"\n=== Training Load ===")
        ctx_lines.append(f"ATL: {metrics.atl:.1f} | CTL: {metrics.ctl:.1f} | TSB: {metrics.tsb:.1f} ({metrics.form_label})")
        if alert:
            ctx_lines.append(f"[RECOVERY ALERT] TSB = {alert.tsb:.1f} — {alert.consecutive_days} consecutive days below threshold")
        ctx_lines.append(mileage['message'])

        ctx_lines.append("\n4-week load:")
        for w in weeks:
            ctx_lines.append(f"  {w.week_start}: load {w.total_load:.0f}, {w.session_count} sessions, TSB avg {w.avg_tsb:.1f}")
    except Exception as e:
        ctx_lines.append(f"\n(Load unavailable: {e})")

    if session:
        pace = TRAINING_PACES.get(session)
        pace_str = format_pace(pace) if pace else "by feel"
        ctx_lines.append(f"\nPlanned session: {session} at {pace_str}")

    try:
        profile = retrieve_profile(MEM_CLIENT, MEM_EF, "recovery readiness training load", n=2)
        if profile:
            ctx_lines.append(f"\nAthlete context:\n{profile}")
    except Exception:
        pass

    context = "\n".join(ctx_lines)

    # ── Prompt ────────────────────────────────────────────────────────────────
    if mode == 'checkin':
        system = """You are a running coach assistant for Will Sutherland (sub-1:24 HM target).

Daily recovery check-in format:

**Recovery Status: [GOOD / MODERATE / LOW / ALERT]**

**Key signals**
- HRV: [value, status]
- Sleep: [hours, quality]
- Body battery at wake: [value]
- Training readiness: [score, level]
- TSB: [value, label]

**Assessment**
[2-3 sentences — overall picture, any conflicting signals]

**Today's recommendation**
[1-2 sentences — train as planned / modify / easy / rest]

Under 250 words. Direct. Real numbers."""
    else:
        system = f"""You are a running coach assistant for Will Sutherland (sub-1:24 HM target).

Pre-workout check for a {session} session format:

**Session Check: [PROCEED / MODIFY / BACK OFF]**

**Planned:** {session} at {format_pace(TRAINING_PACES.get(session)) if TRAINING_PACES.get(session) else 'by feel'}

**Signals**
- [HRV, sleep, BB, readiness, TSB — numbers only]

**Verdict**
[2-3 sentences — proceed or change? Why?]

**If modifying:** [specific modified session]

Under 200 words. [RECOVERY ALERT] if TSB < -30."""

    response = MODEL.generate_content(f"{system}\n\nData:\n{context}").text
    history  = [{"role": "assistant", "content": response}]
    return response, context, history


def recovery_chat(message: str, history: list, context_state: str) -> tuple[str, list]:
    """Handle follow-up messages in the Recovery tab."""
    if not message.strip():
        return "", history

    if _check_injury(message):
        reply = INJURY_DISCLAIMER
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant",  "content": reply})
        return "", history

    gemini_history = []
    if context_state and history:
        gemini_history.append({"role": "user",  "parts": [f"Recovery data:\n{context_state}"]})
        gemini_history.append({"role": "model", "parts": [history[0]['content']]})

    for h in history[1:]:
        role = "user" if h['role'] == "user" else "model"
        gemini_history.append({"role": role, "parts": [h['content']]})

    chat  = MODEL.start_chat(history=gemini_history)
    reply = chat.send_message(message).text
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": reply})
    return "", history


# ── Tab 3 — Dashboard ─────────────────────────────────────────────────────────

def build_dashboard() -> tuple[str, str, str, str]:
    """
    Build dashboard cards. Returns (load_md, race_md, recent_md, splits_md).
    """
    workouts = _workouts()

    # ── Training load card ─────────────────────────────────────────────────────
    try:
        metrics = get_current_metrics(workouts)
        trend   = get_recent_trend(workouts, days=14)
        alert   = check_recovery_alert(workouts)
        weeks   = weekly_load_summary(workouts, weeks=6)

        alert_line = f"\n🚨 **[RECOVERY ALERT]** TSB = {alert.tsb:.1f}" if alert else ""

        load_md = f"""### 📊 Training Load — {metrics.date}
| Metric | Value | Label |
|--------|-------|-------|
| ATL (fatigue) | {metrics.atl:.1f} | — |
| CTL (fitness) | {metrics.ctl:.1f} | {metrics.fitness_label} |
| TSB (form) | {metrics.tsb:.1f} | {metrics.form_label} |
{alert_line}

**14-day trend**
- ATL: {trend.atl_start:.1f} → {trend.atl_end:.1f} ({trend.atl_trend})
- CTL: {trend.ctl_start:.1f} → {trend.ctl_end:.1f} ({trend.ctl_trend})
- TSB: {trend.tsb_start:.1f} → {trend.tsb_end:.1f} ({trend.tsb_trend})

**Weekly load**
| Week | Load | Sessions | Avg TSB |
|------|------|----------|---------|
"""
        for w in weeks:
            change = f"{w.pct_change_load:+.1f}%" if w.pct_change_load is not None else "base"
            load_md += f"| {w.week_start} | {w.total_load:.0f} ({change}) | {w.session_count} | {w.avg_tsb:.1f} |\n"

    except Exception as e:
        load_md = f"Load data unavailable: {e}"

    # ── Race countdown card ────────────────────────────────────────────────────
    try:
        race_date  = date(2026, 5, 10)
        today      = date.today()
        days_left  = (race_date - today).days

        conn = _garmin_conn()
        pred = get_race_predictions(conn)
        conn.close()

        from query_garmin_db import _fmt_race_time
        hm_time = _fmt_race_time(pred.time_half_marathon) if pred else "N/A"

        race_md = f"""### 🏁 Fredericton Half Marathon
**{days_left} days to race day** ({race_date})

| Goal | Target |
|------|--------|
| A+ goal | 1:23:59 |
| B goal | Sub 1:25:00 |
| Garmin prediction | {hm_time} |

**Race strategy:** Negative split — conservative first half, build second half
**HM target pace:** ~3:59/km
"""
    except Exception as e:
        race_md = f"Race data unavailable: {e}"

    # ── Recent runs card ───────────────────────────────────────────────────────
    try:
        recent = get_recent_workouts(workouts, n=7)
        recent_md = "### 🗓️ Recent Runs\n| Date | Type | Distance | Pace | HR | Load |\n|------|------|----------|------|----|------|\n"
        for w in recent:
            wtype = w.get('classification', {}).get('workout_type') or '—'
            recent_md += (
                f"| {w['date']} | {wtype} | {w.get('distance_km')}km "
                f"| {format_pace(w.get('avg_pace_min_km'))} "
                f"| {w.get('avg_hr') or '—'} "
                f"| {w.get('training_load') or '—'} |\n"
            )
    except Exception as e:
        recent_md = f"Recent runs unavailable: {e}"

    # ── Latest splits card ─────────────────────────────────────────────────────
    try:
        conn = _garmin_conn()
        row  = conn.execute("""
            SELECT activity_id, activity_name, DATE(start_time_local)
            FROM activity
            WHERE LOWER(activity_type) LIKE '%run%'
            ORDER BY start_time_local DESC LIMIT 1
        """).fetchone()
        splits_md = f"### 📍 Latest Run Splits — {row[1]} ({row[2]})\n```\n"
        splits    = get_activity_splits(conn, row[0])
        splits_md += format_splits(splits) if splits else "No splits available."
        splits_md += "\n```"
        conn.close()
    except Exception as e:
        splits_md = f"Splits unavailable: {e}"

    return load_md, race_md, recent_md, splits_md


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def launch(share: bool = True):
    """Build and launch the Gradio interface."""

    # Custom theme — dark, minimal, data-forward
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("DM Mono"), "monospace"],
        font_mono=[gr.themes.GoogleFont("DM Mono"), "monospace"],
    ).set(
        body_background_fill="#0f1117",
        body_text_color="#e2e8f0",
        block_background_fill="#1a1f2e",
        block_border_color="#2d3748",
        block_label_text_color="#94a3b8",
        input_background_fill="#1a1f2e",
        input_border_color="#2d3748",
        button_primary_background_fill="#f97316",
        button_primary_background_fill_hover="#ea580c",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(
        theme=theme,
        title="🏃 Running Coach",
        css="""
        .gradio-container { max-width: 1100px !important; }
        h1 { font-size: 1.6rem !important; letter-spacing: -0.03em; }
        h3 { color: #f97316 !important; font-size: 0.95rem !important;
             letter-spacing: 0.08em; text-transform: uppercase; }
        .tab-nav button { font-size: 0.85rem !important; letter-spacing: 0.05em; }
        .chatbot { min-height: 320px; }
        footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# 🏃 Running Coach Agent Swarm\n"
            "Personal training intelligence for Will Sutherland · "
            "Fredericton Half Marathon 2026"
        )

        with gr.Tabs():

            # ── Tab 1 — Feedback ───────────────────────────────────────────────
            with gr.Tab("🏃 Workout Feedback"):
                gr.Markdown("Analyse a completed workout — pace vs target, HR, splits, and patterns.")

                with gr.Row():
                    with gr.Column(scale=2):
                        workout_dropdown = gr.Dropdown(
                            choices=get_workout_options(),
                            label="Select workout",
                            value=None,
                        )
                        custom_date = gr.Textbox(
                            label="Or enter a specific date (YYYY-MM-DD)",
                            placeholder="e.g. 2026-04-22",
                        )
                        type_override = gr.Dropdown(
                            choices=["auto-detect"] + list(TRAINING_PACES.keys()),
                            label="Workout type override",
                            value="auto-detect",
                        )
                        analyse_btn = gr.Button("Analyse workout", variant="primary")

                    with gr.Column(scale=3):
                        feedback_output = gr.Markdown(label="Feedback")

                splits_output = gr.Code(
                    label="Lap splits",
                    language=None,
                    interactive=False,
                )

                gr.Markdown("### 💬 Follow-up")
                feedback_chat_box = gr.Chatbot(
                    label="",
                    type="messages",
                    elem_classes=["chatbot"],
                    show_label=False,
                )
                with gr.Row():
                    feedback_input  = gr.Textbox(
                        placeholder="Ask a follow-up question...",
                        show_label=False,
                        scale=5,
                    )
                    feedback_send   = gr.Button("Send", scale=1)

                feedback_history = gr.State([])
                feedback_context = gr.State("")

                analyse_btn.click(
                    fn=analyse_workout,
                    inputs=[workout_dropdown, custom_date, type_override, feedback_history],
                    outputs=[feedback_output, feedback_history, splits_output],
                ).then(
                    fn=lambda h: h,
                    inputs=[feedback_history],
                    outputs=[feedback_chat_box],
                )

                feedback_send.click(
                    fn=feedback_chat,
                    inputs=[feedback_input, feedback_history, feedback_context],
                    outputs=[feedback_input, feedback_history],
                ).then(
                    fn=lambda h: h,
                    inputs=[feedback_history],
                    outputs=[feedback_chat_box],
                )
                feedback_input.submit(
                    fn=feedback_chat,
                    inputs=[feedback_input, feedback_history, feedback_context],
                    outputs=[feedback_input, feedback_history],
                ).then(
                    fn=lambda h: h,
                    inputs=[feedback_history],
                    outputs=[feedback_chat_box],
                )

            # ── Tab 2 — Recovery ───────────────────────────────────────────────
            with gr.Tab("💤 Recovery"):
                gr.Markdown("Daily check-in or pre-workout readiness assessment.")

                with gr.Row():
                    session_selector = gr.Dropdown(
                        choices=["Daily check-in"] + list(TRAINING_PACES.keys()),
                        label="Check-in type",
                        value="Daily check-in",
                    )
                    recovery_btn = gr.Button("Run check", variant="primary")

                recovery_output = gr.Markdown(label="Recovery assessment")

                gr.Markdown("### 💬 Follow-up")
                recovery_chat_box = gr.Chatbot(
                    label="",
                    type="messages",
                    elem_classes=["chatbot"],
                    show_label=False,
                )
                with gr.Row():
                    recovery_input = gr.Textbox(
                        placeholder="Ask a follow-up question...",
                        show_label=False,
                        scale=5,
                    )
                    recovery_send  = gr.Button("Send", scale=1)

                recovery_history = gr.State([])
                recovery_context = gr.State("")

                recovery_btn.click(
                    fn=run_recovery_checkin,
                    inputs=[session_selector],
                    outputs=[recovery_output, recovery_context, recovery_history],
                ).then(
                    fn=lambda h: h,
                    inputs=[recovery_history],
                    outputs=[recovery_chat_box],
                )

                recovery_send.click(
                    fn=recovery_chat,
                    inputs=[recovery_input, recovery_history, recovery_context],
                    outputs=[recovery_input, recovery_history],
                ).then(
                    fn=lambda h: h,
                    inputs=[recovery_history],
                    outputs=[recovery_chat_box],
                )
                recovery_input.submit(
                    fn=recovery_chat,
                    inputs=[recovery_input, recovery_history, recovery_context],
                    outputs=[recovery_input, recovery_history],
                ).then(
                    fn=lambda h: h,
                    inputs=[recovery_history],
                    outputs=[recovery_chat_box],
                )

            # ── Tab 3 — Dashboard ──────────────────────────────────────────────
            with gr.Tab("📊 Dashboard"):
                gr.Markdown("Live training load, race countdown, recent runs, and latest splits.")

                refresh_btn = gr.Button("🔄 Refresh dashboard", variant="primary")

                with gr.Row():
                    load_card   = gr.Markdown()
                    race_card   = gr.Markdown()

                with gr.Row():
                    recent_card  = gr.Markdown()
                    splits_card  = gr.Markdown()

                refresh_btn.click(
                    fn=build_dashboard,
                    outputs=[load_card, race_card, recent_card, splits_card],
                )

                # Auto-load on tab open
                demo.load(
                    fn=build_dashboard,
                    outputs=[load_card, race_card, recent_card, splits_card],
                )

    demo.launch(share=share, quiet=True)
    print("✅ Running Coach UI launched.")


if __name__ == "__main__":
    launch()
