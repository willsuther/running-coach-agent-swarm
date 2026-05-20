# 🏃 Personal Running Coach Agent Swarm

**Agentic AI Course — Will Sutherland | May 2026**

A multi-agent AI system designed to support half-marathon training through automated workout analysis, recovery monitoring, and weekly planning. Built around real personal training data for the Fredericton Half Marathon (May 2026).

---

## Project Motivation

Rather than building a demo system, this project automates the actual weekly workflow of a competitive runner training under a coach:

- **Did I hit my pace targets today?**
- **Am I recovering well enough to train hard tomorrow?**
- **What should next week look like given where my fitness is right now?**

The agent swarm answers these questions by combining Garmin physiological data (HRV, sleep, body battery, training readiness), Strava activity metrics, and a ChromaDB vector memory of training history — then reasoning over all of it with Gemini 2.5 Flash.

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                  Gradio UI                          │
│  Tab: Workout Feedback | Recovery | Dashboard       │
└────────────┬──────────────┬──────────────┬──────────┘
             │              │              │
             ▼              ▼              ▼
     ┌───────────┐  ┌────────────┐  ┌────────────┐
     │ Feedback  │  │  Recovery  │  │  Planner   │
     │  Agent    │  │   Agent    │  │   Agent    │
     └─────┬─────┘  └─────┬──────┘  └─────┬──────┘
           │              │               │
           └──────────────┼───────────────┘
                          │
              ┌───────────▼────────────┐
              │       Tool Layer       │
              │  parse_workout_data    │
              │  calculate_pace_zones  │
              │  calculate_training_  │
              │  load (ATL/CTL/TSB)   │
              │  query_garmin_db      │
              └───────────┬────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
     ┌──────────┐  ┌──────────┐  ┌──────────────┐
     │ Strava   │  │garmin.db │  │   ChromaDB   │
     │   API    │  │(SQLite)  │  │    Memory    │
     └──────────┘  └──────────┘  └──────────────┘
```

---

## Agent Design

| Agent | Role | Tools | Key Safeguard |
|-------|------|-------|---------------|
| **Coordinator** | Routes user queries to the right specialist agent, orchestrates multi-agent calls, synthesizes final responses, runs self-critique safety pass on all outputs | None (LLM only) | Self-critique reviewer on every subagent output before delivery |
| **Feedback** | Analyses completed workouts — pace vs target, HR, lap splits, patterns | `parse_workout_data`, `calculate_pace_zones`, `query_garmin_db` | Injury disclaimer intercept on pain/symptom keywords |
| **Recovery** | Daily check-in and pre-workout readiness — HRV, sleep, body battery, TSB | `calculate_training_load`, `query_garmin_db` | `[RECOVERY ALERT]` when TSB < −30; session modification suggestions |
| **Planner** | Generates weekly training plans and reviews planned sessions vs recovery | `calculate_training_load`, `query_garmin_db`, ChromaDB memory | 10% mileage rule enforced; automatic taper logic in final 14 days |

### Coordinator Agent

The Coordinator is the user-facing entry point for the swarm. Rather than requiring the user to know which agent to invoke, the Coordinator:

1. **Classifies the query** — determines whether it is a feedback, recovery, or planning question using Gemini's reasoning
2. **Routes to the appropriate specialist** — invokes the Feedback, Recovery, or Planner agent with the relevant context
3. **Synthesizes the response** — combines subagent outputs into a single coherent reply
4. **Self-critique pass** — reviews the combined output before delivery, checking for: unsafe recommendations, violations of the 10% mileage rule, missing injury disclaimers, and contradictions between agents
5. **Multi-agent queries** — for queries that span agents (e.g. "should I do tomorrow's threshold session given how today's run went?"), the Coordinator calls both Feedback and Recovery and synthesizes their outputs

```
User: "How did my run go today and should I do tomorrow's threshold?"
    │
    ▼
Coordinator — classifies as multi-agent (Feedback + Recovery)
    ├── → Feedback Agent (today's run analysis)
    ├── → Recovery Agent (pre-workout check: threshold)
    └── → synthesize + self-critique → final response
```

---

## Technology Stack

| Component | Choice |
|-----------|--------|
| LLM | Gemini 2.5 Flash (Google AI Studio API) |
| Intra-agent orchestration | LangChain tool loop pattern |
| Memory / RAG | ChromaDB (persisted) + Gemini `gemini-embedding-001` |
| UI | Gradio (runs in Colab, shareable public URL) |
| Primary data source | `garmin.db` — SQLite database via `garmin-givemydata` |
| Activity trigger | Strava API (OAuth + stravalib) |
| Environment | Google Colab (Python 3.12) |

---

## Data Pipeline

```
Local machine (Windows)
└── garmin-givemydata → garmin.db
        (headless Chrome bypasses Garmin's Cloudflare protection)
        │
        ▼
Google Drive / running_coach / data / raw / garmin / garmin.db
        │
        ▼
Colab — Phase 1 notebook
  ├── Strava API pull (90-day history)
  ├── Garmin DB enrichment (training load, aerobic effect, HRV, sleep, body battery)
  └── Normalized WorkoutRecord JSON → data/processed/workouts_normalized.json
        │
        ▼
Phase 3 — ChromaDB ingestion
  ├── workout_summaries  (180 days of runs as natural language chunks)
  ├── athlete_profile    (static background, goals, coaching context)
  └── session_notes      (free-text post-session notes)
```

**Key data sources from `garmin.db`:**

| Table | Contents | Used by |
|-------|----------|---------|
| `activity` | All Garmin activities with training load, aerobic/anaerobic TE | Feedback, Planner |
| `activity_splits` | Lap-by-lap pace, HR, distance | Feedback (split analysis) |
| `hrv` | HRV status, weekly avg, baseline bounds | Recovery |
| `sleep` | Sleep stages, total hours, SpO2, feedback | Recovery |
| `body_battery` | Wake value, daily min/max (parsed from raw_json) | Recovery |
| `stress` | Daily avg/max stress (parsed from raw_json) | Recovery |
| `training_readiness` | Score, level, factor breakdown | Recovery, Planner |
| `race_predictions` | Garmin's predicted 5K/10K/HM/marathon times | Planner, Dashboard |

---

## Workout Classification System

Workouts are classified by type using a three-tier detection system:

1. **Known exceptions** — hardcoded by date (e.g. April 8: `8 x 800m @ VO2 Max`)
2. **Garmin date-named workouts** — regex match on `YYYY-MM-DD` in activity name (e.g. `Halifax - 2026-04-15`)
3. **Name pattern matching** — `"Halifax Running"`, `"Morning Run"`, `"easy"` → auto-classified as easy

Pace evaluation uses coach-prescribed targets with ±5 second tolerance:

| Session type | Target pace |
|-------------|-------------|
| Easy | By feel, slower than 4:45/km |
| Marathon | 4:14/km |
| Threshold | 4:01/km |
| 1hr pace | 3:56/km |
| Fartlek | 3:49/km |
| 8K pace | 3:46/km |
| VO2 Max | 3:40/km |

---

## Training Load Model

ATL/CTL/TSB computed from Garmin's per-session `training_load` scores using exponential decay:

```
ATL_today = ATL_yesterday × exp(−1/7)  + load × (1 − exp(−1/7))
CTL_today = CTL_yesterday × exp(−1/42) + load × (1 − exp(−1/42))
TSB = CTL − ATL
```

- **ATL** (7-day): acute fatigue
- **CTL** (42-day): chronic fitness
- **TSB**: form — positive = fresh, negative = fatigued
- **Alert threshold**: TSB < −30 triggers `[RECOVERY ALERT]`

---

## Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Injury disclaimer | Keyword intercept on 20+ injury terms in all agents and follow-up chat |
| Recovery alert | `[RECOVERY ALERT]` flag in all agent responses when TSB < −30 |
| 10% mileage rule | Enforced in Planner with week projection for incomplete weeks |
| Taper logic | Automatic phase detection — reduced volume/intensity in final 14 days |
| Session modification | Recovery and Planner agents recommend PROCEED / MODIFY / BACK OFF |

---

## Repository Structure

```
running_coach/
├── notebooks/
│   ├── 01_phase1_environment_setup.ipynb   # Data ingestion — Strava + Garmin
│   ├── 02_phase2_test_tools.ipynb          # Tool layer test suite (17 tests)
│   ├── 03_phase3_chromadb_memory.ipynb     # ChromaDB memory ingestion
│   └── 04_agents.ipynb                     # Agent runners
├── tools/
│   ├── parse_workout_data.py               # Load, validate, summarise WorkoutRecords
│   ├── calculate_pace_zones.py             # Workout classification and pace evaluation
│   ├── calculate_training_load.py          # ATL, CTL, TSB, mileage rule, weekly summary
│   ├── query_garmin_db.py                  # HRV, sleep, body battery, splits queries
│   └── memory_retrieval.py                 # ChromaDB retrieval wrapper
├── agents/
│   ├── feedback_agent.py                   # Workout analysis vs targets
│   ├── recovery_agent.py                   # Recovery monitoring and readiness
│   ├── planner_agent.py                    # Weekly schedule generation and review
│   └── coordinator_agent.py                # Query routing, orchestration, self-critique
├── ui/
│   └── coach_ui.py                         # Gradio interface (3 tabs)
├── config.py                               # Shared constants and athlete profile
└── .gitignore
```

---

## Setup and Running

### Prerequisites

- Google Colab account
- Google Drive with the `running_coach/` folder structure
- Google AI Studio API key (stored as Colab Secret `key`)
- Strava account with API app configured (secrets: `STRAVA_CLIENT_ID`, `STRAVA_CLIENT_SECRET`, `STRAVA_REFRESH_TOKEN`)
- `garmin.db` generated locally via [garmin-givemydata](https://github.com/nrvim/garmin-givemydata) and uploaded to `data/raw/garmin/`

### Data setup (one-time)

1. Run `garmin-givemydata` locally to generate `garmin.db`
2. Upload `garmin.db` to `Google Drive/running_coach/data/raw/garmin/`
3. Run **Phase 1 notebook** top-to-bottom — connects to Strava, enriches with Garmin data, writes `workouts_normalized.json`
4. Run **Phase 3 notebook** — ingests workout history and athlete profile into ChromaDB

### Running the agents

Open `04_agents.ipynb` in Colab and run the setup cell first (installs dependencies, mounts Drive, loads API key). Then run whichever agent section you want:

```python
# Coordinator Agent — natural language entry point for the full swarm
from coordinator_agent import CoordinatorAgent
agent = CoordinatorAgent(api_key=GEMINI_API_KEY)
agent.run()  # Just ask anything — the Coordinator routes to the right agent(s)

# Example queries the Coordinator handles:
# "How did my run go today?"                    → routes to Feedback
# "Am I recovered enough for tomorrow?"         → routes to Recovery
# "What should next week look like?"            → routes to Planner
# "How did today's run go and should I do      → routes to Feedback + Recovery
#  tomorrow's threshold as planned?"

# Feedback Agent — analyse a completed workout
from feedback_agent import FeedbackAgent
agent = FeedbackAgent(api_key=GEMINI_API_KEY)
agent.run()                        # Interactive menu
agent.run(date='2026-04-22')       # Specific date

# Recovery Agent — check-in or pre-workout readiness
from recovery_agent import RecoveryAgent
agent = RecoveryAgent(api_key=GEMINI_API_KEY)
agent.run()                              # Interactive menu
agent.run(planned_session='threshold')   # Pre-workout check

# Planner Agent — generate or review weekly plan
from planner_agent import PlannerAgent
agent = PlannerAgent(api_key=GEMINI_API_KEY)
agent.run()                   # Interactive menu
agent.run(mode='generate')    # Next week's plan
agent.run(mode='review')      # Review this week's sessions
```

### Running the Gradio UI

```python
%pip install -q gradio
from ui.coach_ui import launch
launch(share=True)  # Generates a public gradio.live URL
```

---

## Athlete Context

- **Athlete:** Will Sutherland, Halifax NS
- **Running since:** Fall 2023
- **PBs:** Half marathon 1:27:50 | Marathon 3:09
- **Target race:** Fredericton Half Marathon, May 10 2026
- **Goal:** Sub 1:23:59 (A+) / Sub 1:25:00 (B)
- **Device:** Garmin Forerunner 265
- **Coach:** Yes — structured 18-week periodized plan

---

## Known Limitations & Workarounds

### ChromaDB Embedding Function Conflict

**Issue:** ChromaDB's built-in `GoogleGenerativeAiEmbeddingFunction` uses an older version of `google-api-core` that conflicts with the current `google-generativeai` package in Colab. Calling it after `genai.configure()` raises:
```
ValueError: ClientOptions does not accept an option 'headers'
```

**Workaround:** All ChromaDB queries use a custom `GeminiEmbedder` class that calls the Gemini embedding REST API directly, bypassing the conflicting client entirely:

```python
import requests

class GeminiEmbedder:
    def __init__(self, api_key, model='models/gemini-embedding-001'):
        self.api_key = api_key
        self.model   = model
        self.name    = 'gemini-embedder'
    def __call__(self, input):
        embeddings = []
        for text in input:
            url  = f'https://generativelanguage.googleapis.com/v1beta/{self.model}:embedContent?key={self.api_key}'
            resp = requests.post(url, json={'model': self.model, 'content': {'parts': [{'text': text}]}})
            embeddings.append(resp.json()['embedding']['values'])
        return embeddings
```

Collections must be retrieved with `get_collection(name)` (no `embedding_function` argument) and queried with `query_embeddings=[vector]` rather than `query_texts`. This pattern is used consistently across the demo notebook and all agent code.

### Gemini API Timeouts

The Gemini API occasionally returns 503 errors or times out under load, particularly for the Planner Agent which sends longer prompts. The agents include retry logic (`for attempt in range(3)`) and the Coordinator degrades gracefully when a subagent fails — delivering the successful agent's response rather than crashing. If timeouts persist, wait a few minutes and retry; this is an API availability issue, not a code issue.

---



Every component of this system runs on real personal training data:

- 807 Garmin activities going back to 2016
- 843 nights of HRV data
- 819 nights of sleep data
- 3,651 days of body battery, stress, and resting HR records
- 70 runs from the last 90 days used for pace zone classification and training load calculation
- ChromaDB populated with 180 days of workout summaries and a complete athlete profile

The agents are used as part of actual race preparation — the Feedback Agent was tested on real workout splits, the Recovery Agent on real HRV and readiness scores, and the Planner on the actual taper week leading into Fredericton.
