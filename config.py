# running_coach/config.py
# Shared constants used across all agents and tools.
# Edit to match your actual athlete profile before running Phase 2+.

# ── Athlete profile ────────────────────────────────────────────────────────────
ATHLETE_NAME        = "Will Sutherland"
TARGET_RACE         = "Fredericton Half Marathon"
RACE_DATE           = "2026-05-10"   # TODO: confirm actual race date
CURRENT_WEEKLY_KM   = 70             # TODO: update with current weekly volume

# ── Training paces (min/km) — +/- 5 seconds tolerance applied in tools ───────
# Easy run is effort-based (by feel) — no strict target, ceiling at 5:30/km
TRAINING_PACES = {
    "easy":       None,            # By feel — no target
    "marathon":   4 + 14/60,       # 4:14/km
    "threshold":  4 + 01/60,       # 4:01/km
    "1hr":        3 + 56/60,       # 3:56/km
    "fartlek":    3 + 49/60,       # 3:49/km
    "8k":         3 + 46/60,       # 3:46/km
    "vo2max":     3 + 40/60,       # 3:40/km
}
EASY_PACE_CEILING  = 4 + 50/60   # Faster than 4:50/km = too hard for easy run
PACE_TOLERANCE_SEC = 5            # +/- 5 seconds counts as on-target

# ── Heart rate zones (bpm) ────────────────────────────────────────────────────
HR_MAX = 195  # TODO: update with your measured max HR
HR_ZONES = {
    "Z1": (0,                    int(HR_MAX * 0.60)),
    "Z2": (int(HR_MAX * 0.60),   int(HR_MAX * 0.70)),
    "Z3": (int(HR_MAX * 0.70),   int(HR_MAX * 0.80)),
    "Z4": (int(HR_MAX * 0.80),   int(HR_MAX * 0.90)),
    "Z5": (int(HR_MAX * 0.90),   HR_MAX),
}

# ── Training load thresholds ───────────────────────────────────────────────────
TSB_ALERT_THRESHOLD  = -30   # [RECOVERY ALERT] when TSB drops below this
MILEAGE_RULE_PCT     = 0.10  # 10% weekly mileage increase cap
ATL_DECAY            = 7     # Acute Training Load decay constant (days)
CTL_DECAY            = 42    # Chronic Training Load decay constant (days)

# ── Data ingestion ─────────────────────────────────────────────────────────────
# Garmin DB: sync locally via garmin-givemydata, copy garmin.db to Drive
# Strava:    automated pull via OAuth on each session
STRAVA_HISTORY_DAYS  = 90

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = "/content/drive/MyDrive/running_coach"
GARMIN_DB_PATH  = f"{BASE_DIR}/data/raw/garmin/garmin.db"
STRAVA_RAW_DIR  = f"{BASE_DIR}/data/raw/strava"
PROC_DATA_DIR   = f"{BASE_DIR}/data/processed"
CHROMA_DIR      = f"{BASE_DIR}/memory/chroma"
AGENTS_DIR      = f"{BASE_DIR}/agents"
TOOLS_DIR       = f"{BASE_DIR}/tools"

# ── Model ──────────────────────────────────────────────────────────────────────
GEMINI_MODEL    = "gemini-2.5-flash"
EMBED_MODEL     = "models/embedding-001"