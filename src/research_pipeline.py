"""
ClārusAI: Simulation Pipeline for Response Generation

Generates 48 simulated sessions (2×2×3×4): 2 expertise levels × 2 AI assistance ×
3 bias types × 4 scenarios per bias. Each session has 4 reasoning stages.
"""

import json
from datetime import datetime
from itertools import product
from src.sim_user_generator import SimulatedUserGenerator
from src.models import UserExpertise, BiasType
from pathlib import Path
from config import EXPORTS_DIR

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S_run")
OUTPUT_DIR = Path(EXPORTS_DIR) / "simulated_datasets" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Bias scenario suffixes: each bias has 4 scenarios with suffixes _001 to _004
SCENARIO_SUFFIXES = ["_001", "_002", "_003", "_004"]

# Factor levels
expertise_levels = [UserExpertise.NOVICE, UserExpertise.EXPERT]
ai_assist_options = [False, True]
bias_types = [BiasType.CONFIRMATION_BIAS, BiasType.ANCHORING_BIAS, BiasType.AVAILABILITY_HEURISTIC]

# Generator instance
generator = SimulatedUserGenerator()

# Load scenario dataframe
if not generator.load_scenarios():
    print("❌ Failed to load scenario CSV. Exiting.")
    raise SystemExit(1)

# Generation loop
session_counter = 0
total_sessions = len(expertise_levels) * len(ai_assist_options) * len(bias_types) * len(SCENARIO_SUFFIXES)

print(f"🚀 Starting full {total_sessions}-session simulation")
start_time = datetime.now()

for expertise, ai_assist, bias_type, suffix in product(expertise_levels, ai_assist_options, bias_types, SCENARIO_SUFFIXES):
    bias_str = bias_type.value  # "confirmation" | "anchoring" | "availability"
    session_id = f"SIM_{expertise.value}_{ai_assist}_{bias_str}_{suffix}"
    output_file = OUTPUT_DIR / f"{session_id}.json"

    # Skip if already exists
    if output_file.exists():
        print(f"⏭️ Skipping existing: {session_id}")
        continue

    # Safety check
    if generator.scenarios_df is None:
        print(f"❌ Scenario dataframe not loaded for {session_id}")
        continue

    # Find matching scenario by bias + suffix
    bias_csv_name = {
        "confirmation": "Confirmation",
        "anchoring": "Anchoring",
        "availability": "Availability"
    }[bias_str]

    matching_scenarios = generator.scenarios_df[
        (generator.scenarios_df["bias_type"].str.lower() == bias_csv_name.lower()) &
        (generator.scenarios_df["scenario_id"].str.endswith(suffix))
    ]

    if matching_scenarios.empty:
        print(f"⚠️ No scenario found for {bias_str} with suffix {suffix}")
        continue

    # Inject temporary scenario override safely
    _full_df = generator.scenarios_df
    try:
        generator.scenarios_df = matching_scenarios
        session_data = generator.create_session_data(
            session_id=session_id,
            expertise=expertise,
            ai_assistance=ai_assist,
            bias_type=bias_type
        )
    finally:
        generator.scenarios_df = _full_df

    if session_data:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        session_counter += 1
        print(f"✅ Generated ({session_counter}/{total_sessions}): {session_id}")
    else:
        print(f"❌ Failed to generate: {session_id}")

# Summary
elapsed = (datetime.now() - start_time).total_seconds()
print(f"🎯 Done. Total generated: {session_counter}/{total_sessions}")
print(f"⏱ Total time: {elapsed:.1f} seconds")