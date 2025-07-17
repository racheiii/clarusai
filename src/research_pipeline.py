"""
ClārusAI: Simulation Pipeline for Response Generation

This script generates 96 simulated sessions (2×2×3×4):
- 2 expertise levels × 2 AI assistance × 3 bias types × 4 scenarios per bias
Each session has 4 reasoning stages and is saved individually.
"""

import os
import json
from datetime import datetime
from itertools import product
from src.sim_user_generator import SimulatedUserGenerator
from src.models import UserExpertise, BiasType
from config import RESPONSES_DIR
from pathlib import Path

# Output directory for generated sessions
OUTPUT_DIR = Path(RESPONSES_DIR)
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
    exit(1)

# Generation loop
session_counter = 0
total_sessions = len(expertise_levels) * len(ai_assist_options) * len(bias_types) * len(SCENARIO_SUFFIXES)

print(f"🚀 Starting full 96-session simulation (target: {total_sessions} sessions)")
start_time = datetime.now()

for expertise, ai_assist, bias_type, suffix in product(expertise_levels, ai_assist_options, bias_types, SCENARIO_SUFFIXES):
    bias_str = bias_type.value
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
        "confirmation_bias": "Confirmation",
        "anchoring_bias": "Anchoring",
        "availability_heuristic": "Availability"
    }.get(bias_str, "")

    matching_scenarios = generator.scenarios_df[
        (generator.scenarios_df["bias_type"].str.lower() == bias_csv_name.lower()) &
        (generator.scenarios_df["scenario_id"].str.endswith(suffix))
    ]

    if matching_scenarios.empty:
        print(f"⚠️ No scenario found for {bias_str} with suffix {suffix}")
        continue

    scenario_row = matching_scenarios.iloc[0]
    scenario_dict = scenario_row.to_dict()

    # Inject temporary scenario override
    generator.scenarios_df = matching_scenarios  # Override DF temporarily
    session_data = generator.create_session_data(
        session_id=session_id,
        expertise=expertise,
        ai_assistance=ai_assist,
        bias_type=bias_type
    )
    generator.load_scenarios()  # Restore full DF

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
