import json

def test_write_response_files_json(tmp_path, monkeypatch):
    import config
    from src.data_collector import DataCollector
    dc = DataCollector()

    # Redirect RESPONSES_DIR to a temporary location
    monkeypatch.setattr(config, "RESPONSES_DIR", str(tmp_path), raising=False)

    payload = {
        "timestamp": "2025-09-04T12:00:00",
        "session_id": "TEST",
        "user_expertise": "novice",
        "ai_assistance_enabled": True,
        "scenario_id": "CONFIRM_001",
        "bias_type": "confirmation",
        "domain": "medical",
        "cognitive_load_level": "Medium",
        "stage_number": 0,
        "stage_name": "Primary Analysis",
        "stage_type": "primary_analysis",
        "stage_prompt": "Analyse the situation",
        "response_text": "Some analysis text.",
        "response_time_seconds": 3.2,
        "word_count": 3,
        "character_count": 20,
        "guidance_requested": False,
        "cumulative_guidance_pattern": [False],
        "scoring_results": {"overall_quality_score": 2.0, "semantic_similarity": 0.5},
        "ideal_answer": "Cross-check evidence.",
        "rubric_focus": "mitigation",
        "bias_learning_objective": "recognise confirmation",
        "condition_code": "novice_True_confirmation",
        "experimental_phase": "live_data_collection",
        "data_quality_flags": {"non_empty": True, "scoring_successful": True},
        "source_type": "live_user_4stage",
        "bias_revelation_status": "pre_revelation",
        "data_collection_version": "2.0",
    }

    ok = dc._write_response_files(payload, stage=0)
    assert ok is True

    # Ensure a stage file was created and is valid JSON
    files = list(tmp_path.glob("*.json"))
    assert files, "No stage file written"
    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("condition_code") == "novice_True_confirmation"