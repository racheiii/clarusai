from datetime import datetime
import importlib

models = importlib.import_module("models")
config = importlib.import_module("config")

def _fake_scenario_row():
    return {
        "scenario_id": "MIL_CONF_001",
        "title": "Intel assessment",
        "bias_type": "Confirmation",  # CSV form on purpose
        "domain": "Military",
        "scenario_text": "You are an analyst...",
        "primary_prompt": "What is your assessment?",
        "follow_up_1": "What biases may be present?",
        "follow_up_2": "How to mitigate?",
        "follow_up_3": "Transfer this reasoning.",
        "ideal_primary_answer": "Balanced assessment…",
        "ideal_answer_1": "Identify confirmation bias…",
        "ideal_answer_2": "Use checklists, counterevidence…",
        "ideal_answer_3": "Apply to medical triage…",
        "cognitive_load_level": "Medium",
        "bias_learning_objective": "Recognise/mitigate confirmation bias",
        "rubric_focus": "similarity,bias,originality,transfer,metacognition",
    }

def test_csv_mapping_helpers_known_values():
    assert models.map_csv_bias_type("Confirmation") == models.BiasType.CONFIRMATION_BIAS
    assert models.map_csv_domain("Medical") == models.Domain.MEDICAL

def test_create_experimental_session_factory_and_types():
    srow = _fake_scenario_row()
    session = models.create_experimental_session(
        session_id="S1",
        user_expertise="novice",            # string form accepted
        ai_assistance_enabled=True,
        scenario_data=srow
    )
    # Types are enforced
    assert isinstance(session.user_expertise, models.UserExpertise)
    assert session.assigned_scenario.bias_type == models.BiasType.CONFIRMATION_BIAS
    assert session.assigned_scenario.domain == models.Domain.MILITARY
    assert session.condition_code.startswith("novice_")

def test_userresponse_post_init_word_char_counts():
    ur = models.UserResponse(
        stage_number=0,
        stage_name="Primary",
        stage_type=models.StageType.PRIMARY_ANALYSIS,
        response_text="This is a short test response.",
        response_time_seconds=12.3,
        timestamp=datetime.now(),
        cumulative_session_time=12.3,
        guidance_requested=False,
        prompt_text="Prompt",
        ideal_answer="Ideal"
    )
    assert ur.word_count >= 5 and ur.character_count >= 10

def test_quality_flags_update_uses_quality_thresholds():
    srow = _fake_scenario_row()
    session = models.create_experimental_session("S2", "expert", False, srow)
    # Two minimal responses to trigger some flags
    for i in range(2):
        session.add_response(models.UserResponse(
            stage_number=i,
            stage_name=f"Stage {i+1}",
            stage_type=list(models.StageType)[i],
            response_text="sufficient words in this response to pass threshold",
            response_time_seconds=15.0,
            timestamp=datetime.now(),
            cumulative_session_time=(i+1)*15.0,
            guidance_requested=False,
            prompt_text="Prompt",
            ideal_answer="Ideal"
        ))
    # Ensure quality flags dictionary is populated sensibly
    q = session.quality_flags
    assert q.get("sufficient_engagement") in {True, False}
    assert q.get("demonstrates_progression") is True
    assert "natural_timing" in q
