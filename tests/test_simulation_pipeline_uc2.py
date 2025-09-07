from typing import Tuple

def test_single_session_smoke_without_ollama(monkeypatch):
    from src.sim_user_generator import SimulatedUserGenerator
    import config                     
    from src.models import UserExpertise, BiasType

    gen = SimulatedUserGenerator(
        scenarios_csv_path=config.SCENARIOS_FILE,
        replicates_per_condition=1,
        seed=123
    )
    assert gen.load_scenarios() is True

    def fake_llm(scenario, stage, expertise, ai_assistance) -> Tuple[str, float]:
        return (f"Dummy response for stage {stage}.", 4.2)
    monkeypatch.setattr(gen, "generate_llama3_response", fake_llm)

    data = gen.create_session_data(
        session_id="SIM_TEST",
        expertise=UserExpertise.NOVICE,
        ai_assistance=True,
        bias_type=BiasType.CONFIRMATION_BIAS,
    )
    assert isinstance(data, dict)
    assert data["experimental_metadata"]["condition_code"].startswith("novice_True_confirmation")
    stages = data.get("stage_responses", [])
    assert len(stages) == 4 and all("response_text" in s for s in stages)