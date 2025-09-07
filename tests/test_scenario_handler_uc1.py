import pandas as pd

def test_load_and_validate_scenarios(monkeypatch):
    import config
    from src.scenario_handler import ScenarioHandler
    monkeypatch.setattr(config, "SCENARIOS_FILE", config.SCENARIOS_FILE, raising=False)

    h = ScenarioHandler()
    df = h.load_scenarios()
    assert isinstance(df, pd.DataFrame) and not df.empty, "scenarios.csv failed to load"

    structure = h._validate_scenarios_structure()
    assert structure.get("valid") is True


def test_weighted_selection_returns_row(monkeypatch):
    import config
    from src.scenario_handler import ScenarioHandler
    from src.models import UserExpertise
    monkeypatch.setattr(config, "SCENARIOS_FILE", config.SCENARIOS_FILE, raising=False)

    h = ScenarioHandler()
    _ = h.load_scenarios()

    row = h.select_balanced_scenario(
        user_expertise=UserExpertise.NOVICE,
        ai_assistance=True,
        domain=None 
    )
    assert row is not None
    assert {"scenario_id", "bias_type", "domain"}.issubset(set(row.index))
