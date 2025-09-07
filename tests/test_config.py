import types
import importlib

# Import once so config auto-creates folders/loads env
config = importlib.import_module("config")

def test_scoring_thresholds_in_range():
    t = config.SCORING_THRESHOLDS
    # numeric and sensible bands
    assert 0 <= t["semantic_similarity_medium"] < t["semantic_similarity_high"] <= 1
    assert 0 <= t["originality_low"] < t["originality_high"] <= 1
    assert isinstance(t["bias_recognition_min_terms"], int) and t["bias_recognition_min_terms"] >= 1

def test_bias_keywords_and_semantic_cues_present():
    # sanity: each bias has a decent vocabulary
    for k, v in config.BIAS_KEYWORDS.items():
        assert isinstance(v, list) and len(v) >= 10, f"Too few keywords for {k}"
    # semantic cue keys align with expected labels
    for k, v in config.SEMANTIC_BIAS_CUES.items():
        assert k in {"confirmation", "anchoring", "availability"}
        assert isinstance(v, list) and len(v) > 0

def test_bias_normalisation_helpers_roundtrip():
    # CSV<->enum-name normalisation and keyword access are consistent
    for csv_label, enum_label in [
        ("Confirmation", "confirmation_bias"),
        ("Anchoring", "anchoring_bias"),
        ("Availability", "availability_heuristic"),
    ]:
        assert config.normalize_bias_type(csv_label) == enum_label
        assert config.normalize_bias_type(enum_label) == csv_label
        # keywords accessible under both forms and identical
        a = config.get_bias_keywords(csv_label)
        b = config.get_bias_keywords(enum_label)
        assert a and b and a == b

def test_validation_rules_and_directories():
    # basic structure present and folders exist (created on import)
    rules = config.VALIDATION_RULES
    assert set(["scenario_id", "bias_type", "domain", "title"]).issubset(set(rules["required_scenario_columns"]))
    assert all(__import__("os").path.exists(p) for p in [config.DATA_DIR, config.RESPONSES_DIR, config.EXPORTS_DIR])

def test_validate_config_passes_core_checks():
    results = config.validate_config()
    assert isinstance(results, dict) and results
    # these are the core gates for experimental integrity
    for key in ["bias_keywords_complete", "thresholds_set", "directories_exist", "validation_rules_complete"]:
        assert results.get(key) is True

def test_get_config_summary_shape():
    summary = config.get_config_summary()
    assert isinstance(summary, dict)
    assert "scoring_thresholds" in summary and summary["scoring_thresholds"] == config.SCORING_THRESHOLDS
