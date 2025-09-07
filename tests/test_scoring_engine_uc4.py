def test_comprehensive_scores_shape_and_bounds():
    """
    Ensures the 6-D scoring runs without downloading an embedding model, and
    tolerates either flat or nested output schema.
    """
    import sys, types
    from typing import Type

    class _NoModel:
        def __init__(self, *a, **k):
            raise RuntimeError("disabled in tests")

    class _STModule(types.ModuleType):
        SentenceTransformer: Type[_NoModel]

    _mock_st = _STModule("sentence_transformers")
    _mock_st.SentenceTransformer = _NoModel
    sys.modules["sentence_transformers"] = _mock_st
    # ---------------------------------------------------------------------

    from src import scoring_engine as se

    response = "I will challenge my initial assumptions and seek disconfirming evidence."
    ideal    = "Seek disconfirming evidence, triangulate sources, and avoid anchoring."
    scenario = {"domain": "Medical", "bias_type": "Confirmation"}

    out = se.calculate_comprehensive_scores(
        response=response,
        ideal_answer=ideal,
        scenario=scenario,
    )

    def _score(flat_key, nested_key):
        val = out.get(flat_key)
        if isinstance(val, dict):
            return float(val.get("score", 0.0))
        if val is not None:
            return float(val)
        obj = out.get(nested_key, {})
        return float(obj.get("score", 0.0))

    def _count(flat_key, nested_key):
        if flat_key in out:
            return int(out[flat_key])
        obj = out.get(nested_key, {})
        return int(obj.get("count", 0))
    # --------------------------------------------------------------------

    # Semantic similarity may be flat or nested (just exercise access)
    _ = _score("semantic_similarity", "semantic_similarity")

    # Tolerate v1 (flat) or v2 (nested) schema keys
    bias_cnt   = _count("bias_recognition_count",   "bias_recognition")
    orig_score = _score("originality_score",        "conceptual_originality")
    strat_cnt  = _count("strategy_count",           "mitigation_strategy")
    trans_cnt  = _count("transfer_count",           "domain_transferability")
    meta_cnt   = _count("metacognition_count",      "metacognitive_awareness")

    # Basic bounds/sanity
    assert 0 <= bias_cnt
    assert 0 <= strat_cnt
    assert 0 <= trans_cnt
    assert 0 <= meta_cnt
    assert 0.0 <= orig_score <= 1.0

    # Always-present keys
    assert "overall_quality_score" in out
    assert "confidence_flags" in out
    assert 0.0 <= float(out["overall_quality_score"]) <= 4.0

    flags = out["confidence_flags"]

    # Accept Python bools and NumPy bools across versions
    try:
        import numpy as np
        _bool_types = (bool, getattr(np, "bool_", bool))
    except Exception:
        _bool_types = (bool,)

    assert isinstance(flags.get("low_effort", False), _bool_types)
    assert isinstance(flags.get("high_similarity_risk", False), _bool_types)
