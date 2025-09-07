def test_generate_ollama_response_offline(monkeypatch):
    from src.llm_utils import generate_ollama_response, check_ollama_availability
    monkeypatch.setattr("src.llm_utils.check_ollama_availability", lambda: False)
    msg = generate_ollama_response("Test prompt")
    assert isinstance(msg, str)
    assert msg.startswith("⚠️"), "Expected graceful offline warning when Ollama unavailable"