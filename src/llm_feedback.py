"""
ClārusAI: LLM Tutor Feedback (Ollama)

Purpose: Generates bias-blind, 1–2 sentence reflective feedback after each reasoning stage.
Uses local Ollama for reproducibility and offline generation.
"""

from src.llm_utils import generate_ollama_response
from config import LLM_ENABLED, OLLAMA_CONFIG
import logging

def generate_stage_feedback(scenario, stage_index, user_response) -> str:
    """Generate short, bias-free tutor-style feedback using LLaMA3 via Ollama."""

    # Respect config flag to disable LLM feedback
    if not LLM_ENABLED:
        return "ℹ️ LLM feedback is disabled."

    # Stage labels to specify task context
    stage_names = [
        "Primary Analysis", "Cognitive Factors",
        "Mitigation Strategies", "Transfer Learning"
    ]

    # Prompt fields mapped from scenario CSV
    prompt_fields = [
        "primary_prompt", "follow_up_1", "follow_up_2", "follow_up_3"
    ]

    # Guard against invalid stage index
    if not (0 <= stage_index < len(stage_names)):
        return f"⚠️ Invalid stage index: {stage_index}"

    # Construct prompt to pass to LLaMA3 model via Ollama
    prompt_text = (
        "You are an expert tutor giving reflective feedback to a trainee.\n"
        "The trainee has completed a professional reasoning task.\n\n"
        "You will see:\n"
        "• The real-world scenario\n"
        "• The task for this stage\n"
        "• The user's written response\n\n"
        "Write a 1–2 sentence feedback:\n"
        "• Start with one positive observation\n"
        "• Then give one area for improvement\n"
        "• Do NOT mention cognitive biases by name\n\n"
        f"--- SCENARIO ---\n{scenario.get('scenario_text', 'N/A')}\n\n"
        f"--- TASK ({stage_names[stage_index]}) ---\n{scenario.get(prompt_fields[stage_index], 'Prompt N/A')}\n\n"
        f"--- USER RESPONSE ---\n{user_response}\n\n"
        "🎓 Feedback:"
    )

    try:
        feedback = generate_ollama_response(
            prompt_text,
            model=str(OLLAMA_CONFIG.get("model") or "llama3.2:instruct")
        )
        return feedback if feedback else "⚠️ No feedback returned by model."
    except Exception as e:
        logging.exception("Error generating LLM feedback")
        return "⚠️ Feedback could not be generated using Ollama."
