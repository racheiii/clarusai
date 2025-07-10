"""
ClārusAI: LLM Tutor Feedback
Generates 1–2 sentence bias-blind feedback for user reflection on each stage.
"""

import config

try:
    import google.generativeai as genai
    from google.generativeai.generative_models import GenerativeModel
    GEMINI_AVAILABLE = config.GEMINI_API_KEY is not None
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    GenerativeModel = None

def generate_stage_feedback(scenario, stage_index, user_response) -> str:
    if not GEMINI_AVAILABLE or GenerativeModel is None:
        return "⚠️ Feedback not available. LLM key is not set."

    try:
        model = GenerativeModel(
            model_name=config.GEMINI_CONFIG["model"],
            generation_config=config.GEMINI_CONFIG["generation_config"],
            safety_settings=config.GEMINI_CONFIG["safety_settings"]
        )

        stage_names = [
            "Primary Analysis", "Cognitive Factors",
            "Mitigation Strategies", "Transfer Learning"
        ]

        prompt_fields = [
            "primary_prompt", "follow_up_1", "follow_up_2", "follow_up_3"
        ]

        prompt_text = f"""You are acting as an expert tutor giving feedback to a professional trainee.
This trainee has completed a cognitive reasoning task. You will be shown:

• The real-world scenario they were given
• The stage-specific task they answered
• Their written response

Please provide feedback on their reasoning in exactly 1–2 sentences. 
Start with one positive insight, followed by one specific improvement.
⚠️ Do NOT mention cognitive biases (e.g., 'confirmation bias') by name.

--- SCENARIO ---
{scenario.get("scenario_text", "N/A")}

--- TASK ({stage_names[stage_index]}) ---
{scenario.get(prompt_fields[stage_index], "Prompt N/A")}

--- USER RESPONSE ---
{user_response}

🎓 Feedback (1–2 sentences only):
"""

        response = model.generate_content(prompt_text)
        return response.text.strip()

    except Exception as e:
        return "⚠️ Feedback could not be generated due to an internal error."