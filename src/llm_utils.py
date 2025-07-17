"""
llm_utils.py - Centralised Ollama LLM invocation for ClārusAI

Provides a shared interface for all LLM-related calls:
- Simulated user generation
- Tutor feedback generation
- AI guidance generation
"""

import subprocess

def generate_ollama_response(prompt: str, model: str = "llama3", temperature: float = 0.7, max_tokens: int = 200) -> str:
    """
    Runs an LLM prompt using Ollama with consistent fallback handling.

    Args:
        prompt (str): The prompt to send to the model
        model (str): The Ollama model to use (default: llama3)
        temperature (float): Temperature parameter for generation
        max_tokens (int): Token cap (approximate control, depends on model config)

    Returns:
        str: Model output or fallback error message
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        response = result.stdout.decode().strip()
        return response if response else "⚠️ Empty response from model."
    except Exception as e:
        return f"⚠️ Ollama generation failed: {str(e)}"
