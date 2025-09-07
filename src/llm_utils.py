"""
llm_utils.py - LLM Interface for ClārusAI Research Framework

Provides standardised Ollama model access for experimental data generation.
Designed for batch processing during 2×2×3 factorial simulation.
"""

import subprocess
from typing import Tuple
from config import OLLAMA_CONFIG
import logging


def generate_ollama_response(
    prompt: str,
    model: str = "",
    timeout: int = 120
) -> str:
    """
    Generate LLM response using Ollama with timeout protection.

    Args:
        prompt: Input text for model generation
        model: Ollama model identifier (default: from config.OLLAMA_CONFIG)
        timeout: Maximum seconds to wait for response

    Returns:
        Model-generated text, or error message prefixed with ⚠️
    """
    if not prompt.strip():
        return "⚠️ Empty prompt provided"

    # Default to model from config if not explicitly provided
    if not model:
        model = OLLAMA_CONFIG.get("model", "llama3.2")

    # Early exit if Ollama is unavailable
    if not check_ollama_availability():
        return "⚠️ Ollama service not available"

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )

        if result.returncode == 0:
            response = result.stdout.decode("utf-8").strip()
            return response if response else "⚠️ Empty response from model"
        else:
            error_output = result.stderr.decode("utf-8").strip()
            return f"⚠️ Model execution failed: {error_output}"

    except subprocess.TimeoutExpired:
        return f"⚠️ Generation timeout after {timeout}s"
    except FileNotFoundError:
        return "⚠️ Ollama not installed or not in PATH"
    except Exception as e:
        logging.exception("Error during Ollama generation")
        return f"⚠️ Generation error: {str(e)}"

def check_ollama_availability() -> bool:
    """
    Verify if Ollama service is accessible and responsive.
    
    Returns:
        True if Ollama responds to basic commands, False otherwise
        
    Purpose:
        Used by fallback logic to determine whether to attempt LLM generation
        or use template responses for reliable experimental completion.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def validate_system_ready() -> Tuple[bool, str]:
    """
    Pre-experiment validation to ensure reliable batch generation.
    
    Returns:
        Tuple of (is_ready, status_message)
        
    Checks:
        - Ollama service availability
        - Model functionality with test prompt
        - System responsiveness under load
    """
    # Check service availability
    if not check_ollama_availability():
        return False, "Ollama service not available"
    
    # Test model functionality
    test_response = generate_ollama_response("Test", timeout=30)
    if test_response.startswith("⚠️"):
        return False, f"Model test failed: {test_response}"
    
    return True, "System ready for experimental generation"


if __name__ == "__main__":
    """Development testing and validation"""
    print("🧪 Testing LLM utilities...")
    
    # System readiness check
    ready, status = validate_system_ready()
    print(f"System status: {'✅' if ready else '❌'} {status}")
    
    # Generation test
    if ready:
        response = generate_ollama_response("Brief test response")
        success = not response.startswith("⚠️")
        print(f"Generation test: {'✅' if success else '❌'}")
        if success:
            print(f"Sample output: {response[:60]}...")