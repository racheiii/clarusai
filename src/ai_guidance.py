"""
src/ai_guidance.py – Local AI guidance generation via LLaMA3 (Ollama)
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit as st

from src.session_manager import safe_get_session_value, safe_set_session_value
from src.llm_utils import generate_ollama_response
from config import OLLAMA_CONFIG
from src import llm_feedback

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# AI GUIDANCE SYSTEM
# =============================================================================

class AIGuidance:   
    def __init__(self) -> None:
        self.fallback_prompts = {
            0: "Consider multiple perspectives and examine your initial assumptions. What evidence supports or contradicts your first impression?",
            1: "Think about cognitive factors that might influence decision-making. What mental shortcuts or patterns could affect analysis?",
            2: "Focus on practical strategies to improve decision quality. What systematic approaches could reduce errors?",
            3: "Consider how these principles apply to other domains. What patterns do you see across different contexts?"
        }


    def get_guidance(self, scenario: Dict[str, Any], current_stage: int,
                    previous_responses: str = "") -> str:
        try:
            text = (self._generate_ollama_guidance(scenario, current_stage, previous_responses) or "").strip()
            if not text:
                raise ValueError("empty_guidance")
            text = self._enforce_brief(text)
            logger.info(f"Generated AI guidance for stage {current_stage}")
            return text
        except Exception as e:
            logger.error(f"Ollama guidance generation error: {e}")
            return self._get_fallback_guidance(current_stage)

    def _get_fallback_guidance(self, current_stage: int) -> str:
        return self.fallback_prompts.get(
            current_stage,
            "Consider multiple perspectives in your analysis."
        )
    
    def render_guidance_interface(self, scenario: Dict[str, Any], current_stage: int) -> Optional[str]:
        # Only render if AI assistance is enabled
        if not safe_get_session_value('ai_assistance_enabled', False):
            return None
        
        # Check if guidance has already been requested for this stage
        guidance_requested_key = f'guidance_requested_stage_{current_stage}'
        guidance_text_key = f'guidance_text_stage_{current_stage}'
        guidance_already_requested = safe_get_session_value(guidance_requested_key, False)
        stored_guidance_text = safe_get_session_value(guidance_text_key, None)
        
        # Render interface
        st.markdown(f"""
        <div class="ai-guidance-panel fade-in">
            <h5 style="color: var(--primary-blue); margin-top: 0;">🤖 AI Guidance Available</h5>
            <p style="color: var(--text-dark); margin-bottom: 0.5rem;">
                AI assistance is enabled for this experimental condition.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show guidance content if already requested
        if guidance_already_requested and stored_guidance_text:
            st.markdown(f"""
            <div class="ai-guidance-content">
                <h6 style="color: var(--accent-orange); margin-top: 0;">🤖 AI Analysis & Guidance</h6>
                <p style="color: var(--text-dark); margin: 0; line-height: 1.5; font-style: italic;">
                    {stored_guidance_text}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show disabled button to indicate guidance was already used
            st.button("✓ AI Guidance Used", 
                     key=f"guidance_stage_{current_stage}_disabled", 
                     disabled=True,
                     help="AI guidance has already been requested for this stage")
            
            return stored_guidance_text
        
        # Guidance request button (only if not already requested)
        if st.button("💡 Get AI Guidance", key=f"guidance_stage_{current_stage}", type="secondary"):
            
            # Show loading indicator
            with st.spinner("🤖 Analyzing scenario and generating guidance..."):
                
                # Get previous responses for context
                previous_responses = self._get_previous_responses()
                
                # Request AI guidance
                guidance_text = self.get_guidance(scenario, current_stage, previous_responses)
            
            # Store guidance text and mark as requested
            safe_set_session_value(guidance_requested_key, True)
            safe_set_session_value(guidance_text_key, guidance_text)
            
            # Track guidance usage
            self._track_guidance_usage(current_stage, guidance_text, scenario)
            
            # Rerun to show the guidance content and disabled button
            st.rerun()
        
        return None
    
    def _enforce_brief(self, text: str) -> str:
        s = (text or "").strip().replace("\n", " ")
        if not s:
            return ""
        import re
        parts = re.split(r"(?<=[.!?])\s+", s)
        clipped = " ".join(parts[:2])
        words = clipped.split()
        if len(words) > 80:
            clipped = " ".join(words[:80]).rstrip() + "..."
        return clipped
    
    def _get_previous_responses(self) -> str:
        previous_responses = []
        
        # Try experimental session first
        experimental_session = safe_get_session_value('experimental_session')
        if experimental_session and experimental_session.stage_responses:
            previous_responses = [r.response_text for r in experimental_session.stage_responses]
        
        # Fallback to legacy session state
        if not previous_responses:
            previous_responses = safe_get_session_value('stage_responses', [])
        
        return " | ".join(previous_responses) if previous_responses else ""
    
    def _track_guidance_usage(self, current_stage: int, guidance_text: str, scenario: Dict[str, Any]) -> None:
        try:
            # Update experimental session if available
            experimental_session = safe_get_session_value('experimental_session')
            if experimental_session and len(experimental_session.stage_responses) > current_stage:
                experimental_session.stage_responses[current_stage].guidance_requested = True
                experimental_session.stage_responses[current_stage].guidance_text = guidance_text
            
            # Update legacy tracking
            guidance_usage = safe_get_session_value('guidance_usage', [])
            while len(guidance_usage) <= current_stage:
                guidance_usage.append(False)
            guidance_usage[current_stage] = True
            safe_set_session_value('guidance_usage', guidance_usage)
            
            # Log guidance interaction
            from src.session_manager import SessionManager
            session_manager = SessionManager()

            # Extract enum values
            bias_type_raw = scenario.get('bias_type', 'unknown')
            bias_type = bias_type_raw.value if hasattr(bias_type_raw, "value") else str(bias_type_raw)

            domain_raw = scenario.get('domain', 'unknown')
            domain = domain_raw.value if hasattr(domain_raw, "value") else str(domain_raw)

            # Log guidance interaction
            session_manager.auto_save_session_data('guidance_requested', {
                'stage': current_stage,
                'guidance_text': guidance_text,
                'scenario_id': scenario.get('scenario_id', 'unknown'),
                'bias_type': bias_type,
                'domain': domain,
                'timestamp': datetime.now().isoformat()
            })
 
        except Exception as e:
            logger.error(f"Failed to track guidance usage: {e}")

    def _generate_ollama_guidance(self, scenario: Dict[str, Any], current_stage: int, previous_responses: str) -> str:
        stage_contexts = {
            0: "initial analysis and decision-making under uncertainty",
            1: "cognitive factors and mental processes that influence professional judgment",
            2: "systematic strategies to improve decision quality and reduce errors",
            3: "cross-domain application of decision-making principles"
        }
        context = stage_contexts.get(current_stage, "decision-making")
        domain_raw = scenario.get("domain", "professional")
        domain = domain_raw.value.lower() if hasattr(domain_raw, "value") else str(domain_raw).lower()
        scenario_text = scenario.get("scenario_text", "A professional decision-making scenario")

        prompt = (
            f"You are providing educational guidance to a professional working through a {domain} scenario involving {context}.\n\n"
            f"### Scenario Context:\n{scenario_text}\n\n"
            f"### Current Stage:\nStage {current_stage + 1} of 4 in a progressive decision-making analysis.\n\n"
            f"### Previous User Responses:\n{previous_responses if previous_responses else 'This is their first response'}\n\n"
            "Write a 1–2 sentence response. Start with one positive insight, then one area for improvement.\n"
            "Avoid naming cognitive biases (e.g., confirmation bias, anchoring). Keep tone professional and bias‑blind.\n"
        )

        return generate_ollama_response(
            prompt,
            model=str(OLLAMA_CONFIG.get("model") or "llama3.2")
        ) or ""