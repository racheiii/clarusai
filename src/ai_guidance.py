"""
ClārusAI: AI Guidance and Assistance System
UCL Master's Dissertation: "Building AI Literacy Through Simulation"

src/ai_guidance.py – Local AI guidance generation via LLaMA3 (Ollama)

Purpose:
Provides locally generated, bias-blind AI guidance to support user decision-making
under uncertainty across staged reasoning tasks. Replaces prior Gemini-based system
with a fully reproducible, offline-compatible architecture using open-source models.
"""

import streamlit as st
import os
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from src.session_manager import safe_get_session_value, safe_set_session_value
import subprocess

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# AI GUIDANCE SYSTEM
# =============================================================================

class AIGuidance:
    """
    Comprehensive AI guidance system for experimental research.
    
    Academic Purpose: Provides reliable AI assistance while maintaining
    experimental integrity and tracking AI dependency patterns.
    """
    
    def __init__(self) -> None:
        """Initialize AI guidance system."""
        self.api_configured = False
        self.fallback_prompts = {
            0: "Consider multiple perspectives and examine your initial assumptions. What evidence supports or contradicts your first impression?",
            1: "Think about cognitive factors that might influence decision-making. What mental shortcuts or patterns could affect analysis?", 
            2: "Focus on practical strategies to improve decision quality. What systematic approaches could reduce errors?",
            3: "Consider how these principles apply to other domains. What patterns do you see across different contexts?"
        }
        
        # Track API status for compatibility (no external API needed)
        self.api_configured = True  # Always true for local Ollama setup
        safe_set_session_value('api_status', 'configured')  # Mark AI as ready

    
    def get_guidance(self, scenario: Dict[str, Any], current_stage: int, 
                    previous_responses: str = "") -> Tuple[str, bool]:
        """
        Generate AI guidance with comprehensive error handling.
        
        Academic Purpose: Provides bias-neutral guidance while maintaining
        experimental integrity and transparent status communication.
        
        Args:
            scenario: Current scenario metadata
            current_stage: Stage number (0-3)
            previous_responses: Context from earlier stages
        
        Returns:
            Tuple[str, bool]: (guidance_text, api_used)
        """
        
        # Check if API is available and configured
        if not self.api_configured:
            logger.warning(f"AI guidance requested but API unavailable - using fallback for stage {current_stage}")
            return self._get_fallback_guidance(current_stage), False
        
        try:
            # Generate guidance using local LLaMA3 model via Ollama
            guidance_text = self._generate_ollama_guidance(scenario, current_stage, previous_responses)
            
            # Update API status to active
            safe_set_session_value('api_status', 'active')
            logger.info(f"Successfully generated AI guidance for stage {current_stage}")
            
            return guidance_text, True
            
        except Exception as e:
            logger.error(f"Ollama guidance generation error: {e}")
            safe_set_session_value('api_status', 'error')
            
            # Log error for research analysis
            self._log_api_error(current_stage, str(e))
            
            # Return fallback guidance
            return self._get_fallback_guidance(current_stage), False
    
    
    
    def _get_fallback_guidance(self, current_stage: int) -> str:
        """Get fallback guidance when API unavailable."""
        return self.fallback_prompts.get(current_stage, "Consider multiple perspectives in your analysis.")
    
    def _log_api_error(self, stage: int, error_message: str) -> None:
        """Log API errors for research analysis."""
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': 'api_failure',
            'stage': stage,
            'error_message': error_message,
            'fallback_used': True
        }
        
        # Add to session errors
        session_errors = safe_get_session_value('session_errors', [])
        session_errors.append(error_entry)
        safe_set_session_value('session_errors', session_errors)
    
    def render_guidance_interface(self, scenario: Dict[str, Any], current_stage: int) -> Optional[str]:
        """
        Render AI guidance interface with status communication.
        
        Academic Purpose: Provides transparent AI assistance interface
        while maintaining experimental integrity.
        
        Returns:
            Optional[str]: Guidance text if requested, None otherwise
        """
        
        # Only render if AI assistance is enabled
        if not safe_get_session_value('ai_assistance_enabled', False):
            return None
        
        # Check scenario appropriateness for AI assistance
        if scenario.get('ai_appropriateness', '').lower() not in ['helpful', 'neutral']:
            return None
        
        # Check if guidance has already been requested for this stage
        guidance_requested_key = f'guidance_requested_stage_{current_stage}'
        guidance_text_key = f'guidance_text_stage_{current_stage}'
        guidance_already_requested = safe_get_session_value(guidance_requested_key, False)
        stored_guidance_text = safe_get_session_value(guidance_text_key, None)
        
        # Display AI status
        api_status = safe_get_session_value('api_status', 'unknown')
        status_display = self._get_status_display(api_status)
        
        # Render interface
        st.markdown(f"""
        <div class="ai-guidance-panel fade-in">
            <h5 style="color: var(--primary-blue); margin-top: 0;">🤖 AI Guidance Available</h5>
            <p style="color: var(--text-dark); margin-bottom: 0.5rem;">
                AI assistance is enabled for this experimental condition. {status_display}
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
                guidance_text, api_used = self.get_guidance(scenario, current_stage, previous_responses)
            
            # Store guidance text and mark as requested
            safe_set_session_value(guidance_requested_key, True)
            safe_set_session_value(guidance_text_key, guidance_text)
            
            # Track guidance usage
            self._track_guidance_usage(current_stage, guidance_text, api_used, scenario)
            
            # Rerun to show the guidance content and disabled button
            st.rerun()
        
        return None
    
    def _get_status_display(self, api_status: str) -> str:
        """Get user-friendly status display message."""
        status_messages = {
            'configured': '🟢 AI ready',
            'active': '🟢 AI active', 
            'error': '🟡 Limited functionality',
            'unavailable': '🔴 Offline mode',
            'unknown': '🟡 Checking status...'
        }
        return status_messages.get(api_status, '🟡 Status unknown')
    
    def _get_previous_responses(self) -> str:
        """Get previous responses for contextual guidance."""
        previous_responses = []
        
        # Try experimental session first
        experimental_session = safe_get_session_value('experimental_session')
        if experimental_session and experimental_session.stage_responses:
            previous_responses = [r.response_text for r in experimental_session.stage_responses]
        
        # Fallback to legacy session state
        if not previous_responses:
            previous_responses = safe_get_session_value('stage_responses', [])
        
        return " | ".join(previous_responses) if previous_responses else ""
    
    def _track_guidance_usage(self, current_stage: int, guidance_text: str, 
                             api_used: bool, scenario: Dict[str, Any]) -> None:
        """Track guidance usage for research analysis."""
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
                'api_used': api_used,
                'api_status': safe_get_session_value('api_status', 'unknown'),
                'scenario_id': scenario.get('scenario_id', 'unknown'),
                'bias_type': bias_type,
                'domain': domain,
                'timestamp': datetime.now().isoformat()
            })
 
        except Exception as e:
            logger.error(f"Failed to track guidance usage: {e}")
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get comprehensive API status for debugging."""
        return {
            'ollama_active': True,
            'api_key_configured': False,
            'api_configured': self.api_configured,
            'current_status': safe_get_session_value('api_status', 'unknown'),
            'fallback_prompts_available': len(self.fallback_prompts) == 4
        }
    def _generate_ollama_guidance(self, scenario: Dict[str, Any], current_stage: int, previous_responses: str) -> str:
        """Generate bias-blind guidance locally using LLaMA3 via Ollama."""
        stage_contexts = {
            0: "initial analysis and decision-making under uncertainty",
            1: "cognitive factors and mental processes that influence professional judgment",
            2: "systematic strategies to improve decision quality and reduce errors",
            3: "cross-domain application of decision-making principles"
        }
        context = stage_contexts.get(current_stage, "decision-making")
        domain_raw = scenario.get('domain', 'professional')
        domain = domain_raw.value.lower() if hasattr(domain_raw, "value") else str(domain_raw).lower()
        scenario_text = scenario.get('scenario_text', 'A professional decision-making scenario')

        prompt = (
            f"You are providing educational guidance to a professional working through a {domain} scenario involving {context}.\n\n"
            f"### Scenario Context:\n{scenario_text}\n\n"
            f"### Current Stage:\nStage {current_stage + 1} of 4 in a progressive decision-making analysis.\n\n"
            f"### Previous User Responses:\n{previous_responses if previous_responses else 'This is their first response'}\n\n"
            "Write a 1–2 sentence response. Start with one positive insight, then one area for improvement.\n"
            "Avoid naming cognitive biases (e.g., confirmation bias, anchoring). Keep tone professional and bias-blind.\n"
        )

        result = subprocess.run(["ollama", "run", "llama3"], input=prompt.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        guidance_text = result.stdout.decode().strip()

        return guidance_text if guidance_text else "⚠️ No guidance generated."