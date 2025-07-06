"""
ClārusAI: User Interface Components
UCL Master's Dissertation: "Building AI Literacy Through Simulation"

src/ui_components.py - Modular UI components for the experimental interface

Purpose:
Provides reusable UI components for the 4-stage experimental protocol

Author: Rachel Seah
Date: July 2025
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Optional

import config
from src.session_manager import safe_get_session_value, safe_set_session_value, SessionManager
from src.models import UserExpertise, UserResponse, StageType
from src.scoring_engine import calculate_comprehensive_scores

# Stage configuration
STAGE_NAMES = [
    "Primary Analysis",
    "Cognitive Factors", 
    "Mitigation Strategies",
    "Transfer Learning"
]

STAGE_PROMPTS = [
    "primary_prompt",
    "follow_up_1",
    "follow_up_2", 
    "follow_up_3"
]

STAGE_TYPES = [
    StageType.PRIMARY_ANALYSIS,
    StageType.COGNITIVE_FACTORS,
    StageType.MITIGATION_STRATEGIES,
    StageType.TRANSFER_LEARNING
]

class UIComponents:
    """
    Comprehensive UI components for experimental interface.
    
    Academic Purpose: Provides modular, reusable components while
    maintaining experimental integrity and data quality.
    """
    
    def __init__(self) -> None:
        """Initialize UI components."""
        pass
    
    def render_experimental_setup(self, scenarios_df, scenario_handler, 
                                 session_manager, data_collector) -> bool:
        """
        Render experimental setup phase.
        
        Returns:
            bool: True if setup completed successfully
        """
        
        st.markdown('<h2 class="section-header">🎯 Experimental Training Setup</h2>', unsafe_allow_html=True)
        
        # Research context
        st.markdown("""
        <div class="research-context fade-in">
            <h4 style="color: var(--text-dark); margin-top: 0;">🔬 Research Participation</h4>
            <p style="color: var(--text-light); margin-bottom: 0.5rem; line-height: 1.6;">
            This is a decision-making training system designed to enhance cognitive bias recognition in high-stakes environments. 
            You will work through a realistic scenario followed by progressive analysis questions. Your responses contribute to research 
            on AI-assisted learning effectiveness and will be analysed to understand how AI guidance affects professional decision-making patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Factor 1: Professional Expertise
        expertise_selected = self._render_expertise_selection()
        
        # Factor 2: AI Assistance
        assistance_selected = self._render_assistance_selection()
        
        # Proceed if both factors selected
        if expertise_selected and assistance_selected is not None:
            return self._render_setup_completion(
                scenarios_df, scenario_handler, session_manager, data_collector
            )
        
        return False
    
    def _render_expertise_selection(self) -> Optional[UserExpertise]:
        """Render expertise selection interface."""
        
        st.markdown("### 👤 Professional Experience Level")
        st.markdown("Please select the option that best describes your experience in high-stakes professional decision-making:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "🔰 Novice Professional\n(2-3 years experience)", 
                use_container_width=True,
                type="primary" if safe_get_session_value('user_expertise') == UserExpertise.NOVICE else "secondary",
                key="expertise_novice"
            ):
                session_manager = SessionManager()
                session_manager.set_user_expertise(UserExpertise.NOVICE)
                st.rerun()
        
        with col2:
            if st.button(
                "🎖️ Expert Professional\n(10+ years experience)", 
                use_container_width=True,
                type="primary" if safe_get_session_value('user_expertise') == UserExpertise.EXPERT else "secondary",
                key="expertise_expert"
            ):
                session_manager = SessionManager()
                session_manager.set_user_expertise(UserExpertise.EXPERT)
                st.rerun()
        
        # Display selection
        current_expertise = safe_get_session_value('user_expertise')
        if current_expertise:
            expertise_label = "🎖️ Expert Professional (10+ years)" if current_expertise == UserExpertise.EXPERT else "🔰 Novice Professional (2-3 years)"
            st.markdown(f"""
            <div class="experimental-condition-summary">
                <strong>Selected Experience Level: {expertise_label}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        return current_expertise
    
    def _render_assistance_selection(self) -> Optional[bool]:
        """Render AI assistance selection interface."""
        
        st.markdown("### 🤖 AI Assistance Configuration")
        st.markdown("Choose whether you would like AI guidance available during scenario analysis:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(
                "🚫 Unassisted Mode\n(Independent analysis)", 
                use_container_width=True,
                type="primary" if safe_get_session_value('ai_assistance_enabled') == False else "secondary",
                key="assistance_disabled"
            ):
                session_manager = SessionManager()
                session_manager.set_ai_assistance(False)
                st.rerun()
        
        with col2:
            if st.button(
                "🤖 AI-Assisted Mode\n(Guidance available)", 
                use_container_width=True,
                type="primary" if safe_get_session_value('ai_assistance_enabled') == True else "secondary",
                key="assistance_enabled"
            ):
                session_manager = SessionManager()
                session_manager.set_ai_assistance(True)
                st.rerun()
        
        # Display selection
        current_assistance = safe_get_session_value('ai_assistance_enabled')
        if current_assistance is not None:
            assistance_label = "🤖 AI-Assisted Mode (Guidance Available)" if current_assistance else "🚫 Unassisted Mode (Independent Analysis)"
            st.markdown(f"""
            <div class="experimental-condition-summary">
                <strong>Selected Mode: {assistance_label}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        return current_assistance
    
    def _render_setup_completion(self, scenarios_df, scenario_handler, 
                                session_manager, data_collector) -> bool:
        """Render setup completion and scenario assignment."""
        
        st.markdown("---")
        
        # Display experimental condition
        current_expertise = safe_get_session_value('user_expertise')
        current_assistance = safe_get_session_value('ai_assistance_enabled')
        
        st.markdown("### 🔬 Experimental Condition Summary")
        st.markdown(f"""
        <div class="experimental-condition-summary">
            <p style="color: var(--text-dark); margin: 0;">
            <strong>Your experimental condition:</strong> {current_expertise.value.title()} Professional with AI Assistance {'Enabled' if current_assistance else 'Disabled'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Begin Training Scenario", type="primary", use_container_width=True):
                
                # Select scenario
                selected_scenario = scenario_handler.select_balanced_scenario(
                    current_expertise, current_assistance
                )
                
                if selected_scenario is None:
                    st.error("❌ Failed to assign training scenario. Please try again.")
                    return False
                
                # Create experimental session
                experimental_session = scenario_handler.create_experimental_session(
                    session_id=SessionManager.generate_unique_session_id(),
                    user_expertise=current_expertise,
                    ai_assistance=current_assistance,
                    selected_scenario=selected_scenario
                )
                
                if experimental_session:
                    session_manager.set_experimental_session(experimental_session)
                    st.success("✅ Training scenario assigned successfully!")
                    time.sleep(1)
                    return True
                else:
                    st.error("❌ Failed to create training session.")
                    return False
        
        return False
    
    def render_scenario_stage(self, scenario: Dict[str, Any], current_stage: int,
                             ai_guidance, session_manager, data_collector) -> str:
        """
        Render individual scenario stage.
        
        Returns:
            str: 'stage_completed', 'reset_requested', or 'continue'
        """
        
        try:
            # Render stage header
            self._render_stage_header(scenario, current_stage)
            
            # Render stage prompt
            self._render_stage_prompt(scenario, current_stage)
            
            # Render AI guidance if enabled
            if safe_get_session_value('ai_assistance_enabled'):
                ai_guidance.render_guidance_interface(scenario, current_stage)
            
            # Render response interface
            return self._render_response_interface(
                scenario, current_stage, session_manager, data_collector
            )
            
        except Exception as e:
            st.error("❌ An error occurred while loading this stage. Please refresh the page.")
            session_manager.log_error('stage_rendering_error', str(e))
            return 'continue'
    
    def _render_stage_header(self, scenario: Dict[str, Any], current_stage: int) -> None:
        """Render stage header with progress indicator."""
        
        # Progress indicator
        progress = (current_stage + 1) / 4
        st.progress(progress)
        
        stage_info = f"Stage {current_stage + 1} of 4: {STAGE_NAMES[current_stage]}"
        st.markdown(f"**{stage_info}**")
        
        if current_stage == 0:
            # Show full scenario on first stage
            st.markdown(f'<h2 class="section-header">📋 {scenario.get("title", "Training Scenario")}</h2>', unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stage-prompt-container">
                <h4 style="color: var(--accent-orange); margin-top: 0;">📖 Professional Scenario</h4>
                <div style="background: var(--background-white); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--accent-orange);">
                    <p style="line-height: 1.6; color: var(--text-dark); margin: 0; font-size: 1.1rem;">
                        {scenario.get("scenario_text", "Scenario text not available")}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f'<h2 class="section-header">📋 {scenario.get("title", "Training Scenario")} - {STAGE_NAMES[current_stage]}</h2>', unsafe_allow_html=True)
    
    def _render_stage_prompt(self, scenario: Dict[str, Any], current_stage: int) -> None:
        """Render current stage prompt."""
        
        prompt_field = STAGE_PROMPTS[current_stage]
        current_prompt = scenario.get(prompt_field, f"Stage {current_stage + 1} prompt not available")
        
        st.markdown(f"""
        <div class="stage-prompt-container">
            <h4 style="color: var(--text-dark); margin-top: 0;">💭 {STAGE_NAMES[current_stage]} Task</h4>
            <p style="color: var(--text-dark); font-size: 1.1rem; line-height: 1.6; margin: 0; font-weight: 500;">
                {current_prompt}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_response_interface(self, scenario: Dict[str, Any], current_stage: int,
                                  session_manager, data_collector) -> str:
        """Render response collection interface."""
        
        st.markdown("---")
        st.markdown(f'<h4 style="color: var(--text-dark);">✍️ Your {STAGE_NAMES[current_stage]} Response</h4>', unsafe_allow_html=True)
        
        # Text area for response
        response_key = f"stage_{current_stage}_response"
        current_response = st.text_area(
            f"Provide your analysis for {STAGE_NAMES[current_stage]}:",
            value="",
            height=200,
            placeholder=f"Please provide your detailed {STAGE_NAMES[current_stage].lower()} response. Take time to think through your reasoning...",
            key=response_key,
            help=f"Stage {current_stage + 1} of 4: {STAGE_NAMES[current_stage]} - No minimum word requirement, respond naturally"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if current_stage > 0:
                st.button("← Previous Stage", use_container_width=True, disabled=True, 
                         help="Previous stages are locked to maintain experimental integrity")
            else:
                if st.button("← Back to Setup", use_container_width=True):
                    return 'reset_requested'
        
        with col3:
            button_text = "Complete Training →" if current_stage == 3 else "Next Stage →"
            response_valid = self._validate_response(current_response, current_stage)
            
            if st.button(button_text, use_container_width=True, 
                        disabled=not response_valid, type="primary"):
                
                # Save response
                if data_collector.save_stage_response(scenario, current_stage, current_response):
                    
                    if current_stage < 3:
                        # Advance stage
                        session_manager.advance_stage(current_stage)
                        st.success("✅ Response saved successfully!")
                        time.sleep(0.5)
                        return 'stage_completed'
                    else:
                        # Complete training
                        session_manager.advance_stage(current_stage)
                        st.success("🎉 Training completed successfully!")
                        time.sleep(1)
                        return 'stage_completed'
                else:
                    st.error("❌ Failed to save response. Please try again.")
        
        # Response feedback
        self._render_response_feedback(current_response, current_stage)
        
        return 'continue'
    
    def _validate_response(self, response: str, stage: int) -> bool:
        """Validate response quality."""
        if not response or not response.strip():
            return False
        
        word_count = len(response.split())
        char_count = len(response.strip())
        
        # Progressive standards by stage
        min_requirements = {
            0: {'words': 5, 'chars': 30},
            1: {'words': 5, 'chars': 30},
            2: {'words': 5, 'chars': 30},
            3: {'words': 5, 'chars': 30}
        }

        
        requirements = min_requirements.get(stage, {'words': 5, 'chars': 20})
        return word_count >= requirements['words'] and char_count >= requirements['chars']
    
    def _render_response_feedback(self, response: str, stage: int) -> None:
        """Render response validation feedback."""
        
        word_count = len(response.split()) if response else 0
        char_count = len(response.strip()) if response else 0
        is_valid = self._validate_response(response, stage)
        
        if not is_valid:
            if char_count == 0:
                message = "📝 Please provide a response to continue"
            else:
                message = f"📝 Please expand your response (currently {word_count} words, {char_count} characters)"
            
            st.markdown(f"""
            <div class="response-validation-feedback status-warning">
                {message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="response-validation-feedback status-success">
                ✅ {word_count} words, {char_count} characters - Ready to proceed
            </div>
            """, unsafe_allow_html=True)
    
    def render_completion_interface(self, scenario: Dict[str, Any], 
                                   session_manager, data_collector) -> str:
        """
        Render completion interface.
        
        Returns:
            str: Navigation choice ('results', 'new_scenario', 'home')
        """
        
        st.markdown('<h2 class="section-header">✅ Training Protocol Completed</h2>', unsafe_allow_html=True)
        
        try:
            # Save complete session
            session_file = data_collector.save_complete_session(scenario)
            
            # Completion celebration
            st.markdown("""
            <div class="session-completion-celebration">
                <h3 style="color: var(--accent-orange); margin-top: 0;">🎉 Excellent Work!</h3>
                <p style="text-align: center; font-size: 1.1rem; margin-bottom: 1.5rem;">
                You have successfully completed all four stages of the cognitive bias recognition training protocol.
                Your responses provide valuable data for understanding AI-assisted learning patterns in professional decision-making.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Session analytics
            self._render_completion_analytics()
            
            # Bias revelation
            self._render_bias_revelation(scenario)
            
            # Navigation options
            return self._render_completion_navigation()
            
        except Exception as e:
            session_manager.log_error('completion_interface_error', str(e))
            st.error("❌ An error occurred while preparing your completion summary.")
            return self._render_minimal_completion()
    
    def _render_completion_analytics(self) -> None:
        """Render session analytics."""
        
        try:
            # Get analytics from experimental session
            experimental_session = safe_get_session_value('experimental_session')
            
            if experimental_session:
                analytics = experimental_session.calculate_analytics()
                total_words = analytics.total_word_count
                total_time = analytics.total_session_time_minutes
                guidance_used = analytics.total_guidance_requests
            else:
                # Fallback calculations
                total_words = sum(len(r.split()) for r in safe_get_session_value('stage_responses', []))
                session_start = safe_get_session_value('session_start_time', datetime.now())
                total_time = (datetime.now() - session_start).total_seconds() / 60
                guidance_used = sum(safe_get_session_value('guidance_usage', []))
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Stages Completed", "4/4")
            with col2:
                st.metric("Total Words", f"{total_words:,}")
            with col3:
                st.metric("Session Duration", f"{total_time:.1f} min")
            with col4:
                st.metric("AI Guidance Used", f"{guidance_used}/4")
            
            # Response progression
            st.markdown("### 📊 Your Response Progression")
            
            response_lengths = []
            if experimental_session and experimental_session.stage_responses:
                response_lengths = [len(r.response_text.split()) for r in experimental_session.stage_responses]
            else:
                response_lengths = [len(r.split()) for r in safe_get_session_value('stage_responses', [])]
            
            for i, (stage_name, word_count) in enumerate(zip(STAGE_NAMES, response_lengths)):
                max_words = max(response_lengths) if response_lengths else 100
                progress = (word_count / max_words) * 100 if max_words > 0 else 0
                
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                    <div class="progress-header">
                        <span>{stage_name}</span>
                        <span>{word_count} words</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown("### 📊 Session Complete")
            st.success("Your training session has been completed successfully.")
    
    def _render_bias_revelation(self, scenario: Dict[str, Any]) -> None:
        """Render bias revelation with educational content."""
        
        st.markdown("---")
        st.markdown("### 🧠 Educational Debrief: Cognitive Bias Revelation")
        
        bias_type = scenario.get('bias_type', 'unknown')
        learning_objective = scenario.get('bias_learning_objective', 'Recognize and mitigate cognitive biases in professional decision-making contexts.')
        
        st.markdown(f"""
        <div class="bias-revelation-panel">
            <h4 style="color: var(--text-dark); margin-top: 0;">🎯 Cognitive Bias Focus</h4>
            <p style="color: var(--text-dark); font-size: 1.2rem; margin-bottom: 1rem;">
                <strong>This scenario was designed to test: {bias_type.replace('_', ' ').title()}</strong>
            </p>
            <p style="color: var(--text-light); line-height: 1.6; margin: 0;">
                <strong>Learning Objective:</strong> {learning_objective}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Expert feedback if available
        llm_feedback = scenario.get('llm_feedback')
        if llm_feedback:
            st.markdown(f"""
            <div class="ai-guidance-content">
                <h5 style="color: var(--accent-orange); margin-top: 0;">💡 Expert Learning Insights</h5>
                <p style="color: var(--text-dark); line-height: 1.6; margin: 0;">
                    {llm_feedback}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_completion_navigation(self) -> str:
        """Render completion navigation options."""
        
        st.markdown("### 🚀 Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 View Results", use_container_width=True, type="primary"):
                return 'results'
        
        with col2:
            if st.button("🎯 Try Another Scenario", use_container_width=True):
                return 'new_scenario'
        
        with col3:
            if st.button("🏠 Return Home", use_container_width=True):
                return 'home'
        
        # Research acknowledgment
        st.markdown("""
        <div class="research-context">
            <h5 style="color: var(--text-dark); margin-top: 0;">🙏 Thank You for Contributing to Research</h5>
            <p style="color: var(--text-light); margin: 0; font-size: 0.9rem; line-height: 1.4;">
                Your responses contribute valuable data to understanding AI-assisted cognitive bias training effectiveness.
                All data is anonymized and used solely for academic research purposes in accordance with UCL research ethics guidelines.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        return 'continue'
    
    def _render_minimal_completion(self) -> str:
        """Render minimal completion interface as fallback."""
        
        st.success("✅ Training completed successfully!")
        
        if st.button("🏠 Return Home", use_container_width=True):
            return 'home'
        
        return 'continue'