"""
ClārusAI: User Interface Components

src/ui_components.py — UI components for a 4‑stage experimental interface.

Purpose:
Provides well‑documented, submission‑ready UI components for the experimental
protocol, including (optional) LLM tutor feedback, performance analysis, and
participant debrief. The code is written for clarity and reproducibility.

Key Features:
- Optional LLM tutor guidance with graceful fallback
- 6‑dimension score visualization using stored results (no duplicate scoring)
- Always‑visible feedback containers for consistent UX
- Bias debrief with concise educational framing
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

import config
from src.session_manager import safe_get_session_value, safe_set_session_value, SessionManager
from src.models import UserExpertise, UserResponse, StageType
from src.llm_feedback import generate_stage_feedback
from src.scoring_engine import map_bias_count_to_level

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
    UI component renderer for stages and results

    This class renders all interactive and analytical elements of the
    cognitive bias training experiment. It ensures that each 4-stage decision-making
    flow is displayed consistently, captures user input for later scoring, and
    displays reflective tutor feedback for research analysis.
    """
    
    def __init__(self) -> None:
        """Initialise the UIComponents instance with configuration and session context"""
        pass
    
    @staticmethod
    def _quartile_label(x: float) -> str:
        """
        Map a 1–4 aggregate score to a plain‑English band used across the UI.
        Keep wording consistent everywhere.
        """
        if x >= 3.5:
            return "Advanced"
        if x >= 2.5:
            return "Proficient"
        if x >= 1.5:
            return "Competent"
        return "Foundational"

    @staticmethod
    def _bands_legend() -> str:
        """Single‑line legend to keep the results page self‑explaining."""
        return "Bands: Advanced (≥3.5) • Proficient (≥2.5) • Competent (≥1.5) • Foundational (<1.5)"

    def show_progress_toast(self):
        """Show progress toast notification after scenario is loaded"""
        if (safe_get_session_value('interaction_flow') == 'scenario' and 
            not safe_get_session_value('progress_toast_shown', False)):
            
            st.toast("Do not refresh this page — your progress will be lost.", icon="⚠️")
            time.sleep(0.5)
            st.toast("Scenario loaded successfully!", icon="🔄")
            safe_set_session_value('progress_toast_shown', True)
    
    def render_experimental_setup(self, scenarios_df, scenario_handler, 
                                 session_manager, data_collector) -> bool:
        """Render the participant setup interface for expertise level and AI assistance selection"""
        
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
        
        # Factor selection
        expertise_selected = self._render_expertise_selection()

        # NEW: Domain selector (optional; bias remains random)
        _ = self._render_domain_selection(scenarios_df)

        assistance_selected = self._render_assistance_selection()
        
        if expertise_selected and assistance_selected is not None:
            return self._render_setup_completion(
                scenarios_df, scenario_handler, session_manager, data_collector
            )
        
        return False
    
    def _render_expertise_selection(self) -> Optional[UserExpertise]:
        """Render the interface for selecting participant expertise level"""
        
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
        
        current_expertise = safe_get_session_value('user_expertise')
        if current_expertise:
            expertise_label = "🎖️ Expert Professional (10+ years)" if current_expertise == UserExpertise.EXPERT else "🔰 Novice Professional (2-3 years)"
            st.markdown(f"""
            <div class="experimental-condition-summary">
                <strong>Selected Experience Level: {expertise_label}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        return current_expertise
    
    def _render_domain_selection(self, scenarios_df) -> Optional[str]:
        """Render the interface for selecting a domain (optional; bias remains random)."""
        st.markdown("### 🏥 Domain Preference")
        st.markdown("Choose a domain for your scenario. The cognitive bias will still be random within the selected domain.")

        # Build options
        domains = sorted({str(d) for d in scenarios_df['domain'].dropna().unique()})
        options = ["All domains"] + domains

        # Respect any previously selected domain in session
        previous = safe_get_session_value('selected_domain', None)
        try:
            default_index = options.index(previous) if previous in options else 0
        except Exception:
            default_index = 0

        choice = st.selectbox(
            "Select a domain:", 
            options=options, 
            index=default_index, 
            help="Lock the domain only; bias remains random within the domain."
        )

        sm = SessionManager()
        if choice == "All domains":
            sm.set_selected_domain(None) 
            return None
        else:
            from src.models import Domain
            try:
                domain_enum = Domain(choice)
            except ValueError:
                domain_enum = None
            sm.set_selected_domain(str(domain_enum) if domain_enum else None)
            return str(domain_enum) if domain_enum else None

    def _render_assistance_selection(self) -> Optional[bool]:
        """Render the interface for selecting whether AI assistance is enabled"""
        
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
        """Display confirmation of experimental condition and assign a scenario"""
        
        st.markdown("---")
        
        current_expertise = safe_get_session_value('user_expertise')
        current_assistance = safe_get_session_value('ai_assistance_enabled')

        st.markdown("### 🔬 Experimental Condition Summary")

        selected_domain = safe_get_session_value('selected_domain', None)
        domain_label = getattr(selected_domain, 'value', selected_domain) if selected_domain else "All domains"

        # Enum-safe → string, then Title Case
        exp_label = getattr(current_expertise, 'value', current_expertise)
        exp_label = (str(exp_label).title() if exp_label else "Unknown")

        assist_label = "Enabled" if current_assistance else "Disabled"

        st.markdown(f"""
        <div class="experimental-condition-summary">
            <p style="color: var(--text-dark); margin: 0;">
            <strong>Your experimental condition:</strong> {exp_label} Professional with AI Assistance {assist_label}
            </p>
            <p style="color: var(--text-dark); margin: 0;">
            <strong>Domain:</strong> {domain_label}
            </p>
        </div>
        """, unsafe_allow_html=True)
  
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Begin Training Scenario", type="primary", use_container_width=True):
                
                selected_scenario = scenario_handler.select_balanced_scenario(
                    current_expertise,
                    current_assistance,
                    domain=(getattr(selected_domain, 'value', selected_domain) if selected_domain else None)
                )
                            
                if selected_scenario is None:
                    st.error("❌ Failed to assign training scenario. Please try again.")
                    return False
                
                experimental_session = scenario_handler.create_experimental_session(
                    session_id=SessionManager.generate_unique_session_id(),
                    user_expertise=current_expertise,
                    ai_assistance=current_assistance,
                    selected_scenario=selected_scenario
                )
                
                if experimental_session:
                    session_manager.set_experimental_session(experimental_session)
                    safe_set_session_value('stage_feedback', [None] * 4)
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
        Render an individual scenario stage, presenting context, prompt, and
        optional AI guidance, and capture the participant’s written response
        for subsequent scoring and analysis.
        """
        try:
            self._render_stage_header(scenario, current_stage)
            self._render_stage_prompt(scenario, current_stage)
            
            if safe_get_session_value('ai_assistance_enabled'):
                ai_guidance.render_guidance_interface(scenario, current_stage)
            
            result = self._render_response_interface(
                scenario, current_stage, session_manager, data_collector
            )
            
            if result == 'continue':
                self._render_unified_feedback(scenario, current_stage)
            
            return result
            
        except Exception as e:
            st.error("❌ An error occurred while loading this stage. Please refresh the page.")
            session_manager.log_error('stage_rendering_error', str(e))
            return 'continue'
    
    def _render_stage_header(self, scenario: Dict[str, Any], current_stage: int) -> None:
        """Render the stage title and progress indicator"""
        
        progress = (current_stage + 1) / 4
        st.progress(progress)
        
        stage_info = f"Stage {current_stage + 1} of 4: {STAGE_NAMES[current_stage]}"
        st.markdown(f"**{stage_info}**")
        
        st.markdown(f'<h2 class="section-header">📋 {scenario.get("title", "Training Scenario")}{" - " + STAGE_NAMES[current_stage] if current_stage > 0 else ""}</h2>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stage-prompt-container">
            <h4 style="color: var(--accent-orange); margin-top: 0;">📖 Professional Scenario Context</h4>
            <div style="background: var(--background-white); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--accent-orange);">
                <p style="line-height: 1.6; color: var(--text-dark); margin: 0; font-size: 1.1rem;">
                    {scenario.get("scenario_text", "Scenario text not available")}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_stage_prompt(self, scenario: Dict[str, Any], current_stage: int) -> None:
        """Render the task prompt for the current stage"""
        
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
        """Render response collection interface"""
        
        st.markdown("---")
        st.markdown(f'<h4 style="color: var(--text-dark);">✍️ Your {STAGE_NAMES[current_stage]} Response</h4>', unsafe_allow_html=True)
        
        response_key = f"stage_{current_stage}_response"
        current_response = st.text_area(
            f"Provide your analysis for {STAGE_NAMES[current_stage]}:",
            value="",
            height=200,
            placeholder=f"Please provide your detailed {STAGE_NAMES[current_stage].lower()} response. Take time to think through your reasoning...",
            key=response_key,
            help=f"Stage {current_stage + 1} of 4: {STAGE_NAMES[current_stage]} - No minimum word requirement, respond naturally"
        )
        
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
                
                # Save response (data_collector handles scoring calculation)
                if data_collector.save_stage_response(scenario, current_stage, current_response):
                    
                    # Generate unified feedback using scores from data_collector
                    self._generate_and_save_unified_feedback(scenario, current_stage, current_response)
                    
                    if current_stage < 3:
                        session_manager.advance_stage(current_stage)
                        st.success("✅ Response saved successfully!")
                        time.sleep(0.5)
                        return 'stage_completed'
                    else:
                        session_manager.advance_stage(current_stage)
                        st.success("🎉 Training completed successfully!")
                        time.sleep(1)
                        return 'stage_completed'
                else:
                    st.error("❌ Failed to save response. Please try again.")
        
        self._render_response_feedback(current_response, current_stage)
        return 'continue'
    
    def _generate_and_save_unified_feedback(self, scenario: Dict[str, Any], current_stage: int, response: str) -> None:
        """
        Generate and store a unified feedback object that merges AI-generated 
        tutor guidance (if enabled) with quantitative performance analysis 
        based on the scoring framework.
        """
        try:
            # Get experimental session scores from prior stage
            experimental_session = safe_get_session_value('experimental_session')
            scores = None

            if experimental_session and len(experimental_session.stage_responses) > current_stage:
                stage_response = experimental_session.stage_responses[current_stage]
                if stage_response.scores:
                    scores = stage_response.scores.__dict__

            # Generate LLM feedback if enabled
            llm_feedback = None
            if getattr(config, "LLM_ENABLED", True):
                try:
                    llm_feedback = generate_stage_feedback(scenario, current_stage, response)
                except Exception:
                    llm_feedback = None  # Don't fallback with generic string

            # Only show fallback message if no LLM and no scores
            if llm_feedback is None and not scores:
                llm_feedback = None  # This will trigger "No feedback generated" in UI

            # Generate performance analysis
            if scores:
                performance_analysis = self._create_performance_analysis_from_scores(scores, current_stage)
            else:
                performance_analysis = self._create_fallback_analysis(current_stage)

            # Store feedback
            unified_feedback = {
                'llm_feedback': llm_feedback,
                'performance_analysis': performance_analysis,
                'has_scores': scores is not None,
                'timestamp': datetime.now().isoformat()
            }

            stage_feedback = safe_get_session_value('stage_feedback', [None] * 4)
            stage_feedback[current_stage] = unified_feedback
            safe_set_session_value('stage_feedback', stage_feedback)

        except Exception:
            fallback_feedback = {
                'llm_feedback': None,
                'performance_analysis': self._create_fallback_analysis(current_stage),
                'has_scores': False,
                'timestamp': datetime.now().isoformat()
            }

            stage_feedback = safe_get_session_value('stage_feedback', [None] * 4)
            stage_feedback[current_stage] = fallback_feedback
            safe_set_session_value('stage_feedback', stage_feedback)

    def _create_performance_analysis_from_scores(self, scores: Dict[str, Any], current_stage: int) -> Dict[str, Any]:
        """Generate performance analysis metrics from existing stage scores"""
        
        # Extract scores safely from scoring engine results
        semantic_score = scores.get('semantic_similarity', 0.5)
        bias_count = scores.get('bias_recognition_count', 0)
        originality_score = scores.get('originality_score', 0.5)
        strategy_count = scores.get('strategy_count', 0)
        metacog_count = scores.get('metacognition_count', 0)
        
        # Calculate scores and performance (unified 1–4 banding)
        avg_score = (semantic_score + originality_score + min(1.0, bias_count / 3.0)) / 3
        composite_1_to_4 = 1.0 + 3.0 * float(max(0.0, min(1.0, avg_score)))
        performance_level = self._quartile_label(composite_1_to_4).lower() 
        
        # Generate strengths based on scores
        strengths = []
        if originality_score > 0.7:
            strengths.append("Original thinking and unique insights")
        if metacog_count > 1:
            strengths.append("Strong self-reflection and reasoning transparency")
        if bias_count > 1:
            strengths.append("Good awareness of cognitive factors")
        if strategy_count > 0 and current_stage >= 2:
            strengths.append("Practical mitigation strategies")
        if not strengths:
            strengths.append("Thoughtful engagement with the scenario")
        
        # Generate improvements based on scores
        improvements = []
        if bias_count < 1:
            improvements.append("Consider cognitive factors that might influence decision-making")
        if metacog_count < 1:
            improvements.append("Provide more explanation of your reasoning process")
        if strategy_count < 1 and current_stage >= 2:
            improvements.append("Develop more specific strategies to improve decision quality")
        if originality_score < 0.3:
            improvements.append("Consider alternative perspectives and approaches")
        if not improvements:
            improvements.append("Continue developing systematic analytical approaches")
        
        # Assess bias recognition level
        bias_recognition_level = map_bias_count_to_level(int(bias_count))
        
        return {
            'performance_summary': f"Your {STAGE_NAMES[current_stage].lower()} shows {performance_level} engagement with the scenario.",
            'strengths': strengths[:3],  # Limit to top 3
            'areas_for_improvement': improvements[:2],  # Limit to top 2
            'bias_recognition_score': bias_recognition_level,
            'detailed_metrics': {
                'originality': originality_score,
                'bias_awareness': bias_count,
                'self_reflection': metacog_count,
                'strategic_thinking': strategy_count
            }
        }
    
    def _create_fallback_analysis(self, current_stage: int) -> Dict[str, Any]:
        """Generate a minimal performance analysis when score data is not available"""
        return {
            'performance_summary': f"Your {STAGE_NAMES[current_stage].lower()} demonstrates thoughtful engagement with the scenario.",
            'strengths': ["Clear reasoning", "Relevant analysis", "Professional approach"],
            'areas_for_improvement': ["Continue developing systematic approaches", "Consider multiple perspectives"],
            'bias_recognition_score': "Moderate",
            'detailed_metrics': {
                'originality': 0.6,
                'bias_awareness': 1,
                'self_reflection': 1,
                'strategic_thinking': 1
            }
        }
    
    def _render_unified_feedback(self, scenario: Dict[str, Any], current_stage: int) -> None:
        """
        Render a single, consolidated feedback section for each stage

        Content is shown only when available:
        - Expert Guidance (AI-generated feedback) is displayed if present
        - Performance Analysis is displayed if corresponding score data is available
        """
        stage_feedback = safe_get_session_value('stage_feedback', [None] * 4)
        stage_responses = safe_get_session_value('stage_responses', [])

        # Show feedback only for submitted stages
        if not (len(stage_responses) > current_stage and stage_responses[current_stage]):
            return

        feedback = stage_feedback[current_stage]
        if not feedback:
            return

        llm_text = feedback.get('llm_feedback')
        has_scores = bool(feedback.get('has_scores'))
        analysis = feedback.get('performance_analysis')

        # If there is nothing useful to show, skip rendering quietly
        if not llm_text and not has_scores:
            return

        with st.expander(f"🎓📊 Feedback for {STAGE_NAMES[current_stage]}", expanded=False):
            # Expert Tutor Feedback (only if provided)
            if llm_text:
                st.markdown("### 🎓 Expert Guidance")
                st.markdown(
                    f"""
                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 1rem;">
                        <p style="color: #2c3e50; margin: 0; line-height: 1.6; font-style: italic;">
                            {llm_text}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Performance Analysis (only if scores exist)
            if has_scores and analysis:
                st.markdown("### 📊 Performance Analysis")
                st.markdown(
                    f"""
                    <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196f3; margin-bottom: 1rem;">
                        <h6 style="color: #1976d2; margin: 0 0 0.5rem 0;">📈 Summary</h6>
                        <p style="color: #424242; margin: 0; line-height: 1.5;">
                            {analysis.get('performance_summary', '')}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Key Strengths:**")
                    for s in analysis.get('strengths', [])[:3]:
                        st.markdown(f"• {s}")
                with col2:
                    st.markdown("**💡 Development Areas:**")
                    for a in analysis.get('areas_for_improvement', [])[:2]:
                        st.markdown(f"• {a}")

                # Cognitive awareness indicator
                st.markdown(
                    f"""
                    <div style="background: #fff3e0; padding: 0.75rem; border-radius: 6px; border-left: 3px solid #ff9800; margin-top: 1rem;">
                        <strong>🧠 Cognitive Awareness: {analysis.get('bias_recognition_score', '—')}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    def _validate_response(self, response: str, stage: int) -> bool:
        """Check whether the participant's response meets word and character count thresholds"""
        if not response or not response.strip():
            return False
        
        word_count = len(response.split())
        char_count = len(response.strip())
        
        min_requirements = {
            0: {'words': 5, 'chars': 30},
            1: {'words': 5, 'chars': 30},
            2: {'words': 5, 'chars': 30},
            3: {'words': 5, 'chars': 30}
        }
        
        requirements = min_requirements.get(stage, {'words': 5, 'chars': 20})
        return word_count >= requirements['words'] and char_count >= requirements['chars']
    
    def _render_response_feedback(self, response: str, stage: int) -> None:
        """Render response validation feedback"""
        
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
        """Render completion interface with summary feedback"""
        
        st.markdown('<h2 class="section-header">✅ Training Protocol Completed</h2>', unsafe_allow_html=True)
        
        try:
            session_file = data_collector.save_complete_session(scenario)
            
            # Session analytics
            self._render_session_analytics()
            
            # Summary feedback using integrated AI guidance and performance analysis
            self._render_optimized_summary_feedback(scenario)
            
            # Bias revelation
            self._render_bias_revelation(scenario)
            
            return self._render_completion_navigation()
            
        except Exception as e:
            session_manager.log_error('completion_interface_error', str(e))
            st.error("❌ An error occurred while preparing your completion summary.")
            return self._render_minimal_completion()
    
    def _render_session_analytics(self) -> None:
        """Render session analytics"""
        
        try:
            experimental_session = safe_get_session_value('experimental_session')
            
            if experimental_session:
                analytics = experimental_session.calculate_analytics()
                total_time = analytics.total_session_time_minutes
                guidance_used = analytics.total_guidance_requests
            else:
                session_start = safe_get_session_value('session_start_time', datetime.now())
                total_time = (datetime.now() - session_start).total_seconds() / 60
                guidance_used = sum(safe_get_session_value('guidance_usage', []))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Session Duration", f"{total_time:.1f} min")
            with col2:
                st.metric("AI Guidance Used", f"{guidance_used}/4")
            
        except Exception:
            st.markdown("### 📊 Session Complete")
            st.success("Your training session has been completed successfully.")
    
    def _render_optimized_summary_feedback(self, scenario: Dict[str, Any]) -> None:
        """Display a detailed summary using all available feedback and performance data"""
        
        st.markdown("---")
        st.markdown("### 📊 Comprehensive Performance Summary")
        
        stage_feedback = safe_get_session_value('stage_feedback', [None] * 4)
        valid_feedback = [f for f in stage_feedback if f is not None and f.get('performance_analysis')]
        
        if not valid_feedback:
            st.info("📈 Performance analysis will be available after completing all stages with the scoring system active.")
            return
        
        # Calculate overall metrics from unified feedback
        performance_levels = []
        bias_scores = []
        
        for feedback in valid_feedback:
            analysis = feedback['performance_analysis']
            
            # Extract performance level from summary
            summary = str(analysis.get('performance_summary', '')).lower()
            if 'advanced' in summary:
                performance_levels.append(4)
            elif 'proficient' in summary:
                performance_levels.append(3)
            elif 'competent' in summary:
                performance_levels.append(2)
            else:
                performance_levels.append(1)
            
            # Extract bias recognition level
            bias_level = str(analysis.get('bias_recognition_score', ''))
            bias_mapping = {'Advanced': 4, 'Proficient': 3, 'Competent': 2, 'Foundational': 1}
            bias_scores.append(bias_mapping.get(bias_level, 2))
        
        # Calculate averages
        avg_performance = sum(performance_levels) / len(performance_levels)
        avg_bias = sum(bias_scores) / len(bias_scores)
        
        # Display overall metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            overall_label = self._quartile_label(avg_performance)
            st.metric("Overall Performance", overall_label)

        with col2:
            bias_label = self._quartile_label(avg_bias)
            st.metric("Cognitive Awareness", bias_label)
        
        with col3:
            st.metric("Analysis Depth", f"{avg_performance * 2.5:.1f}/10")
        
        # Stage-by-stage breakdown
        st.markdown("#### 📈 Stage Performance Overview")

        for i, feedback in enumerate(stage_feedback):
            title = f"Stage {i + 1}: {STAGE_NAMES[i]}"
            with st.expander(title, expanded=False):
                if not feedback:
                    st.info("No feedback available for this stage.")
                    continue

                llm = feedback.get("llm_feedback")
                has_scores = bool(feedback.get("has_scores"))
                analysis = feedback.get("performance_analysis")

                # Show concise, professional copy only when content exists
                if llm:
                    st.markdown("**Expert Guidance**")
                    st.markdown(f"> {llm}")

                if has_scores and analysis:
                    st.markdown("**Performance Summary**")
                    st.markdown(f"- {analysis.get('performance_summary', '')}")

                    details = analysis.get('detailed_metrics', {})
                    if details:
                        st.markdown(
                            "- **Originality**: {0:.2f} | **Bias Awareness**: {1} | **Self‑reflection**: {2} | **Strategies**: {3}".format(
                                float(details.get('originality', 0)),
                                int(details.get('bias_awareness', 0)),
                                int(details.get('self_reflection', 0)),
                                int(details.get('strategic_thinking', 0)),
                            )
                        )

                if not llm and not (has_scores and analysis):
                    st.info("This stage has recorded a response but no additional feedback was generated.")
    
    def _render_bias_revelation(self, scenario: Dict[str, Any]) -> None:
        try:
            st.markdown("---")
            st.markdown("### 🧠 Educational Debrief: Cognitive Bias Revelation")

            bias_type = scenario.get('bias_type') or None
            learning_objective = scenario.get('bias_learning_objective') or None

            if not bias_type:
                st.info("⚠️ Bias type is not available for this scenario.")
                return

            bias_type_display = (
                bias_type.replace('_', ' ').title()
                if isinstance(bias_type, str)
                else str(bias_type).split('.')[-1].replace('_', ' ').title()
            )

            st.markdown(f"""
            <div style="
                background-color: #f9fafb;
                border-left: 4px solid #2563eb;
                padding: 1.2rem 1.5rem;
                border-radius: 0.75rem;
                margin-bottom: 1.5rem;
            ">
                <h4 style="margin-top: 0; color: #111827;">🎯 <span style="color: #1f2937;">Cognitive Bias Focus</span></h4>
                <p style="font-size: 1.1rem; margin-bottom: 0.75rem;">
                    This scenario was designed to test: {bias_type_display} 
                </p>
                <p style="font-size: 0.95rem; color: #4b5563;">
                    <strong>Learning Objective:</strong> {learning_objective}
                </p>
            </div>
            """, unsafe_allow_html=True)

        except Exception:
            st.info("⚠️ Bias information is currently unavailable.")

    def _render_experimental_condition_summary(self, expertise: str, ai_assistance: bool, bias_type: str, domain: str) -> None:
        """Render the summary of experimental conditions for the current session"""
        st.markdown(f"""
            <div class="experimental-condition-summary">
                <p><strong>Expertise:</strong> {expertise}</p>
                <p><strong>AI Assistance:</strong> {'Enabled' if ai_assistance else 'Disabled'}</p>
                <p><strong>Bias Type:</strong> {bias_type}</p>
                <p><strong>Domain:</strong> {domain}</p>
            </div>
        """, unsafe_allow_html=True)

    def _render_completion_navigation(self) -> str:
        """Render completion navigation options"""
        
        st.markdown("### 🚀 Next Steps")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Try Another Scenario", use_container_width=True):
                return 'new_scenario'
        
        with col2:
            if st.button("🏠 Return Home", use_container_width=True):
                return 'home'
        
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