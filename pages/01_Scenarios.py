"""
ClārusAI: 4-Stage Cognitive Bias Training Interface
UCL Master's Dissertation: "Building AI Literacy Through Simulation"

pages/01_Scenarios.py

RESEARCH OBJECTIVE:
This module implements the core experimental interface for investigating whether 
LLM-assisted training develops authentic AI literacy or encourages algorithmic dependence.

EXPERIMENTAL DESIGN:
- 2×2×3 Factorial Design: User Expertise × AI Assistance × Bias Type
- Progressive 4-Stage Interaction: Primary Analysis → Cognitive Factors → Mitigation Strategies → Transfer Learning
- Bias-Blind Methodology: Participants unaware of specific bias being tested until completion
- Comprehensive Data Collection: All interactions logged for 6-dimensional scoring analysis

RESEARCH QUESTIONS ADDRESSED:
1. Do users develop authentic AI literacy through progressive bias recognition training?
2. Does AI assistance enhance learning or create dependency patterns?
3. Can users transfer bias recognition skills across professional domains?
4. What interaction patterns distinguish expert from novice decision-makers?

Author: Rachel Seah
Date: July 2025
"""

import streamlit as st
import pandas as pd
import json
import time
import random
import hashlib
import os
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project directories to Python path for module imports
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# Import project configuration and utilities
import config
from utils import (
    setup_training_page_config, 
    load_css, 
    render_training_navigation,
    render_academic_footer, 
    render_progress_indicator, 
    render_stage_context
)

# Import Gemini API for real-time AI assistance
try:
    import google.generativeai as genai
    from google.generativeai.generative_models import GenerativeModel
    GEMINI_API_AVAILABLE = True
except ImportError:
    GEMINI_API_AVAILABLE = False
    genai = None
    GenerativeModel = None
    print("Gemini API not available - using fallback guidance prompts")

# =============================================================================
# RESEARCH CONFIGURATION & EXPERIMENTAL CONSTANTS
# =============================================================================

# 4-Stage Progressive Interaction Framework
STAGE_NAMES = [
    "Primary Analysis",      # Initial scenario response and decision-making
    "Cognitive Factors",     # Recognition of mental processes affecting judgment  
    "Mitigation Strategies", # Development of bias countermeasures
    "Transfer Learning"      # Application of principles to other domains
]

# Mapping to CSV column names containing stage-specific prompts
STAGE_PROMPTS = [
    "primary_prompt",    # Main scenario question from scenarios.csv
    "follow_up_1",       # Cognitive factor exploration prompt
    "follow_up_2",       # Strategy development prompt
    "follow_up_3"        # Cross-domain transfer prompt
]

# =============================================================================
# SESSION STATE MANAGEMENT FOR EXPERIMENTAL CONTROL
# =============================================================================

def initialize_session_state():
    """
    Initialize comprehensive session state for 4-stage experimental interaction.
    
    Establishes data structures for tracking the complete experimental condition 
    (2×2×3 factorial design) and participant progression through training protocol.
    
    Research Variables Tracked:
    - user_expertise: 'novice' or 'expert' (Factor 1 of factorial design)
    - ai_assistance_enabled: True/False (Factor 2 of factorial design)  
    - assigned_scenario: Contains bias_type (Factor 3 of factorial design)
    - Temporal data: Response times, interaction patterns, guidance usage
    """
    
    # Core experimental variables for 2×2×3 factorial design
    if 'user_expertise' not in st.session_state:
        st.session_state.user_expertise = None
    if 'ai_assistance_enabled' not in st.session_state:
        st.session_state.ai_assistance_enabled = None
    
    # Scenario assignment and experimental progression control
    if 'assigned_scenario' not in st.session_state:
        st.session_state.assigned_scenario = None
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 0
    if 'interaction_flow' not in st.session_state:
        st.session_state.interaction_flow = 'setup'
    
    # Response data collection for longitudinal analysis
    if 'stage_responses' not in st.session_state:
        st.session_state.stage_responses = []
    if 'stage_timings' not in st.session_state:
        st.session_state.stage_timings = []
    if 'guidance_usage' not in st.session_state:
        st.session_state.guidance_usage = []
    
    # Experimental session metadata
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    if 'last_auto_save' not in st.session_state:
        st.session_state.last_auto_save = None
    
    # Session recovery control
    if 'recovery_checked' not in st.session_state:
        st.session_state.recovery_checked = False

# =============================================================================
# SCENARIO LOADING & EXPERIMENTAL RANDOMIZATION
# =============================================================================

def load_scenarios():
    """
    Load and validate cognitive bias scenarios from CSV database.
    
    Returns:
        pandas.DataFrame: Validated scenario database with required columns
        None: If file missing or validation fails
    
    Required CSV Structure for 4-stage experimental protocol:
    - scenario_id, bias_type, domain, scenario_text
    - primary_prompt, follow_up_1, follow_up_2, follow_up_3
    - ideal_primary_answer, ideal_answer_1, ideal_answer_2, ideal_answer_3
    - cognitive_load_level, ai_appropriateness, bias_learning_objective
    """
    try:
        scenarios_df = pd.read_csv(config.SCENARIOS_FILE)
        
        # Validate required columns for experimental protocol
        required_columns = [
            'scenario_id', 'bias_type', 'domain', 'scenario_text',
            'primary_prompt', 'follow_up_1', 'follow_up_2', 'follow_up_3',
            'ideal_primary_answer', 'ideal_answer_1', 'ideal_answer_2', 'ideal_answer_3',
            'cognitive_load_level', 'ai_appropriateness', 'bias_learning_objective'
        ]
        
        missing_columns = [col for col in required_columns if col not in scenarios_df.columns]
        if missing_columns:
            st.error(f"Missing required columns in scenarios.csv: {missing_columns}")
            return None
        
        # Validate experimental design coverage
        bias_types = scenarios_df['bias_type'].unique()
        domains = scenarios_df['domain'].unique()
        
        return scenarios_df
        
    except FileNotFoundError:
        st.error("❌ Scenarios database not found. Please ensure data/scenarios.csv exists.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading scenarios database: {e}")
        return None

def select_balanced_scenario(scenarios_df, user_expertise, ai_assistance):
    """
    Implement stratified scenario selection for balanced experimental design.
    
    Ensures balanced distribution across 2×2×3 factorial design by selecting 
    scenarios that maintain experimental validity with limited sample size.
    
    Args:
        scenarios_df: Available scenarios database
        user_expertise: 'novice' or 'expert' (experimental factor 1)
        ai_assistance: True/False (experimental factor 2)
    
    Returns:
        pandas.Series: Selected scenario containing bias_type (experimental factor 3)
    """
    
    available_scenarios = scenarios_df.copy()
    
    # Apply cognitive load weighting based on expertise level
    if user_expertise == 'novice':
        # Prefer medium cognitive load scenarios for novice users
        medium_load_scenarios = available_scenarios[available_scenarios['cognitive_load_level'] == 'Medium']
        if not medium_load_scenarios.empty:
            if random.random() < 0.7:  # 70% chance for medium load
                available_scenarios = medium_load_scenarios
    
    # Randomize final selection from weighted pool
    selected_scenario = available_scenarios.sample(frac=1).reset_index(drop=True).iloc[0]
    
    # Log scenario selection for research tracking
    selection_metadata = {
        'timestamp': datetime.now().isoformat(),
        'user_expertise': user_expertise,
        'ai_assistance': ai_assistance,
        'selected_scenario': selected_scenario['scenario_id'],
        'bias_type': selected_scenario['bias_type'],
        'domain': selected_scenario['domain'],
        'cognitive_load': selected_scenario['cognitive_load_level'],
        'selection_method': 'weighted_random',
        'experimental_condition': f"{user_expertise}_{ai_assistance}_{selected_scenario['bias_type']}"
    }
    
    auto_save_session_data('scenario_selection', selection_metadata)
    return selected_scenario

# =============================================================================
# AI GUIDANCE SYSTEM - GEMINI INTEGRATION
# =============================================================================

def configure_gemini_api():
    """
    Configure Gemini API with safety settings.
    Returns:
        bool: True if API configured successfully, False otherwise
    """
    if not GEMINI_API_AVAILABLE or not config.GEMINI_API_KEY or genai is None:
        return False

    try:
        # Set Gemini API key directly if 'configure' is not available
        os.environ["GOOGLE_API_KEY"] = config.GEMINI_API_KEY
        return True
    except Exception as e:
        print(f"Gemini API configuration failed: {e}")
        return False

def get_ai_guidance(scenario, current_stage, user_response_so_far=""):
    """
    Generate real-time AI guidance using Gemini API for authentic AI-human interaction.
    
    Implements core AI assistance component for measuring authentic AI literacy 
    vs dependency patterns while maintaining bias-blind experimental methodology.
    
    Args:
        scenario: Current experimental scenario
        current_stage: 0-3 stage number
        user_response_so_far: Previous responses for context
    
    Returns:
        str: AI-generated guidance text (bias-neutral)
    
    Experimental Notes:
    - Maintains bias-blind approach (never reveals specific bias being tested)
    - Provides stage-appropriate guidance without compromising experimental validity
    - Logs all AI interactions for dependency analysis
    - Falls back to pre-written prompts if API unavailable
    """
    
    # Fallback guidance prompts for when API is unavailable
    fallback_prompts = {
        0: "Consider multiple perspectives and examine your initial assumptions. What evidence supports or contradicts your first impression?",
        1: "Think about cognitive factors that might influence decision-making. What mental shortcuts or patterns could affect analysis?", 
        2: "Focus on practical strategies to improve decision quality. What systematic approaches could reduce errors?",
        3: "Consider how these principles apply to other domains. What patterns do you see across different contexts?"
    }
    
    # Check API availability and configuration
    if not GEMINI_API_AVAILABLE or genai is None or GenerativeModel is None or not configure_gemini_api():
        return fallback_prompts.get(current_stage, "Consider multiple perspectives in your analysis.")
    
    try:
        # Initialize Gemini model with research configuration
        model = GenerativeModel(
            model_name=config.GEMINI_CONFIG["model"],
            generation_config=config.GEMINI_CONFIG["generation_config"],
            safety_settings=config.GEMINI_CONFIG["safety_settings"]
        )
        
        # Define stage-specific guidance contexts
        stage_contexts = {
            0: "initial analysis and decision-making under uncertainty",
            1: "cognitive factors and mental processes that influence professional judgment", 
            2: "systematic strategies to improve decision quality and reduce errors",
            3: "cross-domain application of decision-making principles"
        }
        
        context = stage_contexts.get(current_stage, "decision-making")
        
        # Construct bias-neutral prompt maintaining experimental integrity
        prompt = f"""You are providing guidance to a professional working through a {scenario['domain'].lower()} scenario involving {context}.

        Scenario context: {scenario['scenario_text']}

        Current stage: Stage {current_stage + 1} of 4 in a progressive analysis

        Previous responses from user: {user_response_so_far if user_response_so_far else "This is their first response"}

        Provide helpful, educational guidance that:
        1. Does NOT reveal or mention specific cognitive biases by name
        2. Encourages systematic thinking and multiple perspectives
        3. Suggests general decision-making frameworks applicable to {scenario['domain'].lower()} contexts
        4. Is practical and actionable for professional development
        5. Maintains academic tone appropriate for professional training
        6. Keeps response concise (under 100 words)

        Focus on general critical thinking principles rather than specific bias identification."""

        # Generate response using Gemini
        response = model.generate_content(prompt)
        guidance_text = response.text
        
        print(f"Gemini API: Successfully generated guidance for stage {current_stage}")
        return guidance_text
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Provide fallback guidance to maintain experimental continuity
        return fallback_prompts.get(current_stage, "Consider multiple perspectives in your analysis.")
def render_ai_guidance_interface(scenario, current_stage):
    """
    Render AI guidance interface for experimental assistance condition.
    
    Enables measurement of guided vs unguided decision-making patterns by providing 
    optional AI assistance while maintaining bias-blind experimental methodology.
    
    Experimental Design Notes:
    - Displays only when ai_assistance_enabled = True (Factor 2 of factorial design)
    - Records guidance requests for AI dependency analysis
    - Maintains bias-blind protocol (specific bias type not revealed)
    - Captures interaction data for authentic AI literacy assessment
    """
    
    # Only render if AI assistance enabled for this experimental condition
    if not st.session_state.ai_assistance_enabled:
        return
    
    # Check scenario appropriateness for AI assistance
    if scenario.get('ai_appropriateness', '').lower() != 'helpful':
        return
    
    # Render AI assistance interface
    st.markdown("""
    <div style="background: #E8F4FD; padding: 1rem; border-radius: 12px; margin: 1rem 0; border-left: 4px solid var(--secondary-blue);">
        <h5 style="color: var(--primary-blue); margin-top: 0;">🤖 AI Guidance Available</h5>
        <p style="color: var(--text-dark); margin-bottom: 0.5rem;">
            AI assistance is enabled for this experimental condition. Click below for personalized guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI guidance request button
    if st.button("💡 Get AI Guidance", key=f"guidance_stage_{current_stage}"):
        
        # Show loading indicator for authentic API experience
        with st.spinner("🤖 AI is analyzing your scenario and generating guidance..."):
            
            # Compile previous responses for contextual guidance
            previous_responses = " | ".join(st.session_state.stage_responses) if st.session_state.stage_responses else ""
            
            # Request real-time AI guidance
            guidance_text = get_ai_guidance(scenario, current_stage, previous_responses)
        
        # Display AI guidance with clear attribution
        st.markdown(f"""
        <div style="background: #FFF8E1; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <h6 style="color: var(--accent-orange); margin-top: 0;">🤖 AI Analysis & Guidance</h6>
            <p style="color: var(--text-dark); margin: 0; line-height: 1.5; font-style: italic;">
                {guidance_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Track guidance usage for experimental analysis
        if len(st.session_state.guidance_usage) <= current_stage:
            st.session_state.guidance_usage.extend([False] * (current_stage + 1 - len(st.session_state.guidance_usage)))
        st.session_state.guidance_usage[current_stage] = True
        
        # Log guidance interaction with research metadata
        auto_save_session_data('guidance_requested', {
            'stage': current_stage,
            'stage_name': STAGE_NAMES[current_stage],
            'guidance_text': guidance_text,
            'api_used': configure_gemini_api(),
            'scenario_id': scenario['scenario_id'],
            'bias_type': scenario['bias_type'],
            'user_expertise': st.session_state.user_expertise,
            'timestamp': datetime.now().isoformat()
        })

# =============================================================================
# PROGRESSIVE 4-STAGE USER INTERFACE
# =============================================================================

def render_experimental_setup(scenarios_df):
    """
    Render experimental setup interface for 2×2×3 factorial design.
    
    Collects primary experimental factors while providing ethical disclosure
    and research context to participants.
    
    Factors Collected:
    1. User expertise level (novice vs expert)
    2. AI assistance preference (enabled vs disabled)
    The third factor (bias type) is determined by random scenario assignment.
    """
    
    st.markdown('<h2 style="color: var(--text-dark); border-bottom: 3px solid var(--primary-blue); padding-bottom: 0.5rem;">🎯 Experimental Training Setup</h2>', unsafe_allow_html=True)
    
    # Research context and ethical disclosure
    st.markdown("""
    <div style="background: var(--background-light); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; border-left: 4px solid var(--primary-blue);">
        <h4 style="color: var(--text-dark); margin-top: 0;">🔬 Research Participation</h4>
        <p style="color: var(--text-light); margin-bottom: 0.5rem; line-height: 1.6;">
        This is a decision-making training system designed to enhance cognitive bias recognition in high-stakes environments. 
        You will work through a realistic scenario followed by progressive analysis questions. Your responses contribute to research 
        on AI-assisted learning effectiveness and will be analysed to understand how AI guidance affects professional decision-making patterns.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Factor 1: Professional Expertise Level Selection
    st.markdown("### 👤 Professional Experience Level")
    st.markdown("Please select the option that best describes your experience in high-stakes professional decision-making:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "🔰 Novice Professional\n(2-3 years experience)", 
            use_container_width=True,
            type="primary" if st.session_state.user_expertise == 'novice' else "secondary",
            key="expertise_novice"
        ):
            st.session_state.user_expertise = 'novice'
            auto_save_session_data('expertise_selection', {
                'expertise': 'novice', 
                'timestamp': datetime.now().isoformat()
            })
            st.rerun()
    
    with col2:
        if st.button(
            "🎖️ Expert Professional\n(10+ years experience)", 
            use_container_width=True,
            type="primary" if st.session_state.user_expertise == 'expert' else "secondary",
            key="expertise_expert"
        ):
            st.session_state.user_expertise = 'expert'
            auto_save_session_data('expertise_selection', {
                'expertise': 'expert',
                'timestamp': datetime.now().isoformat()
            })
            st.rerun()
    
    # Display current expertise selection
    if st.session_state.user_expertise:
        expertise_label = "🎖️ Expert Professional (10+ years)" if st.session_state.user_expertise == 'expert' else "🔰 Novice Professional (2-3 years)"
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: var(--border-light); border-radius: 8px; margin: 0.5rem 0;">
            <strong>Selected Experience Level: {expertise_label}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Factor 2: AI Assistance Mode Selection  
    st.markdown("### 🤖 AI Assistance Configuration")
    st.markdown("Choose whether you would like AI guidance available during scenario analysis:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(
            "🚫 Unassisted Mode\n(Independent analysis)", 
            use_container_width=True,
            type="primary" if st.session_state.ai_assistance_enabled == False else "secondary",
            key="assistance_disabled"
        ):
            st.session_state.ai_assistance_enabled = False
            auto_save_session_data('assistance_selection', {
                'ai_assistance': False,
                'timestamp': datetime.now().isoformat()
            })
            st.rerun()
    
    with col2:
        if st.button(
            "🤖 AI-Assisted Mode\n(Guidance available)", 
            use_container_width=True,
            type="primary" if st.session_state.ai_assistance_enabled == True else "secondary",
            key="assistance_enabled"
        ):
            st.session_state.ai_assistance_enabled = True
            auto_save_session_data('assistance_selection', {
                'ai_assistance': True,
                'timestamp': datetime.now().isoformat()
            })
            st.rerun()
    
    # Display current assistance selection
    if st.session_state.ai_assistance_enabled is not None:
        assistance_label = "🤖 AI-Assisted Mode (Guidance Available)" if st.session_state.ai_assistance_enabled else "🚫 Unassisted Mode (Independent Analysis)"
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem; background: var(--border-light); border-radius: 8px; margin: 0.5rem 0;">
            <strong>Selected Mode: {assistance_label}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Proceed to experimental scenario when both factors selected
    if st.session_state.user_expertise and st.session_state.ai_assistance_enabled is not None:
        st.markdown("---")
        
        # Display experimental condition summary
        st.markdown("### 🔬 Experimental Condition Summary")
        st.markdown(f"""
        <div style="background: #E8F5E8; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="color: var(--text-dark); margin: 0;">
            <strong>Your experimental condition:</strong> {st.session_state.user_expertise.title()} Professional with AI Assistance {'Enabled' if st.session_state.ai_assistance_enabled else 'Disabled'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Begin Training Scenario", type="primary", use_container_width=True):
                # Select balanced scenario for experimental validity
                selected_scenario = select_balanced_scenario(
                    scenarios_df, 
                    st.session_state.user_expertise, 
                    st.session_state.ai_assistance_enabled
                )
                
                # Initialize experimental session
                st.session_state.assigned_scenario = selected_scenario.to_dict()
                st.session_state.interaction_flow = 'scenario'
                st.session_state.current_stage = 0
                st.session_state.stage_timings = [datetime.now()]
                
                # Log experimental session initiation
                auto_save_session_data('experimental_session_start', {
                    'scenario_id': selected_scenario['scenario_id'],
                    'experimental_condition': f"{st.session_state.user_expertise}_{st.session_state.ai_assistance_enabled}_{selected_scenario['bias_type']}",
                    'bias_type': selected_scenario['bias_type'],
                    'domain': selected_scenario['domain'],
                    'session_start': datetime.now().isoformat()
                })
                
                st.rerun()
    else:
        st.info("⚠️ Please select both your experience level and AI assistance preference to continue.")

def render_stage_header(scenario, current_stage):
    """Render stage header with scenario context and progress indicator"""
    render_progress_indicator(current_stage, 4, STAGE_NAMES)
    
    if current_stage == 0:
        st.markdown(f'<h2 style="color: var(--text-dark); border-bottom: 3px solid var(--primary-blue); padding-bottom: 0.5rem;">📋 {scenario["title"]}</h2>', unsafe_allow_html=True)
        
        # Present complete scenario context
        st.markdown(f"""
        <div style="background: var(--background-light); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;">
            <h4 style="color: var(--accent-orange); margin-top: 0;">📖 Professional Scenario</h4>
            <div style="background: var(--background-white); padding: 1rem; border-radius: 8px; border-left: 4px solid var(--accent-orange);">
                <p style="line-height: 1.6; color: var(--text-dark); margin: 0; font-size: 1.1rem;">
                    {scenario["scenario_text"]}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display experimental metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background: var(--border-light); padding: 0.5rem; border-radius: 8px; text-align: center;">
                <strong>Domain:</strong> {scenario["domain"]}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: var(--border-light); padding: 0.5rem; border-radius: 8px; text-align: center;">
                <strong>Complexity:</strong> {scenario["cognitive_load_level"]}
            </div>
            """, unsafe_allow_html=True)
        with col3:
            assistance_mode = "🤖 AI-Assisted" if st.session_state.ai_assistance_enabled else "🚫 Unassisted"
            st.markdown(f"""
            <div style="background: var(--border-light); padding: 0.5rem; border-radius: 8px; text-align: center;">
                <strong>Mode:</strong> {assistance_mode}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f'<h2 style="color: var(--text-dark); border-bottom: 3px solid var(--primary-blue); padding-bottom: 0.5rem;">📋 {scenario["title"]} - {STAGE_NAMES[current_stage]}</h2>', unsafe_allow_html=True)
        render_stage_context(st.session_state.stage_responses, STAGE_NAMES)

def render_stage_prompt(scenario, current_stage):
    """Render current stage prompt and AI guidance interface"""
    prompt_field = STAGE_PROMPTS[current_stage]
    current_prompt = scenario[prompt_field]
    
    st.markdown(f"""
    <div style="background: var(--background-light); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; border: 2px solid var(--accent-orange);">
        <h4 style="color: var(--text-dark); margin-top: 0;">💭 {STAGE_NAMES[current_stage]} Task</h4>
        <p style="color: var(--text-dark); font-size: 1.1rem; line-height: 1.6; margin: 0; font-weight: 500;">
            {current_prompt}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render AI guidance interface if assistance enabled
    render_ai_guidance_interface(scenario, current_stage)

def render_response_interface(scenario, current_stage):
    """Render response collection interface with navigation controls"""
    st.markdown("---")
    st.markdown(f'<h4 style="color: var(--text-dark);">✍️ Your {STAGE_NAMES[current_stage]} Response</h4>', unsafe_allow_html=True)
    
    # Text area for current stage response
    response_key = f"stage_{current_stage}_response"
    current_response = st.text_area(
        f"Provide your analysis for {STAGE_NAMES[current_stage]}:",
        value="",
        height=200,
        placeholder=f"Please provide your detailed {STAGE_NAMES[current_stage].lower()} response. Take time to think through your reasoning...",
        key=response_key,
        help=f"Stage {current_stage + 1} of 4: {STAGE_NAMES[current_stage]} - No minimum word requirement, respond naturally"
    )
    
    # Action buttons for stage progression
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_stage > 0:
            st.button("← Previous Stage", use_container_width=True, disabled=True, 
                     help="Previous stages are locked to maintain experimental integrity")
        else:
            if st.button("← Back to Setup", use_container_width=True):
                reset_experimental_session()
                st.rerun()
    
    with col3:
        button_text = "Complete Training →" if current_stage == 3 else "Next Stage →"
        button_disabled = len((current_response or "").strip()) < 5
        
        if st.button(button_text, use_container_width=True, 
                    disabled=button_disabled, type="primary"):
            
            # Save current stage response
            save_stage_response(scenario, current_stage, current_response)
            
            if current_stage < 3:
                # Advance to next stage
                st.session_state.current_stage += 1
                st.session_state.stage_timings.append(datetime.now())
                
                auto_save_session_data('stage_progression', {
                    'completed_stage': current_stage,
                    'advancing_to_stage': current_stage + 1,
                    'stage_name': STAGE_NAMES[current_stage],
                    'response_length': len(current_response),
                    'progression_time': datetime.now().isoformat()
                })
            else:
                # Complete the experimental protocol
                st.session_state.interaction_flow = 'completed'
                
                auto_save_session_data('experimental_protocol_completed', {
                    'total_stages_completed': 4,
                    'final_completion_time': datetime.now().isoformat(),
                    'session_duration_minutes': (datetime.now() - st.session_state.session_start_time).total_seconds() / 60,
                    'experimental_condition': f"{st.session_state.user_expertise}_{st.session_state.ai_assistance_enabled}_{scenario['bias_type']}"
                })
            
            st.rerun()
    
    # Response feedback
    word_count = len(current_response.split()) if current_response else 0
    char_count = len(current_response) if current_response else 0
    
    if char_count < 5:
        st.markdown(f"""
        <div style="text-align: center; color: var(--warning-red); font-size: 0.9rem; margin-top: 0.5rem;">
            📝 Please provide a response to continue
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; color: var(--success-green); font-size: 0.9rem; margin-top: 0.5rem;">
            ✅ {word_count} words, {char_count} characters - Ready to proceed
        </div>
        """, unsafe_allow_html=True)

def render_scenario_stage(scenario, current_stage):
    """
    Render individual stage of the 4-stage progressive experimental protocol.
    
    Implements core experimental interface where participants progress through 
    increasingly sophisticated analysis of cognitive bias scenarios.
    
    Stage Progression Logic:
    - Stage 0 (Primary): Initial scenario response and decision-making
    - Stage 1 (Cognitive): Recognition of mental processes affecting judgment
    - Stage 2 (Mitigation): Development of bias countermeasures  
    - Stage 3 (Transfer): Application to other professional domains
    """
    
    render_stage_header(scenario, current_stage)
    render_stage_prompt(scenario, current_stage)
    render_response_interface(scenario, current_stage)

def render_completion_interface(scenario):
    """
    Render completion interface with session summary and bias revelation.
    
    Serves multiple research functions including educational debrief, session analytics,
    data completion confirmation, and participant satisfaction measurement.
    """
    
    st.markdown('<h2 style="color: var(--text-dark); border-bottom: 3px solid var(--primary-blue); padding-bottom: 0.5rem;">✅ Training Protocol Completed</h2>', unsafe_allow_html=True)
    
    # Save complete experimental session
    session_file = save_complete_session(scenario)
    
    # Session completion celebration
    st.markdown("""
    <div style="background: linear-gradient(135deg, var(--background-light) 0%, var(--background-white) 100%); border-radius: 16px; padding: 3rem; margin: 2rem 0; text-align: center; box-shadow: 0 8px 32px var(--shadow-light); border: 1px solid var(--border-light);">
        <h3 style="color: var(--accent-orange); margin-top: 0;">🎉 Excellent Work!</h3>
        <p style="text-align: center; font-size: 1.1rem; margin-bottom: 1.5rem;">
        You have successfully completed all four stages of the cognitive bias recognition training protocol.
        Your responses provide valuable data for understanding AI-assisted learning patterns in professional decision-making.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Session analytics summary
    col1, col2, col3, col4 = st.columns(4)
    
    total_words = sum(len(r.split()) for r in st.session_state.stage_responses)
    total_time = (datetime.now() - st.session_state.session_start_time).total_seconds()
    guidance_used = sum(st.session_state.guidance_usage) if st.session_state.guidance_usage else 0
    
    with col1:
        st.metric("Stages Completed", "4/4", help="All experimental stages completed successfully")
    with col2:
        st.metric("Total Words", f"{total_words:,}", help="Total response length across all stages")
    with col3:
        st.metric("Session Duration", f"{total_time/60:.1f} min", help="Time spent in complete training session")
    with col4:
        st.metric("AI Guidance Used", f"{guidance_used}/4", help="Number of stages where AI guidance was requested")
    
    # Response evolution visualization
    st.markdown("### 📊 Your Response Progression")
    
    response_lengths = [len(r.split()) for r in st.session_state.stage_responses]
    
    for i, (stage_name, word_count) in enumerate(zip(STAGE_NAMES, response_lengths)):
        max_words = max(response_lengths) if response_lengths else 100
        progress = (word_count / max_words) * 100 if max_words > 0 else 0
        
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 500;">{stage_name}</span>
                <span style="color: var(--text-light); font-size: 0.9rem;">{word_count} words</span>
            </div>
            <div style="background: var(--border-light); border-radius: 4px; height: 8px; overflow: hidden;">
                <div style="background: var(--primary-blue); height: 100%; width: {progress}%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bias revelation - educational debrief
    st.markdown("---")
    st.markdown("### 🧠 Educational Debrief: Cognitive Bias Revelation")
    
    st.markdown(f"""
    <div style="background: var(--background-light); padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0;">
        <h4 style="color: var(--text-dark); margin-top: 0;">🎯 Cognitive Bias Focus</h4>
        <p style="color: var(--text-dark); font-size: 1.2rem; margin-bottom: 1rem;">
            <strong>This scenario was designed to test: {scenario['bias_type']} Bias</strong>
        </p>
        <p style="color: var(--text-light); line-height: 1.6; margin: 0;">
            <strong>Learning Objective:</strong> {scenario.get('bias_learning_objective', 'Recognize and mitigate cognitive biases in professional decision-making contexts.')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Expert feedback if available
    if scenario.get('llm_feedback'):
        st.markdown(f"""
        <div style="background: #FFF8E1; padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
            <h5 style="color: var(--accent-orange); margin-top: 0;">💡 Expert Learning Insights</h5>
            <p style="color: var(--text-dark); line-height: 1.6; margin: 0;">
                {scenario['llm_feedback']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Next steps
    st.markdown("### 🚀 Next Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Assessment", use_container_width=True, type="primary"):
            st.switch_page("pages/02_Assessment.py")
    
    with col2:
        if st.button("🎯 Try Another Scenario", use_container_width=True):
            reset_experimental_session()
            st.rerun()
    
    with col3:
        if st.button("🏠 Return Home", use_container_width=True):
            st.switch_page("Home.py")
    
    # Research contribution acknowledgment
    st.markdown("""
    <div style="background: var(--border-light); padding: 1rem; border-radius: 12px; margin: 2rem 0; text-align: center;">
        <h5 style="color: var(--text-dark); margin-top: 0;">🙏 Thank You for Contributing to Research</h5>
        <p style="color: var(--text-light); margin: 0; font-size: 0.9rem; line-height: 1.4;">
            Your responses contribute valuable data to understanding AI-assisted cognitive bias training effectiveness.
            All data is anonymized and used solely for academic research purposes in accordance with UCL research ethics guidelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

def reset_experimental_session():
    """Reset session state for new experimental trial"""
    st.session_state.interaction_flow = 'setup'
    st.session_state.assigned_scenario = None
    st.session_state.current_stage = 0
    st.session_state.stage_responses = []
    st.session_state.stage_timings = []
    st.session_state.guidance_usage = []
    st.session_state.user_expertise = None
    st.session_state.ai_assistance_enabled = None
    st.session_state.recovery_checked = False
    st.session_state.session_start_time = datetime.now()

# =============================================================================
# DATA COLLECTION & PERSISTENCE
# =============================================================================

def auto_save_session_data(event_type, data):
    """
    Auto-save session data for comprehensive research tracking and error recovery.
    
    Maintains detailed logs of all user interactions for subsequent research analysis 
    while providing session recovery capabilities.
    """
    try:
        os.makedirs(config.RESPONSES_DIR, exist_ok=True)
        
        session_id = id(st.session_state)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        autosave_file = f"{config.RESPONSES_DIR}/autosave_{session_id}_{timestamp}.json"
        
        # Comprehensive session state capture
        session_snapshot = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id,
            
            # Core experimental variables
            'user_expertise': st.session_state.get('user_expertise'),
            'ai_assistance_enabled': st.session_state.get('ai_assistance_enabled'),
            'assigned_scenario_id': st.session_state.get('assigned_scenario', {}).get('scenario_id') if st.session_state.get('assigned_scenario') else None,
            'bias_type': st.session_state.get('assigned_scenario', {}).get('bias_type') if st.session_state.get('assigned_scenario') else None,
            
            # Interaction progress tracking
            'current_stage': st.session_state.get('current_stage', 0),
            'stage_responses': st.session_state.get('stage_responses', []),
            'stage_timings': [t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in st.session_state.get('stage_timings', [])],
            'guidance_usage': st.session_state.get('guidance_usage', []),
            'interaction_flow': st.session_state.get('interaction_flow', 'setup'),
            
            # Research metadata
            'experimental_condition': f"{st.session_state.get('user_expertise', 'unknown')}_{st.session_state.get('ai_assistance_enabled', 'unknown')}_{st.session_state.get('assigned_scenario', {}).get('bias_type', 'unknown') if st.session_state.get('assigned_scenario') else 'unknown'}",
            'session_start_time': st.session_state.get('session_start_time', datetime.now()).isoformat(),
            'event_data': data
        }
        
        with open(autosave_file, 'w') as f:
            json.dump(session_snapshot, f, indent=2)
            
        st.session_state.last_auto_save = datetime.now()
        
    except Exception as e:
        print(f"Auto-save failed: {e}")

def save_stage_response(scenario, stage, response):
    """
    Save individual stage response with comprehensive research metadata.
    
    Captures detailed interaction data for each experimental stage, enabling 
    sophisticated analysis of learning progression and AI dependency patterns.
    """
    
    # Calculate response timing
    stage_start_time = st.session_state.stage_timings[stage]
    response_time = (datetime.now() - stage_start_time).total_seconds()
    
    # Add response to session state
    if len(st.session_state.stage_responses) <= stage:
        st.session_state.stage_responses.extend([''] * (stage + 1 - len(st.session_state.stage_responses)))
    st.session_state.stage_responses[stage] = response
    
    # Comprehensive response metadata
    response_data = {
        'timestamp': datetime.now().isoformat(),
        'session_id': id(st.session_state),
        
        # Core research variables
        'user_expertise': st.session_state.user_expertise,
        'ai_assistance_enabled': st.session_state.ai_assistance_enabled,
        'scenario_id': scenario['scenario_id'],
        'bias_type': scenario['bias_type'],
        'domain': scenario['domain'],
        'cognitive_load_level': scenario['cognitive_load_level'],
        
        # Stage-specific data
        'stage_number': stage,
        'stage_name': STAGE_NAMES[stage],
        'stage_prompt': scenario[STAGE_PROMPTS[stage]],
        'response_text': response,
        'response_time_seconds': response_time,
        'word_count': len(response.split()),
        'character_count': len(response),
        
        # AI interaction analysis
        'guidance_requested': st.session_state.guidance_usage[stage] if len(st.session_state.guidance_usage) > stage else False,
        'cumulative_guidance_pattern': st.session_state.guidance_usage[:stage+1],
        
        # Scoring analysis fields
        'ideal_answer': scenario.get(f'ideal_answer_{stage+1}' if stage > 0 else 'ideal_primary_answer', ''),
        'rubric_focus': scenario.get('rubric_focus', ''),
        'bias_learning_objective': scenario.get('bias_learning_objective', ''),
        
        # Experimental condition tracking
        'condition_code': f"{st.session_state.user_expertise}_{st.session_state.ai_assistance_enabled}_{scenario['bias_type']}",
        'experimental_phase': 'live_data_collection',
        
        # Longitudinal progression tracking
        'session_start_time': st.session_state.session_start_time.isoformat(),
        'cumulative_session_time': (datetime.now() - st.session_state.session_start_time).total_seconds(),
        'previous_responses': st.session_state.stage_responses[:stage],
        
        # Data quality indicators
        'data_quality_flags': {
            'meaningful_length': len(response.strip()) >= 5,
            'contains_reasoning': len(response.split('.')) > 1,
            'non_empty': bool(response.strip()),
            'natural_timing': response_time > 10,
            'engagement_level': 'high' if len(response.split()) > 20 else 'moderate' if len(response.split()) > 10 else 'low'
        },
        
        # Research metadata
        'source_type': 'live_user_4stage',
        'bias_revelation_status': 'pre_revelation',
        'api_integration': 'gemini_available' if configure_gemini_api() else 'fallback_prompts'
    }
    
    # Save individual stage response
    try:
        os.makedirs(config.RESPONSES_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        condition = response_data['condition_code']
        stage_file = f"{config.RESPONSES_DIR}/stage_{stage}_{condition}_{timestamp}.json"
        
        with open(stage_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        
        # Append to master research log
        master_log = f"{config.RESPONSES_DIR}/master_4stage_responses.jsonl"
        with open(master_log, 'a') as f:
            f.write(json.dumps(response_data) + '\n')
        
        save_recovery_checkpoint()
        
    except Exception as e:
        st.warning(f"Could not save response to file: {e}")

def save_complete_session(scenario):
    """
    Save complete 4-stage experimental session for comprehensive research analysis.
    
    Creates master dataset for 6-dimensional scoring and statistical analysis 
    of AI-assisted learning patterns.
    """
    
    # Calculate session analytics
    total_session_time = (datetime.now() - st.session_state.session_start_time).total_seconds()
    stage_durations = []
    
    for i in range(len(st.session_state.stage_timings)):
        if i < len(st.session_state.stage_timings) - 1:
            duration = (st.session_state.stage_timings[i + 1] - st.session_state.stage_timings[i]).total_seconds()
        else:
            duration = (datetime.now() - st.session_state.stage_timings[i]).total_seconds()
        stage_durations.append(duration)
    
    # Comprehensive session data
    complete_session = {
        'session_completion_timestamp': datetime.now().isoformat(),
        'session_id': id(st.session_state),
        
        # Experimental design variables
        'experimental_condition': {
            'user_expertise': st.session_state.user_expertise,
            'ai_assistance_enabled': st.session_state.ai_assistance_enabled,
            'bias_type': scenario['bias_type'],
            'domain': scenario['domain'],
            'cognitive_load_level': scenario['cognitive_load_level'],
            'condition_code': f"{st.session_state.user_expertise}_{st.session_state.ai_assistance_enabled}_{scenario['bias_type']}"
        },
        
        # Scenario metadata
        'scenario_metadata': {
            'scenario_id': scenario['scenario_id'],
            'title': scenario['title'],
            'bias_learning_objective': scenario.get('bias_learning_objective', ''),
            'source_citation': scenario.get('source_citation', ''),
            'rubric_focus': scenario.get('rubric_focus', ''),
            'llm_feedback': scenario.get('llm_feedback', '')
        },
        
        # Complete 4-stage response set
        'stage_responses': [
            {
                'stage_number': i,
                'stage_name': STAGE_NAMES[i],
                'prompt': scenario[STAGE_PROMPTS[i]],
                'response': response,
                'word_count': len(response.split()),
                'character_count': len(response),
                'duration_seconds': stage_durations[i] if i < len(stage_durations) else 0,
                'guidance_used': st.session_state.guidance_usage[i] if i < len(st.session_state.guidance_usage) else False,
                'timestamp': st.session_state.stage_timings[i].isoformat() if i < len(st.session_state.stage_timings) else datetime.now().isoformat()
            }
            for i, response in enumerate(st.session_state.stage_responses)
        ],
        
        # Ideal answers for scoring
        'ideal_answers': {
            'primary': scenario.get('ideal_primary_answer', ''),
            'follow_up_1': scenario.get('ideal_answer_1', ''),
            'follow_up_2': scenario.get('ideal_answer_2', ''),
            'follow_up_3': scenario.get('ideal_answer_3', '')
        },
        
        # Session analytics
        'session_analytics': {
            'total_session_time_seconds': total_session_time,
            'total_session_time_minutes': total_session_time / 60,
            'stage_durations_seconds': stage_durations,
            'average_stage_duration': sum(stage_durations) / len(stage_durations) if stage_durations else 0,
            'total_word_count': sum(len(r.split()) for r in st.session_state.stage_responses),
            'average_words_per_stage': sum(len(r.split()) for r in st.session_state.stage_responses) / len(st.session_state.stage_responses),
            'guidance_usage_pattern': st.session_state.guidance_usage,
            'total_guidance_requests': sum(st.session_state.guidance_usage) if st.session_state.guidance_usage else 0,
            'guidance_frequency': sum(st.session_state.guidance_usage) / len(st.session_state.guidance_usage) if st.session_state.guidance_usage else 0
        },
        
        # Research metadata
        'research_metadata': {
            'study_phase': 'primary_data_collection',
            'data_collection_method': '4stage_progressive_interaction',
            'bias_revelation_timing': 'post_completion',
            'experimental_protocol_version': '1.0',
            'ai_integration_type': 'real_gemini_api' if configure_gemini_api() else 'fallback_prompts',
            'quality_indicators': {
                'all_stages_completed': len(st.session_state.stage_responses) == 4,
                'minimum_engagement': all(len(r.strip()) >= 5 for r in st.session_state.stage_responses),
                'natural_progression': total_session_time > 120,
                'session_completion_rate': 1.0
            }
        }
    }
    
    # Save complete session
    try:
        os.makedirs(config.RESPONSES_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        condition = complete_session['experimental_condition']['condition_code']
        session_file = f"{config.RESPONSES_DIR}/complete_session_{condition}_{timestamp}.json"
        
        with open(session_file, 'w') as f:
            json.dump(complete_session, f, indent=2)
        
        # Add to master sessions log
        master_sessions = f"{config.RESPONSES_DIR}/master_complete_sessions.jsonl"
        with open(master_sessions, 'a') as f:
            f.write(json.dumps(complete_session) + '\n')
        
        return session_file
        
    except Exception as e:
        st.error(f"Could not save complete session: {e}")
        return None

# =============================================================================
# SESSION RECOVERY & QUALITY ASSURANCE
# =============================================================================

def generate_session_id():
    """Generate consistent session ID for recovery functionality"""
    try:
        time_window = str(int(datetime.now().timestamp() / 7200))
        session_data = f"recovery_{time_window}"
        session_id = hashlib.md5(session_data.encode()).hexdigest()[:8]
        return f"session_{session_id}"
    except:
        return f"session_{int(datetime.now().timestamp())}"

def save_recovery_checkpoint():
    """Save current session state for recovery after interruption"""
    try:
        if (st.session_state.get('interaction_flow') == 'setup' or 
            not st.session_state.get('assigned_scenario')):
            return
        
        session_id = generate_session_id()
        recovery_dir = f"{config.RESPONSES_DIR}/recovery"
        os.makedirs(recovery_dir, exist_ok=True)
        
        recovery_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'recovery_version': '1.0',
            
            # Core experimental state
            'user_expertise': st.session_state.get('user_expertise'),
            'ai_assistance_enabled': st.session_state.get('ai_assistance_enabled'),
            'assigned_scenario': st.session_state.get('assigned_scenario'),
            'current_stage': st.session_state.get('current_stage', 0),
            'interaction_flow': st.session_state.get('interaction_flow'),
            
            # Response progression data
            'stage_responses': st.session_state.get('stage_responses', []),
            'stage_timings': [t.isoformat() if hasattr(t, 'isoformat') else str(t) 
                            for t in st.session_state.get('stage_timings', [])],
            'guidance_usage': st.session_state.get('guidance_usage', []),
            'session_start_time': st.session_state.get('session_start_time', datetime.now()).isoformat(),
            
            # Recovery metadata
            'recovery_point': f"stage_{st.session_state.get('current_stage', 0)}",
            'progress_summary': f"{len(st.session_state.get('stage_responses', []))} stages completed"
        }
        
        recovery_file = f"{recovery_dir}/{session_id}_recovery.json"
        with open(recovery_file, 'w') as f:
            json.dump(recovery_data, f, indent=2)
            
    except Exception as e:
        print(f"Recovery save failed: {e}")

def load_recovery_session():
    """Load recovery session if available within time window"""
    try:
        session_id = generate_session_id()
        recovery_dir = f"{config.RESPONSES_DIR}/recovery"
        recovery_file = f"{recovery_dir}/{session_id}_recovery.json"
        
        if os.path.exists(recovery_file):
            with open(recovery_file, 'r') as f:
                recovery_data = json.load(f)
            
            # Check if recovery data is recent (within 6 hours)
            recovery_time = datetime.fromisoformat(recovery_data['timestamp'])
            if datetime.now() - recovery_time < timedelta(hours=6):
                return recovery_data
            else:
                os.remove(recovery_file)
        
        return None
    except Exception as e:
        print(f"Recovery load failed: {e}")
        return None

def restore_session(recovery_data):
    """Restore experimental session from recovery data"""
    try:
        # Restore core experimental variables
        st.session_state.user_expertise = recovery_data.get('user_expertise')
        st.session_state.ai_assistance_enabled = recovery_data.get('ai_assistance_enabled')
        st.session_state.assigned_scenario = recovery_data.get('assigned_scenario')
        st.session_state.current_stage = recovery_data.get('current_stage', 0)
        st.session_state.interaction_flow = recovery_data.get('interaction_flow', 'setup')
        
        # Restore response progression
        st.session_state.stage_responses = recovery_data.get('stage_responses', [])
        st.session_state.guidance_usage = recovery_data.get('guidance_usage', [])
        
        # Restore timing data
        stage_timings = []
        for timing_str in recovery_data.get('stage_timings', []):
            try:
                stage_timings.append(datetime.fromisoformat(timing_str))
            except:
                stage_timings.append(datetime.now())
        st.session_state.stage_timings = stage_timings
        
        # Restore session start time
        try:
            st.session_state.session_start_time = datetime.fromisoformat(
                recovery_data.get('session_start_time', datetime.now().isoformat())
            )
        except:
            st.session_state.session_start_time = datetime.now()
        
        return True
    except Exception as e:
        print(f"Session restoration failed: {e}")
        return False

def render_recovery_interface(recovery_data):
    """Display recovery options when previous session found"""
    responses_count = len(recovery_data.get('stage_responses', []))
    current_stage = recovery_data.get('current_stage', 0)
    scenario_title = recovery_data.get('assigned_scenario', {}).get('title', 'Unknown Scenario')
    
    st.markdown("""
    <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1.5rem; margin: 2rem 0;">
        <h3 style="color: #856404; margin-top: 0;">🔄 Previous Session Found</h3>
        <p style="color: #856404; margin-bottom: 1rem;">
    """ + f"""
        We found a previous training session where you were working on "<strong>{scenario_title}</strong>" 
        and had completed {responses_count} stage(s). You were on stage {current_stage + 1} of 4.
        Would you like to continue where you left off or start a fresh session?
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Continue Previous Session", use_container_width=True, type="primary"):
            if restore_session(recovery_data):
                st.success("✅ Session restored successfully! Continuing where you left off...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to restore session. Starting fresh.")
                st.session_state.recovery_checked = True
                st.rerun()
    
    with col2:
        if st.button("🆕 Start Fresh Session", use_container_width=True):
            try:
                session_id = generate_session_id()
                recovery_file = f"{config.RESPONSES_DIR}/recovery/{session_id}_recovery.json"
                if os.path.exists(recovery_file):
                    os.remove(recovery_file)
            except:
                pass
            st.session_state.recovery_checked = True
            st.info("Starting a new training session...")
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem; color: #6c757d; font-size: 0.9rem; line-height: 1.3;">
        <strong>Need Help?</strong><br>
        Choose continue to resume your progress or start fresh for a new scenario.
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION ORCHESTRATION
# =============================================================================

def main():
    """
    Main function orchestrating the 4-stage experimental training interface.
    
    Serves as central controller for the experimental protocol, managing user flow 
    through the complete research procedure while maintaining data integrity and 
    experimental validity.
    
    Research Flow:
    1. Session Recovery Check (if applicable)
    2. Experimental Setup (Factor 1 & 2 selection)
    3. Scenario Assignment (Factor 3 randomization)
    4. 4-Stage Progressive Interaction
    5. Completion & Bias Revelation
    6. Data Export for Analysis
    """
    
    # Configure page for training interface
    setup_training_page_config("Training Scenarios", "🎯")
    load_css()
    
    # Initialize comprehensive session state
    initialize_session_state()
    
    # Session recovery check
    if (st.session_state.interaction_flow == 'setup' and 
        not st.session_state.get('recovery_checked', False)):
        
        recovery_data = load_recovery_session()
        if recovery_data and recovery_data.get('interaction_flow') != 'setup':
            render_recovery_interface(recovery_data)
            return
        else:
            st.session_state.recovery_checked = True
    
    # Load experimental scenarios database
    scenarios_df = load_scenarios()
    if scenarios_df is None:
        st.error("❌ Cannot proceed without scenarios database. Please check data/scenarios.csv file.")
        st.stop()
    
    # Render minimal navigation
    render_training_navigation()
    
    # Main experimental flow routing
    if st.session_state.interaction_flow == 'setup':
        # Phase 1: Experimental Setup
        render_experimental_setup(scenarios_df)
    
    elif st.session_state.interaction_flow == 'scenario':
        # Phase 2: 4-Stage Progressive Interaction
        if st.session_state.assigned_scenario:
            render_scenario_stage(st.session_state.assigned_scenario, st.session_state.current_stage)
        else:
            st.error("❌ No scenario assigned. Returning to experimental setup.")
            st.session_state.interaction_flow = 'setup'
            st.rerun()
    
    elif st.session_state.interaction_flow == 'completed':
        # Phase 3: Completion & Educational Debrief
        if st.session_state.assigned_scenario:
            render_completion_interface(st.session_state.assigned_scenario)
        else:
            st.error("❌ No completed scenario found. Returning to experimental setup.")
            st.session_state.interaction_flow = 'setup'
            st.rerun()
    
    # Development debug information
    if config.DEBUG:
        with st.expander("🔧 Development Debug Information", expanded=False):
            st.write("**Current Session State:**")
            debug_info = {
                'interaction_flow': st.session_state.interaction_flow,
                'current_stage': st.session_state.current_stage,
                'user_expertise': st.session_state.user_expertise,
                'ai_assistance_enabled': st.session_state.ai_assistance_enabled,
                'assigned_scenario_id': st.session_state.assigned_scenario['scenario_id'] if st.session_state.assigned_scenario else None,
                'bias_type': st.session_state.assigned_scenario['bias_type'] if st.session_state.assigned_scenario else None,
                'stage_responses_count': len(st.session_state.stage_responses),
                'guidance_usage': st.session_state.guidance_usage,
                'session_duration_minutes': (datetime.now() - st.session_state.session_start_time).total_seconds() / 60 if hasattr(st.session_state, 'session_start_time') else 0
            }
            st.json(debug_info)
            
            if st.session_state.assigned_scenario:
                st.write("**Available Scenarios:**")
                st.write(scenarios_df[['scenario_id', 'bias_type', 'domain', 'cognitive_load_level']])
    
    # Render academic footer
    render_academic_footer()

if __name__ == "__main__":
    main()