"""
Training Scenarios Page - Interactive Cognitive Bias Training
pages/02_Scenarios.py
"""

import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

import config

def load_css():
    """Load custom CSS"""
    with open("assets/styles/main.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def setup_page():
    """Page configuration and setup"""
    st.set_page_config(
        page_title="Training Scenarios - ClārusAI",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "ClārusAI - UCL Master's Dissertation Research"
        }
    )
    
    # Hide default Streamlit elements
    st.markdown("""
    <style>
    .stSidebar {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .stApp > header {display: none;}
    </style>
    """, unsafe_allow_html=True)

def render_academic_footer():
    """Academic footer"""
    st.markdown("""
    <div class="academic-footer">
        <h4 style="margin-top: 0; color: white;">UCL Master's Dissertation Research</h4>
        <p style="margin-bottom: 0; opacity: 0.9; font-size: 1rem;">
        Building AI Literacy Through Simulation: Evaluating LLM-Assisted Cognitive Bias Training
        </p>
    </div>
    """, unsafe_allow_html=True)

def load_scenarios():
    """Load scenarios from CSV file"""
    try:
        scenarios_df = pd.read_csv(config.SCENARIOS_FILE)
        return scenarios_df
    except FileNotFoundError:
        st.error("Scenarios file not found. Please ensure data/scenarios.csv exists.")
        return None

def get_scenario_by_id(scenarios_df, scenario_id):
    """Get specific scenario by ID"""
    scenario = scenarios_df[scenarios_df['scenario_id'] == scenario_id]
    if not scenario.empty:
        return scenario.iloc[0]
    return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = None
    if 'ai_assistance_enabled' not in st.session_state:
        st.session_state.ai_assistance_enabled = False
    if 'scenario_start_time' not in st.session_state:
        st.session_state.scenario_start_time = None
    if 'user_responses' not in st.session_state:
        st.session_state.user_responses = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'selection'  # selection, scenario, response
    if 'response_text' not in st.session_state:
        st.session_state.response_text = ""

def render_scenario_selection(scenarios_df):
    """Render scenario selection interface"""
    st.markdown('<h2 class="section-header">🎯 Select Training Scenario</h2>', unsafe_allow_html=True)
    
    # User Expertise Level Selection (FOR RESEARCH DESIGN)
    st.markdown("""
    <div style="background: var(--background-secondary); padding: var(--space-md); border-radius: var(--radius-lg); margin: var(--space-lg) 0; border: 1px solid var(--border-light);">
        <h4 style="color: var(--text-primary); margin-top: 0;">👤 Professional Experience Level</h4>
        <p style="color: var(--text-secondary); margin-bottom: var(--space-sm);">
        Please indicate your experience level in high-stakes professional decision-making:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔰 Novice (2-3 years)", use_container_width=True, 
                    type="primary" if st.session_state.get('user_expertise') == 'novice' else "secondary"):
            st.session_state.user_expertise = 'novice'
            st.rerun()
    
    with col2:
        if st.button("🎖️ Expert (10+ years)", use_container_width=True,
                    type="primary" if st.session_state.get('user_expertise') == 'expert' else "secondary"):
            st.session_state.user_expertise = 'expert'
            st.rerun()
    
    # Show current selection
    if 'user_expertise' in st.session_state:
        expertise_label = "🎖️ Expert Professional" if st.session_state.user_expertise == 'expert' else "🔰 Novice Professional"
        st.markdown(f"""
        <div style="text-align: center; padding: var(--space-sm); background: var(--background-tertiary); border-radius: var(--radius-md); margin: var(--space-sm) 0;">
            <strong>Experience Level: {expertise_label}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Assistance Toggle
    st.markdown("""
    <div style="background: var(--background-secondary); padding: var(--space-md); border-radius: var(--radius-lg); margin: var(--space-lg) 0; border: 1px solid var(--border-light);">
        <h4 style="color: var(--text-primary); margin-top: 0;">🤖 AI Assistance Setting</h4>
        <p style="color: var(--text-secondary); margin-bottom: var(--space-sm);">
        Choose whether to receive AI guidance during the scenario. This affects how your responses will be analyzed.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚫 Unassisted Mode", use_container_width=True, 
                    type="secondary" if st.session_state.ai_assistance_enabled else "primary"):
            st.session_state.ai_assistance_enabled = False
            st.rerun()
    
    with col2:
        if st.button("🤖 AI-Assisted Mode", use_container_width=True,
                    type="primary" if st.session_state.ai_assistance_enabled else "secondary"):
            st.session_state.ai_assistance_enabled = True
            st.rerun()
    
    # Display current selection
    assistance_status = "🤖 AI-Assisted" if st.session_state.ai_assistance_enabled else "🚫 Unassisted"
    st.markdown(f"""
    <div style="text-align: center; padding: var(--space-sm); background: var(--background-tertiary); border-radius: var(--radius-md); margin: var(--space-sm) 0;">
        <strong>Current Mode: {assistance_status}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if both expertise and assistance are selected
    if 'user_expertise' not in st.session_state:
        st.warning("Please select your experience level to continue.")
        return
    
    st.markdown("---")
    
    # Scenario Cards
    st.markdown('<h3 style="color: var(--text-primary); margin-bottom: var(--space-md);">Available Scenarios</h3>', unsafe_allow_html=True)
    
    # Group scenarios by domain
    domains = scenarios_df['domain'].unique()
    
    for domain in domains:
        domain_scenarios = scenarios_df[scenarios_df['domain'] == domain]
        
        st.markdown(f"""
        <div style="background: var(--background-secondary); padding: var(--space-md); border-radius: var(--radius-lg); margin: var(--space-md) 0;">
            <h4 style="color: var(--claude-orange); margin-top: 0;">{domain} Domain</h4>
        </div>
        """, unsafe_allow_html=True)
        
        for _, scenario in domain_scenarios.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="bias-card" style="margin: var(--space-sm) 0;">
                    <h5 style="color: var(--text-primary); margin-top: 0;">{scenario['title']}</h5>
                    <p style="color: var(--text-secondary); margin-bottom: var(--space-sm); line-height: 1.5;">
                        <strong>Bias Focus:</strong> {scenario['bias_type']}<br>
                        <strong>Cognitive Load:</strong> {scenario['cognitive_load_level']}<br>
                        <strong>Objective:</strong> {scenario['bias_learning_objective'][:100]}...
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col3:
                    if st.button(f"Select Scenario", key=f"select_{scenario['scenario_id']}", use_container_width=True):
                        st.session_state.current_scenario = scenario['scenario_id']
                        st.session_state.current_step = 'scenario'
                        st.session_state.scenario_start_time = time.time()
                        st.rerun()

def render_scenario_interface(scenario):
    """Render the selected scenario interface"""
    st.markdown(f'<h2 class="section-header">📋 {scenario["title"]}</h2>', unsafe_allow_html=True)
    
    # Scenario context
    st.markdown(f"""
    <div class="hero-container" style="text-align: left; margin: var(--space-lg) 0;">
        <h4 style="color: var(--claude-orange); margin-top: 0;">Scenario Context</h4>
        <div style="background: var(--background-primary); padding: var(--space-md); border-radius: var(--radius-md); border-left: 4px solid var(--claude-orange);">
            <p style="line-height: 1.6; color: var(--text-primary); margin: 0; font-size: 1.1rem;">
                {scenario["scenario_text"]}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Training information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-highlight">
            <h5 style="margin-top: 0; color: #856404;">🎯 Focus Area</h5>
            <p style="margin-bottom: 0;"><strong>{scenario["bias_type"]}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        assistance_mode = "🤖 AI-Assisted" if st.session_state.ai_assistance_enabled else "🚫 Unassisted"
        st.markdown(f"""
        <div class="feature-highlight">
            <h5 style="margin-top: 0; color: #856404;">🔧 Mode</h5>
            <p style="margin-bottom: 0;">{assistance_mode}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Primary prompt
    st.markdown(f"""
    <div style="background: var(--background-secondary); padding: var(--space-lg); border-radius: var(--radius-lg); margin: var(--space-lg) 0; border: 2px solid var(--claude-orange);">
        <h4 style="color: var(--text-primary); margin-top: 0;">💭 Your Task</h4>
        <p style="color: var(--text-primary); font-size: 1.1rem; line-height: 1.6; margin: 0; font-weight: 500;">
            {scenario["primary_prompt"]}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Assistance (if enabled)
    if st.session_state.ai_assistance_enabled:
        st.markdown("""
        <div style="background: #E8F4FD; padding: var(--space-md); border-radius: var(--radius-lg); margin: var(--space-md) 0; border-left: 4px solid var(--secondary-blue);">
            <h5 style="color: var(--secondary-blue); margin-top: 0;">🤖 AI Guidance Available</h5>
            <p style="color: var(--text-primary); margin: 0;">
                AI assistance is enabled. You can request guidance while formulating your response.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("💡 Get AI Guidance", key="ai_guidance"):
            st.markdown("""
            <div style="background: #FFF8E1; padding: var(--space-md); border-radius: var(--radius-md); margin: var(--space-sm) 0;">
                <p style="color: var(--text-primary); margin: 0; font-style: italic;">
                "Consider multiple perspectives and potential biases that might influence your initial judgment. 
                Look for evidence that both supports and contradicts your first impression."
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Response interface
    st.markdown("---")
    st.markdown('<h4 style="color: var(--text-primary);">✍️ Your Response</h4>', unsafe_allow_html=True)
    
    # Text area for response
    response = st.text_area(
        "Provide your detailed analysis and decision:",
        value=st.session_state.response_text,
        height=200,
        placeholder="Take your time to think through this scenario. Consider potential biases and explain your reasoning...",
        key="response_input",
        help="Explain your thought process, identify any potential biases, and justify your final decision."
    )
    
    # Update session state
    st.session_state.response_text = response
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back to Selection", use_container_width=True):
            st.session_state.current_step = 'selection'
            st.session_state.current_scenario = None
            st.session_state.response_text = ""
            st.rerun()
    
    with col3:
        if st.button("Submit Response →", use_container_width=True, 
                    disabled=len((response or "").strip()) < 50,
                    type="primary"):
            save_response(scenario, response)
            st.session_state.current_step = 'response'
            st.rerun()
    
    # Word count and guidance
    word_count = len(response.split()) if response else 0
    min_words = 50
    
    if word_count < min_words:
        st.markdown(f"""
        <div style="text-align: center; color: var(--warning); font-size: 0.9rem; margin-top: var(--space-sm);">
            📝 {word_count} words (minimum {min_words} words required)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align: center; color: var(--success); font-size: 0.9rem; margin-top: var(--space-sm);">
            ✅ {word_count} words - Ready to submit!
        </div>
        """, unsafe_allow_html=True)

def save_response(scenario, response):
    """Save user response with metadata for research analysis"""
    response_time = time.time() - st.session_state.scenario_start_time
    
    # Enhanced research metadata
    response_data = {
        'timestamp': datetime.now().isoformat(),
        'scenario_id': scenario['scenario_id'],
        'scenario_title': scenario['title'],
        'bias_type': scenario['bias_type'],
        'domain': scenario['domain'],
        
        # RESEARCH DESIGN VARIABLES (2x2x3 Factorial)
        'user_expertise': st.session_state.get('user_expertise', 'unknown'),  # novice/expert
        'ai_assistance_enabled': st.session_state.ai_assistance_enabled,      # assisted/unassisted
        'bias_category': scenario['bias_type'],                               # 3 bias types
        
        # RESPONSE DATA FOR 6-DIMENSIONAL SCORING
        'response_text': response,
        'response_time_seconds': response_time,
        'word_count': len(response.split()),
        'ai_guidance_requested': st.session_state.get('ai_guidance_used', False),
        
        # RESEARCH METADATA
        'session_id': id(st.session_state),
        'ideal_answer': scenario.get('ideal_primary_answer', ''),
        'rubric_focus': scenario.get('rubric_focus', ''),
        'source_type': 'live_user',  # vs 'automated_persona'
        
        # FOR STATISTICAL ANALYSIS
        'condition_code': f"{st.session_state.get('user_expertise', 'unknown')}_{st.session_state.ai_assistance_enabled}_{scenario['bias_type']}"
    }
    
    # Add to session state
    st.session_state.user_responses.append(response_data)
    
    # Save to file with research-ready format
    try:
        import os
        os.makedirs(config.RESPONSES_DIR, exist_ok=True)
        
        # Create research-compatible filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        condition = response_data['condition_code']
        response_file = f"{config.RESPONSES_DIR}/response_{condition}_{timestamp}.json"
        
        with open(response_file, 'w') as f:
            json.dump(response_data, f, indent=2)
            
        # Also append to master research log
        master_log = f"{config.RESPONSES_DIR}/master_responses.jsonl"
        with open(master_log, 'a') as f:
            f.write(json.dumps(response_data) + '\n')
            
    except Exception as e:
        st.warning(f"Could not save response to file: {e}")

def render_response_submitted():
    """Render confirmation after response submission"""
    st.markdown('<h2 class="section-header">✅ Response Submitted Successfully</h2>', unsafe_allow_html=True)
    
    last_response = st.session_state.user_responses[-1] if st.session_state.user_responses else None
    
    if last_response:
        st.markdown("""
        <div class="hero-container">
            <h3 style="color: var(--claude-orange); margin-top: 0;">📊 Response Summary</h3>
            <div style="text-align: left;">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Response Time", f"{last_response['response_time_seconds']:.1f}s")
        with col2:
            st.metric("Word Count", last_response['word_count'])
        with col3:
            mode = "AI-Assisted" if last_response['ai_assistance_enabled'] else "Unassisted"
            st.metric("Mode", mode)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Next steps
    st.markdown("""
    <div style="background: var(--background-secondary); padding: var(--space-lg); border-radius: var(--radius-lg); margin: var(--space-lg) 0;">
        <h4 style="color: var(--text-primary); margin-top: 0;">🎯 What's Next?</h4>
        <p style="color: var(--text-secondary); margin-bottom: var(--space-md);">
        Your response has been recorded and is ready for analysis. You can now:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Assessment", use_container_width=True, type="primary"):
            st.switch_page("pages/03_Assessment.py")
    
    with col2:
        if st.button("🎯 Try Another Scenario", use_container_width=True):
            st.session_state.current_step = 'selection'
            st.session_state.current_scenario = None
            st.session_state.response_text = ""
            st.rerun()
    
    with col3:
        if st.button("🏠 Return Home", use_container_width=True):
            st.switch_page("Home.py")

def main():
    """Main function for scenarios page"""
    # Setup page
    setup_page()
    
    # Load CSS
    load_css()
    
    # Initialize session state
    initialize_session_state()
    
    # Load scenarios
    scenarios_df = load_scenarios()
    if scenarios_df is None:
        st.stop()
    
    # Main page header
    st.markdown('<h1 class="main-header">🎯 Interactive Training Scenarios</h1>', unsafe_allow_html=True)
    
    # Render appropriate interface based on current step
    if st.session_state.current_step == 'selection':
        render_scenario_selection(scenarios_df)
    
    elif st.session_state.current_step == 'scenario':
        scenario = get_scenario_by_id(scenarios_df, st.session_state.current_scenario)
        if scenario is not None:
            render_scenario_interface(scenario)
        else:
            st.error("Scenario not found. Returning to selection.")
            st.session_state.current_step = 'selection'
            st.rerun()
    
    elif st.session_state.current_step == 'response':
        render_response_submitted()
    
    # Debug info (can be removed in production)
    if config.DEBUG:
        with st.expander("🔧 Debug Information"):
            st.write("Session State:", st.session_state)
            st.write("Available Scenarios:", scenarios_df['scenario_id'].tolist() if scenarios_df is not None else "None")
    
    # Render academic footer
    render_academic_footer()

if __name__ == "__main__":
    main()