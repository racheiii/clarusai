"""
ClārusAI: Enhanced Shared Navigation and Utility Functions
UCL Master's Dissertation - AI Literacy Through Cognitive Bias Training
utils.py - Reusable navigation and page configuration functions + 4-stage training utilities
"""

import streamlit as st
from pathlib import Path
from typing import List, Optional

def load_css() -> None:
    """Load custom CSS safely with error handling - NO FALLBACK CSS"""
    try:
        css_path = Path("assets/styles/main.css")
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        else:
            # Only essential Streamlit cleanup, no visual styles
            st.markdown("""
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stHeader"] {display: none !important;}
            .main .block-container {padding-top: 2rem; max-width: 1200px;}
            </style>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS: {e}")

def setup_page_config(page_title: str, page_icon: str = "🧠", enable_sidebar: bool = True) -> None:
    """Configure Streamlit page settings consistently"""
    st.set_page_config(
        page_title=f"ClārusAI - {page_title}",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded" if enable_sidebar else "collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "ClārusAI - UCL Master's Dissertation Research"
        }
    )

def get_current_page() -> str:
    """Determine current page from URL or session state"""
    try:
        # Get current page from Streamlit's query params or URL
        if hasattr(st, 'session_state') and 'current_page' in st.session_state:
            return st.session_state.current_page
        
        # Fallback to detecting from file path
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            filename = frame.f_back.f_globals.get('__file__', '')
            if 'Home.py' in filename:
                return 'home'
            elif '01_Scenarios.py' in filename:
                return 'scenarios'
            elif '02_Results.py' in filename:
                return 'results'
            elif '03_Dashboard.py' in filename:
                return 'dashboard'
            elif '04_Methodology.py' in filename:
                return 'methodology'
        
        return 'home'
    except:
        return 'home'

def render_navigation(current_page: Optional[str] = None) -> None:
    """Render consistent navigation sidebar"""
    if current_page is None:
        current_page = get_current_page()
    
    with st.sidebar:
        # Logo/Header
        st.markdown("""
        <div class="nav-header">
            <h2>🧠 ClārusAI</h2>
            <p>AI Literacy Training</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Research Tools Section
        st.markdown("### 🔬 Research Tools")
        
        dashboard_current = (current_page == 'dashboard')
        if dashboard_current:
            st.markdown("""
            <div class="nav-dashboard-active">
                📊 Research Dashboard
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("📊 Research Dashboard", key="nav_dashboard", use_container_width=True):
                st.session_state.current_page = 'dashboard'
                st.switch_page("pages/03_Dashboard.py")
        
        st.markdown("---")
        
        # Project Info
        st.markdown("""
        <div class="nav-project-info">
            <strong>UCL Research Project</strong><br>
            Building AI Literacy Through Simulation<br>
            <em>Master's Dissertation</em>
        </div>
        """, unsafe_allow_html=True)

def render_academic_footer() -> None:
    """Render consistent academic footer"""
    st.markdown("---")
    st.markdown("""
    <div class="academic-footer-content">
        <h4>UCL Master's Dissertation Research</h4>
        <p>Building AI Literacy Through Simulation: Evaluating LLM-Assisted Cognitive Bias Training</p>
        <p>Cognitive Bias Recognition • Professional Training • AI Assistance Research</p>
    </div>
    """, unsafe_allow_html=True)

def render_page_header(title: str, subtitle: Optional[str] = None, icon: str = "🧠") -> None:
    """Render consistent page headers"""
    st.markdown(f"""
    <div class="page-header">
        <h1>{icon} {title}</h1>
        {f'<p>{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def render_bias_cards() -> None:
    """Render the three main cognitive bias cards"""
    st.markdown('<h2 class="bias-section-header">🧠 Cognitive Biases We Target</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="bias-card-confirmation">
            <h4>⚡ Confirmation Bias</h4>
            <p>
            The tendency to search for, interpret, and recall information that confirms pre-existing beliefs 
            while giving disproportionately less consideration to alternative possibilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="bias-card-anchoring">
            <h4>⚓ Anchoring Bias</h4>
            <p>
            The tendency to rely too heavily on the first piece of information encountered 
            when making decisions, serving as an "anchor" for subsequent judgments.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="bias-card-availability">
            <h4>🧩 Availability Heuristic</h4>
            <p>
            Estimating the likelihood of events based on their availability in memory, 
            influenced by how recent or emotionally charged the examples are.
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_domain_cards() -> None:
    """Render the three professional domain cards"""
    st.markdown("""
    <h2 style="
        font-size: 1.8rem;
        color: var(--text-dark);
        border-bottom: 3px solid var(--primary-blue);
        padding-bottom: 0.5rem;
        margin: 2.5rem 0 1.5rem 0;
        font-weight: 600;
    ">
        🏥 High-Stakes Professional Domains
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: #f8fffe;
            border: 1px solid #d4edda;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h4 style="color: #2e7d32; margin: 0 0 1rem 0; font-size: 1.2rem;">🎖️ Military Intelligence</h4>
            <p style="margin: 0; line-height: 1.5; color: var(--text-dark);">
            Strategic threat assessment and tactical decision-making under time pressure
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: #f8fffe;
            border: 1px solid #d4edda;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h4 style="color: #2e7d32; margin: 0 0 1rem 0; font-size: 1.2rem;">🏥 Medical Emergency</h4>
            <p style="margin: 0; line-height: 1.5; color: var(--text-dark);">
            Critical patient diagnosis and treatment decisions in emergency settings
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: #f8fffe;
            border: 1px solid #d4edda;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            height: 100%;
        ">
            <h4 style="color: #2e7d32; margin: 0 0 1rem 0; font-size: 1.2rem;">🚨 Emergency Management</h4>
            <p style="margin: 0; line-height: 1.5; color: var(--text-dark);">
            Crisis response coordination and resource allocation during disasters
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_hero_section() -> None:
    """Render main hero section"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, var(--background-light) 0%, var(--background-white) 100%);
        border-radius: 16px;
        padding: 3rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 32px var(--shadow-light);
        border: 1px solid var(--border-light);
    ">
        <h2 style="
            font-size: 2rem;
            color: var(--primary-blue);
            margin-bottom: 1rem;
            font-weight: 600;
        ">
            AI Literacy Training Through Cognitive Bias Recognition
        </h2>
        <p style="
            font-size: 1.2rem;
            color: var(--text-dark);
            margin-bottom: 2rem;
            line-height: 1.6;
            opacity: 0.9;
        ">
            Professional decision-making training for high-stakes environments.<br>
            Develop critical thinking skills by recognising and overcoming cognitive biases.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_cta_section() -> None:
    """Render call-to-action section"""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, var(--primary-blue) 0%, #1565c0 100%);
        color: white;
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin: 3rem 0;
        box-shadow: 0 8px 32px rgba(31, 119, 180, 0.2);
    ">
        <h2 style="margin-top: 0; color: white;">Ready to Begin Your Training?</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem; opacity: 0.9;">
        Experience professional-grade cognitive bias training with AI-assisted learning
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_research_access_section() -> None:
    """Render the research team access section"""
    st.markdown("---")
    st.markdown("""
    <div class="home-research-access">
        <h3>🔬 Research Team Access</h3>
        <p>
        Access automated testing, statistical analysis, and methodology validation tools
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 4-STAGE TRAINING SPECIFIC UTILITIES - ENHANCED
# =============================================================================

def render_progress_indicator(current_stage: int, total_stages: int, stage_names: List[str]) -> None:
    """
    Render enhanced progress indicator for multi-stage interactions.
    
    Args:
        current_stage: Current stage number (0-based)
        total_stages: Total number of stages
        stage_names: List of stage names
    """
    progress_percentage = ((current_stage + 1) / total_stages) * 100
    
    st.markdown(f"""
    <div style="background: var(--background-light); padding: 1rem; border-radius: 12px; margin: 1.5rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-weight: 600; color: var(--text-dark);">Progress: Stage {current_stage + 1} of {total_stages}</span>
            <span style="color: var(--text-light); font-size: 0.9rem;">{progress_percentage:.0f}% Complete</span>
        </div>
        <div style="background: var(--border-light); border-radius: 4px; height: 8px; overflow: hidden;">
            <div style="background: var(--primary-blue); height: 100%; width: {progress_percentage}%; transition: width 0.3s ease;"></div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: var(--text-light);">
            Current: {stage_names[current_stage]}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visual stage progression
    cols = st.columns(total_stages)
    for i in range(total_stages):
        with cols[i]:
            if i < current_stage:
                # Completed stage
                st.markdown(f"✅ **{stage_names[i]}**")
            elif i == current_stage:
                # Current stage
                st.markdown(f"🔄 **{stage_names[i]}**")
            else:
                # Future stage
                st.markdown(f"⏳ {stage_names[i]}")

def render_stage_context(previous_responses: List[str], stage_names: List[str]) -> None:
    """
    Show previous responses as read-only context.
    
    Args:
        previous_responses: List of previous response texts
        stage_names: List of stage names
    """
    if not previous_responses:
        return
    
    with st.expander("📚 Previous Stage Responses", expanded=False):
        for i, (response, stage_name) in enumerate(zip(previous_responses, stage_names)):
            if response and response.strip():
                st.markdown(f"**{stage_name}:**")
                st.markdown(f"> {response[:200]}{'...' if len(response) > 200 else ''}")
                if i < len(previous_responses) - 1:  # Don't add separator after last item
                    st.markdown("---")

def render_training_navigation() -> None:
    """
    Enhanced navigation for training interface without sidebar.
    Provides clean, minimal navigation that doesn't interfere with training flow.
    """
    st.markdown("""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 1000;">
        <div style="background: rgba(255, 255, 255, 0.95); padding: 0.5rem; border-radius: 8px; border: 1px solid #e0e0e0; backdrop-filter: blur(10px);">
            <span style="color: #1976d2; font-weight: 600; font-size: 0.9rem;">🧠 ClārusAI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Return to home button in top right
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("🏠", help="Return to Home", key="nav_home"):
            st.switch_page("Home.py")

def setup_training_page_config(page_title: str, page_icon: str = "🎯") -> None:
    """
    Enhanced page configuration specifically for training interface (no sidebar).
    
    Args:
        page_title: Title for the browser tab
        page_icon: Icon for the browser tab
    """
    st.set_page_config(
        page_title=f"ClārusAI - {page_title}",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "ClārusAI - UCL Master's Dissertation Research"
        }
    )
    
    # Hide sidebar and streamlit elements for clean training interface
    st.markdown("""
    <style>
    .stSidebar {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .stApp > header {display: none;}
    [data-testid="stHeader"] {display: none !important;}
    
    /* Ensure main container has proper spacing without sidebar */
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# EXPERIMENTAL INTERFACE UTILITIES
# =============================================================================

def render_experimental_progress(current_stage: int, total_stages: int = 4) -> None:
    """
    Render experimental progress indicator specifically for research interface.
    
    Args:
        current_stage: Current stage (0-based)
        total_stages: Total number of stages (default 4)
    """
    progress = (current_stage + 1) / total_stages
    
    st.markdown(f"""
    <div class="experimental-progress">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-weight: 600;">Experimental Progress</span>
            <span style="color: #666;">{current_stage + 1}/{total_stages}</span>
        </div>
        <div style="background: #e0e0e0; height: 6px; border-radius: 3px; overflow: hidden;">
            <div style="background: #4caf50; height: 100%; width: {progress * 100}%; transition: width 0.3s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_condition_indicator(user_expertise, ai_assistance_enabled: bool) -> None:
    """
    Render experimental condition indicator for research tracking.
    
    Args:
        user_expertise: UserExpertise enum value
        ai_assistance_enabled: Boolean indicating AI assistance status
    """
    expertise_label = "Expert" if user_expertise and user_expertise.value == "expert" else "Novice"
    assistance_label = "AI-Assisted" if ai_assistance_enabled else "Unassisted"
    
    st.markdown(f"""
    <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 0.5rem; margin: 0.5rem 0; text-align: center;">
        <small style="color: #6c757d;">Experimental Condition: <strong>{expertise_label} • {assistance_label}</strong></small>
    </div>
    """, unsafe_allow_html=True)

def render_research_metadata(scenario_id: str, bias_type: str, domain: str) -> None:
    """
    Render research metadata for debugging and validation.
    
    Args:
        scenario_id: Unique scenario identifier
        bias_type: Type of cognitive bias being tested
        domain: Professional domain of the scenario
    """
    if st.secrets.get("DEBUG", False):  # Only show in debug mode
        with st.expander("🔍 Research Metadata", expanded=False):
            st.code(f"""
Scenario ID: {scenario_id}
Bias Type: {bias_type}
Domain: {domain}
            """)

def render_session_quality_indicator(word_count: int, response_time: float, guidance_used: bool) -> None:
    """
    Render session quality indicators for research validation.
    
    Args:
        word_count: Number of words in current response
        response_time: Time spent on current response (seconds)
        guidance_used: Whether AI guidance was used
    """
    quality_indicators = []
    
    if word_count >= 20:
        quality_indicators.append("✅ Sufficient detail")
    elif word_count >= 10:
        quality_indicators.append("⚠️ Moderate detail")
    else:
        quality_indicators.append("❌ Brief response")
    
    if response_time >= 30:
        quality_indicators.append("✅ Thoughtful timing")
    elif response_time >= 15:
        quality_indicators.append("⚠️ Moderate timing")
    else:
        quality_indicators.append("❌ Quick response")
    
    if guidance_used:
        quality_indicators.append("🤖 AI guidance used")
    else:
        quality_indicators.append("🧠 Independent analysis")
    
    st.markdown(f"""
    <div style="background: #f8f9fa; border-radius: 4px; padding: 0.5rem; margin: 0.5rem 0; font-size: 0.85rem;">
        {" • ".join(quality_indicators)}
    </div>
    """, unsafe_allow_html=True)