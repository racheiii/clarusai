"""
ClārusAI: Enhanced Shared Navigation and Utility Functions
UCL Master's Dissertation - AI Literacy Through Cognitive Bias Training
utils.py - Reusable navigation and page configuration functions + 4-stage training utilities
"""

import streamlit as st
from pathlib import Path

def load_css():
    """Load custom CSS safely with error handling"""
    try:
        css_path = Path("assets/styles/main.css")
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        else:
            # Fallback minimal CSS if file not found
            st.markdown("""
            <style>
            :root {
                --primary-blue: #1f77b4;
                --secondary-blue: #aec7e8;
                --accent-orange: #ff7f0e;
                --success-green: #2ca02c;
                --warning-red: #d62728;
                --warning-yellow: #ffc107;
                --text-dark: #2c3e50;
                --text-light: #7f8c8d;
                --background-white: #ffffff;
                --background-light: #f8f9fa;
                --border-light: #e9ecef;
                --shadow-light: rgba(0, 0, 0, 0.08);
                --shadow-medium: rgba(0, 0, 0, 0.12);
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stHeader"] {display: none !important;}
            .main .block-container {
                padding-top: 2rem;
                max-width: 1200px;
            }
            </style>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS: {e}")

def setup_page_config(page_title, page_icon="🧠", enable_sidebar=True):
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

def get_current_page():
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
            elif '02_Assessment.py' in filename:
                return 'assessment'
            elif '03_Results.py' in filename:
                return 'results'
            elif '04_Dashboard.py' in filename:
                return 'dashboard'
            elif '05_Methodology.py' in filename:
                return 'methodology'
        
        return 'home'
    except:
        return 'home'

def render_navigation(current_page=None):
    """Render consistent navigation sidebar"""
    if current_page is None:
        current_page = get_current_page()
    
    with st.sidebar:
        # Logo/Header
        st.markdown("""
        <div style="text-align: center; padding: 1rem; margin-bottom: 1rem;">
            <h2 style="color: var(--primary-blue); margin: 0; font-size: 1.5rem;">🧠 ClārusAI</h2>
            <p style="color: var(--text-light); margin: 0.5rem 0 0 0; font-size: 0.9rem;">AI Literacy Training</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Research Tools Section
        st.markdown("### 🔬 Research Tools")
        
        dashboard_current = (current_page == 'dashboard')
        if dashboard_current:
            st.markdown("""
            <div style="
                background: var(--primary-blue);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                margin: 0.25rem 0;
                font-weight: 600;
                text-align: center;
            ">
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
        <div style="
            background: var(--background-light);
            border-left: 3px solid var(--primary-blue);
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 6px 6px 0;
            font-size: 0.85rem;
            color: var(--text-dark);
        ">
            <strong>UCL Research Project</strong><br>
            Building AI Literacy Through Simulation<br>
            <em>Master's Dissertation</em>
        </div>
        """, unsafe_allow_html=True)

def render_academic_footer():
    """Render consistent academic footer"""
    st.markdown("---")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, var(--text-dark) 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        text-align: center;
        border-radius: 12px;
        margin-top: 3rem;
        box-shadow: 0 8px 24px var(--shadow-medium);
    ">
        <h4 style="margin-top: 0; color: white; font-weight: 600;">
            UCL Master's Dissertation Research
        </h4>
        <p style="margin-bottom: 0; opacity: 0.9; font-size: 1rem; line-height: 1.5;">
            Building AI Literacy Through Simulation: Evaluating LLM-Assisted Cognitive Bias Training
        </p>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.7; font-size: 0.9rem;">
            Cognitive Bias Recognition • Professional Training • AI Assistance Research
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_page_header(title, subtitle=None, icon="🧠"):
    """Render consistent page headers"""
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="
            color: var(--primary-blue);
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
            text-shadow: 0 2px 4px rgba(31, 119, 180, 0.1);
        ">
            {icon} {title}
        </h1>
        {f'<p style="color: var(--text-light); font-size: 1.2rem; margin-bottom: 2rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def render_bias_cards():
    """Render the three main cognitive bias cards"""
    st.markdown("""
    <h2 style="
        font-size: 1.8rem;
        color: var(--text-dark);
        border-bottom: 3px solid var(--primary-blue);
        padding-bottom: 0.5rem;
        margin: 2.5rem 0 1.5rem 0;
        font-weight: 600;
    ">
        🧠 Cognitive Biases We Target
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 16px var(--shadow-light);
            border-left: 4px solid var(--accent-orange);
            height: 100%;
        ">
            <h4 style="color: var(--warning-red); margin-top: 0; font-size: 1.3rem;">⚡ Confirmation Bias</h4>
            <p style="line-height: 1.6; color: var(--text-dark); margin-bottom: 0;">
            The tendency to search for, interpret, and recall information that confirms pre-existing beliefs 
            while giving disproportionately less consideration to alternative possibilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 16px var(--shadow-light);
            border-left: 4px solid var(--accent-orange);
            height: 100%;
        ">
            <h4 style="color: var(--warning-red); margin-top: 0; font-size: 1.3rem;">⚓ Anchoring Bias</h4>
            <p style="line-height: 1.6; color: var(--text-dark); margin-bottom: 0;">
            The tendency to rely too heavily on the first piece of information encountered 
            when making decisions, serving as an "anchor" for subsequent judgments.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: white;
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 16px var(--shadow-light);
            border-left: 4px solid var(--accent-orange);
            height: 100%;
        ">
            <h4 style="color: var(--warning-red); margin-top: 0; font-size: 1.3rem;">🧩 Availability Heuristic</h4>
            <p style="line-height: 1.6; color: var(--text-dark); margin-bottom: 0;">
            Estimating the likelihood of events based on their availability in memory, 
            influenced by how recent or emotionally charged the examples are.
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_domain_cards():
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

def render_hero_section():
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

def render_cta_section():
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

# =============================================================================
# 4-STAGE TRAINING SPECIFIC UTILITIES
# =============================================================================

def render_progress_indicator(current_stage, total_stages, stage_names):
    """Render progress indicator for multi-stage interactions"""
    progress_percentage = (current_stage / total_stages) * 100
    
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

def render_stage_context(previous_responses, stage_names):
    """Show previous responses as read-only context"""
    if not previous_responses:
        return
    
    st.markdown("### 📝 Your Previous Responses")
    
    for idx, (stage_name, response) in enumerate(zip(stage_names[:len(previous_responses)], previous_responses)):
        st.markdown(f"""
        <div style="background: var(--border-light); padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid var(--secondary-blue);">
            <h5 style="color: var(--text-dark); margin-top: 0; font-size: 1rem;">{stage_name}</h5>
            <p style="color: var(--text-light); margin: 0; font-style: italic; line-height: 1.4;">
                {response[:200]}{"..." if len(response) > 200 else ""}
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_training_navigation():
    """Minimal navigation for training interface without sidebar"""
    st.markdown("""
    <div style="position: fixed; top: 10px; left: 10px; z-index: 1000;">
        <div style="background: var(--background-light); padding: 0.5rem; border-radius: 8px; border: 1px solid var(--border-light);">
            <span style="color: var(--primary-blue); font-weight: 600; font-size: 0.9rem;">🧠 ClārusAI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Return to home button in top right
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("🏠", help="Return to Home", key="nav_home"):
            st.switch_page("Home.py")

def setup_training_page_config(page_title, page_icon="🎯"):
    """Page configuration specifically for training interface (no sidebar)"""
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
    </style>
    """, unsafe_allow_html=True)