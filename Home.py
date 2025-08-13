"""
Main Landing Page
"""

import streamlit as st
import sys
from pathlib import Path
from src.session_manager import SessionManager

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Import utilities and configuration
from utils import (
    setup_page_config, 
    load_css, 
    render_navigation, 
    render_page_header,
    render_hero_section,
    render_bias_cards,
    render_domain_cards,
    render_cta_section,
    render_research_access_section,
    render_academic_footer
)
import config

def main():
    """Main application entry point"""
    
    # Page configuration
    setup_page_config("AI Literacy Training", "🧠", enable_sidebar=True)
    
    # Load styling
    load_css()
    
    # Render navigation (passing current page)
    render_navigation("home")
    
    # Main page header
    render_page_header(
        "ClārusAI Research Platform",
        "Building AI Literacy Through Cognitive Bias Recognition",
        "🧠"
    )
    
    # Hero section
    render_hero_section()
    
    # Core cognitive biases section
    render_bias_cards()
    
    # Professional domains section
    render_domain_cards()
    
    # Call to action section
    render_cta_section()
    
    # Primary CTA button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🚀 Start Training", 
            type="primary", 
            use_container_width=True, 
            help="Begin interactive bias recognition training (demo mode — not used for dataset generation)"
        ):
            SessionManager().reset_experimental_session()
            st.session_state.current_page = "scenarios"
            try:
                st.switch_page("pages/01_Scenarios.py")
            except AttributeError:
                st.session_state['interaction_flow'] = 'setup'
                st.rerun()

    # Research access section
    render_research_access_section()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "📊 Access Research Dashboard", 
            use_container_width=True, 
            help="Research automation and analysis tools"
        ):
            st.session_state.current_page = "dashboard"
            st.switch_page("pages/02_Dashboard.py")
    
    # Academic footer
    render_academic_footer()

if __name__ == "__main__":
    main()