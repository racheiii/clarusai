"""
ClārusAI: Enhanced Main Landing Page
UCL Master's Dissertation - AI Literacy Through Cognitive Bias Training
"""

import streamlit as st
import sys
from pathlib import Path

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
            help="Begin interactive bias recognition training"
        ):
            st.session_state.current_page = "scenarios"
            st.switch_page("pages/01_Scenarios.py")
    
    # Research access section
    st.markdown("---")
    st.markdown("""
    <div style="
        background: #f8f9fa;
        border: 2px dashed #1f77b4;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    ">
        <h3 style="color: #1f77b4; margin-top: 0;">🔬 Research Team Access</h3>
        <p style="color: #2c3e50; margin-bottom: 1.5rem;">
        Access automated testing, statistical analysis, and methodology validation tools
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "📊 Access Research Dashboard", 
            use_container_width=True, 
            help="Research automation and analysis tools"
        ):
            st.session_state.current_page = "dashboard"
            st.switch_page("pages/04_Dashboard.py")
    
    # Academic footer
    render_academic_footer()

if __name__ == "__main__":
    main()