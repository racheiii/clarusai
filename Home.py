"""
ClārusAI: Main Streamlit Application Entry Point
UCL Master's Dissertation - AI Literacy Through Cognitive Bias Training
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Import configuration
import config

def main():
    """Main application entry point with routing logic"""
    
    # Page configuration
    st.set_page_config(
        page_title="ClārusAI - AI Literacy Training",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "ClārusAI - UCL Master's Dissertation Research"
        }
    )
    
    # Custom CSS injection (will be implemented later)
    inject_custom_css()
    
    # Sidebar navigation info
    with st.sidebar:
        st.markdown("# 🧠 ClārusAI")
        st.markdown("*AI Literacy Training System*")
        st.markdown("---")
        
        # Navigation guidance
        st.markdown("### 📋 Navigation Guide")
        st.markdown("""
        **Training Path:**
        - 🏠 Landing → Overview
        - 🎯 Training → Interactive scenarios  
        - 📊 Assessment → Live scoring
        - 📈 Results → Your analytics
        
        **Research Path:**
        - 🔬 Dashboard → Research controls
        - 🔍 Methodology → Algorithm transparency
        """)
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Scenarios", "6", help="Training scenarios available")
            st.metric("Biases", "3", help="Cognitive biases targeted")
        with col2:
            st.metric("Domains", "3", help="Professional contexts")
            st.metric("Dimensions", "6", help="Assessment criteria")
        
        # System status
        st.markdown("---")
        st.markdown("### ⚙️ System Status")
        
        if config.API_ENABLED:
            st.success("🟢 API Connected")
        else:
            st.info("🟡 Development Mode")
            
        if config.ENABLE_RESEARCH_MODE:
            st.success("🟢 Research Enabled")
        else:
            st.warning("🟡 Research Disabled")
        
        if config.DEBUG:
            st.warning("🔧 Debug Mode Active")
    
    # Main page content - Clean and focused
    st.markdown('<h1 class="main-header">🧠 ClārusAI Research Platform</h1>', unsafe_allow_html=True)
    
    # Quick welcome message
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 12px; margin: 2rem 0;">
        <h3 style="color: #1f77b4; margin-top: 0;">AI Literacy Training Through Cognitive Bias Recognition</h3>
        <p style="font-size: 1.1rem; color: #2c3e50; margin-bottom: 1rem;">
        Professional decision-making training for high-stakes environments
        </p>
        <p style="color: #7f8c8d; margin-bottom: 0;">
        Navigate using the sidebar menu or visit the <strong>🏠 Landing</strong> page for detailed information
        </p>
    </div>
    """, unsafe_allow_html=True)

def inject_custom_css():
    """Inject custom CSS styling"""
    try:
        with open("assets/styles/main.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback basic styling if CSS file not found
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()