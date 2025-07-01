"""
ClārusAI: Enhanced Main Landing Page
UCL Master's Dissertation - AI Literacy Through Cognitive Bias Training
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

import config

def load_css():
    """Load custom CSS"""
    with open("assets/styles/main.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def render_navigation():
    """Render collapsible sidebar navigation"""
    with st.sidebar:
        st.markdown('<div class="nav-logo">🧠 ClārusAI</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Main navigation
        st.markdown("### 📋 Navigation")
        if st.button("🏠 Home", use_container_width=True):
            st.rerun()
        if st.button("🎯 Training Scenarios", use_container_width=True):
            st.switch_page("pages/01_Scenarios.py")
        if st.button("📊 Assessment", use_container_width=True):
            st.switch_page("pages/02_Assessment.py")
        if st.button("📈 Results", use_container_width=True):
            st.switch_page("pages/03_Results.py")
        if st.button("🔍 Methodology", use_container_width=True):
            st.switch_page("pages/05_Methodology.py")
        
        st.markdown("---")
        
        # Research access
        st.markdown("### 🔬 Research Tools")
        if st.button("📊 Research Dashboard", use_container_width=True):
            st.switch_page("pages/04_Dashboard.py")
        
        st.markdown("---")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #7f8c8d; text-align: center;">
        <strong>UCL Research Project</strong><br>
        AI Literacy Training System
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
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
    
    # Load styling
    load_css()
    
    # Render navigation
    render_navigation()
    
    # Main header
    st.markdown('<h1 class="main-header">🧠 ClārusAI Research Platform</h1>', unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
    <div class="hero-container">
        <h2 class="hero-title">AI Literacy Training Through Cognitive Bias Recognition</h2>
        <p class="hero-subtitle">
        Professional decision-making training for high-stakes environments.<br>
        Develop critical thinking skills by recognising and overcoming cognitive biases.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Core focus areas
    st.markdown('<h2 class="section-header">🧠 Cognitive Biases We Target</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="bias-grid">
        <div class="bias-card">
            <h4 style="color: #d62728; margin-top: 0; font-size: 1.3rem;">⚡ Confirmation Bias</h4>
            <p style="line-height: 1.6; color: #2c3e50;">
            The tendency to search for, interpret, and recall information that confirms pre-existing beliefs 
            while giving disproportionately less consideration to alternative possibilities.
            </p>
        </div>
        <div class="bias-card">
            <h4 style="color: #d62728; margin-top: 0; font-size: 1.3rem;">⚓ Anchoring Bias</h4>
            <p style="line-height: 1.6; color: #2c3e50;">
            The tendency to rely too heavily on the first piece of information encountered 
            when making decisions, serving as an "anchor" for subsequent judgments.
            </p>
        </div>
        <div class="bias-card">
            <h4 style="color: #d62728; margin-top: 0; font-size: 1.3rem;">🧩 Availability Heuristic</h4>
            <p style="line-height: 1.6; color: #2c3e50;">
            Estimating the likelihood of events based on their availability in memory, 
            influenced by how recent or emotionally charged the examples are.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Professional domains
    st.markdown('<h2 class="section-header">🏥 High-Stakes Professional Domains</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="domain-grid">
        <div class="domain-card">
            <h4 style="color: #2e7d32; margin: 0 0 1rem 0; font-size: 1.2rem;">🎖️ Military Intelligence</h4>
            <p style="margin: 0; line-height: 1.5; color: #2c3e50;">
            Strategic threat assessment and tactical decision-making under time pressure
            </p>
        </div>
        <div class="domain-card">
            <h4 style="color: #2e7d32; margin: 0 0 1rem 0; font-size: 1.2rem;">🏥 Medical Emergency</h4>
            <p style="margin: 0; line-height: 1.5; color: #2c3e50;">
            Critical patient diagnosis and treatment decisions in emergency settings
            </p>
        </div>
        <div class="domain-card">
            <h4 style="color: #2e7d32; margin: 0 0 1rem 0; font-size: 1.2rem;">🚨 Emergency Management</h4>
            <p style="margin: 0; line-height: 1.5; color: #2c3e50;">
            Crisis response coordination and resource allocation during disasters
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features highlight
    st.markdown('<h2 class="section-header">✨ Training Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-highlight">
            <h4 style="margin-top: 0; color: #856404;">🎯 Interactive Scenarios</h4>
            <p style="margin-bottom: 0; line-height: 1.5;">
            Six realistic professional scenarios based on actual high-stakes decision contexts
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-highlight">
            <h4 style="margin-top: 0; color: #856404;">🤖 AI Assistance Toggle</h4>
            <p style="margin-bottom: 0; line-height: 1.5;">
            Optional AI guidance to compare assisted vs unassisted decision-making
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-highlight">
            <h4 style="margin-top: 0; color: #856404;">📊 Real-time Assessment</h4>
            <p style="margin-bottom: 0; line-height: 1.5;">
            Six-dimensional scoring system providing immediate bias recognition feedback
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-highlight">
            <h4 style="margin-top: 0; color: #856404;">📈 Progress Analytics</h4>
            <p style="margin-bottom: 0; line-height: 1.5;">
            Track improvement in bias recognition and mitigation strategies over time
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action section
    st.markdown("""
    <div class="cta-section">
        <h2 style="margin-top: 0; color: white;">Ready to Begin Your Training?</h2>
        <p style="font-size: 1.1rem; margin-bottom: 2rem; opacity: 0.9;">
        Experience professional-grade cognitive bias training with AI-assisted learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Primary CTA button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Training", type="primary", use_container_width=True, help="Begin interactive bias recognition training"):
            st.switch_page("pages/01_Scenarios.py")
    
    # Research access section
    st.markdown("---")
    st.markdown("""
    <div class="research-access">
        <h3 style="color: #1f77b4; margin-top: 0;">🔬 Research Team Access</h3>
        <p style="color: #2c3e50; margin-bottom: 1.5rem;">
        Access automated testing, statistical analysis, and methodology validation tools
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📊 Access Research Dashboard", use_container_width=True, help="Research automation and analysis tools"):
            st.switch_page("pages/04_Dashboard.py")
    
    # Academic footer
    st.markdown("""
    <div class="academic-footer">
        <h4 style="margin-top: 0; color: white;">UCL Master's Dissertation Research</h4>
        <p style="margin-bottom: 0; opacity: 0.9; font-size: 1rem;">
        Building AI Literacy Through Simulation: Evaluating LLM-Assisted Cognitive Bias Training
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()