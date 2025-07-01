"""
Landing Page - Project Introduction & Pathway Selection
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

import config

def load_css():
    """Load custom CSS with error handling"""
    try:
        with open("assets/styles/main.css") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback inline styles for development
        st.markdown("""
        <style>
        .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
        .section-header { font-size: 1.8rem; color: #2c3e50; border-bottom: 2px solid #1f77b4; padding-bottom: 0.5rem; margin-bottom: 1.5rem; }
        .feature-card { background: #ffffff; border: 1px solid #e9ecef; border-radius: 12px; padding: 2rem; margin: 1rem 0; box-shadow: 0 4px 12px rgba(0,0,0,0.08); transition: transform 0.3s ease; }
        .feature-card:hover { transform: translateY(-4px); box-shadow: 0 8px 24px rgba(0,0,0,0.12); }
        .pathway-button { background: linear-gradient(135deg, #1f77b4 0%, #1565c0 100%); color: white; padding: 1rem 2rem; border-radius: 8px; border: none; font-weight: 600; font-size: 1.1rem; cursor: pointer; transition: all 0.3s ease; width: 100%; }
        .pathway-button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(31, 119, 180, 0.3); }
        .overview-box { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-left: 5px solid #1f77b4; padding: 2rem; margin: 2rem 0; border-radius: 0 12px 12px 0; }
        .bias-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 1.5rem 0; }
        .bias-item { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #ff7f0e; }
        .domain-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
        .domain-item { background: #fff3cd; padding: 1rem; border-radius: 6px; text-align: center; border: 1px solid #ffeaa7; }
        </style>
        """, unsafe_allow_html=True)

def main():
    # Load styling
    load_css()
    
    # Main header with proper spacing
    st.markdown('<div class="main-header">🏠 Welcome to ClārusAI</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Project overview section
    st.markdown("""
    <div class="overview-box">
        <h2 style="color: #1f77b4; margin-top: 0;">🎯 AI Literacy Training Through Cognitive Bias Recognition</h2>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #2c3e50; margin-bottom: 0;">
        ClārusAI helps professionals develop critical thinking skills by recognising and overcoming 
        cognitive biases in high-stakes decision-making environments through interactive scenario-based training.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Core focus areas
    st.markdown('<h2 class="section-header">🧠 Cognitive Biases We Target</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="bias-grid">
        <div class="bias-item">
            <h4 style="color: #d62728; margin-top: 0;">⚡ Confirmation Bias</h4>
            <p>Tendency to search for information that confirms pre-existing beliefs while ignoring contradictory evidence</p>
        </div>
        <div class="bias-item">
            <h4 style="color: #d62728; margin-top: 0;">⚓ Anchoring Bias</h4>
            <p>Over-reliance on the first piece of information encountered when making decisions</p>
        </div>
        <div class="bias-item">
            <h4 style="color: #d62728; margin-top: 0;">🧩 Availability Heuristic</h4>
            <p>Judging probability by how easily examples or instances come to mind</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # High-stakes domains
    st.markdown('<h2 class="section-header">🏥 Professional Domains</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="domain-grid">
        <div class="domain-item">
            <h4 style="color: #2e7d32; margin: 0;">🎖️ Military Intelligence</h4>
            <p style="margin: 0.5rem 0 0 0;">Strategic decision-making under pressure</p>
        </div>
        <div class="domain-item">
            <h4 style="color: #2e7d32; margin: 0;">🏥 Medical Emergency</h4>
            <p style="margin: 0.5rem 0 0 0;">Critical patient care scenarios</p>
        </div>
        <div class="domain-item">
            <h4 style="color: #2e7d32; margin: 0;">🚨 Emergency Management</h4>
            <p style="margin: 0.5rem 0 0 0;">Crisis response coordination</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Pathway selection
    st.markdown("---")
    st.markdown('<h2 class="section-header">🚀 Choose Your Experience</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #1f77b4; margin-top: 0;">👤 Interactive Training</h3>
            <p style="line-height: 1.6; color: #2c3e50;">
            Experience hands-on cognitive bias training through realistic professional scenarios 
            with optional AI assistance and immediate feedback.
            </p>
            <br>
            <ul style="color: #2c3e50; line-height: 1.6;">
                <li><strong>6 Professional Scenarios</strong> - Real-world decision contexts</li>
                <li><strong>AI Assistance Toggle</strong> - Compare assisted vs unassisted performance</li>
                <li><strong>Real-time Scoring</strong> - Immediate bias recognition feedback</li>
                <li><strong>Progress Analytics</strong> - Track your improvement over time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Start Training Experience", type="primary", use_container_width=True):
            st.switch_page("pages/02_Scenarios.py")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #1f77b4; margin-top: 0;">🔬 Research Dashboard</h3>
            <p style="line-height: 1.6; color: #2c3e50;">
            Access comprehensive research tools for automated testing, statistical analysis, 
            and methodology validation with live hypothesis testing.
            </p>
            <br>
            <ul style="color: #2c3e50; line-height: 1.6;">
                <li><strong>Automated Testing</strong> - Generate 72 research responses</li>
                <li><strong>Statistical Analysis</strong> - Live ANOVA and effect size calculations</li>
                <li><strong>Algorithm Transparency</strong> - Inspect scoring methodologies</li>
                <li><strong>Data Export</strong> - Download results for external analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Access Research Dashboard", type="secondary", use_container_width=True):
            st.switch_page("pages/05_Dashboard.py")
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
        <p style="margin: 0;"><strong>UCL Master's Dissertation Research</strong></p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Building AI Literacy Through Simulation: Evaluating LLM-Assisted Cognitive Bias Training
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()