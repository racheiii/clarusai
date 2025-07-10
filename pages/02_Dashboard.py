"""
ClārusAI: Research Dashboard
UCL MSc Dissertation – Building AI Literacy Through Simulation

02_Dashboard.py – Research analysis interface for simulated data validation,
and statistical hypothesis testing.

Purpose:
- Generate and manage simulated experimental datasets
- Validate 6-dimensional scoring framework reliability  
- Test core research hypotheses with statistical rigor
- Export research-ready visualisations and datasets
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback

# Add project paths
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# Import project modules
import config
from utils import load_css, render_academic_footer

# Visualization imports with error handling
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Plotting libraries not available: {e}")
    st.error("Please install: pip install matplotlib seaborn plotly")
    PLOTTING_AVAILABLE = False
    # Create dummy objects to prevent unbound variable errors
    plt = None
    sns = None
    px = None
    go = None
    make_subplots = None

# Import simulation generator
try:
    from sim_user_generator import SimulatedUserGenerator
    SIMULATION_AVAILABLE = True
except ImportError:
    try:
        from src.sim_user_generator import SimulatedUserGenerator
        SIMULATION_AVAILABLE = True
    except ImportError:
        SIMULATION_AVAILABLE = False
        SimulatedUserGenerator = None

# ================================
# CONFIGURATION
# ================================

# Use project configuration
EXPORTS_DIR = Path(config.EXPORTS_DIR) / "simulated_datasets"
CHART_EXPORT_DIR = Path(config.EXPORTS_DIR) / "research_outputs"
DATA_DIR = Path(config.DATA_DIR)

# Ensure directories exist
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHART_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Research parameters
STATISTICAL_SIGNIFICANCE_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}

# ================================
# UTILITY FUNCTIONS
# ================================

def safe_numeric_operation(value1: Any, value2: Any, operation: str) -> float:
    """Safely perform numeric operations with type conversion."""
    try:
        # Convert to numeric values
        num1 = float(pd.to_numeric(value1, errors='coerce'))
        num2 = float(pd.to_numeric(value2, errors='coerce'))
        
        # Check for NaN values
        if pd.isna(num1) or pd.isna(num2):
            return 0.0
        
        # Perform operation
        if operation == "add":
            return num1 + num2
        elif operation == "subtract":
            return num1 - num2
        elif operation == "divide":
            return num1 / num2 if num2 != 0 else 0.0
        elif operation == "multiply":
            return num1 * num2
        else:
            return 0.0
            
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0

def safe_variance_calculation(series: pd.Series) -> float:
    """Safely calculate variance with proper type handling."""
    try:
        # Convert to numeric and drop NaN values
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) == 0:
            return 0.0
        
        # Calculate variance using numpy for type safety
        return float(np.var(numeric_series, ddof=1))
        
    except (ValueError, TypeError):
        return 0.0

def safe_mean_calculation(series: pd.Series) -> float:
    """Safely calculate mean with proper type handling."""
    try:
        # Convert to numeric and drop NaN values
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) == 0:
            return 0.0
        
        return float(numeric_series.mean())
        
    except (ValueError, TypeError):
        return 0.0

# ================================
# DATA LOADING FUNCTIONS
# ================================

@st.cache_data(show_spinner=False)
def load_simulation_dataset(simulation_folder: str) -> Optional[pd.DataFrame]:
    """Load simulated dataset from specified folder."""
    try:
        folder_path = EXPORTS_DIR / simulation_folder
        if not folder_path.exists():
            st.error(f"Simulation folder not found: {folder_path}")
            return None
        
        records = []
        session_files = list(folder_path.glob("*.json"))
        
        if not session_files:
            st.warning(f"No session files found in {folder_path}")
            return None
        
        # Load each session file
        for file_path in session_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Extract session metadata
                session_meta = session_data.get("session_metadata", {})
                experimental_meta = session_data.get("experimental_metadata", {})
                
                # Process stage responses
                stage_responses = session_data.get("stage_responses", [])
                
                for response in stage_responses:
                    # Combine all metadata into flat record
                    record = {
                        # Session identifiers
                        "session_id": session_meta.get("session_id", "unknown"),
                        "simulation_version": session_meta.get("simulation_version", "unknown"),
                        "is_simulated": session_meta.get("is_simulated", True),
                        
                        # Experimental factors (2×2×3 design)
                        "user_expertise": session_meta.get("user_expertise", "unknown"),
                        "ai_assistance_enabled": session_meta.get("ai_assistance_enabled", False),
                        "bias_type": session_meta.get("bias_type", "unknown"),
                        "domain": session_meta.get("domain", "unknown"),
                        "scenario_id": session_meta.get("scenario_id", "unknown"),
                        "condition_code": experimental_meta.get("condition_code", "unknown"),
                        
                        # Stage-level data
                        "stage_number": response.get("stage_number", 0),
                        "stage_name": response.get("stage_name", "unknown"),
                        "response_text": response.get("response_text", ""),
                        "word_count": response.get("word_count", 0),
                        "character_count": response.get("character_count", 0),
                        "response_time_seconds": response.get("response_time_seconds", 0),
                        "guidance_requested": response.get("guidance_requested", False),
                        
                        # Session analytics
                        "total_session_time_minutes": session_meta.get("total_session_time_minutes", 0),
                        "total_guidance_requests": session_meta.get("total_guidance_requests", 0),
                        "total_word_count": session_meta.get("total_word_count", 0),
                        "session_quality": session_meta.get("session_quality", "unknown"),
                        
                        # Scoring results (6-dimensional)
                        **extract_scoring_results(response.get("scores", {})),
                        
                        # Quality flags
                        **extract_quality_flags(response.get("scores", {})),
                        
                        # Timestamps
                        "generated_timestamp": session_meta.get("generated_timestamp", ""),
                        "stage_timestamp": response.get("timestamp", "")
                    }
                    
                    records.append(record)
                    
            except Exception as e:
                st.warning(f"Failed to load session file {file_path.name}: {e}")
                continue
        
        if not records:
            st.error("No valid session data found")
            return None
        
        df = pd.DataFrame(records)
        st.success(f"✅ Loaded {len(df)} responses from {df['session_id'].nunique()} sessions")
        return df
        
    except Exception as e:
        st.error(f"Failed to load simulation dataset: {e}")
        return None

def extract_scoring_results(scores: Dict) -> Dict:
    """Extract 6-dimensional scores from scoring results."""
    default_scores = {
        "semantic_similarity": 0.0,
        "semantic_tag": "unknown",
        "bias_recognition_count": 0,
        "bias_recognition_tag": "unknown", 
        "originality_score": 0.0,
        "originality_tag": "unknown",
        "strategy_count": 0,
        "strategy_tag": "unknown",
        "transfer_count": 0,
        "transfer_tag": "unknown",
        "metacognition_count": 0,
        "metacognition_tag": "unknown",
        "overall_quality_score": 0.0
    }
    
    if not scores or "error" in scores:
        return default_scores
    
    # Extract semantic similarity
    semantic_data = scores.get("semantic_similarity", {})
    if isinstance(semantic_data, dict):
        default_scores["semantic_similarity"] = semantic_data.get("score", 0.0)
        default_scores["semantic_tag"] = semantic_data.get("tag", "unknown")
    
    # Extract bias recognition
    bias_data = scores.get("bias_recognition", {})
    if isinstance(bias_data, dict):
        default_scores["bias_recognition_count"] = bias_data.get("count", 0)
        default_scores["bias_recognition_tag"] = bias_data.get("tag", "unknown")
    
    # Extract originality
    originality_data = scores.get("conceptual_originality", {})
    if isinstance(originality_data, dict):
        default_scores["originality_score"] = originality_data.get("score", 0.0)
        default_scores["originality_tag"] = originality_data.get("tag", "unknown")
    
    # Extract strategy
    strategy_data = scores.get("mitigation_strategy", {})
    if isinstance(strategy_data, dict):
        default_scores["strategy_count"] = strategy_data.get("count", 0)
        default_scores["strategy_tag"] = strategy_data.get("tag", "unknown")
    
    # Extract transfer
    transfer_data = scores.get("domain_transferability", {})
    if isinstance(transfer_data, dict):
        default_scores["transfer_count"] = transfer_data.get("count", 0)
        default_scores["transfer_tag"] = transfer_data.get("tag", "unknown")
    
    # Extract metacognition
    metacog_data = scores.get("metacognitive_awareness", {})
    if isinstance(metacog_data, dict):
        default_scores["metacognition_count"] = metacog_data.get("count", 0)
        default_scores["metacognition_tag"] = metacog_data.get("tag", "unknown")
    
    # Overall score
    default_scores["overall_quality_score"] = scores.get("overall_quality_score", 0.0)
    
    return default_scores

def extract_quality_flags(scores: Dict) -> Dict:
    """Extract quality flags from scoring results."""
    default_flags = {
        "low_effort_flag": False,
        "high_similarity_risk": False,
        "scoring_error": False
    }
    
    if not scores:
        return default_flags
    
    if "error" in scores:
        default_flags["scoring_error"] = True
        return default_flags
    
    confidence_flags = scores.get("confidence_flags", {})
    if isinstance(confidence_flags, dict):
        default_flags["low_effort_flag"] = confidence_flags.get("low_effort", False)
        default_flags["high_similarity_risk"] = confidence_flags.get("high_similarity_risk", False)
    
    return default_flags

def get_available_simulations() -> List[str]:
    """Get list of available simulation folders."""
    try:
        if not EXPORTS_DIR.exists():
            return []
        
        folders = [
            f.name for f in EXPORTS_DIR.iterdir() 
            if f.is_dir() and not f.name.startswith('.')
        ]
        return sorted(folders, reverse=True)  # Most recent first
        
    except Exception as e:
        st.error(f"Failed to get simulation folders: {e}")
        return []

# ================================
# STATISTICAL ANALYSIS FUNCTIONS
# ================================

def calculate_basic_statistics(df: pd.DataFrame) -> Dict:
    """Calculate basic descriptive statistics."""
    try:
        stats = {
            "total_sessions": df["session_id"].nunique(),
            "total_responses": len(df),
            "expertise_distribution": df["user_expertise"].value_counts().to_dict(),
            "ai_assistance_distribution": df["ai_assistance_enabled"].value_counts().to_dict(),
            "bias_type_distribution": df["bias_type"].value_counts().to_dict(),
            "domain_distribution": df["domain"].value_counts().to_dict(),
            "session_quality_distribution": df["session_quality"].value_counts().to_dict(),
            "average_response_length": safe_mean_calculation(df["word_count"]),
            "average_session_time": safe_mean_calculation(df["total_session_time_minutes"]),
            "guidance_usage_rate": safe_mean_calculation(df["guidance_requested"]),
            "low_effort_rate": safe_mean_calculation(df["low_effort_flag"]),
            "high_similarity_risk_rate": safe_mean_calculation(df["high_similarity_risk"])
        }
        return stats
    except Exception as e:
        st.error(f"Error calculating statistics: {e}")
        return {}

def test_research_hypotheses(df: pd.DataFrame) -> Dict:
    """Test core research hypotheses with statistical analysis - TYPE SAFE VERSION."""
    try:
        results = {}
        
        # H1: AI assistance creates dependency (higher semantic similarity)
        try:
            ai_users = df[df["ai_assistance_enabled"] == True]["semantic_similarity"]
            no_ai_users = df[df["ai_assistance_enabled"] == False]["semantic_similarity"]
            
            # Convert to numeric and filter out non-numeric values
            ai_users_numeric = pd.to_numeric(ai_users, errors='coerce').dropna()
            no_ai_users_numeric = pd.to_numeric(no_ai_users, errors='coerce').dropna()
            
            if len(ai_users_numeric) > 0 and len(no_ai_users_numeric) > 0:
                try:
                    from scipy import stats
                    from typing import Any
                    
                    # Perform t-test with type bypass for Pylance
                    t_test_result: Any = stats.ttest_ind(ai_users_numeric, no_ai_users_numeric)
                    
                    # Modern scipy (1.16.0) returns TtestResult object with attributes
                    try:
                        # Try modern scipy first (your version)
                        t_stat: float = float(t_test_result.statistic)
                        p_value: float = float(t_test_result.pvalue)
                    except AttributeError:
                        # Fallback for older scipy (if needed)
                        t_stat: float = float(t_test_result[0])
                        p_value: float = float(t_test_result[1])
                    
                    # Use safe variance calculation
                    ai_var = safe_variance_calculation(ai_users_numeric)
                    no_ai_var = safe_variance_calculation(no_ai_users_numeric)
                    ai_mean = safe_mean_calculation(ai_users_numeric)
                    no_ai_mean = safe_mean_calculation(no_ai_users_numeric)
                    
                    # Calculate pooled standard deviation safely
                    pooled_variance = safe_numeric_operation(ai_var, no_ai_var, "add") / 2
                    pooled_std = np.sqrt(pooled_variance) if pooled_variance > 0 else 0.0
                    
                    # Calculate effect size safely
                    mean_diff = safe_numeric_operation(ai_mean, no_ai_mean, "subtract")
                    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0
                    
                    results["h1_ai_dependency"] = {
                        "hypothesis": "AI assistance increases semantic similarity (dependency)",
                        "ai_mean": ai_mean,
                        "no_ai_mean": no_ai_mean,
                        "difference": mean_diff,
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < STATISTICAL_SIGNIFICANCE_THRESHOLD,
                        "effect_size": effect_size
                    }
                except ImportError:
                    results["h1_ai_dependency"] = {"error": "scipy not available for statistical testing"}
            else:
                results["h1_ai_dependency"] = {"error": "Insufficient numeric data for AI dependency test"}
        except Exception as e:
            results["h1_ai_dependency"] = {"error": f"Could not test H1: {e}"}
        
        # H2: Expert vs Novice differences in AI usage
        try:
            expert_guidance = df[df["user_expertise"] == "expert"]["guidance_requested"]
            novice_guidance = df[df["user_expertise"] == "novice"]["guidance_requested"]
            
            # Convert to numeric
            expert_guidance_numeric = pd.to_numeric(expert_guidance, errors='coerce').dropna()
            novice_guidance_numeric = pd.to_numeric(novice_guidance, errors='coerce').dropna()
            
            if len(expert_guidance_numeric) > 0 and len(novice_guidance_numeric) > 0:
                expert_mean = safe_mean_calculation(expert_guidance_numeric)
                novice_mean = safe_mean_calculation(novice_guidance_numeric)
                
                results["h2_expertise_difference"] = {
                    "hypothesis": "Experts and novices differ in AI guidance usage",
                    "expert_guidance_rate": expert_mean,
                    "novice_guidance_rate": novice_mean,
                    "difference": safe_numeric_operation(expert_mean, novice_mean, "subtract")
                }
            else:
                results["h2_expertise_difference"] = {"error": "Insufficient data for expertise comparison"}
        except Exception as e:
            results["h2_expertise_difference"] = {"error": f"Could not test H2: {e}"}
        
        # H3: Transfer learning effectiveness
        try:
            stage_3_transfer = df[df["stage_number"] == 3]["transfer_count"]
            stage_2_transfer = df[df["stage_number"] == 2]["transfer_count"]
            
            # Convert to numeric
            stage_3_numeric = pd.to_numeric(stage_3_transfer, errors='coerce').dropna()
            stage_2_numeric = pd.to_numeric(stage_2_transfer, errors='coerce').dropna()
            
            if len(stage_3_numeric) > 0 and len(stage_2_numeric) > 0:
                stage_3_mean = safe_mean_calculation(stage_3_numeric)
                stage_2_mean = safe_mean_calculation(stage_2_numeric)
                
                results["h3_transfer_learning"] = {
                    "hypothesis": "Transfer learning improves from stage 2 to 3",
                    "stage_2_mean": stage_2_mean,
                    "stage_3_mean": stage_3_mean,
                    "improvement": safe_numeric_operation(stage_3_mean, stage_2_mean, "subtract")
                }
            else:
                results["h3_transfer_learning"] = {"error": "Insufficient data for transfer learning analysis"}
        except Exception as e:
            results["h3_transfer_learning"] = {"error": f"Could not test H3: {e}"}
        
        return results
        
    except Exception as e:
        st.error(f"Error in hypothesis testing: {e}")
        return {}

def analyze_factorial_effects(df: pd.DataFrame) -> Dict:
    """Analyze 2×2×3 factorial design effects."""
    try:
        # Group by experimental conditions
        grouped = df.groupby(["user_expertise", "ai_assistance_enabled", "bias_type"]).agg({
            "overall_quality_score": ["mean", "std", "count"],
            "semantic_similarity": "mean",
            "originality_score": "mean", 
            "bias_recognition_count": "mean",
            "strategy_count": "mean",
            "guidance_requested": "mean"
        }).round(3)
        
        return {
            "factorial_means": grouped.to_dict(),
            "sample_sizes": df.groupby(["user_expertise", "ai_assistance_enabled", "bias_type"]).size().to_dict()
        }
    except Exception as e:
        st.error(f"Error in factorial analysis: {e}")
        return {}

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def create_factorial_visualization(df: pd.DataFrame):
    """Create 2×2×3 factorial design visualization."""
    if not PLOTTING_AVAILABLE or go is None or make_subplots is None:
        st.error("Plotting libraries not available")
        return None
    
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Overall Quality Score", "Originality Score", 
                          "Bias Recognition", "AI Guidance Usage"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Group data by experimental factors
        grouped = df.groupby(["user_expertise", "ai_assistance_enabled", "bias_type"]).agg({
            "overall_quality_score": "mean",
            "originality_score": "mean",
            "bias_recognition_count": "mean", 
            "guidance_requested": "mean"
        }).reset_index()
        
        # Create condition labels
        grouped["condition"] = grouped["user_expertise"] + "_" + grouped["ai_assistance_enabled"].astype(str)
        
        # Add traces for each bias type
        for i, bias_type in enumerate(grouped["bias_type"].unique()):
            bias_data = grouped[grouped["bias_type"] == bias_type]
            
            # Overall quality
            fig.add_trace(
                go.Bar(name=f"{bias_type}", x=bias_data["condition"], y=bias_data["overall_quality_score"],
                      showlegend=(i==0)), row=1, col=1
            )
            
            # Originality  
            fig.add_trace(
                go.Bar(name=f"{bias_type}", x=bias_data["condition"], y=bias_data["originality_score"],
                      showlegend=False), row=1, col=2
            )
            
            # Bias recognition
            fig.add_trace(
                go.Bar(name=f"{bias_type}", x=bias_data["condition"], y=bias_data["bias_recognition_count"],
                      showlegend=False), row=2, col=1
            )
            
            # Guidance usage
            fig.add_trace(
                go.Bar(name=f"{bias_type}", x=bias_data["condition"], y=bias_data["guidance_requested"],
                      showlegend=False), row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="2×2×3 Factorial Design Results")
        return fig
        
    except Exception as e:
        st.error(f"Error creating factorial visualization: {e}")
        return None

def create_hypothesis_testing_charts(hypothesis_results: Dict):
    """Create visualizations for hypothesis testing results."""
    if not PLOTTING_AVAILABLE or go is None or make_subplots is None:
        return None
    
    try:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("H1: AI Dependency", "H2: Expertise Differences", "H3: Transfer Learning")
        )
        
        # H1: AI Dependency
        h1 = hypothesis_results.get("h1_ai_dependency", {})
        if "error" not in h1:
            fig.add_trace(
                go.Bar(name="AI Dependency", x=["No AI", "AI Assisted"], 
                      y=[h1.get("no_ai_mean", 0), h1.get("ai_mean", 0)]),
                row=1, col=1
            )
        
        # H2: Expertise 
        h2 = hypothesis_results.get("h2_expertise_difference", {})
        if "error" not in h2:
            fig.add_trace(
                go.Bar(name="Expertise", x=["Expert", "Novice"],
                      y=[h2.get("expert_guidance_rate", 0), h2.get("novice_guidance_rate", 0)]),
                row=1, col=2
            )
        
        # H3: Transfer
        h3 = hypothesis_results.get("h3_transfer_learning", {})
        if "error" not in h3:
            fig.add_trace(
                go.Bar(name="Transfer", x=["Stage 2", "Stage 3"],
                      y=[h3.get("stage_2_mean", 0), h3.get("stage_3_mean", 0)]),
                row=1, col=3
            )
        
        fig.update_layout(height=400, title_text="Research Hypothesis Testing Results")
        return fig
        
    except Exception as e:
        st.error(f"Error creating hypothesis charts: {e}")
        return None

# ================================
# MAIN DASHBOARD INTERFACE
# ================================

def main():
    """Main dashboard interface."""
    try:
        # Page configuration
        st.set_page_config(
            page_title="ClārusAI Research Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        load_css()
        
        # Header
        st.markdown("# 📊 ClārusAI Research Dashboard")
        st.markdown("*UCL Master's Dissertation: Building AI Literacy Through Simulation*")
        st.markdown("---")
        
        # Check system requirements
        if not SIMULATION_AVAILABLE or SimulatedUserGenerator is None:
            st.error("⚠️ Simulation generator not available. Please check sim_user_generator.py")
        
        if not PLOTTING_AVAILABLE:
            st.warning("⚠️ Some visualization features unavailable. Install matplotlib, seaborn, plotly")
        
        # Main tabs
        tabs = st.tabs([
            "📊 Dataset Generation", 
            "📈 Statistical Analysis", 
            "🧮 Scoring Validation", 
            "📤 Research Exports"
        ])
        
        # ================================
        # TAB 1: Dataset Generation
        # ================================
        with tabs[0]:
            st.header("📊 Simulated Dataset Generation & Management")
            
            # Available simulations
            available_sims = get_available_simulations()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📁 Available Datasets")
                if available_sims:
                    selected_sim = st.selectbox(
                        "Select simulation dataset:",
                        available_sims,
                        key="dataset_selector"
                    )
                    
                    # Load and display dataset info
                    if selected_sim:
                        df = load_simulation_dataset(selected_sim)
                        
                        if df is not None:
                            st.session_state['current_dataset'] = df
                            st.session_state['current_sim_name'] = selected_sim
                            
                            # Dataset summary
                            stats = calculate_basic_statistics(df)
                            
                            col_a, col_b, col_c, col_d = st.columns(4)
                            col_a.metric("Sessions", stats.get("total_sessions", 0))
                            col_b.metric("Responses", stats.get("total_responses", 0))
                            col_c.metric("Avg Words", f"{stats.get('average_response_length', 0):.1f}")
                            col_d.metric("Quality Rate", f"{(1 - stats.get('low_effort_rate', 0)) * 100:.1f}%")
                            
                            # Distribution summaries
                            st.write("**Experimental Conditions Distribution:**")
                            dist_col1, dist_col2, dist_col3 = st.columns(3)
                            
                            with dist_col1:
                                st.write("*Expertise:*", stats.get("expertise_distribution", {}))
                            with dist_col2:
                                st.write("*AI Assistance:*", stats.get("ai_assistance_distribution", {}))
                            with dist_col3:
                                st.write("*Bias Types:*", stats.get("bias_type_distribution", {}))
                
                else:
                    st.info("No simulation datasets found. Generate one below.")
            
            with col2:
                st.subheader("🧪 Generate New Dataset")
                
                if SIMULATION_AVAILABLE and SimulatedUserGenerator is not None:
                    # Generation parameters
                    st.write("**2×2×3 Factorial Design:**")
                    st.write("• User Expertise: Novice, Expert")
                    st.write("• AI Assistance: Enabled, Disabled") 
                    st.write("• Bias Type: Confirmation, Anchoring, Availability")
                    st.write("• Replicates: 3 per condition")
                    st.write("• **Total: 144 responses (36 sessions × 4 stages)**")
                    
                    if st.button("🚀 Generate Simulated Dataset", type="primary"):
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def dashboard_progress_callback(current, total, session_id):
                            progress = float(current) / float(total)
                            progress_bar.progress(progress)
                            status_text.text(f"Generating: {session_id} ({current}/{total})")
                        
                        try:
                            with st.spinner("Initializing simulation generator..."):
                                generator = SimulatedUserGenerator()
                            
                            # Generate dataset
                            result = generator.generate_full_dataset(
                                progress_callback=dashboard_progress_callback
                            )
                            
                            if result["success"]:
                                st.success(f"✅ Generated {result['summary']['generation_metadata']['total_generated']} sessions!")
                                st.info("🔄 Refresh the page to see the new dataset in the dropdown.")
                                
                                # Show generation summary
                                summary = result["summary"]
                                with st.expander("📋 Generation Summary"):
                                    st.json(summary["generation_metadata"])
                            else:
                                st.error(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"❌ Generation error: {e}")
                            if config.DEBUG:
                                st.exception(e)
                else:
                    st.error("Simulation generator not available")
        
        # ================================
        # TAB 2: Statistical Analysis  
        # ================================
        with tabs[1]:
            st.header("📈 Statistical Analysis & Hypothesis Testing")
            
            if 'current_dataset' not in st.session_state:
                st.warning("⚠️ Please select a dataset in Tab 1 first.")
            else:
                df = st.session_state['current_dataset']
                
                # Research hypotheses testing
                st.subheader("🎯 Core Research Hypotheses")
                
                with st.spinner("Running statistical analyses..."):
                    hypothesis_results = test_research_hypotheses(df)
                
                # Display hypothesis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**H1: AI Dependency Effect**")
                    h1 = hypothesis_results.get("h1_ai_dependency", {})
                    if "error" not in h1:
                        st.metric("AI Users Similarity", f"{h1.get('ai_mean', 0):.3f}")
                        st.metric("Non-AI Users Similarity", f"{h1.get('no_ai_mean', 0):.3f}")
                        st.metric("P-value", f"{h1.get('p_value', 1):.4f}")
                        
                        if h1.get('significant', False):
                            st.success("✅ Statistically significant")
                        else:
                            st.info("ℹ️ Not statistically significant")
                    else:
                        st.error(f"H1 Error: {h1.get('error', 'Unknown')}")
                
                with col2:
                    st.write("**H2: Expertise Differences**")
                    h2 = hypothesis_results.get("h2_expertise_difference", {})
                    if "error" not in h2:
                        st.metric("Expert Guidance Rate", f"{h2.get('expert_guidance_rate', 0):.3f}")
                        st.metric("Novice Guidance Rate", f"{h2.get('novice_guidance_rate', 0):.3f}")
                        st.metric("Difference", f"{h2.get('difference', 0):.3f}")
                    else:
                        st.error(f"H2 Error: {h2.get('error', 'Unknown')}")
                
                # Factorial analysis
                st.subheader("🔬 2×2×3 Factorial Analysis")
                factorial_results = analyze_factorial_effects(df)
                
                if PLOTTING_AVAILABLE and make_subplots is not None and go is not None:
                    factorial_fig = create_factorial_visualization(df)
                    if factorial_fig:
                        st.plotly_chart(factorial_fig, use_container_width=True)
                
                    # Hypothesis testing visualization
                    hyp_fig = create_hypothesis_testing_charts(hypothesis_results)
                    if hyp_fig:
                        st.plotly_chart(hyp_fig, use_container_width=True)
                
                # Detailed statistics
                with st.expander("📊 Detailed Statistical Results"):
                    st.write("**Factorial Means:**")
                    if factorial_results:
                        st.json(factorial_results.get("sample_sizes", {}))
                    
                    st.write("**Hypothesis Testing Details:**")
                    st.json(hypothesis_results)
        
        # ================================
        # TAB 3: Scoring Validation
        # ================================
        with tabs[2]:
            st.header("🧮 6-Dimensional Scoring Framework Validation")
            
            if 'current_dataset' not in st.session_state:
                st.warning("⚠️ Please select a dataset in Tab 1 first.")
            else:
                df = st.session_state['current_dataset']
                
                # Scoring distribution analysis
                st.subheader("📊 Scoring Distribution Analysis")
                
                scoring_cols = [
                    "semantic_similarity", "originality_score", "bias_recognition_count",
                    "strategy_count", "transfer_count", "metacognition_count"
                ]
                
                # Check scoring completeness
                scoring_errors = df["scoring_error"].sum()
                total_responses = len(df)
                scoring_success_rate = (total_responses - scoring_errors) / total_responses * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Scoring Success Rate", f"{scoring_success_rate:.1f}%")
                col2.metric("Scoring Errors", scoring_errors)
                col3.metric("Low Effort Flags", df["low_effort_flag"].sum())
                col4.metric("High Similarity Flags", df["high_similarity_risk"].sum())
                
                if PLOTTING_AVAILABLE and px is not None:
                    # Scoring correlation matrix
                    st.subheader("🔗 Scoring Dimension Correlations")
                    
                    try:
                        # Create correlation matrix for valid scores only
                        valid_scores = df[df["scoring_error"] == False][scoring_cols]
                        
                        if not valid_scores.empty:
                            correlation_matrix = valid_scores.corr()
                            
                            fig = px.imshow(
                                correlation_matrix,
                                labels=dict(x="Scoring Dimension", y="Scoring Dimension", color="Correlation"),
                                x=scoring_cols,
                                y=scoring_cols,
                                color_continuous_scale="RdBu",
                                aspect="auto"
                            )
                            fig.update_layout(title="6-Dimensional Scoring Correlation Matrix")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Scoring distribution by expertise
                            st.subheader("📈 Score Distributions by Expertise")
                            
                            if make_subplots is not None and go is not None:
                                fig2 = make_subplots(
                                    rows=2, cols=3,
                                    subplot_titles=scoring_cols
                                )
                                
                                for i, col in enumerate(scoring_cols):
                                    row = (i // 3) + 1
                                    col_pos = (i % 3) + 1
                                    
                                    for expertise in df["user_expertise"].unique():
                                        expertise_data = pd.to_numeric(df[df["user_expertise"] == expertise][col], errors='coerce').dropna()
                                        
                                        fig2.add_trace(
                                            go.Histogram(
                                                x=expertise_data,
                                                name=f"{expertise}",
                                                opacity=0.7,
                                                showlegend=(i == 0)
                                            ),
                                            row=row, col=col_pos
                                        )
                                
                                fig2.update_layout(height=500, title_text="Score Distributions by User Expertise")
                                st.plotly_chart(fig2, use_container_width=True)
                            else:
                                st.info("Advanced plotting features not available - install plotly for visualizations")
                            
                        else:
                            st.warning("No valid scoring data available for correlation analysis")
                            
                    except Exception as e:
                        st.error(f"Error creating scoring visualizations: {e}")
                
                # Algorithm transparency
                st.subheader("🔍 Algorithm Transparency")
                
                # Show scoring thresholds
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Scoring Thresholds:**")
                    thresholds = {
                        "Semantic Similarity High": config.SCORING_THRESHOLDS.get("semantic_similarity_high", 0.75),
                        "Semantic Similarity Medium": config.SCORING_THRESHOLDS.get("semantic_similarity_medium", 0.5),
                        "Bias Recognition Min Terms": config.SCORING_THRESHOLDS.get("bias_recognition_min_terms", 2),
                        "Originality Low": config.SCORING_THRESHOLDS.get("originality_low", 0.25),
                        "Originality High": config.SCORING_THRESHOLDS.get("originality_high", 0.5)
                    }
                    st.json(thresholds)
                
                with col2:
                    st.write("**Quality Indicators:**")
                    quality_metrics = {
                        "Response Length Threshold": config.QUALITY_THRESHOLDS.get("minimum_response_length", 10),
                        "Min Words Per Stage": config.QUALITY_THRESHOLDS.get("minimum_words_per_stage", 15),
                        "Engagement Threshold": config.QUALITY_THRESHOLDS.get("engagement_threshold", 0.3)
                    }
                    st.json(quality_metrics)
                
                # Cross-validation analysis
                st.subheader("✅ Cross-Validation Results")
                
                # Simulate cross-validation by splitting data
                try:
                    # Group by session to maintain session integrity
                    sessions = df["session_id"].unique()
                    n_sessions = len(sessions)
                    
                    if n_sessions >= 4:  # Need minimum for cross-validation
                        split_point = n_sessions // 2
                        
                        train_sessions = sessions[:split_point]
                        test_sessions = sessions[split_point:]
                        
                        train_df = df[df["session_id"].isin(train_sessions)]
                        test_df = df[df["session_id"].isin(test_sessions)]
                        
                        # Compare distributions
                        cv_metrics = {}
                        for col in scoring_cols:
                            if col in train_df.columns and col in test_df.columns:
                                train_mean = safe_mean_calculation(train_df[col])
                                test_mean = safe_mean_calculation(test_df[col])
                                difference = abs(train_mean - test_mean)
                                cv_metrics[col] = {
                                    "train_mean": train_mean,
                                    "test_mean": test_mean,
                                    "difference": difference,
                                    "consistency": difference < 0.1  # Threshold for consistency
                                }
                        
                        # Display cross-validation results
                        cv_df = pd.DataFrame(cv_metrics).T
                        st.dataframe(cv_df)
                        
                        # Overall consistency score
                        consistent_dimensions = sum(1 for metrics in cv_metrics.values() if metrics["consistency"])
                        consistency_rate = consistent_dimensions / len(cv_metrics) * 100
                        
                        st.metric("Cross-Validation Consistency", f"{consistency_rate:.1f}%")
                        
                        if consistency_rate >= 80:
                            st.success("✅ High scoring consistency across data splits")
                        elif consistency_rate >= 60:
                            st.warning("⚠️ Moderate scoring consistency")
                        else:
                            st.error("❌ Low scoring consistency - review algorithm")
                    
                    else:
                        st.info("Insufficient data for cross-validation (need 4+ sessions)")
                        
                except Exception as e:
                    st.error(f"Error in cross-validation analysis: {e}")
        
        # ================================
        # TAB 4: Research Exports
        # ================================
        with tabs[3]:
            st.header("📤 Research Exports & Documentation")
            
            if 'current_dataset' not in st.session_state:
                st.warning("⚠️ Please select a dataset in Tab 1 first.")
            else:
                df = st.session_state['current_dataset']
                sim_name = st.session_state.get('current_sim_name', 'unknown')
                
                # Export options
                st.subheader("📊 Dataset Exports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Full Dataset Export**")
                    
                    # CSV export
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📄 Download Full Dataset (CSV)",
                        data=csv_data,
                        file_name=f"clarusai_simulation_{sim_name}.csv",
                        mime="text/csv"
                    )
                    
                    # JSON export
                    json_data = df.to_json(orient='records', indent=2).encode('utf-8')
                    st.download_button(
                        label="📋 Download Full Dataset (JSON)",
                        data=json_data,
                        file_name=f"clarusai_simulation_{sim_name}.json",
                        mime="application/json"
                    )
                
                with col2:
                    st.write("**Statistical Summary Export**")
                    
                    # Calculate comprehensive summary
                    stats = calculate_basic_statistics(df)
                    factorial_results = analyze_factorial_effects(df)
                    hypothesis_results = test_research_hypotheses(df)
                    
                    summary_report = {
                        "dataset_info": {
                            "simulation_name": sim_name,
                            "generated_timestamp": datetime.now().isoformat(),
                            "export_timestamp": datetime.now().isoformat()
                        },
                        "descriptive_statistics": stats,
                        "factorial_analysis": factorial_results,
                        "hypothesis_testing": hypothesis_results,
                        "research_metadata": {
                            "experimental_design": "2x2x3 factorial",
                            "factors": ["user_expertise", "ai_assistance", "bias_type"],
                            "scoring_dimensions": 6,
                            "total_conditions": 12
                        }
                    }
                    
                    summary_json = json.dumps(summary_report, indent=2, default=str).encode('utf-8')
                    st.download_button(
                        label="📈 Download Statistical Summary",
                        data=summary_json,
                        file_name=f"clarusai_statistical_summary_{sim_name}.json",
                        mime="application/json"
                    )
                
                # Research documentation
                st.subheader("📚 Research Documentation")
                
                # Calculate stats for documentation
                stats = calculate_basic_statistics(df)
                
                # Methodology documentation
                methodology_doc = f"""
                # ClārusAI Simulation Study Methodology

                ## Dataset: {sim_name}
                **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                ## Experimental Design
                - **Type:** 2×2×3 Factorial Design
                - **Factors:**
                - User Expertise: Novice vs Expert (2 levels)
                - AI Assistance: Enabled vs Disabled (2 levels)
                - Bias Type: Confirmation vs Anchoring vs Availability (3 levels)
                - **Replicates:** 3 per condition
                - **Total Sessions:** {stats.get('total_sessions', 0)}
                - **Total Responses:** {stats.get('total_responses', 0)}

                ## 6-Dimensional Scoring Framework
                1. **Semantic Similarity** - Measures AI dependency vs independence
                2. **Bias Recognition** - Detects awareness of cognitive biases
                3. **Conceptual Originality** - Evaluates independent reasoning
                4. **Mitigation Strategy** - Assesses strategic thinking quality
                5. **Domain Transferability** - Tests cross-context application
                6. **Metacognitive Awareness** - Captures self-reflective reasoning

                ## Quality Control
                - **Scoring Success Rate:** {((len(df) - df['scoring_error'].sum()) / len(df) * 100):.1f}%
                - **Low Effort Rate:** {(df['low_effort_flag'].mean() * 100):.1f}%
                - **High Similarity Risk:** {(df['high_similarity_risk'].mean() * 100):.1f}%

                ## Research Hypotheses
                1. **H1:** AI assistance increases semantic similarity (dependency indicator)
                2. **H2:** Expert and novice users show different AI utilization patterns
                3. **H3:** Transfer learning improves from mitigation to transfer stages

                ## Data Quality Indicators
                - **Average Response Length:** {stats.get('average_response_length', 0):.1f} words
                - **Average Session Time:** {stats.get('average_session_time', 0):.1f} minutes
                - **Guidance Usage Rate:** {(stats.get('guidance_usage_rate', 0) * 100):.1f}%

                ## Statistical Thresholds
                - **Significance Level:** α = 0.05
                - **Effect Size Thresholds:** Small (0.2), Medium (0.5), Large (0.8)
                - **Scoring Thresholds:** See configuration documentation

                ---
                *Generated by ClārusAI Research Dashboard*
                *UCL Master's Dissertation: Building AI Literacy Through Simulation*
                """
                
                st.download_button(
                    label="📖 Download Methodology Documentation",
                    data=methodology_doc.encode('utf-8'),
                    file_name=f"clarusai_methodology_{sim_name}.md",
                    mime="text/markdown"
                )
                
                # Configuration export
                config_export = {
                    "scoring_thresholds": config.SCORING_THRESHOLDS,
                    "quality_thresholds": config.QUALITY_THRESHOLDS,
                    "validation_rules": config.VALIDATION_RULES,
                    "bias_keywords_count": {bias: len(keywords) for bias, keywords in config.BIAS_KEYWORDS.items()},
                    "simulation_parameters": {
                        "replicates_per_condition": 3,
                        "stages_per_session": 4,
                        "total_conditions": 12
                    }
                }
                
                config_json = json.dumps(config_export, indent=2).encode('utf-8')
                st.download_button(
                    label="⚙️ Download Configuration Settings",
                    data=config_json,
                    file_name=f"clarusai_config_{sim_name}.json",
                    mime="application/json"
                )
                
                # Publication-ready summary
                st.subheader("📊 Publication-Ready Results Summary")
                
                hypothesis_results = test_research_hypotheses(df)
                
                with st.expander("📋 Key Findings Summary", expanded=True):
                    st.write("**Sample Characteristics:**")
                    st.write(f"- Total sessions: {stats.get('total_sessions', 0)}")
                    st.write(f"- Total responses: {stats.get('total_responses', 0)}")
                    st.write(f"- Mean response length: {stats.get('average_response_length', 0):.1f} words")
                    
                    st.write("**Experimental Conditions:**")
                    exp_dist = stats.get('expertise_distribution', {})
                    ai_dist = stats.get('ai_assistance_distribution', {})
                    st.write(f"- Novice users: {exp_dist.get('novice', 0)}, Expert users: {exp_dist.get('expert', 0)}")
                    st.write(f"- AI-assisted: {ai_dist.get(True, 0)}, Unassisted: {ai_dist.get(False, 0)}")
                    
                    st.write("**Research Hypothesis Results:**")
                    h1 = hypothesis_results.get("h1_ai_dependency", {})
                    if "error" not in h1:
                        sig_text = "significant" if h1.get('significant', False) else "not significant"
                        st.write(f"- H1 (AI Dependency): {sig_text} (p = {h1.get('p_value', 1):.4f})")
                    
                    h2 = hypothesis_results.get("h2_expertise_difference", {})
                    if "error" not in h2:
                        st.write(f"- H2 (Expertise Difference): Expert guidance rate = {h2.get('expert_guidance_rate', 0):.3f}, Novice = {h2.get('novice_guidance_rate', 0):.3f}")
                    
                    h3 = hypothesis_results.get("h3_transfer_learning", {})
                    if "error" not in h3:
                        improvement = h3.get('improvement', 0)
                        st.write(f"- H3 (Transfer Learning): Stage improvement = {improvement:.3f}")
        
        # Footer
        render_academic_footer()
        
    except Exception as e:
        st.error("❌ Critical dashboard error occurred")
        st.error(f"Error: {e}")
        
        if config.DEBUG:
            st.exception(e)
            st.write("**Traceback:**")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()