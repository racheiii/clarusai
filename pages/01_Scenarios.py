"""
pages/01_Scenarios.py — Live Demonstration Interface

Purpose:
Provides the interactive 4-stage scenario interface used in ClārusAI for demonstration
purposes. While fully functional, this mode is not used in the main dataset generation
for the research; instead, all analysed data is produced by the automated simulation
pipeline for consistency and reproducibility.

Role in Research Pipeline:
Implements the progressive decision-making sequence for the 2×2×3 factorial design
(User Expertise × AI Assistance × Bias Type), collecting structured responses across:
1. Primary Analysis
2. Cognitive Factors
3. Mitigation Strategies
4. Transfer Learning

Key Features:
- Bias-blind design: The targeted cognitive bias is revealed only after scenario completion.
- Modular integration: Uses shared controllers (SessionManager, ScenarioHandler,
  UIComponents, AIGuidance, DataCollector) to ensure consistent behaviour with
  simulation mode.
- Comprehensive logging: Records all user inputs in a structure compatible with the
  6-dimensional scoring rubric.

This page exists in the submission to illustrate how the experiment could run with
real participants, complementing the automated simulation-based evaluation.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import sys
from pathlib import Path

# Add project paths
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# Import configuration and utilities
import config
from utils import (
    setup_training_page_config, 
    load_css, 
    render_academic_footer
)

# Import modular components
try:
    from src.session_manager import SessionManager, safe_get_session_value, safe_set_session_value
    from src.scenario_handler import ScenarioHandler
    from src.ai_guidance import AIGuidance
    from src.ui_components import UIComponents
    from src.data_collector import DataCollector
    
except ImportError as e:
    st.error(f"❌ Component modules not available: {e}")
    st.error("Please ensure all src/ modules are available")
    st.stop()

# =============================================================================
# MAIN APPLICATION CONTROLLER
# =============================================================================

class ScenariosPageController:
    """
    Main controller for the 4-stage experimental training interface.
    
    Purpose: Orchestrates the complete experimental protocol while
    maintaining clean separation of concerns and error handling.
    """
    
    def __init__(self):
        """Initialize controller with all required components"""
        self.session_manager = SessionManager()
        self.scenario_handler = ScenarioHandler()
        self.ai_guidance = AIGuidance()
        self.ui_components = UIComponents()
        self.data_collector = DataCollector()
        
        # Initialize session state
        self.session_manager.initialize_session_state()
    
    def run(self):
        """
        Main execution flow for the experimental interface.
        
        Research Flow:
        1. Setup & Configuration Validation
        2. Experimental Factor Selection (2×2×3 design)
        3. 4-Stage Progressive Interaction
        4. Completion & Bias Revelation
        """
        try:
            # Page configuration
            setup_training_page_config("Training Scenarios", "🎯")
            load_css()

            # Demo mode banner
            st.info("💡 Demo Mode: This interface is for demonstration purposes only. All analysis uses simulated datasets on Dashboard.")

            # Validate system configuration
            self._validate_configuration()

            # Handle session recovery if needed
            self._handle_session_recovery()

            if "cached_scenarios" not in st.session_state:
                loaded_df = self.scenario_handler.load_scenarios()
                if loaded_df is not None:
                    st.session_state.cached_scenarios = loaded_df
                    self.scenario_handler.scenarios_df = loaded_df  # rehydrate
                else:
                    st.error("❌ Scenario database could not be loaded.")
                    st.stop()
            else:
                self.scenario_handler.scenarios_df = st.session_state.cached_scenarios

            scenarios_df = self.scenario_handler.scenarios_df
            if scenarios_df is None or scenarios_df.empty:
                st.error("❌ No scenarios available after loading. Please check your CSV file.")
                st.stop()

            # ===========================
            # RENDER ROUTING AND FLOW
            # ===========================
            self._render_compact_navigation()
            self._route_experimental_flow(scenarios_df)

            # Developer Debug (optional)
            if config.DEBUG:
                self._render_debug_interface(scenarios_df)

            render_academic_footer()

        except Exception as e:
            self._handle_critical_error(e)
    
    def _render_compact_navigation(self):
        col1, col2 = st.columns([10, 1])
        
        with col1:
            st.markdown("### ClārusAI Training Interface")
        
        with col2: 
            if st.button("🏠", key="nav_home", help="Return to main menu"):
                # Show confirmation dialog
                self._show_navigation_confirmation()

    @st.dialog("Confirm Navigation")
    def _show_navigation_confirmation(self):
        st.write("⚠️ **Warning**: Returning home will lose your current progress.")
        st.write("Your responses will not be saved unless you complete all 4 stages.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("Return Home", type="primary", use_container_width=True):
                try:
                    st.switch_page("Home.py")
                except AttributeError:
                    st.session_state['interaction_flow'] = 'setup'
                    st.rerun()
    
    def _validate_configuration(self):
        """Validate system configuration for research integrity"""
        
        config_validation = config.validate_config()
        if not all(config_validation.values()):
            st.warning("⚠️ Some configuration issues detected. Functionality may be limited.")
            
            if config.DEBUG:
                st.write("Configuration Status:", config_validation)
    
    def _handle_session_recovery(self):
        """Handle session recovery if applicable"""
        
        # For now, skip recovery to avoid complexity
        # Can be enhanced later if needed
        recovery_checked = safe_get_session_value('recovery_checked', False)
        if not recovery_checked:
            safe_set_session_value('recovery_checked', True)
    
    def _route_experimental_flow(self, scenarios_df):
        """Route user to appropriate experimental phase"""
        
        current_flow = safe_get_session_value('interaction_flow', 'setup')
        
        if current_flow == 'setup':
            # Phase 1: Experimental Setup (Factor Selection)
            self._render_setup_phase(scenarios_df)
        
        elif current_flow == 'scenario':
            # Show progress toast only when entering scenario phase
            self.ui_components.show_progress_toast()
            # Phase 2: 4-Stage Progressive Interaction
            self._render_scenario_phase()
        
        elif current_flow == 'completed':
            # Phase 3: Completion & Educational Debrief
            self._render_completion_phase()
        
        else:
            # Unknown state - reset to setup
            st.warning(f"Unknown interaction flow: {current_flow}. Resetting to setup.")
            safe_set_session_value('interaction_flow', 'setup')
            st.rerun()
    
    def _render_setup_phase(self, scenarios_df):
        """Render experimental setup phase"""
        
        success = self.ui_components.render_experimental_setup(
            scenarios_df=scenarios_df,
            scenario_handler=self.scenario_handler,
            session_manager=self.session_manager,
            data_collector=self.data_collector
        )
        
        if success:
            # Setup completed, move to scenario phase
            safe_set_session_value('interaction_flow', 'scenario')
            st.rerun()
    
    def _render_scenario_phase(self):
        """Render 4-stage progressive interaction phase"""
        
        experimental_session = safe_get_session_value('experimental_session')
        
        if not experimental_session or not experimental_session.assigned_scenario:
            st.error("❌ No experimental session found. Returning to setup.")
            safe_set_session_value('interaction_flow', 'setup')
            st.rerun()
            return
        
        scenario = experimental_session.assigned_scenario.__dict__
        current_stage = safe_get_session_value('current_stage', 0)
        
        # Render current stage
        completion_status = self.ui_components.render_scenario_stage(
            scenario=scenario,
            current_stage=current_stage,
            ai_guidance=self.ai_guidance,
            session_manager=self.session_manager,
            data_collector=self.data_collector
        )
        
        if completion_status == 'stage_completed':
            # Advance to next stage or complete
            TOTAL_STAGES = 4
            if current_stage < TOTAL_STAGES - 1:
                safe_set_session_value('current_stage', current_stage + 1)
                st.rerun()
            else:
                # All stages completed
                safe_set_session_value('interaction_flow', 'completed')
                st.rerun()
        
        elif completion_status == 'reset_requested':
            # User requested reset
            self.session_manager.reset_experimental_session()
            st.rerun()
    
    def _render_completion_phase(self):
        """Render completion and bias revelation phase"""
        
        experimental_session = safe_get_session_value('experimental_session')
        
        if not experimental_session or not experimental_session.assigned_scenario:
            st.error("❌ No completed session found. Returning to setup.")
            safe_set_session_value('interaction_flow', 'setup')
            st.rerun()
            return
        
        scenario = experimental_session.assigned_scenario.__dict__
        
        # Render completion interface
        navigation_choice = self.ui_components.render_completion_interface(
            scenario=scenario,
            session_manager=self.session_manager,
            data_collector=self.data_collector
        )
        
        # Handle navigation choices 
        if navigation_choice == 'new_scenario':
            # Reset experimental session and stay on same page
            self.session_manager.reset_experimental_session()
            # Clear toast flags for new session
            safe_set_session_value('progress_toast_shown', False)
            st.rerun()
        elif navigation_choice == 'home':
            st.switch_page("Home.py")
    
    def _handle_critical_error(self, error):
        st.error("❌ A critical error occurred. Please refresh the page and try again.")
        if hasattr(self, 'session_manager'):
            self.session_manager.log_error('critical_application_error', str(error))
        # Reset potentially corrupted session
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        if config.DEBUG:
            st.exception(error)
    
    def _render_debug_interface(self, scenarios_df):
        """Render development debug interface"""
        
        with st.expander("🔧 Development Debug Information", expanded=False):
            
            # Session state summary
            st.write("**Session State Summary:**")
            debug_info = {
                'interaction_flow': safe_get_session_value('interaction_flow', 'unset'),
                'current_stage': safe_get_session_value('current_stage', 'unset'),
                'user_expertise': getattr(safe_get_session_value('user_expertise'), 'value', None),
                'ai_assistance_enabled': safe_get_session_value('ai_assistance_enabled', False),
                'selected_domain': safe_get_session_value('selected_domain', None),
                'session_id': getattr(getattr(st.session_state, 'experimental_session', None), 'session_id', None),
                'api_status': safe_get_session_value('api_status', 'unknown'),
                'session_errors': len(safe_get_session_value('session_errors', []))
            }
            st.json(debug_info)
            
            # Available scenarios
            st.write("**Available Scenarios:**")
            st.dataframe(scenarios_df[['scenario_id', 'bias_type', 'domain', 'cognitive_load_level']])
            
            # Configuration status
            st.write("**Configuration Status:**")
            st.json(config.validate_config())
            
            # Component status
            st.write("**Component Status:**")
            component_status = {
                'session_manager': 'initialized' if hasattr(self, 'session_manager') else 'error',
                'scenario_handler': 'initialized' if hasattr(self, 'scenario_handler') else 'error',
                'ai_guidance': 'initialized' if hasattr(self, 'ai_guidance') else 'error',
                'ui_components': 'initialized' if hasattr(self, 'ui_components') else 'error',
                'data_collector': 'initialized' if hasattr(self, 'data_collector') else 'error'
            }
            st.json(component_status)
            
            # Data collector summary
            st.write("**Data Collection Summary:**")
            st.json(self.data_collector.get_data_summary())

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main entry point for the 01_Scenarios.py page.
    
    Purpose: Provides clean entry point for the experimental
    training interface with proper error boundaries.
    """
    
    try:
        # Initialize and run the main controller
        controller = ScenariosPageController()
        controller.run()
        
    except Exception as e:
        # Fallback error handling
        st.error("❌ Failed to initialize the training interface.")
        st.error(f"Error: {e}")
        
        if config.DEBUG:
            st.exception(e)
        
        # Provide recovery options
        if st.button("🔄 Refresh Page"):
            st.rerun()

if __name__ == "__main__":
    main()