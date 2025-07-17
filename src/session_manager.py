"""
ClārusAI: Session State Management System

src/session_manager.py - Comprehensive session state management

Academic Purpose:
Handles all session state operations with null safety, error tracking,
and recovery functionality for experimental data integrity.

Author: Rachel Seah
Date: July 2025
"""

import streamlit as st
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os

import config
from src.models import UserExpertise, ExperimentalSession
from enum import Enum

def enum_to_str(obj):
    """Recursively convert Enums and NumPy types (incl. bool_) to JSON-safe values."""
    from enum import Enum
    import numpy as np
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: enum_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [enum_to_str(i) for i in obj]
    else:
        return obj

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# SAFE SESSION STATE OPERATIONS
# =============================================================================

def safe_get_session_value(key: str, default: Any = None) -> Any:
    """
    Safely retrieve session state values with null checking.

    Academic Purpose: Prevents crashes during experimental sessions while
    maintaining research data integrity.
    """
    try:
        return getattr(st.session_state, key, default)
    except AttributeError:
        return default

def safe_set_session_value(key: str, value: Any) -> bool:
    """
    Safely set session state values with error logging.

    Academic Purpose: Ensures reliable session state management while
    tracking potential issues for research validation.
    """
    try:
        setattr(st.session_state, key, value)
        return True
    except Exception as e:
        logger.error(f"Failed to set session state {key}: {e}")

        # Track error for research analysis
        if 'session_errors' in st.session_state:
            st.session_state.session_errors.append({
                'timestamp': datetime.now().isoformat(),
                'error_type': 'session_state_error',
                'key': key,
                'error': str(e)
            })
        return False

# =============================================================================
# SESSION MANAGER CLASS
# =============================================================================

class SessionManager:
    """
    Comprehensive session state management for experimental research.

    Academic Purpose: Provides robust session management with error tracking,
    recovery capabilities, and research data integrity validation.
    """

    def __init__(self):
        """Initialize session manager."""
        self.session_initialized = False

    def initialize_session_state(self):
        """
        Initialize comprehensive session state with null safety.

        Academic Purpose: Establishes robust data structures for tracking
        the complete experimental condition while ensuring data integrity.
        """

        if self.session_initialized:
            return

        # Core experimental variables with null safety
        if 'user_expertise' not in st.session_state:
            st.session_state.user_expertise = None
        if 'ai_assistance_enabled' not in st.session_state:
            st.session_state.ai_assistance_enabled = None

        # Primary session management
        if 'experimental_session' not in st.session_state:
            st.session_state.experimental_session = None

        # Flow control
        if 'current_stage' not in st.session_state:
            st.session_state.current_stage = 0
        if 'interaction_flow' not in st.session_state:
            st.session_state.interaction_flow = 'setup'

        # Legacy support (maintained for compatibility)
        if 'stage_responses' not in st.session_state:
            st.session_state.stage_responses = []
        if 'stage_timings' not in st.session_state:
            st.session_state.stage_timings = []
        if 'guidance_usage' not in st.session_state:
            st.session_state.guidance_usage = []

        # Session metadata
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = datetime.now()
        if 'last_auto_save' not in st.session_state:
            st.session_state.last_auto_save = None
        if 'recovery_checked' not in st.session_state:
            st.session_state.recovery_checked = False

        # Error tracking for research quality
        if 'session_errors' not in st.session_state:
            st.session_state.session_errors = []
        if 'api_status' not in st.session_state:
            st.session_state.api_status = 'unknown'

        self.session_initialized = True
        logger.info("Session state initialized successfully")

    def set_user_expertise(self, expertise: UserExpertise) -> bool:
        """
        Set user expertise with validation and logging.

        Academic Purpose: Factor 1 of 2×2×3 factorial design.
        """
        if safe_set_session_value('user_expertise', expertise):
            self.auto_save_session_data('expertise_selection', {
                'expertise': expertise.value,
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"User expertise set to: {expertise.value}")
            return True
        return False

    def set_ai_assistance(self, enabled: bool) -> bool:
        """
        Set AI assistance preference with validation and logging.

        Academic Purpose: Factor 2 of 2×2×3 factorial design.
        """
        if safe_set_session_value('ai_assistance_enabled', enabled):
            self.auto_save_session_data('assistance_selection', {
                'ai_assistance': enabled,
                'timestamp': datetime.now().isoformat()
            })
            logger.info(f"AI assistance set to: {enabled}")
            return True
        return False

    def set_experimental_session(self, session: ExperimentalSession) -> bool:
        """
        Set experimental session with validation.

        Academic Purpose: Complete experimental condition assignment.
        """
        if safe_set_session_value('experimental_session', session):
            safe_set_session_value('interaction_flow', 'scenario')
            safe_set_session_value('current_stage', 0)
            safe_set_session_value('stage_timings', [datetime.now()])

            self.auto_save_session_data('experimental_session_start', {
                'session_id': session.session_id,
                'experimental_condition': session.condition_code,
                'bias_type': session.bias_type.value,
                'domain': session.domain.value,
                'session_start': datetime.now().isoformat()
            })

            logger.info(f"Experimental session created: {session.session_id}")
            return True
        return False

    def advance_stage(self, current_stage: int) -> bool:
        """
        Advance to next experimental stage with validation.

        Academic Purpose: Progress through 4-stage experimental protocol.
        """
        if current_stage < 3:
            if safe_set_session_value('current_stage', current_stage + 1):
                # Update stage timings safely
                current_timings = safe_get_session_value('stage_timings', [])
                current_timings.append(datetime.now())
                safe_set_session_value('stage_timings', current_timings)

                self.auto_save_session_data('stage_progression', {
                    'completed_stage': current_stage,
                    'advancing_to_stage': current_stage + 1,
                    'progression_time': datetime.now().isoformat()
                })

                logger.info(f"Advanced from stage {current_stage} to {current_stage + 1}")
                return True
        else:
            # Complete the experimental protocol
            if safe_set_session_value('interaction_flow', 'completed'):
                # Finalize experimental session
                if st.session_state.experimental_session:
                    st.session_state.experimental_session.completion_time = datetime.now()
                    st.session_state.experimental_session.calculate_analytics()

                self.auto_save_session_data('experimental_protocol_completed', {
                    'total_stages_completed': 4,
                    'final_completion_time': datetime.now().isoformat(),
                    'session_duration_minutes': self.get_session_duration_minutes()
                })

                logger.info("Experimental protocol completed")
                return True

        return False

    def reset_experimental_session(self):
        """
        Reset session state with comprehensive cleanup.

        Academic Purpose: Enable new experimental trials while
        maintaining data integrity.
        """
        try:
            # Reset core experimental variables
            safe_set_session_value('interaction_flow', 'setup')
            safe_set_session_value('experimental_session', None)
            safe_set_session_value('current_stage', 0)
            safe_set_session_value('user_expertise', None)
            safe_set_session_value('ai_assistance_enabled', None)
            safe_set_session_value('recovery_checked', False)
            safe_set_session_value('session_start_time', datetime.now())

            # Reset legacy tracking
            safe_set_session_value('stage_responses', [])
            safe_set_session_value('stage_timings', [])
            safe_set_session_value('guidance_usage', [])

            # Reset error tracking
            safe_set_session_value('session_errors', [])
            safe_set_session_value('api_status', 'unknown')

            logger.info("Experimental session reset successfully")

        except Exception as e:
            logger.error(f"Error resetting experimental session: {e}")

    def log_error(self, error_type: str, error_message: str):
        """
        Log errors for research analysis.

        Academic Purpose: Track system issues for data quality assessment.
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'session_id': self.get_session_id(),
            'current_stage': safe_get_session_value('current_stage', 0),
            'interaction_flow': safe_get_session_value('interaction_flow', 'unknown')
        }

        current_errors = safe_get_session_value('session_errors', [])
        current_errors.append(error_entry)
        safe_set_session_value('session_errors', current_errors)

        logger.error(f"Session error logged: {error_type} - {error_message}")

    def get_session_id(self) -> str:
        """Get current session ID or generate new one."""
        if st.session_state.experimental_session:
            return st.session_state.experimental_session.session_id
        return self.generate_unique_session_id()

    def get_session_duration_minutes(self) -> float:
        """Calculate session duration in minutes."""
        session_start = safe_get_session_value('session_start_time', datetime.now())
        return (datetime.now() - session_start).total_seconds() / 60

    def get_experimental_condition(self) -> str:
        """Get current experimental condition code."""
        if st.session_state.experimental_session:
            return st.session_state.experimental_session.condition_code

        expertise = safe_get_session_value('user_expertise')
        assistance = safe_get_session_value('ai_assistance_enabled')

        expertise_str = expertise.value if expertise else 'unknown'
        assistance_str = str(assistance) if assistance is not None else 'unknown'

        return f"{expertise_str}_{assistance_str}_unknown"

    def validate_session_state(self) -> Dict[str, bool]:
        """
        Validate session state for research integrity.

        Returns validation flags for quality control.
        """
        validations = {
            'session_initialized': self.session_initialized,
            'has_experimental_session': st.session_state.experimental_session is not None,
            'valid_flow_state': safe_get_session_value('interaction_flow') in ['setup', 'scenario', 'completed'],
            'valid_stage': 0 <= safe_get_session_value('current_stage', 0) <= 3,
            'session_timing_valid': safe_get_session_value('session_start_time') is not None,
            'minimal_errors': len(safe_get_session_value('session_errors', [])) < 10
        }

        return validations

    @staticmethod
    def generate_unique_session_id() -> str:
        """
        Generate cryptographically unique session ID.

        Academic Purpose: Prevents session ID collisions in multi-user
        research environments while maintaining participant anonymity.
        """
        try:
            # Use UUID4 for cryptographic uniqueness
            unique_id = str(uuid.uuid4())

            # Add timestamp component for sorting
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Combine for final session ID
            session_id = f"session_{timestamp}_{unique_id[:8]}"

            logger.info(f"Generated unique session ID: {session_id}")
            return session_id

        except Exception as e:
            logger.error(f"Session ID generation failed: {e}")
            # Fallback to timestamp-based ID with random component
            import random
            fallback_id = f"session_{int(datetime.now().timestamp())}_{random.randint(1000, 9999)}"
            return fallback_id

    def auto_save_session_data(self, event_type: str, data: Dict[str, Any]):
        """
        Auto-save session data with enhanced error handling.

        Academic Purpose: Maintains detailed logs of user interactions
        for research analysis while providing session recovery.
        """
        try:
            os.makedirs(config.RESPONSES_DIR, exist_ok=True)

            # Generate unique autosave filename
            session_id = id(st.session_state)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            autosave_file = f"{config.RESPONSES_DIR}/autosave_{session_id}_{timestamp}.json"

            # Comprehensive session state capture
            session_snapshot = {
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,

                # Core experimental variables
                'user_expertise': safe_get_session_value('user_expertise').value if safe_get_session_value('user_expertise') else None,
                'ai_assistance_enabled': safe_get_session_value('ai_assistance_enabled'),
                'experimental_condition': self.get_experimental_condition(),

                # Progress tracking
                'current_stage': safe_get_session_value('current_stage', 0),
                'interaction_flow': safe_get_session_value('interaction_flow', 'setup'),
                'session_duration_minutes': self.get_session_duration_minutes(),

                # Error tracking
                'session_errors': safe_get_session_value('session_errors', []),
                'api_status': safe_get_session_value('api_status', 'unknown'),

                # Event data
                'event_data': data
            }

            # Safely export experimental session if available
            if st.session_state.experimental_session:
                try:
                    session_snapshot['experimental_session_data'] = st.session_state.experimental_session.export_for_analysis()
                except Exception as e:
                    logger.warning(f"Failed to export experimental session data: {e}")
                    session_snapshot['experimental_session_data'] = {'error': str(e)}

            # Write autosave file
            with open(autosave_file, 'w') as f:
                json.dump(enum_to_str(enum_to_str(session_snapshot)), f, indent=2)

            safe_set_session_value('last_auto_save', datetime.now())

        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
            self.log_error('auto_save_failure', str(e))

# =============================================================================
# SESSION RECOVERY SYSTEM
# =============================================================================

class SessionRecovery:
    """
    Session recovery and checkpoint management.

    Academic Purpose: Maintains experimental continuity by enabling
    session recovery within reasonable time windows.
    """

    @staticmethod
    def save_recovery_checkpoint(session_manager: SessionManager):
        """Save recovery checkpoint with enhanced data integrity."""

        try:
            if (safe_get_session_value('interaction_flow') == 'setup' or
                not safe_get_session_value('experimental_session')):
                return

            session_id = session_manager.generate_unique_session_id()
            recovery_dir = f"{config.RESPONSES_DIR}/recovery"
            os.makedirs(recovery_dir, exist_ok=True)

            recovery_data = {
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'recovery_version': '2.0',

                # Core experimental state
                'user_expertise': safe_get_session_value('user_expertise').value if safe_get_session_value('user_expertise') else None,
                'ai_assistance_enabled': safe_get_session_value('ai_assistance_enabled'),
                'current_stage': safe_get_session_value('current_stage', 0),
                'interaction_flow': safe_get_session_value('interaction_flow'),

                # Experimental session data
                'experimental_session_data': st.session_state.experimental_session.export_for_analysis() if st.session_state.experimental_session else None,

                # Legacy compatibility
                'stage_responses': safe_get_session_value('stage_responses', []),
                'stage_timings': [t.isoformat() if hasattr(t, 'isoformat') else str(t)
                                for t in safe_get_session_value('stage_timings', [])],
                'guidance_usage': safe_get_session_value('guidance_usage', []),
                'session_start_time': safe_get_session_value('session_start_time', datetime.now()).isoformat(),

                # Recovery metadata
                'recovery_point': f"stage_{safe_get_session_value('current_stage', 0)}",
                'progress_summary': f"{len(safe_get_session_value('stage_responses', []))} stages completed"
            }

            recovery_file = f"{recovery_dir}/{session_id}_recovery.json"
            with open(recovery_file, 'w') as f:
                json.dump(enum_to_str(enum_to_str(recovery_data)), f, indent=2)

            logger.info(f"Recovery checkpoint saved: {recovery_file}")

        except Exception as e:
            logger.error(f"Recovery save failed: {e}")

    @staticmethod
    def load_recovery_session() -> Optional[Dict[str, Any]]:
        """
        Load recovery session if available within time window.

        Academic Purpose: Enables continuation of interrupted experimental
        sessions while maintaining data quality and experimental validity.
        """
        try:
            # For now, skip recovery to keep implementation simple
            # Can be enhanced later if needed
            return None

        except Exception as e:
            logger.error(f"Recovery load failed: {e}")
            return None

    @staticmethod
    def restore_session(recovery_data: Dict[str, Any], session_manager: SessionManager) -> bool:
        """
        Restore experimental session from recovery data.

        Academic Purpose: Maintains experimental continuity by restoring
        complete session state while preserving data integrity.
        """
        try:
            # Implementation would restore session state from recovery data
            # Skipping for now to keep implementation manageable
            return False

        except Exception as e:
            logger.error(f"Session restoration failed: {e}")
            return False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_session_summary() -> Dict[str, Any]:
    """
    Get comprehensive session summary for debugging and validation.

    Returns summary of current session state and experimental progress.
    """

    experimental_session = safe_get_session_value('experimental_session')

    summary = {
        'session_metadata': {
            'session_id': experimental_session.session_id if experimental_session else 'none',
            'interaction_flow': safe_get_session_value('interaction_flow', 'setup'),
            'current_stage': safe_get_session_value('current_stage', 0),
            'session_duration_minutes': (datetime.now() - safe_get_session_value('session_start_time', datetime.now())).total_seconds() / 60
        },
        'experimental_condition': {
            'user_expertise': safe_get_session_value('user_expertise').value if safe_get_session_value('user_expertise') else None,
            'ai_assistance_enabled': safe_get_session_value('ai_assistance_enabled'),
            'bias_type': experimental_session.bias_type.value if experimental_session else None,
            'domain': experimental_session.domain.value if experimental_session else None
        },
        'progress_tracking': {
            'stages_completed': len(safe_get_session_value('stage_responses', [])),
            'guidance_requests': sum(safe_get_session_value('guidance_usage', [])),
            'session_errors': len(safe_get_session_value('session_errors', [])),
            'api_status': safe_get_session_value('api_status', 'unknown')
        },
        'data_quality': {
            'session_initialized': True,
            'has_experimental_session': experimental_session is not None,
            'session_valid': experimental_session.is_completed if experimental_session else False,
            'minimal_errors': len(safe_get_session_value('session_errors', [])) < 5
        }
    }

    return summary
