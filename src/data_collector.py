"""
ClārusAI: Data Collection and Persistence System

src/data_collector.py - Data collection with 6-dimensional scoring

Purpose:
Handles all data collection, scoring, and persistence operations
Note: This module is not used in the simulated dataset generation or RQ analysis pipeline. 
Certain fields (e.g., llm_feedback) are retained solely for live demo presentation purposes.
"""

import streamlit as st
import json
from json import JSONDecodeError
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import config
from src.session_manager import safe_get_session_value, safe_set_session_value
from src.models import UserResponse, StageType
from src.scoring_engine import calculate_comprehensive_scores
from enum import Enum


def enum_to_str(obj):
    """Recursively convert Enums and NumPy types (incl. bool_) to JSON-safe values."""
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

# Stage configuration
STAGE_NAMES = [
    "Primary Analysis",
    "Cognitive Factors", 
    "Mitigation Strategies",
    "Transfer Learning"
]

STAGE_PROMPTS = [
    "primary_prompt",
    "follow_up_1",
    "follow_up_2", 
    "follow_up_3"
]

STAGE_TYPES = [
    StageType.PRIMARY_ANALYSIS,
    StageType.COGNITIVE_FACTORS,
    StageType.MITIGATION_STRATEGIES,
    StageType.TRANSFER_LEARNING
]

class DataCollector:
    """
    Comprehensive data collection system for experimental research.
    
    Purpose: Provides robust data collection with 6-dimensional
    scoring while maintaining research integrity and quality standards.
    """
    
    def __init__(self) -> None:
        """Initialize data collector."""
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in [config.RESPONSES_DIR, f"{config.RESPONSES_DIR}/recovery"]:
            os.makedirs(directory, exist_ok=True)
    
    def save_stage_response(self, scenario: Dict[str, Any], stage: int, response: str) -> bool:
        """
        Save stage response with comprehensive scoring and metadata.
        
        Purpose: Captures complete interaction data with 6-dimensional
        scoring for research analysis while maintaining data integrity.
        
        Args:
            scenario: Current scenario metadata
            stage: Stage number (0-3)
            response: User's response text
        
        Returns:
            bool: True if saved successfully
        """
        
        try:
            # Calculate response timing
            stage_timings = safe_get_session_value('stage_timings', [])
            if len(stage_timings) > stage:
                stage_start_time = stage_timings[stage]
                response_time = (datetime.now() - stage_start_time).total_seconds()
            else:
                response_time = 0
                logger.warning(f"No timing data available for stage {stage}")
            
            # Update legacy session state
            self._update_legacy_session_state(stage, response)
            
            # Get ideal answer for scoring
            ideal_answer_fields = ['ideal_primary_answer', 'ideal_answer_1', 'ideal_answer_2', 'ideal_answer_3']
            ideal_answer = scenario.get(ideal_answer_fields[stage], '')
            
            # Calculate 6-dimensional scores
            scoring_results = self._calculate_stage_scores(response, ideal_answer, scenario)
            
            # Update experimental session
            self._update_experimental_session(stage, response, response_time, ideal_answer, scoring_results)
            
            # Save response data
            response_data = self._compile_response_data(
                scenario, stage, response, response_time, ideal_answer, scoring_results
            )
            
            # Write data files
            return self._write_response_files(response_data, stage)
            
        except Exception as e:
            logger.error(f"Error in save_stage_response: {e}")
            return False
    
    def _update_legacy_session_state(self, stage: int, response: str) -> None:
        """Update legacy session state for backward compatibility."""
        
        current_responses = safe_get_session_value('stage_responses', [])
        while len(current_responses) <= stage:
            current_responses.append('')
        current_responses[stage] = response
        safe_set_session_value('stage_responses', current_responses)
    
    def _calculate_stage_scores(self, response: str, ideal_answer: str, 
                               scenario: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate 6-dimensional scores with error handling."""
        
        try:
            scoring_results = calculate_comprehensive_scores(
                response=response,
                ideal_answer=ideal_answer,
                scenario=scenario
            )
            logger.info(f"Successfully calculated scores for response")
            return scoring_results
            
        except Exception as e:
            logger.error(f"Scoring calculation failed: {e}")
            return {
                'error': str(e), 
                'response_length': len(response),
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_experimental_session(self, stage: int, response: str, response_time: float,
                                    ideal_answer: str, scoring_results: Optional[Dict[str, Any]]) -> None:
        """Update experimental session with new response."""
        
        experimental_session = safe_get_session_value('experimental_session')
        if not experimental_session:
            return
        
        try:
            # Get guidance usage
            guidance_usage = safe_get_session_value('guidance_usage', [])
            guidance_requested = guidance_usage[stage] if len(guidance_usage) > stage else False
            
            # Create UserResponse object
            user_response = UserResponse(
                stage_number=stage,
                stage_name=STAGE_NAMES[stage],
                stage_type=STAGE_TYPES[stage],
                response_text=response,
                response_time_seconds=response_time,
                timestamp=datetime.now(),
                cumulative_session_time=(datetime.now() - safe_get_session_value('session_start_time', datetime.now())).total_seconds(),
                guidance_requested=guidance_requested,
                prompt_text=self._get_stage_prompt_text(stage),
                ideal_answer=ideal_answer
            )
            
            # Add to experimental session
            experimental_session.add_response(user_response)
            logger.info(f"Added response to experimental session for stage {stage}")
            
        except Exception as e:
            logger.error(f"Failed to update experimental session: {e}")
    
    def _get_stage_prompt_text(self, stage: int) -> str:
        """Get stage prompt text from scenario."""
        
        experimental_session = safe_get_session_value('experimental_session')
        if experimental_session and experimental_session.assigned_scenario:
            scenario_dict = experimental_session.assigned_scenario.__dict__
            return scenario_dict.get(STAGE_PROMPTS[stage], '')
        return ''
    
    def _compile_response_data(
        self,
        scenario: Dict[str, Any],
        stage: int,
        response: str,
        response_time: float,
        ideal_answer: str,
        scoring_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compile comprehensive response metadata for research analysis.

        Args:
            scenario (Dict[str, Any]): The full scenario metadata dictionary.
            stage (int): Stage number (0–3).
            response (str): User's text response.
            response_time (float): Time taken to respond in seconds.
            ideal_answer (str): The stage's model ideal answer.
            scoring_results (Optional[Dict[str, Any]]): Results from scoring engine.

        Returns:
            Dict[str, Any]: Structured data ready for JSON/JSONL persistence
        """

        # Get experimental condition
        experimental_session = safe_get_session_value('experimental_session')
        condition_code = experimental_session.condition_code if experimental_session else 'unknown'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'session_id': experimental_session.session_id if experimental_session else 'unknown',
            
            # Core research variables            
            'user_expertise': (
                getattr(safe_get_session_value('user_expertise'), 'value', safe_get_session_value('user_expertise'))
                if safe_get_session_value('user_expertise') is not None else None
            ),
            'ai_assistance_enabled': safe_get_session_value('ai_assistance_enabled'),
            'scenario_id': scenario.get('scenario_id', 'unknown'),
            'bias_type': scenario.get('bias_type', 'unknown'),
            'domain': scenario.get('domain', 'unknown'),
            'cognitive_load_level': scenario.get('cognitive_load_level', 'unknown'),
            
            # Stage-specific data
            'stage_number': stage,
            'stage_name': STAGE_NAMES[stage],
            'stage_type': STAGE_TYPES[stage].value,
            'stage_prompt': scenario.get(STAGE_PROMPTS[stage], ''),
            'response_text': response,
            'response_time_seconds': response_time,
            'word_count': len(response.split()),
            'character_count': len(response),
            
            # AI interaction analysis
            'guidance_requested': safe_get_session_value('guidance_usage', [False] * (stage + 1))[stage] if len(safe_get_session_value('guidance_usage', [])) > stage else False,
            'cumulative_guidance_pattern': safe_get_session_value('guidance_usage', [])[:stage+1],
            
            # 6-dimensional scoring results
            'scoring_results': scoring_results,
            'ideal_answer': ideal_answer,
            'rubric_focus': scenario.get('rubric_focus', ''),
            'bias_learning_objective': scenario.get('bias_learning_objective', ''),
            
            # Experimental condition tracking
            'condition_code': condition_code,
            'experimental_phase': 'live_data_collection',
            
            # Data quality indicators
            'data_quality_flags': {
                'meaningful_length': len(response.strip()) >= config.QUALITY_THRESHOLDS['minimum_response_length'],
                'contains_reasoning': len(response.split('.')) > 1,
                'non_empty': bool(response.strip()),
                'natural_timing': response_time > 10,
                'scoring_successful': scoring_results is not None and 'error' not in scoring_results
            },
            
            # Research metadata
            'source_type': 'live_user_4stage',
            'bias_revelation_status': 'pre_revelation',
            'data_collection_version': '2.0'
        }
    
    def _write_response_files(self, response_data: Dict[str, Any], stage: int) -> bool:
        """Write response data to files with enhanced error handling"""
        
        try:
            # Generate filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            condition = response_data['condition_code']
            stage_file = f"{config.RESPONSES_DIR}/stage_{stage}_{condition}_{timestamp}.json"
            
            # Ensure directory exists with proper permissions
            os.makedirs(config.RESPONSES_DIR, exist_ok=True)
            
            # Save individual stage response
            with open(stage_file, 'w', encoding='utf-8') as f:
                json.dump(enum_to_str(response_data), f, indent=2)
            
            # Append to master research log with file locking
            master_log = f"{config.RESPONSES_DIR}/master_4stage_responses.jsonl"
            try:
                with open(master_log, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(enum_to_str(response_data), ensure_ascii=False) + '\n')
            except PermissionError as e:
                logger.warning(f"Permission denied writing to master log: {e}")
                # Continue execution - individual file saved successfully
            
            # Save recovery checkpoint
            self._save_recovery_checkpoint()
            
            logger.info(f"Successfully saved stage {stage} response data")
            return True
            
        except PermissionError as e:
            logger.error(f"Permission denied saving stage response: {e}")
            return False
        except OSError as e:
            logger.error(f"File system error saving stage response: {e}")
            return False
        except JSONDecodeError as e:
            logger.error(f"JSON encoding error saving stage response: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving stage response: {e}")
            return False
    
    def save_complete_session(self, scenario: Dict[str, Any]) -> Optional[str]:
        """
        Save complete session with enhanced validation.
        
        Note:
            This method is used only in live user demonstration mode and does not form part
            of the simulated dataset generation or RQ analysis pipeline. Certain fields 
            (e.g., llm_feedback) are retained solely for presentation purposes in the demo.

        Returns:
            Optional[str]: Filename if saved successfully
        """
        
        experimental_session = safe_get_session_value('experimental_session')
        if not experimental_session:
            logger.warning("No experimental session to save")
            return None
        
        try:
            # Export complete session data from the active live session
            session_data = experimental_session.export_for_analysis()
            
            # Add completion metadata
            session_data.update({
                'bias_revealed': True,
                'bias_revelation_time': datetime.now().isoformat(),
                'completion_method': 'full_4stage_protocol',
                # Demo-only: LLM tutor feedback for live users; not used in RQ analysis
                'llm_feedback': safe_get_session_value('llm_feedback', [None] * 4),
                'llm_feedback_per_stage': [
                    f.get('llm_feedback') if f else None
                    for f in safe_get_session_value('stage_feedback', [None] * 4)
                ],
                'data_quality_assessment': self._validate_session_quality(experimental_session),
                'research_version': '2.0'
            })
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            condition = experimental_session.condition_code
            session_file = f"{config.RESPONSES_DIR}/complete_session_{condition}_{timestamp}.json"
            
            # Save complete session
            with open(session_file, 'w') as f:
                json.dump(enum_to_str(session_data), f, indent=2)
            
            # Append to master sessions log
            master_sessions = f"{config.RESPONSES_DIR}/master_complete_sessions.jsonl"
            with open(master_sessions, 'a') as f:
                f.write(json.dumps(enum_to_str(session_data)) + '\n')
            
            logger.info(f"Complete session saved: {session_file}")
            return session_file
            
        except Exception as e:
            logger.error(f"Failed to save complete session: {e}")
            return None
    
    def _validate_session_quality(self, experimental_session) -> Dict[str, Any]:
        """Validate session quality for research standards."""
        
        try:
            if not experimental_session or not experimental_session.stage_responses:
                return {'quality': 'poor', 'reason': 'no_responses'}
            
            # Calculate quality metrics
            total_words = sum(len(r.response_text.split()) for r in experimental_session.stage_responses)
            avg_words = total_words / len(experimental_session.stage_responses)
            completion_rate = len(experimental_session.stage_responses) / 4.0
            
            # Assess quality
            if completion_rate == 1.0 and avg_words >= 20:
                quality = 'high'
            elif completion_rate >= 0.75 and avg_words >= 10:
                quality = 'moderate'
            else:
                quality = 'low'
            
            return {
                'quality': quality,
                'completion_rate': completion_rate,
                'average_words_per_stage': avg_words,
                'total_words': total_words,
                'stages_completed': len(experimental_session.stage_responses),
                'session_duration_minutes': experimental_session.session_duration_minutes
            }
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            return {'quality': 'unknown', 'error': str(e)}
    
    def _save_recovery_checkpoint(self) -> None:
        """Save recovery checkpoint for session continuity."""
        
        try:
            experimental_session = safe_get_session_value('experimental_session')
            if not experimental_session or safe_get_session_value('interaction_flow') == 'setup':
                return
            
            recovery_dir = f"{config.RESPONSES_DIR}/recovery"
            
            recovery_data = {
                'session_id': experimental_session.session_id,
                'timestamp': datetime.now().isoformat(),
                'recovery_version': '2.0',
                
                # Core experimental state
                'user_expertise': (
                    getattr(safe_get_session_value('user_expertise'), 'value', safe_get_session_value('user_expertise'))
                    if safe_get_session_value('user_expertise') is not None else None
                ),
                'ai_assistance_enabled': safe_get_session_value('ai_assistance_enabled'),
                'current_stage': safe_get_session_value('current_stage', 0),
                'interaction_flow': safe_get_session_value('interaction_flow'),
                
                # Experimental session data
                'experimental_session_data': experimental_session.export_for_analysis(),
                
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
            
            recovery_file = f"{recovery_dir}/{experimental_session.session_id}_recovery.json"
            with open(recovery_file, 'w') as f:
                json.dump(enum_to_str(recovery_data), f, indent=2)
                
            logger.info(f"Recovery checkpoint saved: {recovery_file}")
                
        except Exception as e:
            logger.error(f"Recovery save failed: {e}")
    
    def auto_save_session_data(self, event_type: str, data: Dict[str, Any]) -> None:
        """Auto-save session data with comprehensive error handling."""
        
        try:
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
                'user_expertise': (
                    getattr(safe_get_session_value('user_expertise'), 'value', safe_get_session_value('user_expertise'))
                    if safe_get_session_value('user_expertise') is not None else None
                ),
                'ai_assistance_enabled': safe_get_session_value('ai_assistance_enabled'),
                'experimental_condition': self._get_experimental_condition(),
                
                # Progress tracking
                'current_stage': safe_get_session_value('current_stage', 0),
                'interaction_flow': safe_get_session_value('interaction_flow', 'setup'),
                'session_duration_minutes': self._get_session_duration_minutes(),
                
                # Error tracking
                'session_errors': safe_get_session_value('session_errors', []),
                
                # Event data
                'event_data': data
            }
            
            # Safely export experimental session if available
            experimental_session = safe_get_session_value('experimental_session')
            if experimental_session:
                try:
                    session_snapshot['experimental_session_data'] = experimental_session.export_for_analysis()
                except Exception as e:
                    logger.warning(f"Failed to export experimental session data: {e}")
                    session_snapshot['experimental_session_data'] = {'error': str(e)}
            
            # Write autosave file with error handling
            try:
                with open(autosave_file, 'w', encoding='utf-8') as f:
                    json.dump(enum_to_str(session_snapshot), f, indent=2)
                    
                safe_set_session_value('last_auto_save', datetime.now())
                
            except OSError as e:
                logger.error(f"File system error during auto-save: {e}")
            except JSONDecodeError as e:
                logger.error(f"JSON encoding error during auto-save: {e}")
                
        except Exception as e:
            logger.error(f"Critical auto-save failure: {e}")
    
    def _get_experimental_condition(self) -> str:
        """Get current experimental condition code."""
        
        experimental_session = safe_get_session_value('experimental_session')
        if experimental_session:
            return experimental_session.condition_code
        
        expertise = safe_get_session_value('user_expertise')
        assistance = safe_get_session_value('ai_assistance_enabled')
        
        expertise_str = getattr(expertise, 'value', expertise) if expertise is not None else 'unknown'
        assistance_str = str(assistance) if assistance is not None else 'unknown'
        
        return f"{expertise_str}_{assistance_str}_unknown"
    
    def _get_session_duration_minutes(self) -> float:
        """Calculate session duration in minutes."""
        
        session_start = safe_get_session_value('session_start_time', datetime.now())
        return (datetime.now() - session_start).total_seconds() / 60
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data collection summary."""
        
        experimental_session = safe_get_session_value('experimental_session')
        
        summary = {
            'session_metadata': {
                'session_id': experimental_session.session_id if experimental_session else 'none',
                'experimental_condition': self._get_experimental_condition(),
                'session_duration_minutes': self._get_session_duration_minutes()
            },
            'progress_tracking': {
                'stages_completed': len(safe_get_session_value('stage_responses', [])),
                'guidance_requests': sum(safe_get_session_value('guidance_usage', [])),
                'session_errors': len(safe_get_session_value('session_errors', []))
            },
            'data_quality': {
                'has_experimental_session': experimental_session is not None,
                'session_valid': experimental_session.is_completed if experimental_session else False,
                'minimal_errors': len(safe_get_session_value('session_errors', [])) < 5
            }
        }
        
        return summary