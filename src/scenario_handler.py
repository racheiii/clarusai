"""
ClārusAI: Scenario Loading and Selection System
UCL Master's Dissertation: "Building AI Literacy Through Simulation"

src/scenario_handler.py - Scenario management and balanced selection

Purpose:
Handles CSV loading, validation, and stratified scenario selection
for maintaining experimental validity in 2×2×3 factorial design.

Author: Rachel Seah
Date: July 2025
"""

import pandas as pd
import random
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

import config
from src.models import UserExpertise, create_experimental_session, ExperimentalSession

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# SCENARIO HANDLER CLASS
# =============================================================================

class ScenarioHandler:
    """
    Comprehensive scenario management for experimental research.
    
    Academic Purpose: Ensures experimental integrity through validated
    scenario loading and balanced selection for 2×2×3 factorial design.
    """
    
    def __init__(self):
        """Initialize scenario handler."""
        self.scenarios_df = None
        self.loaded = False
    
    def load_scenarios(self) -> Optional[pd.DataFrame]:
        """
        Load and validate cognitive bias scenarios from CSV.
        
        Academic Purpose: Ensures experimental integrity by validating
        all required columns and data quality for research protocol.
        
        Returns:
            pandas.DataFrame: Validated scenario database
            None: If file missing or validation fails
        """
        
        if self.loaded and self.scenarios_df is not None:
            return self.scenarios_df
        
        try:
            # Check file existence
            if not os.path.exists(config.SCENARIOS_FILE):
                logger.error(f"Scenarios file not found: {config.SCENARIOS_FILE}")
                return None
            
            # Load CSV data
            self.scenarios_df = pd.read_csv(config.SCENARIOS_FILE)
            
            # Validate file structure
            validation_result = self._validate_scenarios_structure()
            if not validation_result['valid']:
                logger.error(f"Scenario validation failed: {validation_result['errors']}")
                return None
            
            # Validate data content
            content_validation = self._validate_scenarios_content()
            if not content_validation['valid']:
                logger.warning(f"Scenario content issues: {content_validation['warnings']}")
            
            self.loaded = True
            logger.info(f"Successfully loaded {len(self.scenarios_df)} scenarios")
            
            return self.scenarios_df
            
        except pd.errors.EmptyDataError:
            logger.error("Scenarios file is empty or corrupted")
            return None
        except Exception as e:
            logger.error(f"Error loading scenarios database: {e}")
            return None
    
    def _validate_scenarios_structure(self) -> Dict[str, Any]:
        """Validate required CSV structure."""
        
        required_columns = config.VALIDATION_RULES["required_scenario_columns"]
        if self.scenarios_df is None:
            return {
                'valid': False,
                'errors': "Scenarios DataFrame is None"
            }
        missing_columns = [col for col in required_columns if col not in self.scenarios_df.columns]
        
        if missing_columns:
            return {
                'valid': False,
                'errors': f"Missing required columns: {missing_columns}"
            }
        
        if self.scenarios_df.empty:
            return {
                'valid': False,
                'errors': "Scenarios file is empty"
            }
        
        return {'valid': True, 'errors': None}
    
    def _validate_scenarios_content(self) -> Dict[str, Any]:
        """Validate scenario content quality."""
        
        warnings = []
        
        # Validate bias types
        valid_bias_types = set(config.VALIDATION_RULES["valid_bias_types"])
        if self.scenarios_df is None or 'bias_type' not in self.scenarios_df.columns:
            warnings.append("Scenarios DataFrame is None or missing 'bias_type' column")
            return {'valid': False, 'warnings': warnings}
        invalid_bias_types = set(self.scenarios_df['bias_type']) - valid_bias_types
        if invalid_bias_types:
            warnings.append(f"Unknown bias types: {invalid_bias_types}")
        
        # Validate domains
        valid_domains = set(config.VALIDATION_RULES["valid_domains"])
        invalid_domains = set(self.scenarios_df['domain']) - valid_domains
        if invalid_domains:
            warnings.append(f"Unknown domains: {invalid_domains}")
        
        # Check for missing text content
        text_columns = ['scenario_text', 'primary_prompt', 'follow_up_1', 'follow_up_2', 'follow_up_3']
        for col in text_columns:
            if col in self.scenarios_df.columns:
                missing_text = self.scenarios_df[col].isna().sum()
                if missing_text > 0:
                    warnings.append(f"Missing text in {col}: {missing_text} rows")
        
        return {
            'valid': len(warnings) == 0,
            'warnings': warnings
        }
    
    def select_balanced_scenario(self, user_expertise: UserExpertise, ai_assistance: bool) -> Optional[pd.Series]:
        """
        Implement stratified scenario selection for balanced experimental design.
        
        Academic Purpose: Ensures balanced distribution across experimental
        conditions while maintaining randomization and research validity.
        
        Args:
            user_expertise: UserExpertise enum (Factor 1)
            ai_assistance: Boolean (Factor 2)
        
        Returns:
            pandas.Series: Selected scenario (Factor 3 determined)
            None: If selection fails
        """
        
        if self.scenarios_df is None or self.scenarios_df.empty:
            logger.error("Cannot select scenario - no scenarios available")
            return None
        
        try:
            # Start with all available scenarios
            available_scenarios = self.scenarios_df.copy()
            
            # Apply cognitive load weighting based on expertise
            if user_expertise == UserExpertise.NOVICE:
                # Prefer medium cognitive load for novice users
                medium_load_scenarios = available_scenarios[
                    available_scenarios['cognitive_load_level'] == 'Medium'
                ]
                if not medium_load_scenarios.empty and random.random() < 0.7:
                    available_scenarios = medium_load_scenarios
                    logger.info("Applied medium cognitive load weighting for novice user")
            
            # Ensure we have scenarios to select from
            if available_scenarios.empty:
                logger.warning("No scenarios match selection criteria, using all available")
                available_scenarios = self.scenarios_df.copy()
            
            # Apply AI assistance appropriateness filter
            if ai_assistance:
                # Prefer scenarios marked as helpful for AI assistance
                helpful_scenarios = available_scenarios[
                    available_scenarios['ai_appropriateness'] == 'helpful'
                ]
                if not helpful_scenarios.empty and random.random() < 0.8:
                    available_scenarios = helpful_scenarios
                    logger.info("Applied AI-helpful scenario weighting")
            
            # Randomize final selection from weighted pool
            selected_scenario = available_scenarios.sample(frac=1).reset_index(drop=True).iloc[0]
            
            # Log selection for research tracking
            selection_metadata = {
                'timestamp': datetime.now().isoformat(),
                'user_expertise': user_expertise.value,
                'ai_assistance': ai_assistance,
                'selected_scenario': selected_scenario['scenario_id'],
                'bias_type': selected_scenario['bias_type'],
                'domain': selected_scenario['domain'],
                'cognitive_load': selected_scenario['cognitive_load_level'],
                'ai_appropriateness': selected_scenario['ai_appropriateness'],
                'selection_method': 'weighted_random',
                'available_pool_size': len(available_scenarios),
                'total_scenarios': len(self.scenarios_df)
            }
            
            logger.info(f"Selected scenario {selected_scenario['scenario_id']} for condition {user_expertise.value}_{ai_assistance}")
            
            return selected_scenario
            
        except Exception as e:
            logger.error(f"Scenario selection failed: {e}")
            return None
    
    def create_experimental_session(self, session_id: str, user_expertise: UserExpertise, 
                                   ai_assistance: bool, selected_scenario: pd.Series) -> Optional['ExperimentalSession']:
        """
        Create experimental session from selected scenario.
        
        Academic Purpose: Converts scenario selection into properly
        structured experimental session for research tracking.
        
        Returns:
            ExperimentalSession: Configured session object
            None: If creation fails
        """
        
        try:
            # Convert pandas Series to dictionary
            scenario_data = selected_scenario.to_dict()
            
            # Create experimental session using models
            experimental_session = create_experimental_session(
                session_id=session_id,
                user_expertise=user_expertise,
                ai_assistance_enabled=ai_assistance,
                scenario_data=scenario_data
            )
            
            logger.info(f"Created experimental session: {session_id}")
            return experimental_session
            
        except Exception as e:
            logger.error(f"Failed to create experimental session: {e}")
            return None
    
    def get_scenario_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive scenario database statistics.
        
        Academic Purpose: Provides overview of experimental design
        coverage and scenario distribution for research validation.
        """
        
        if self.scenarios_df is None:
            return {'error': 'No scenarios loaded'}
        
        try:
            stats = {
                'total_scenarios': len(self.scenarios_df),
                'bias_type_distribution': self.scenarios_df['bias_type'].value_counts().to_dict(),
                'domain_distribution': self.scenarios_df['domain'].value_counts().to_dict(),
                'cognitive_load_distribution': self.scenarios_df['cognitive_load_level'].value_counts().to_dict(),
                'ai_appropriateness_distribution': self.scenarios_df['ai_appropriateness'].value_counts().to_dict(),
                'experimental_coverage': self._calculate_experimental_coverage()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to calculate scenario statistics: {e}")
            return {'error': str(e)}
    
    def _calculate_experimental_coverage(self) -> Dict[str, Any]:
        """Calculate 2×2×3 factorial design coverage."""
        
        if self.scenarios_df is None:
            logger.error("No scenarios loaded for experimental coverage calculation")
            return {'error': 'No scenarios loaded'}
        try:
            # Calculate theoretical and actual combinations
            bias_types = self.scenarios_df['bias_type'].nunique()
            domains = self.scenarios_df['domain'].nunique()
            cognitive_loads = self.scenarios_df['cognitive_load_level'].nunique()
            
            # Calculate coverage for each experimental factor
            coverage = {
                'bias_types_covered': bias_types,
                'domains_covered': domains,
                'cognitive_loads_covered': cognitive_loads,
                'total_combinations': bias_types * domains * cognitive_loads,
                'actual_scenarios': len(self.scenarios_df),
                'coverage_ratio': len(self.scenarios_df) / (bias_types * domains * cognitive_loads) if bias_types * domains * cognitive_loads > 0 else 0
            }
            
            # Check for balanced distribution
            coverage['balanced_bias_distribution'] = all(
                count >= 1 for count in self.scenarios_df['bias_type'].value_counts()
            )
            coverage['balanced_domain_distribution'] = all(
                count >= 1 for count in self.scenarios_df['domain'].value_counts()
            )
            
            return coverage
            
        except Exception as e:
            logger.error(f"Failed to calculate experimental coverage: {e}")
            return {'error': str(e)}
    
    def validate_scenario_content(self, scenario_id: str) -> Dict[str, Any]:
        """
        Validate specific scenario content quality.
        
        Academic Purpose: Ensures individual scenario meets research
        standards for experimental validity.
        """
        
        if self.scenarios_df is None:
            return {'valid': False, 'error': 'No scenarios loaded'}
        
        try:
            scenario_row = self.scenarios_df[self.scenarios_df['scenario_id'] == scenario_id]
            
            if scenario_row.empty:
                return {'valid': False, 'error': f'Scenario {scenario_id} not found'}
            
            scenario = scenario_row.iloc[0]
            
            validation_checks = {
                'has_scenario_text': pd.notna(scenario['scenario_text']) and len(scenario['scenario_text'].strip()) > 50,
                'has_all_prompts': all(
                    pd.notna(scenario[col]) and len(scenario[col].strip()) > 10
                    for col in ['primary_prompt', 'follow_up_1', 'follow_up_2', 'follow_up_3']
                ),
                'has_ideal_answers': all(
                    pd.notna(scenario[col]) and len(scenario[col].strip()) > 20
                    for col in ['ideal_primary_answer', 'ideal_answer_1', 'ideal_answer_2', 'ideal_answer_3']
                ),
                'valid_bias_type': scenario['bias_type'] in config.VALIDATION_RULES["valid_bias_types"],
                'valid_domain': scenario['domain'] in config.VALIDATION_RULES["valid_domains"],
                'has_learning_objective': pd.notna(scenario['bias_learning_objective']) and len(scenario['bias_learning_objective'].strip()) > 10
            }
            
            all_valid = all(validation_checks.values())
            
            return {
                'valid': all_valid,
                'scenario_id': scenario_id,
                'validation_checks': validation_checks,
                'summary': f"Scenario {scenario_id} {'passes' if all_valid else 'fails'} validation"
            }
            
        except Exception as e:
            logger.error(f"Scenario validation failed for {scenario_id}: {e}")
            return {'valid': False, 'error': str(e)}
    
    def get_available_scenarios_for_condition(self, user_expertise: UserExpertise, ai_assistance: bool) -> Dict[str, Any]:
        """
        Get scenarios available for specific experimental condition.
        
        Academic Purpose: Preview scenario pool for experimental condition
        to ensure adequate coverage and balance.
        """
        
        if self.scenarios_df is None:
            return {'available': [], 'count': 0}
        
        try:
            # Apply same filtering logic as scenario selection
            available_scenarios = self.scenarios_df.copy()
            
            # Apply cognitive load weighting for novice users
            if user_expertise == UserExpertise.NOVICE:
                medium_load_scenarios = available_scenarios[
                    available_scenarios['cognitive_load_level'] == 'Medium'
                ]
                if not medium_load_scenarios.empty:
                    available_scenarios = medium_load_scenarios
            
            # Apply AI assistance weighting
            if ai_assistance:
                helpful_scenarios = available_scenarios[
                    available_scenarios['ai_appropriateness'] == 'helpful'
                ]
                if not helpful_scenarios.empty:
                    available_scenarios = helpful_scenarios
            
            scenario_info = []
            for _, scenario in available_scenarios.iterrows():
                scenario_info.append({
                    'scenario_id': scenario['scenario_id'],
                    'title': scenario['title'],
                    'bias_type': scenario['bias_type'],
                    'domain': scenario['domain'],
                    'cognitive_load': scenario['cognitive_load_level']
                })
            
            return {
                'available': scenario_info,
                'count': len(available_scenarios),
                'condition': f"{user_expertise.value}_{ai_assistance}",
                'total_pool': len(self.scenarios_df)
            }
            
        except Exception as e:
            logger.error(f"Failed to get available scenarios: {e}")
            return {'available': [], 'count': 0, 'error': str(e)}