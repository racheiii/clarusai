"""
ClārusAI: Data Models for Experimental Session Management

models.py - Data structures for 2×2×3 factorial design research

Purpose:
Provides type-safe data structures for experimental session management,
ensuring research integrity and facilitating statistical analysis of
AI-assisted cognitive bias training effectiveness.

Research Framework:
- ExperimentalSession: Complete session data for factorial analysis
- UserResponse: Individual stage response with comprehensive metadata
- ScoringResults: 6-dimensional assessment outcomes
- SessionAnalytics: Temporal and behavioral analysis data
"""



from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

# =============================================================================
# EXPERIMENTAL DESIGN ENUMERATIONS
# =============================================================================

class UserExpertise(Enum):
    """User expertise levels for factorial design Factor 1"""
    NOVICE = "novice"
    EXPERT = "expert"

class BiasType(Enum):
    """Cognitive bias types for factorial design Factor 3"""
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING_BIAS = "anchoring_bias" 
    AVAILABILITY_HEURISTIC = "availability_heuristic"

class Domain(Enum):
    """Professional domains for scenario contexts"""
    MEDICAL = "medical"
    MILITARY = "military"
    EMERGENCY = "emergency"

class StageType(Enum):
    """4-stage progressive interaction framework"""
    PRIMARY_ANALYSIS = "primary_analysis"
    COGNITIVE_FACTORS = "cognitive_factors"
    MITIGATION_STRATEGIES = "mitigation_strategies"
    TRANSFER_LEARNING = "transfer_learning"

# =============================================================================
# CSV FORMAT MAPPING - HANDLES SCENARIOS.CSV VALUES
# =============================================================================

# CSV Format Mapping - converts CSV values to enum values
CSV_TO_ENUM_MAPPING = {
    # Bias type mapping (CSV format → Enum)
    "Confirmation": BiasType.CONFIRMATION_BIAS,
    "Anchoring": BiasType.ANCHORING_BIAS,
    "Availability": BiasType.AVAILABILITY_HEURISTIC,
    
    # Domain mapping (CSV format → Enum)
    "Medical": Domain.MEDICAL,
    "Military": Domain.MILITARY,
    "Emergency": Domain.EMERGENCY
}

def map_csv_bias_type(csv_value: str) -> BiasType:
    """Convert CSV bias type to BiasType enum."""
    return CSV_TO_ENUM_MAPPING.get(csv_value, BiasType.CONFIRMATION_BIAS)

def map_csv_domain(csv_value: str) -> Domain:
    """Convert CSV domain to Domain enum."""
    return CSV_TO_ENUM_MAPPING.get(csv_value, Domain.MEDICAL)

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class ScoringResults:
    """
    6-dimensional assessment results for research analysis.
    
    Academic Purpose: Structured storage of all scoring dimensions
    for statistical analysis and educational feedback generation.
    """
    
    # Individual dimension scores
    semantic_similarity: float
    semantic_tag: str
    bias_recognition_count: int
    bias_recognition_tag: str
    originality_score: float
    originality_tag: str
    strategy_count: int
    strategy_tag: str
    transfer_count: int
    transfer_tag: str
    metacognition_count: int
    metacognition_tag: str
    
    # Composite scores
    overall_quality_score: float
    ai_dependency_risk: str
    learning_authenticity: str
    
    # Response analytics
    word_count: int
    character_count: int
    sentence_count: int
    response_complexity: str
    
    # Quality flags for research validation
    sufficient_length: bool
    demonstrates_reasoning: bool
    shows_engagement: bool
    potential_ai_dependency: bool
    excellent_performance: bool
    requires_review: bool
    
    # Assessment metadata
    assessment_timestamp: datetime
    scoring_version: str
    model_availability: bool

@dataclass
class UserResponse:
    """
    Individual stage response with comprehensive metadata for research analysis.
    
    Academic Purpose: Captures complete interaction data for each experimental
    stage, enabling longitudinal analysis of learning progression and AI dependency patterns.
    """
    
    # Core response data
    stage_number: int  # 0-3 for 4-stage progression
    stage_name: str    # From STAGE_NAMES constant
    stage_type: StageType
    response_text: str
    
    # Temporal analysis data
    response_time_seconds: float
    timestamp: datetime
    cumulative_session_time: float
    
    # Interaction metadata
    guidance_requested: bool
    guidance_text: Optional[str] = None
    prompt_text: str = ""
    ideal_answer: str = ""
    
    # Assessment results
    scores: Optional[ScoringResults] = None
    
    # Response analytics
    word_count: int = field(init=False)
    character_count: int = field(init=False)
    
    def __post_init__(self):
        """Calculate response analytics automatically"""
        self.word_count = len(self.response_text.split()) if self.response_text else 0
        self.character_count = len(self.response_text) if self.response_text else 0

@dataclass
class ScenarioMetadata:
    """
    Comprehensive scenario information for research tracking.
    
    Academic Purpose: Stores complete scenario context for experimental
    validity and enables bias-blind protocol maintenance.
    """
    
    scenario_id: str
    title: str
    bias_type: BiasType
    domain: Domain
    scenario_text: str
    
    # Stage prompts
    primary_prompt: str
    follow_up_1: str
    follow_up_2: str
    follow_up_3: str
    
    # Ideal answers for scoring
    ideal_primary_answer: str
    ideal_answer_1: str
    ideal_answer_2: str
    ideal_answer_3: str
    
    # Research metadata
    cognitive_load_level: str
    ai_appropriateness: str
    bias_learning_objective: str
    rubric_focus: str
    llm_feedback: Optional[str] = None
    source_citation: Optional[str] = None

@dataclass
class SessionAnalytics:
    """
    Comprehensive session analytics for research analysis.
    
    Academic Purpose: Provides temporal and behavioral analysis data
    for understanding learning patterns and AI interaction effects.
    """
    
    # Temporal metrics
    total_session_time_minutes: float
    average_stage_duration: float
    stage_durations: List[float]
    
    # Interaction patterns
    total_guidance_requests: int
    guidance_frequency: float  # Requests per stage
    guidance_usage_pattern: List[bool]  # Per-stage usage
    
    # Response progression
    total_word_count: int
    average_words_per_stage: float
    word_count_progression: List[int]
    
    # Quality metrics
    overall_engagement_score: float
    learning_progression_score: float
    consistency_score: float
    
    # Completion metrics
    all_stages_completed: bool
    completion_rate: float
    session_quality: str  # 'excellent', 'good', 'acceptable', 'poor'

@dataclass
class ExperimentalSession:
    """
    Complete experimental session for 2×2×3 factorial design research.
    
    Academic Purpose: Primary data structure for statistical analysis,
    containing all variables and measurements for hypothesis testing.
    
    Research Design:
    - Factor 1: user_expertise (novice vs expert)
    - Factor 2: ai_assistance_enabled (enabled vs disabled)  
    - Factor 3: bias_type (confirmation vs anchoring vs availability)
    """
    
    # Session identification
    session_id: str
    session_start_time: datetime

    # Experimental factors (2×2×3 design)
    user_expertise: UserExpertise
    ai_assistance_enabled: bool  # Factor 2
    assigned_scenario: ScenarioMetadata

    completion_time: Optional[datetime] = None

    # Response progression data
    stage_responses: List[UserResponse] = field(default_factory=list)
    current_stage: int = 0
    interaction_flow: str = "setup"  # setup, scenario, completed
    
    # Session analytics
    analytics: Optional[SessionAnalytics] = None
    
    # Recovery and quality control
    recovery_data: Dict[str, Any] = field(default_factory=dict)
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    
    # Research metadata
    experimental_protocol_version: str = "1.0"
    data_collection_method: str = "4stage_progressive_interaction"
    bias_revelation_timing: str = "post_completion"
    
    @property
    def condition_code(self) -> str:
        """Generate experimental condition identifier for analysis"""
        return f"{self.user_expertise.value}_{self.ai_assistance_enabled}_{self.assigned_scenario.bias_type.value}"
    
    @property
    def bias_type(self) -> BiasType:
        """Get bias type from assigned scenario"""
        return self.assigned_scenario.bias_type
    
    @property
    def domain(self) -> Domain:
        """Get domain from assigned scenario"""
        return self.assigned_scenario.domain
    
    @property
    def session_duration_minutes(self) -> float:
        """Calculate session duration in minutes"""
        end_time = self.completion_time or datetime.now()
        return (end_time - self.session_start_time).total_seconds() / 60
    
    @property
    def is_completed(self) -> bool:
        """Check if all 4 stages are completed"""
        return len(self.stage_responses) == 4 and self.completion_time is not None
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        return (len(self.stage_responses) / 4.0) * 100
    
    def add_response(self, response: UserResponse) -> None:
        """Add response and update session state"""
        self.stage_responses.append(response)
        self.current_stage = len(self.stage_responses)
        
        # Update quality flags
        self.update_quality_flags()
    
    def update_quality_flags(self) -> None:
        """Update session quality indicators for research validation"""
        if not self.stage_responses:
            return
        
        # Calculate quality metrics
        total_words = sum(r.word_count for r in self.stage_responses)
        avg_words = total_words / len(self.stage_responses) if self.stage_responses else 0
        guidance_usage = sum(1 for r in self.stage_responses if r.guidance_requested)
        
        self.quality_flags.update({
            'sufficient_engagement': avg_words >= 20,
            'demonstrates_progression': len(self.stage_responses) >= 2,
            'natural_timing': self.session_duration_minutes >= 5,
            'balanced_ai_usage': 0 < guidance_usage < len(self.stage_responses) if self.ai_assistance_enabled else True,
            'high_quality_responses': all(r.word_count >= 10 for r in self.stage_responses),
            'research_ready': self.is_completed and total_words >= 100
        })
    
    def calculate_analytics(self) -> SessionAnalytics:
        """Calculate comprehensive session analytics"""
        if not self.stage_responses:
            return SessionAnalytics(
                total_session_time_minutes=0,
                average_stage_duration=0,
                stage_durations=[],
                total_guidance_requests=0,
                guidance_frequency=0,
                guidance_usage_pattern=[],
                total_word_count=0,
                average_words_per_stage=0,
                word_count_progression=[],
                overall_engagement_score=0,
                learning_progression_score=0,
                consistency_score=0,
                all_stages_completed=False,
                completion_rate=0,
                session_quality='poor'
            )
        
        # Calculate temporal metrics
        stage_durations = [r.response_time_seconds for r in self.stage_responses]
        avg_duration = sum(stage_durations) / len(stage_durations)
        
        # Calculate interaction patterns
        guidance_requests = sum(1 for r in self.stage_responses if r.guidance_requested)
        guidance_pattern = [r.guidance_requested for r in self.stage_responses]
        
        # Calculate response progression
        word_counts = [r.word_count for r in self.stage_responses]
        total_words = sum(word_counts)
        avg_words = total_words / len(self.stage_responses)
        
        # Calculate quality scores
        engagement_score = min(1.0, avg_words / 50.0)  # Target: 50 words per stage
        progression_score = self.calculate_learning_progression()
        consistency_score = self.calculate_response_consistency()
        
        # Determine session quality
        quality = self.determine_session_quality(engagement_score, progression_score, consistency_score)
        
        analytics = SessionAnalytics(
            total_session_time_minutes=self.session_duration_minutes,
            average_stage_duration=avg_duration,
            stage_durations=stage_durations,
            total_guidance_requests=guidance_requests,
            guidance_frequency=guidance_requests / len(self.stage_responses),
            guidance_usage_pattern=guidance_pattern,
            total_word_count=total_words,
            average_words_per_stage=avg_words,
            word_count_progression=word_counts,
            overall_engagement_score=engagement_score,
            learning_progression_score=progression_score,
            consistency_score=consistency_score,
            all_stages_completed=self.is_completed,
            completion_rate=self.completion_percentage / 100,
            session_quality=quality
        )
        
        self.analytics = analytics
        return analytics
    
    def calculate_learning_progression(self) -> float:
        """Calculate learning progression score based on response development"""
        if len(self.stage_responses) < 2:
            return 0.0
        
        # Look for improvement in response quality over stages
        word_trend = 0
        for i in range(1, len(self.stage_responses)):
            if self.stage_responses[i].word_count > self.stage_responses[i-1].word_count:
                word_trend += 1
        
        # Calculate progression score
        progression = word_trend / (len(self.stage_responses) - 1) if len(self.stage_responses) > 1 else 0
        return min(1.0, progression)
    
    def calculate_response_consistency(self) -> float:
        """Calculate response consistency score"""
        if len(self.stage_responses) < 2:
            return 1.0
        
        word_counts = [r.word_count for r in self.stage_responses]
        mean_words = sum(word_counts) / len(word_counts)
        
        if mean_words == 0:
            return 0.0
        
        # Calculate coefficient of variation (lower = more consistent)
        variance = sum((x - mean_words) ** 2 for x in word_counts) / len(word_counts)
        std_dev = variance ** 0.5
        cv = std_dev / mean_words
        
        # Convert to consistency score (higher = more consistent)
        consistency = max(0.0, 1.0 - cv)
        return min(1.0, consistency)
    
    def determine_session_quality(self, engagement: float, progression: float, consistency: float) -> str:
        """Determine overall session quality for research validation"""
        composite_score = (engagement * 0.5 + progression * 0.3 + consistency * 0.2)
        
        if composite_score >= 0.8:
            return 'excellent'
        elif composite_score >= 0.6:
            return 'good'
        elif composite_score >= 0.4:
            return 'acceptable'
        else:
            return 'poor'
    
    def export_for_analysis(self) -> Dict[str, Any]:
        """Export session data for statistical analysis"""
        # Ensure analytics are calculated
        if self.analytics is None:
            self.calculate_analytics()
        
        return {
            'session_metadata': {
                'session_id': self.session_id,
                'condition_code': self.condition_code,
                'user_expertise': self.user_expertise.value,
                'ai_assistance_enabled': self.ai_assistance_enabled,
                'bias_type': self.bias_type.value,
                'domain': self.domain.value,
                'scenario_id': self.assigned_scenario.scenario_id,
                'session_start_time': self.session_start_time.isoformat(),
                'completion_time': self.completion_time.isoformat() if self.completion_time else None,
                'session_duration_minutes': self.session_duration_minutes,
                'is_completed': self.is_completed,
                'completion_percentage': self.completion_percentage
            },
            'response_data': [
                {
                    'stage_number': r.stage_number,
                    'stage_name': r.stage_name,
                    'response_text': r.response_text,
                    'word_count': r.word_count,
                    'character_count': r.character_count,
                    'response_time_seconds': r.response_time_seconds,
                    'guidance_requested': r.guidance_requested,
                    'scores': r.scores.__dict__ if r.scores else None
                }
                for r in self.stage_responses
            ],
            'session_analytics': self.analytics.__dict__ if self.analytics else None,
            'quality_flags': self.quality_flags,
            'experimental_metadata': {
                'protocol_version': self.experimental_protocol_version,
                'data_collection_method': self.data_collection_method,
                'bias_revelation_timing': self.bias_revelation_timing
            }
        }

# =============================================================================
# UTILITY FUNCTIONS FOR DATA MANAGEMENT
# =============================================================================

def create_experimental_session(
    session_id: str,
    user_expertise: Union[str, UserExpertise],
    ai_assistance_enabled: bool,
    scenario_data: Dict[str, Any]
) -> ExperimentalSession:
    """
    Factory function to create properly initialized experimental session.
    
    Academic Purpose: Ensures consistent session creation with proper
    type validation and experimental integrity.
    """
    
    # Convert string expertise to enum if needed
    if isinstance(user_expertise, str):
        user_expertise = UserExpertise(user_expertise)
    
    # FIXED: Map CSV values to enums using mapping functions
    bias_type_str = scenario_data['bias_type']
    domain_str = scenario_data['domain']
    
    # Create scenario metadata with proper enum mapping
    scenario = ScenarioMetadata(
        scenario_id=scenario_data['scenario_id'],
        title=scenario_data['title'],
        bias_type=map_csv_bias_type(bias_type_str),  # FIXED: Use mapping function
        domain=map_csv_domain(domain_str),          # FIXED: Use mapping function
        scenario_text=scenario_data['scenario_text'],
        primary_prompt=scenario_data['primary_prompt'],
        follow_up_1=scenario_data['follow_up_1'],
        follow_up_2=scenario_data['follow_up_2'],
        follow_up_3=scenario_data['follow_up_3'],
        ideal_primary_answer=scenario_data['ideal_primary_answer'],
        ideal_answer_1=scenario_data['ideal_answer_1'],
        ideal_answer_2=scenario_data['ideal_answer_2'],
        ideal_answer_3=scenario_data['ideal_answer_3'],
        cognitive_load_level=scenario_data['cognitive_load_level'],
        ai_appropriateness=scenario_data['ai_appropriateness'],
        bias_learning_objective=scenario_data['bias_learning_objective'],
        rubric_focus=scenario_data['rubric_focus'],
        llm_feedback=scenario_data.get('llm_feedback'),
        source_citation=scenario_data.get('source_citation')
    )
    
    # Create experimental session
    session = ExperimentalSession(
        session_id=session_id,
        session_start_time=datetime.now(),
        user_expertise=user_expertise,
        ai_assistance_enabled=ai_assistance_enabled,
        assigned_scenario=scenario
    )
    
    return session

def validate_session_data(session: ExperimentalSession) -> Dict[str, bool]:
    """
    Validate experimental session data for research integrity.
    
    Returns validation flags for quality control and data filtering.
    """
    
    validations = {
        'valid_experimental_design': (
            session.user_expertise in UserExpertise and
            isinstance(session.ai_assistance_enabled, bool) and
            session.bias_type in BiasType
        ),
        'sufficient_responses': len(session.stage_responses) >= 2,
        'complete_session': session.is_completed,
        'adequate_engagement': (
            session.analytics.overall_engagement_score > 0.3 
            if session.analytics else False
        ),
        'natural_timing': session.session_duration_minutes >= 2,
        'quality_responses': all(
            r.word_count >= 5 for r in session.stage_responses
        ) if session.stage_responses else False,
        'research_ready': (
            session.is_completed and
            len(session.stage_responses) == 4 and
            sum(r.word_count for r in session.stage_responses) >= 50
        )
    }
    
    return validations