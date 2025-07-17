"""
Configuration settings for ClārusAI

config.py - Centralises platform-wide constants, thresholds, and API keys
to ensure reproducibility and integrity in cognitive bias training experiments.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AI API Configuration

# Warning if API not configured

# Application Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Research Configuration
ENABLE_RESEARCH_MODE = os.getenv("ENABLE_RESEARCH_MODE", "True").lower() == "true"
MAX_RESPONSES_PER_SESSION = int(os.getenv("MAX_RESPONSES_PER_SESSION", "10"))

# Scoring Algorithm Thresholds
SCORING_THRESHOLDS = {
    "semantic_similarity_high": 0.75,
    "semantic_similarity_medium": 0.5,
    "originality_low": 0.25,
    "originality_high": 0.5,
    "bias_recognition_min_terms": 2
}

# Bias Recognition Keywords: keyword sets for detecting bias recognition
# in user responses across different cognitive bias types
BIAS_KEYWORDS = {
    # Primary keys matching CSV format (what's in your scenarios.csv)
    "Confirmation": [
        # Core bias terminology
        "confirmation bias", "selective attention", "cherry picking", 
        "preconceived notions", "confirmation", "bias", "selective",
        # Research and evidence terminology
        "contradictory evidence", "disconfirming", "alternative explanations",
        "counter-evidence", "opposing views", "different perspectives",
        # Cognitive process terminology  
        "seeking information", "information seeking", "evidence gathering",
        "motivated reasoning", "wishful thinking", "belief perseverance",
        # Professional context terminology
        "verify assumptions", "challenge beliefs", "question premises",
        "examine alternatives", "consider opposition", "devil's advocate"
    ],
    
    "Anchoring": [
        # Core bias terminology
        "anchoring bias", "first impression", "initial information",
        "anchor", "adjustment", "starting point", "reference point",
        # Cognitive process terminology
        "insufficient adjustment", "anchoring effect", "initial value",
        "baseline", "benchmark", "reference frame", "starting figure",
        # Professional adjustment terminology
        "adjust upward", "adjust downward", "recalibrate", "reassess",
        "independent assessment", "fresh perspective", "clean slate",
        # Mitigation terminology
        "multiple estimates", "different starting points", "range of values",
        "bracketing", "scenario planning", "alternative baselines"
    ],
    
    "Availability": [
        # Core bias terminology
        "availability heuristic", "recent examples", "memorable cases",
        "availability", "recency", "salience", "media coverage",
        # Frequency and probability terminology
        "representative", "base rate", "frequency", "probability",
        "statistical likelihood", "actual rates", "true prevalence",
        # Memory and recall terminology
        "vivid examples", "memorable incidents", "dramatic cases",
        "media attention", "publicity", "newsworthy events",
        # Professional context terminology
        "systematic data", "comprehensive statistics", "full dataset",
        "population data", "historical trends", "objective measures",
        # Mitigation terminology
        "seek statistics", "consult data", "review records",
        "systematic analysis", "broader sample", "representative data"
    ],
    
    # Enum format aliases for backward compatibility with existing code
    "confirmation_bias": [
        "confirmation bias", "selective attention", "cherry picking", 
        "preconceived notions", "confirmation", "bias", "selective",
        "contradictory evidence", "disconfirming", "alternative explanations",
        "counter-evidence", "opposing views", "different perspectives",
        "seeking information", "information seeking", "evidence gathering",
        "motivated reasoning", "wishful thinking", "belief perseverance",
        "verify assumptions", "challenge beliefs", "question premises",
        "examine alternatives", "consider opposition", "devil's advocate"
    ],
    
    "anchoring_bias": [
        "anchoring bias", "first impression", "initial information",
        "anchor", "adjustment", "starting point", "reference point",
        "insufficient adjustment", "anchoring effect", "initial value",
        "baseline", "benchmark", "reference frame", "starting figure",
        "adjust upward", "adjust downward", "recalibrate", "reassess",
        "independent assessment", "fresh perspective", "clean slate",
        "multiple estimates", "different starting points", "range of values",
        "bracketing", "scenario planning", "alternative baselines"
    ],
    
    "availability_heuristic": [
        "availability heuristic", "recent examples", "memorable cases",
        "availability", "recency", "salience", "media coverage",
        "representative", "base rate", "frequency", "probability",
        "statistical likelihood", "actual rates", "true prevalence",
        "vivid examples", "memorable incidents", "dramatic cases",
        "media attention", "publicity", "newsworthy events",
        "systematic data", "comprehensive statistics", "full dataset",
        "population data", "historical trends", "objective measures",
        "seek statistics", "consult data", "review records",
        "systematic analysis", "broader sample", "representative data"
    ]
}

# Semantic bias cue patterns used to detect implicit bias recognition in responses
SEMANTIC_BIAS_CUES = {
    "confirmation": [
        "ignored conflicting", "sought evidence", "only looked for supporting", 
        "focused on confirming", "cherry-picked", "prior belief", "interpreted as support"
    ],
    "anchoring": [
        "initial value", "anchored to", "first estimate", "starting number", 
        "adjusted from baseline", "fixed to initial", "influenced by original"
    ],
    "availability": [
        "recent events", "media coverage", "vivid memory", "easily recalled", 
        "emotionally charged", "frequent exposure", "recent experience shaped"
    ]
}


# Bias Format Conversion Functions
def normalize_bias_type(bias_type):
    """Convert between CSV format and enum format for bias types."""
    bias_mapping = {
        # CSV to enum
        "Confirmation": "confirmation_bias",
        "Anchoring": "anchoring_bias", 
        "Availability": "availability_heuristic",
        # Enum to CSV
        "confirmation_bias": "Confirmation",
        "anchoring_bias": "Anchoring",
        "availability_heuristic": "Availability"
    }
    return bias_mapping.get(bias_type, bias_type)

def get_bias_keywords(bias_type):
    """Get keywords for bias recognition, handling both CSV and enum formats."""
    # Try direct lookup first
    if bias_type in BIAS_KEYWORDS:
        return BIAS_KEYWORDS[bias_type]
    
    # Try normalized lookup
    normalized = normalize_bias_type(bias_type)
    if normalized in BIAS_KEYWORDS:
        return BIAS_KEYWORDS[normalized]
    
    # Return empty list if not found
    return []

# Persona Definitions for Research Simulation
PERSONAS = {
    "novice": {
        "description": "Junior professional with 2-3 years experience, limited bias training, relies on intuition",
        "characteristics": ["intuitive", "confident", "limited experience", "pattern-focused"],
        "typical_responses": [
            "goes with gut feeling",
            "focuses on immediate patterns", 
            "limited systematic approaches",
            "high confidence despite uncertainty"
        ]
    },
    "expert": {
        "description": "Senior professional with 10+ years experience, some bias awareness, systematic approach", 
        "characteristics": ["systematic", "experienced", "cautious", "evidence-based"],
        "typical_responses": [
            "seeks multiple sources",
            "applies systematic frameworks",
            "acknowledges uncertainty",
            "considers alternative explanations"
        ]
    }
}

# Session Management Configuration
SESSION_CONFIG = {
    "recovery_window_hours": int(os.getenv("RECOVERY_WINDOW_HOURS", "6")),
    "auto_save_interval_minutes": int(os.getenv("AUTO_SAVE_INTERVAL", "2")),
    "session_timeout_hours": int(os.getenv("SESSION_TIMEOUT_HOURS", "24")),
    "max_concurrent_sessions": int(os.getenv("MAX_CONCURRENT_SESSIONS", "100"))
}

# Data Quality Thresholds
QUALITY_THRESHOLDS = {
    "minimum_response_length": int(os.getenv("MIN_RESPONSE_LENGTH", "10")),
    "minimum_session_duration_minutes": int(os.getenv("MIN_SESSION_DURATION", "5")),
    "maximum_session_duration_hours": int(os.getenv("MAX_SESSION_DURATION", "2")),
    "minimum_words_per_stage": int(os.getenv("MIN_WORDS_PER_STAGE", "15")),
    "engagement_threshold": float(os.getenv("ENGAGEMENT_THRESHOLD", "0.3"))
}

# File Paths
DATA_DIR = "data"
SCENARIOS_FILE = f"{DATA_DIR}/scenarios.csv"
RESPONSES_DIR = f"{DATA_DIR}/responses"
EXPORTS_DIR = "exports"
ASSETS_DIR = "assets"
STYLES_DIR = f"{ASSETS_DIR}/styles"

# Ensure directories exist
for directory in [DATA_DIR, RESPONSES_DIR, EXPORTS_DIR, ASSETS_DIR, STYLES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Research Logging Configuration
LOGGING_CONFIG = {
    "log_level": LOG_LEVEL,
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": f"{DATA_DIR}/research.log",
    "max_log_size_mb": int(os.getenv("MAX_LOG_SIZE_MB", "50")),
    "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5"))
}

# Error Messages for User Communication
ERROR_MESSAGES = {
    "api_failure": "AI assistance is temporarily unavailable. You can continue with the training using your own analysis.",
    "session_recovery_failed": "Unable to restore your previous session. Starting a fresh training session.",
    "data_save_failed": "Your response has been recorded, but there was an issue saving additional metadata.",
    "invalid_response": "Please provide a more detailed response to continue with the training.",
    "scenario_load_failed": "Unable to load training scenarios. Please refresh the page and try again.",
    "scoring_failed": "Response recorded successfully, but detailed analysis is temporarily unavailable."
}

# Success Messages for User Communication  
SUCCESS_MESSAGES = {
    "session_restored": "Welcome back! Your previous training session has been restored.",
    "stage_completed": "Response saved successfully. Proceeding to the next stage.",
    "training_completed": "Congratulations! You have completed the full training protocol.",
    "data_exported": "Your training data has been successfully exported for analysis.",
    "api_connected": "AI assistance is available for this training session."
}

# Research Validation Rules - Match CSV format
VALIDATION_RULES = {
    "required_scenario_columns": [
        "scenario_id", "bias_type", "domain", "title", "scenario_text",
        "primary_prompt", "follow_up_1", "follow_up_2", "follow_up_3",
        "ideal_primary_answer", "ideal_answer_1", "ideal_answer_2", "ideal_answer_3",
        "cognitive_load_level", "ai_appropriateness", "bias_learning_objective", "rubric_focus"
    ],
    # Match actual CSV values
    "valid_bias_types": ["Confirmation", "Anchoring", "Availability"],
    "valid_domains": ["Medical", "Military", "Emergency"],
    "valid_expertise_levels": ["novice", "expert"],
    "valid_cognitive_loads": ["Low", "Medium", "High"],
    "valid_ai_appropriateness": ["helpful", "neutral", "unhelpful"]
}

# Domain Format Conversion Functions
def normalize_domain(domain):
    """Convert between CSV format and lowercase format for domains."""
    domain_mapping = {
        # CSV to lowercase
        "Medical": "medical",
        "Military": "military", 
        "Emergency": "emergency",
        # Lowercase to CSV
        "medical": "Medical",
        "military": "Military",
        "emergency": "Emergency"
    }
    return domain_mapping.get(domain, domain)

# Enhanced validation function
def validate_config():
    """
    Validate configuration completeness for research integrity.
    
    Returns:
        Dict[str, bool]: Validation results for each configuration section
    """
    validations = {
        
        "bias_keywords_complete": all(
            len(keywords) >= 10 for keywords in BIAS_KEYWORDS.values()
        ),
        "thresholds_set": all(
            isinstance(threshold, (int, float)) for threshold in SCORING_THRESHOLDS.values()
        ),
        "directories_exist": all(
            os.path.exists(directory) for directory in [DATA_DIR, RESPONSES_DIR, EXPORTS_DIR]
        ),
        "validation_rules_complete": len(VALIDATION_RULES["required_scenario_columns"]) >= 14,
        "bias_format_support": all(
            bias_type in BIAS_KEYWORDS for bias_type in VALIDATION_RULES["valid_bias_types"]
        ),
        "enum_format_support": all(
            normalize_bias_type(bias_type) in BIAS_KEYWORDS for bias_type in VALIDATION_RULES["valid_bias_types"]
        )
    }
    
    return validations

# Development helper function
def get_config_summary():
    """
    Get configuration summary for debugging and validation.
    
    Returns:
        Dict[str, Any]: Summary of current configuration state
    """
    return {
        "api_status": "not_applicable",
        "bias_keyword_counts": {bias: len(keywords) for bias, keywords in BIAS_KEYWORDS.items()},
        "scoring_thresholds": SCORING_THRESHOLDS,
        "quality_thresholds": QUALITY_THRESHOLDS,
        "session_config": SESSION_CONFIG,
        "debug_mode": DEBUG,
        "validation_rules": VALIDATION_RULES,
        "directories": {
            "data_dir": DATA_DIR,
            "responses_dir": RESPONSES_DIR,
            "exports_dir": EXPORTS_DIR
        }
    }

# Testing functions for format conversion
def test_bias_format_conversion():
    """Test bias type format conversion functions."""
    test_cases = [
        ("Confirmation", "confirmation_bias"),
        ("Anchoring", "anchoring_bias"),
        ("Availability", "availability_heuristic")
    ]
    
    results = {}
    for csv_format, enum_format in test_cases:
        # Test CSV to enum
        csv_to_enum = normalize_bias_type(csv_format)
        # Test enum to CSV  
        enum_to_csv = normalize_bias_type(enum_format)
        # Test keyword lookup
        csv_keywords = get_bias_keywords(csv_format)
        enum_keywords = get_bias_keywords(enum_format)
        
        results[csv_format] = {
            "csv_to_enum": csv_to_enum == enum_format,
            "enum_to_csv": enum_to_csv == csv_format,
            "keywords_accessible": len(csv_keywords) > 0 and len(enum_keywords) > 0,
            "keywords_match": csv_keywords == enum_keywords
        }
    
    return results

if __name__ == "__main__":
    # Configuration validation on direct execution
    print("ClārusAI Configuration Validation")
    print("=" * 50)
    
    # Basic validation
    validations = validate_config()
    for check, result in validations.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check}: {status}")
    
    print("\n" + "=" * 50)
    print("Bias Format Conversion Testing")
    print("=" * 50)
    
    # Test bias format conversion
    conversion_tests = test_bias_format_conversion()
    for bias_type, tests in conversion_tests.items():
        print(f"\n{bias_type}:")
        for test_name, result in tests.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 50)
    print("Configuration Summary")
    print("=" * 50)
    
    summary = get_config_summary()
    for section, data in summary.items():
        if isinstance(data, dict) and len(data) > 3:
            print(f"{section}: {len(data)} items configured")
        else:
            print(f"{section}: {data}")
    
    print("\n🚀 Configuration validation complete!")

# Research-Based Scoring Weights for Overall Quality
WEIGHTS = {
    "independence": 0.30,
    "bias_awareness": 0.25,
    "originality": 0.20,
    "strategy": 0.15,
    "transfer": 0.05,
    "metacognition": 0.05
}

# Strategy/Mitigation Detection Terms
MITIGATION_TERMS = [
    "second opinion", "alternative explanation", "cross-check",
    "evidence-based", "structured approach", "slow down",
    "disconfirming evidence", "check assumptions", "re-evaluate",
    "consult others", "meta-analysis", "reduce bias",
    "systematic review", "peer review", "independent verification",
    "multiple sources", "devil's advocate", "base rate",
    "statistical analysis", "objective criteria", "blind review",
    "control for", "standardized process", "checklist approach"
]

# Transferability Terms by Domain
TRANSFER_TERMS = {
    "medical": [
        "military", "war", "battlefield", "combat", "strategic",
        "emergency", "disaster", "crisis", "evacuation", "triage",
        "business", "corporate", "management", "finance", "investment"
    ],
    "military": [
        "hospital", "patient", "medical", "diagnosis", "treatment",
        "emergency", "disaster", "rescue", "civilian", "humanitarian",
        "business", "corporate", "strategic planning", "risk assessment"
    ],
    "emergency": [
        "military", "combat", "tactical", "strategic", "defense",
        "medical", "hospital", "clinical", "patient care", "triage",
        "business", "corporate", "project management", "resource allocation"
    ]
}

GENERAL_TRANSFER_TERMS = [
    "in other fields", "cross-domain", "any context", "transferable",
    "applies elsewhere", "similar situations", "other areas",
    "broader application", "general principle", "universal approach",
    "across disciplines", "various contexts", "different domains",
    "analogous situations", "parallel cases", "similar patterns"
]

# Metacognitive Awareness Indicators
METACOGNITIVE_TERMS = [
    "I realised", "upon reflection", "I initially thought",
    "I changed my mind", "in hindsight", "at first",
    "I questioned", "reconsidered", "as I analysed",
    "I noticed", "I became aware", "I recognized",
    "looking back", "on second thought", "I revised",
    "I corrected", "I adjusted", "I learned",
    "I discovered", "it occurred to me", "I reflected",
    "I challenged", "I examined", "I evaluated",
    "I deliberated", "I contemplated", "I reassessed"
]

OLLAMA_CONFIG = {
    "model": "llama3",
    "temperature": 0.7,
    "max_tokens": 200
}