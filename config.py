"""
Configuration settings for ClārusAI
UCL Master's Dissertation: "Building AI Literacy Through Simulation"
config.py - Configuration settings and constants
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AI API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_ENABLED = GEMINI_API_KEY is not None

# Warning if API not configured
if not API_ENABLED:
    print("⚠️  GEMINI_API_KEY not configured - AI guidance features disabled")

# Application Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Research Configuration
ENABLE_RESEARCH_MODE = os.getenv("ENABLE_RESEARCH_MODE", "True").lower() == "true"
MAX_RESPONSES_PER_SESSION = int(os.getenv("MAX_RESPONSES_PER_SESSION", "10"))

# Scoring Algorithm Thresholds
SCORING_THRESHOLDS = {
    "semantic_similarity": float(os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.75")),
    "originality": float(os.getenv("ORIGINALITY_THRESHOLD", "0.25")),
    "bias_recognition_min_terms": int(os.getenv("BIAS_RECOGNITION_MIN_TERMS", "2"))
}

# Bias Recognition Keywords for Experimental Analysis
BIAS_KEYWORDS = {
    "confirmation_bias": [
        "confirmation bias", "selective attention", "cherry picking", 
        "preconceived notions", "confirmation", "bias", "selective",
        "contradictory evidence", "disconfirming", "alternative explanations"
    ],
    "anchoring_bias": [
        "anchoring bias", "first impression", "initial information",
        "anchor", "adjustment", "starting point", "reference point",
        "insufficient adjustment", "anchoring effect"
    ],
    "availability_heuristic": [
        "availability heuristic", "recent examples", "memorable cases",
        "availability", "recency", "salience", "media coverage",
        "representative", "base rate", "frequency"
    ]
}

# Persona Definitions for Research Simulation
PERSONAS = {
    "novice": {
        "description": "Junior professional with 2-3 years experience, limited bias training, relies on intuition",
        "characteristics": ["intuitive", "confident", "limited experience", "pattern-focused"]
    },
    "expert": {
        "description": "Senior professional with 10+ years experience, some bias awareness, systematic approach", 
        "characteristics": ["systematic", "experienced", "cautious", "evidence-based"]
    }
}

# Gemini API Configuration
GEMINI_CONFIG = {
    "model": "gemini-pro",
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 512,
        "candidate_count": 1,
    },
    "safety_settings": [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH", 
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
}

# File Paths
DATA_DIR = "data"
SCENARIOS_FILE = f"{DATA_DIR}/scenarios.csv"
RESPONSES_DIR = f"{DATA_DIR}/responses"
EXPORTS_DIR = "exports"

# Ensure directories exist
for directory in [DATA_DIR, RESPONSES_DIR, EXPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)