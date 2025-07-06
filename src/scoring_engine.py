
""" 
ClārusAI: 6-Dimensional Cognitive Bias Assessment Engine
UCL Master's Dissertation: "Building AI Literacy Through Simulation"

scoring_engine.py - Complete assessment system for authentic AI literacy measurement

Academic Purpose:
This module implements the core 6-dimensional scoring framework for evaluating
authentic AI literacy vs algorithmic dependency patterns in cognitive bias training.
Each dimension captures specific aspects of learning internalization and critical thinking.

Research Framework:
- Semantic Similarity: Detects AI parroting vs independent analysis
- Bias Recognition: Measures learning of cognitive bias concepts  
- Conceptual Originality: Assesses independent reasoning vs mimicry
- Mitigation Strategy: Evaluates strategic thinking sophistication
- Domain Transferability: Tests cross-context application ability
- Metacognitive Awareness: Captures self-reflective reasoning patterns

Statistical Integration:
All functions return both quantitative scores and qualitative tags for
comprehensive statistical analysis and real-time educational feedback.

Author: Rachel Seah
Date: July 2025
Dependencies: sentence-transformers, scikit-learn, nltk, scipy
"""

from typing import Tuple, Dict, Any, List, cast
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
import config
from scipy.sparse import csr_matrix

# Load transformer model once
try:
    nltk.download("punkt", quiet=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    MODEL_LOADED = True
except Exception as e:
    print(f"Model failed to load: {e}")
    model = None
    MODEL_LOADED = False

def validate_response(response: str, min_length: int = 10) -> bool:
    return isinstance(response, str) and len(response.strip()) >= min_length

def safe_enum_str(value):
    return value.value if isinstance(value, Enum) else str(value)

def calculate_semantic_similarity(response: str, ideal_answer: str) -> Tuple[float, str]:
    if not validate_response(response) or not validate_response(ideal_answer):
        return 0.0, "invalid"

    if not MODEL_LOADED or model is None:
        return 0.0, "fallback"

    try:
        embeddings = model.encode([response, ideal_answer])
        # Fix: convert to np.array to satisfy MatrixLike type requirement
        similarity = cosine_similarity(np.array([embeddings[0]]), np.array([embeddings[1]]))[0][0]
    except Exception as e:
        print(f"Semantic similarity error: {e}")
        similarity = 0.0

    high = config.SCORING_THRESHOLDS.get("semantic_similarity_high", 0.75)
    medium = config.SCORING_THRESHOLDS.get("semantic_similarity_medium", 0.5)

    if similarity > high:
        tag = "high"
    elif similarity > medium:
        tag = "medium"
    else:
        tag = "low"

    return similarity, tag

def detect_bias_recognition(response: str, bias_type: Any) -> Tuple[int, str]:
    if not validate_response(response):
        return 0, "invalid"

    bias_key = safe_enum_str(bias_type).lower()
    keywords = config.BIAS_KEYWORDS.get(bias_key, [])
    found = [kw for kw in keywords if kw in response.lower()]
    count = len(found)

    threshold = config.SCORING_THRESHOLDS.get("bias_recognition_min_terms", 2)

    if count >= threshold:
        tag = "strong"
    elif count == 1:
        tag = "partial"
    else:
        tag = "none"

    return count, tag

def measure_originality(response: str, ideal_answer: str) -> Tuple[float, str]:
    if not validate_response(response) or not validate_response(ideal_answer):
        return 0.0, "invalid"

    try:
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 2),
            token_pattern=r'\b[A-Za-z]{2,}\b'
        )
        tfidf_matrix = tfidf.fit_transform([response, ideal_answer])
        csr = cast(csr_matrix, tfidf_matrix)
        dense = csr.toarray()
        # Fix: convert to np.array to satisfy MatrixLike requirement
        similarity = cosine_similarity(np.array([dense[0]]), np.array([dense[1]]))[0][0]
        originality = 1.0 - similarity
    except Exception as e:
        print(f"Originality error: {e}")
        originality = 0.0

    low = config.SCORING_THRESHOLDS.get("originality_low", 0.25)
    high = config.SCORING_THRESHOLDS.get("originality_high", 0.5)

    if originality < low:
        tag = "low"
    elif originality < high:
        tag = "moderate"
    else:
        tag = "high"

    return originality, tag

def assess_mitigation_strategy(response: str) -> Tuple[int, str]:
    if not validate_response(response):
        return 0, "invalid"

    strategies = config.MITIGATION_TERMS
    found = [term for term in strategies if term in response.lower()]
    count = len(found)

    if count >= 2:
        tag = "clear strategy"
    elif count == 1:
        tag = "partial strategy"
    else:
        tag = "no strategy"

    return count, tag

def evaluate_transferability(response: str, original_domain: Any) -> Tuple[int, str]:
    if not validate_response(response):
        return 0, "invalid"

    domain = safe_enum_str(original_domain).lower()
    terms = config.TRANSFER_TERMS.get(domain, []) + config.GENERAL_TRANSFER_TERMS
    found = [term for term in terms if term in response.lower()]
    count = len(found)

    if count >= 2:
        tag = "strong transfer"
    elif count == 1:
        tag = "limited transfer"
    else:
        tag = "no transfer"

    return count, tag

def measure_metacognition(response: str) -> Tuple[int, str]:
    if not validate_response(response):
        return 0, "invalid"

    markers = config.METACOGNITIVE_TERMS
    found = [marker for marker in markers if marker in response.lower()]
    count = len(found)

    if count >= 2:
        tag = "high awareness"
    elif count == 1:
        tag = "some awareness"
    else:
        tag = "no awareness"

    return count, tag

def calculate_comprehensive_scores(response: str, ideal_answer: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    bias_type = scenario.get("bias_type", "unknown")
    domain = scenario.get("domain", "unknown")

    semantic_score, semantic_tag = calculate_semantic_similarity(response, ideal_answer)
    bias_count, bias_tag = detect_bias_recognition(response, bias_type)
    originality_score, originality_tag = measure_originality(response, ideal_answer)
    strategy_count, strategy_tag = assess_mitigation_strategy(response)
    transfer_count, transfer_tag = evaluate_transferability(response, domain)
    metacog_count, metacog_tag = measure_metacognition(response)

    overall = (
        config.WEIGHTS["independence"] * (1 - semantic_score) +
        config.WEIGHTS["bias_awareness"] * min(1, bias_count / 3) +
        config.WEIGHTS["originality"] * originality_score +
        config.WEIGHTS["strategy"] * min(1, strategy_count / 3) +
        config.WEIGHTS["transfer"] * min(1, transfer_count / 2) +
        config.WEIGHTS["metacognition"] * min(1, metacog_count / 2)
    )

    return {
        "semantic_similarity": {"score": semantic_score, "tag": semantic_tag},
        "bias_recognition": {"count": bias_count, "tag": bias_tag},
        "conceptual_originality": {"score": originality_score, "tag": originality_tag},
        "mitigation_strategy": {"count": strategy_count, "tag": strategy_tag},
        "domain_transferability": {"count": transfer_count, "tag": transfer_tag},
        "metacognitive_awareness": {"count": metacog_count, "tag": metacog_tag},
        "overall_quality_score": overall,
        "confidence_flags": {
            "low_effort": len(response.split()) < 10,
            "high_similarity_risk": semantic_score > 0.75 and originality_score < 0.25
        },
        "metadata": {
            "bias_type": safe_enum_str(bias_type),
            "domain": safe_enum_str(domain),
            "model_loaded": MODEL_LOADED
        }
    }
