"""
scoring_engine.py

This module contains the backend scoring logic for ClārusAI's 6-dimensional AI literacy assessment system.
Each function evaluates a user response against a specific cognitive learning dimension: 
semantic understanding, bias recognition, conceptual originality, strategic thinking, domain transfer, and metacognitive insight.

The aim is to provide a fine-grained and explainable evaluation of how well users internalise bias mitigation principles.
Written to support real-time Streamlit feedback and later statistical analysis.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import nltk
import re
from scipy.sparse import issparse
import config

# Load English tokenizer if not already downloaded
nltk.download("punkt", quiet=True)

# Load SentenceTransformer model once globally for efficiency
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 1. Semantic Similarity
# -----------------------------
def calculate_semantic_similarity(response: str, ideal_answer: str) -> Tuple[float, str]:
    """
    Computes cosine similarity between user response and ideal answer embeddings.
    High similarity may indicate reliance on AI hints.
    """
    embeddings = model.encode([response, ideal_answer])
    sim_score = cosine_similarity(np.array([embeddings[0]]), np.array([embeddings[1]]))[0][0]

    threshold = config.SCORING_THRESHOLDS.get("semantic_similarity", 0.75)
    if sim_score > threshold:
        tag = "high"  # Potential parroting
    elif sim_score > 0.5:
        tag = "medium"
    else:
        tag = "low"  # Divergent / original interpretation

    return sim_score, tag


# -----------------------------
# 2. Bias Recognition
# -----------------------------
def detect_bias_recognition(response: str, bias_type: str) -> Tuple[int, str]:
    """
    Checks how many bias-related keywords appear in the response.
    Assumes keyword sets are defined in config.BIAS_KEYWORDS.
    """
    bias_keywords = config.BIAS_KEYWORDS.get(bias_type.lower(), [])
    found_terms = [term for term in bias_keywords if re.search(rf"\\b{re.escape(term)}\\b", response, re.IGNORECASE)]
    count = len(found_terms)

    threshold = config.SCORING_THRESHOLDS.get("bias_recognition_min_terms", 2)
    if count >= threshold:
        tag = "strong"
    elif count == 1:
        tag = "partial"
    else:
        tag = "none"

    return count, tag


# -----------------------------
# 3. Conceptual Originality
# -----------------------------
def measure_originality(response: str, ideal_answer: str) -> Tuple[float, str]:
    """
    Uses TF-IDF cosine similarity to check surface overlap with ideal answer.
    Lower similarity => more original phrasing or ideas.
    
    Args:
        response: User's response text
        ideal_answer: Reference ideal answer for comparison
    
    Returns:
        Tuple[float, str]: (originality_score, originality_tag)
        
    Research Notes:
    - Originality score ranges from 0.0 (identical) to 1.0 (completely different)
    - Uses TF-IDF vectorization to capture semantic rather than surface differences
    - Inverts similarity to treat dissimilarity as originality
    - Thresholds calibrated for academic assessment standards
    """
    
    # Handle edge cases
    if not response.strip() or not ideal_answer.strip():
        return 0.0, "low"
    
    # Create TF-IDF vectors for both texts
    try:
        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=1000,  # Limit features for efficiency
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1,  # Include all terms (important for small text samples)
            lowercase=True,
            token_pattern=r'\b[A-Za-z]{2,}\b'  # Only words with 2+ letters
        )
        
        # Fit and transform both texts
        tfidf_matrix = tfidf.fit_transform([response, ideal_answer])
        
        # Convert sparse matrix to dense array for cosine similarity
        if issparse(tfidf_matrix):
            dense_matrix = tfidf_matrix.toarray()
        else:
            dense_matrix = np.asarray(tfidf_matrix)
        
        # Calculate cosine similarity between the two vectors
        similarity = cosine_similarity(
            dense_matrix[0].reshape(1, -1),  # User response vector
            dense_matrix[1].reshape(1, -1)   # Ideal answer vector
        )[0][0]
        
        # Invert similarity to treat as originality score
        # High similarity = Low originality, Low similarity = High originality
        originality = 1.0 - similarity
        
        # Apply research-calibrated thresholds
        threshold_low = config.SCORING_THRESHOLDS.get("originality", 0.25)
        threshold_high = 0.5
        
        # Classify originality level
        if originality < threshold_low:
            tag = "low"
        elif originality < threshold_high:
            tag = "moderate"
        else:
            tag = "high"
        
        return float(originality), tag
        
    except Exception as e:
        # Fallback scoring if TF-IDF fails
        print(f"TF-IDF originality scoring failed: {e}")
        
        # Simple word overlap fallback
        response_words = set(response.lower().split())
        ideal_words = set(ideal_answer.lower().split())
        
        if not response_words or not ideal_words:
            return 0.0, "low"
        
        # Calculate Jaccard similarity as fallback
        intersection = len(response_words.intersection(ideal_words))
        union = len(response_words.union(ideal_words))
        
        if union == 0:
            fallback_originality = 0.0
        else:
            fallback_similarity = intersection / union
            fallback_originality = 1.0 - fallback_similarity
        
        # Apply same thresholds
        threshold_low = config.SCORING_THRESHOLDS.get("originality", 0.25)
        
        if fallback_originality < threshold_low:
            tag = "low"
        elif fallback_originality < 0.5:
            tag = "moderate"
        else:
            tag = "high"
        
        return float(fallback_originality), tag
# -----------------------------
# 4. Mitigation Strategy
# -----------------------------
def assess_mitigation_strategy(response: str) -> Tuple[int, str]:
    """
    Searches for common strategic language used in bias mitigation (e.g., check assumptions).
    """
    mitigation_terms = [
        "second opinion", "alternative explanation", "cross-check",
        "evidence-based", "structured approach", "slow down",
        "disconfirming evidence", "check assumptions", "re-evaluate",
        "consult others", "meta-analysis", "reduce bias"
    ]
    count = sum(1 for term in mitigation_terms if term in response.lower())

    if count >= 2:
        tag = "clear strategy"
    elif count == 1:
        tag = "partial strategy"
    else:
        tag = "no strategy"

    return count, tag


# -----------------------------
# 5. Domain Transferability
# -----------------------------
def evaluate_transferability(response: str, original_domain: str) -> Tuple[int, str]:
    """
    Detects reference to other domains or generalised decision strategies.
    """
    domain_terms = {
        "medical": ["military", "war", "emergency", "disaster", "conflict"],
        "military": ["hospital", "patient", "triage", "ambulance", "diagnosis"],
        "emergency": ["battlefield", "war", "surgery", "hospital", "naval"]
    }
    fallback_terms = ["in other fields", "cross-domain", "any context", "transferable", "applies elsewhere"]

    cross_refs = domain_terms.get(original_domain.lower(), []) + fallback_terms
    count = sum(1 for term in cross_refs if term in response.lower())

    if count >= 2:
        tag = "strong transfer"
    elif count == 1:
        tag = "limited transfer"
    else:
        tag = "no transfer"

    return count, tag


# -----------------------------
# 6. Metacognitive Awareness
# -----------------------------
def measure_metacognition(response: str) -> Tuple[int, str]:
    """
    Looks for reflective phrases suggesting awareness of one's own reasoning.
    """
    reflective_markers = [
        "I realised", "upon reflection", "I initially thought",
        "I changed my mind", "in hindsight", "at first",
        "I questioned", "reconsidered", "as I analysed"
    ]
    count = sum(1 for marker in reflective_markers if marker.lower() in response.lower())

    if count >= 2:
        tag = "high awareness"
    elif count == 1:
        tag = "some awareness"
    else:
        tag = "no awareness"

    return count, tag
