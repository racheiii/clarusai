""" 
6-Dimensional Cognitive Bias Assessment Engine

Measures:
  1) Semantic similarity (MiniLM cosine)
  2) Bias recognition (keywords/semantic cues)
  3) Conceptual originality (TF‑IDF char n‑grams)
  4) Mitigation strategy (domain terms)
  5) Domain transferability (cross‑domain terms)
  6) Metacognitive awareness (self‑reflection markers)

Config‑driven thresholds and weights; returns a structured dict for the dashboard.
"""

from __future__ import annotations

# Standard library
from enum import Enum
from typing import Any, Dict, List, Sequence, Tuple, cast

# Third-party
import numpy as np
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local
import config

def validate_response(response: str, min_length: int = 10) -> bool:
    """Basic sanity check for non-empty, minimally-long responses."""
    return isinstance(response, str) and len(response.strip()) >= min_length

def safe_enum_str(value: Any) -> str:
    """Return Enum.value if Enum, else str(value)."""
    return value.value if isinstance(value, Enum) else str(value)

def _collect_transfer_terms_from_config(domain_key: str) -> List[str]:
    """
    Collect domain-specific and general transfer vocabulary from `config`
    without direct attribute access (static-checker friendly).
    Returns a flat, deduplicated list of lowercase terms.
    """
    per_domain = cast(Dict[str, List[str]], getattr(config, "TRANSFER_TERMS", {}))
    general = cast(Sequence[str], getattr(config, "GENERAL_TRANSFER_TERMS", ()))
    verbs = cast(Sequence[str], getattr(config, "TRANSFER_VERBS", ()))  # optional

    ordered: List[str] = []
    ordered.extend(t.lower() for t in per_domain.get(domain_key, []))
    ordered.extend(t.lower() for t in general)
    ordered.extend(t.lower() for t in verbs)

    seen: set[str] = set()
    unique: List[str] = []
    for t in ordered:
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique

# Load transformer model once (no external downloads here)
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    MODEL_LOADED = True
except Exception as e:
    print(f"Model failed to load: {e}")
    model = None
    MODEL_LOADED = False

def calculate_semantic_similarity(response: str, ideal_answer: str) -> Tuple[float, str]:
    if not validate_response(response) or not validate_response(ideal_answer):
        return 0.0, "invalid"

    if not MODEL_LOADED or model is None:
        print("⚠️ Semantic model not loaded — returning fallback similarity score of 0.0")
        return 0.0, "fallback"

    try:
        embeddings = model.encode([response, ideal_answer])
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

    response_lc = response.lower()

    # Use config helper to support both CSV ("Confirmation") and enum ("confirmation_bias")
    bias_raw = safe_enum_str(bias_type)
    keywords = config.get_bias_keywords(bias_raw)  # robust lookup via config
    keyword_hits = [kw for kw in keywords if kw.lower() in response_lc]

    # Normalise to simple cue key: "confirmation" | "anchoring" | "availability"
    cue_key = bias_raw.lower().replace("_bias", "").replace("_heuristic", "")
    semantic_patterns = config.SEMANTIC_BIAS_CUES.get(cue_key, [])
    phrase_hits = [phrase for phrase in semantic_patterns if phrase.lower() in response_lc]

    total_hits = len(set(keyword_hits + phrase_hits))

    threshold = config.SCORING_THRESHOLDS.get("bias_recognition_min_terms", 2)

    if total_hits >= threshold:
        tag = "strong"
    elif total_hits == 1:
        tag = "partial"
    else:
        tag = "none"

    return total_hits, tag

def measure_originality(response: str, ideal_answer: str) -> Tuple[float, str]:
    """
    Originality = 1 - cosine(tfidf_char_ngrams).
    Uses character n-grams at word boundaries
    """
    if not validate_response(response) or not validate_response(ideal_answer):
        return 0.0, "invalid"

    try:
        tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=20000,
        )
        sparse = tfidf.fit_transform([response, ideal_answer])
        csr_mat = cast(csr_matrix, sparse)
        sim = float(cosine_similarity(csr_mat.getrow(0), csr_mat.getrow(1))[0, 0])
        originality = float(max(0.0, min(1.0, 1.0 - sim)))
    except Exception as e:
        print(f"Originality error: {e}")
        originality = 0.0

    low = config.SCORING_THRESHOLDS.get("originality_low", 0.25)
    high = config.SCORING_THRESHOLDS.get("originality_high", 0.5)
    tag = "low" if originality < low else ("moderate" if originality < high else "high")
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
    """
    Detect transfer intent by counting soft substring matches against a
    domain-aware vocabulary (verbs + cross-domain phrases)
    """
    if not validate_response(response):
        return 0, "invalid"

    text = response.lower().strip()
    domain_key = safe_enum_str(original_domain).lower()  # e.g., "medical", "military", "emergency"

    terms = _collect_transfer_terms_from_config(domain_key)
    hits = [t for t in terms if t and t in text]
    count = len(set(hits))

    if count >= 3:
        tag = "strong transfer"
    elif count == 2:
        tag = "clear transfer"
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
