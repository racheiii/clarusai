"""
ClārusAI Research Dashboard


02_Dashboard.py — Interface for statistical analysis of simulated experimental datasets

PURPOSE:
Multi-tab dashboard for RQ-aligned analysis:
- RQ1: AI vs No-AI reasoning quality comparison  
- RQ2: Parroting detection via Mimicry Index
- RQ3: Cross-domain transfer learning evaluation
- Dataset management and research exports

APPROACH:
- Welch t-tests with Cohen's d effect sizes
- Type-safe pandas operations with graceful error handling
- Plotly visualisations with fallback UI for missing dependencies
- Reproducible statistical analysis for dissertation findings
"""

from __future__ import annotations

# ================================
# STANDARD LIBRARY IMPORTS
# ================================
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import sys
import traceback

# ================================
# THIRD‑PARTY IMPORTS
# ================================
import numpy as np
import pandas as pd
import streamlit as st

# Plotting (optional; guarded)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except Exception:
    PLOTTING_AVAILABLE = False
    px = None  
    go = None 
    make_subplots = None  

# ================================
# LOCAL IMPORTS (PROJECT MODULES)
# ================================
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

import config 
from utils import load_css, render_academic_footer 

try:
    from sim_user_generator import SimulatedUserGenerator  
    SIMULATION_AVAILABLE = True
except Exception:
    try:
        from src.sim_user_generator import SimulatedUserGenerator 
        SIMULATION_AVAILABLE = True
    except Exception:  
        SimulatedUserGenerator = None  
        SIMULATION_AVAILABLE = False

# ================================
# CONFIGURATION & CONSTANTS
# ================================
EXPORTS_DIR: Path = Path(getattr(config, "EXPORTS_DIR", "exports")) / "simulated_datasets"
CHART_EXPORT_DIR: Path = Path(getattr(config, "EXPORTS_DIR", "exports")) / "research_outputs"
DATA_DIR: Path = Path(getattr(config, "DATA_DIR", "data"))

# Statistical controls (centralised)
STATISTICAL_SIGNIFICANCE_THRESHOLD: float = float(
    getattr(config, "STATISTICAL_SIGNIFICANCE_THRESHOLD", 0.05)
)
MIMICRY_SD_MULTIPLIER: float = float(
    getattr(config, "MIMICRY_THRESHOLD_SD_MULTIPLIER", 0.5)
)

# Ensure directories exist
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHART_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ================================
# NUMERIC SAFETY HELPERS
# ================================

def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric and drop NaNs.

    Args:
        s: Input Series (mixed types allowed).
    Returns:
        Numeric Series with NaNs removed.
    """
    return pd.to_numeric(s, errors="coerce").dropna()


def safe_mean(s: pd.Series) -> float:
    """Return mean of a Series safely (0.0 if empty)."""
    s2 = to_numeric_series(s)
    return float(s2.mean()) if len(s2) else 0.0

def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Compute Cohen's d for two independent samples (safe).

    Returns 0.0 if inputs are empty or pooled SD is zero.
    """
    x2, y2 = to_numeric_series(x), to_numeric_series(y)
    if not len(x2) or not len(y2):
        return 0.0
    sx2 = float(np.var(x2, ddof=1)) if len(x2) > 1 else 0.0
    sy2 = float(np.var(y2, ddof=1)) if len(y2) > 1 else 0.0
    denom_n = (len(x2) + len(y2) - 2)
    sp = np.sqrt(((len(x2) - 1) * sx2 + (len(y2) - 1) * sy2) / denom_n) if denom_n > 0 else 0.0
    if sp == 0:
        return 0.0
    return float((x2.mean() - y2.mean()) / sp)

# ================================
# DATA LOADING & VALIDATION
# ================================

@st.cache_data(show_spinner=False)
def _score_stage(response_text: str, ideal_answer: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute 6D scores for a single stage response on demand.
    Cached to avoid recomputation across UI interactions.
    """
    try:
        import importlib
        scoring = importlib.import_module("src.scoring_engine")
        return scoring.calculate_comprehensive_scores(
            response=response_text or "",
            ideal_answer=ideal_answer or "",
            scenario=scenario or {},
        )
    except Exception as e:
        # Return a structured error; downstream flattening is defensive
        return {"error": f"scoring_failed: {e}"}


@st.cache_data(show_spinner=False)
def load_simulation_dataset(simulation_folder: str) -> Optional[pd.DataFrame]:
    """
    Load a simulation dataset (folder of JSON sessions) into a flat DataFrame.

    Supports two data shapes:
      1) Sessions where each stage already has 'scores' (legacy datasets)
      2) Sessions where stages have 'ideal_answer' but no 'scores'
         -> we compute scores on load using the scoring engine (cached)

    Returns:
        A stage-level DataFrame or None on failure. All numeric fields are
        coerced safely; missing values are handled gracefully.
    """
    try:
        # Resolve the dataset folder using a simple, ordered fallback strategy
        candidates = [
            EXPORTS_DIR / simulation_folder,                         # if EXPORTS_DIR == .../simulated_datasets
            (EXPORTS_DIR / "simulated_datasets" / simulation_folder) # if EXPORTS_DIR == .../exports
        ]
        folder = next((p for p in candidates if p.exists()), candidates[0])

        if not folder.exists():
            st.error(f"Simulation folder not found: {folder}")
            return None

        files = sorted(folder.glob("*.json"))
        if not files:
            st.warning(f"No session files found in {folder}")
            return None

        records: List[Dict[str, Any]] = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    session: Dict[str, Any] = json.load(f)
            except Exception as e:
                st.warning(f"Failed to load {fp.name}: {e}")
                continue

            meta = session.get("session_metadata", {}) or {}
            expt = session.get("experimental_metadata", {}) or {}
            scenario_snapshot = session.get("scenario_data", {}) or {}
            stages = session.get("stage_responses", []) or []

            # Defensive mapping for bias/domain fields (string or {csv: ...})
            def _maybe_csv(x: Any) -> Any:
                return x.get("csv") if isinstance(x, dict) and "csv" in x else x

            for r in stages:
                # If scores are missing or error-tagged, score on load (cached)
                scores = r.get("scores")
                if (not scores) or (isinstance(scores, dict) and scores.get("error")):
                    scores = _score_stage(
                        response_text=r.get("response_text", ""),
                        ideal_answer=r.get("ideal_answer", ""),
                        scenario=scenario_snapshot,
                    )

                # Flatten out the stage into a single record (defensive conversions)
                sem = scores.get("semantic_similarity", {}) if isinstance(scores, dict) else {}
                bri = scores.get("bias_recognition", {}) if isinstance(scores, dict) else {}
                org = scores.get("conceptual_originality", {}) if isinstance(scores, dict) else {}
                mit = scores.get("mitigation_strategy", {}) if isinstance(scores, dict) else {}
                trn = scores.get("domain_transferability", {}) if isinstance(scores, dict) else {}
                met = scores.get("metacognitive_awareness", {}) if isinstance(scores, dict) else {}
                flg = scores.get("confidence_flags", {}) if isinstance(scores, dict) else {}

                record = {
                    # Identifiers
                    "session_id": meta.get("session_id", "unknown"),
                    "simulation_version": meta.get("simulation_version", "unknown"),
                    "is_simulated": bool(meta.get("is_simulated", True)),

                    # Factors (2×2×3)
                    "user_expertise": meta.get("user_expertise", "unknown"),
                    "ai_assistance_enabled": bool(meta.get("ai_assistance_enabled", False)),
                    "bias_type": _maybe_csv(meta.get("bias_type", "unknown")),
                    "domain": _maybe_csv(meta.get("domain", "unknown")),
                    "scenario_id": meta.get("scenario_id", "unknown"),
                    "condition_code": expt.get("condition_code", "unknown"),

                    # Stage
                    "stage_number": int(r.get("stage_number", 0) or 0),
                    "stage_name": r.get("stage_name", "unknown"),
                    "response_text": r.get("response_text", ""),
                    "ideal_answer": r.get("ideal_answer", ""),  # for audit/debug
                    "word_count": int(r.get("word_count", 0) or 0),
                    "character_count": int(r.get("character_count", 0) or 0),
                    "response_time_seconds": float(r.get("response_time_seconds", 0) or 0.0),
                    "guidance_requested": bool(r.get("guidance_requested", False)),

                    # Session analytics
                    "total_session_time_minutes": float(meta.get("total_session_time_minutes", 0) or 0.0),
                    "total_guidance_requests": int(meta.get("total_guidance_requests", 0) or 0),
                    "total_word_count": int(meta.get("total_word_count", 0) or 0),
                    "session_quality": meta.get("session_quality", "unknown"),

                    # Scores (6-D flattened; safe numeric conversions)
                    "semantic_similarity": float(sem.get("score", 0.0) or 0.0),
                    "semantic_tag": sem.get("tag", "unknown"),
                    "bias_recognition_count": int(bri.get("count", 0) or 0),
                    "bias_recognition_tag": bri.get("tag", "unknown"),
                    "originality_score": float(org.get("score", 0.0) or 0.0),
                    "originality_tag": org.get("tag", "unknown"),
                    "strategy_count": int(mit.get("count", 0) or 0),
                    "strategy_tag": mit.get("tag", "unknown"),
                    "transfer_count": int(trn.get("count", 0) or 0),
                    "transfer_tag": trn.get("tag", "unknown"),
                    "metacognition_count": int(met.get("count", 0) or 0),
                    "metacognition_tag": met.get("tag", "unknown"),
                    "overall_quality_score": float(scores.get("overall_quality_score", 0.0) or 0.0) if isinstance(scores, dict) else 0.0,

                    # Flags
                    "low_effort_flag": bool(flg.get("low_effort", False)),
                    "high_similarity_risk": bool(flg.get("high_similarity_risk", False)),
                    "scoring_error": bool(isinstance(scores, dict) and ("error" in scores)),

                    # Timestamps
                    "generated_timestamp": meta.get("generated_timestamp", ""),
                    "stage_timestamp": r.get("timestamp", ""),
                }
                records.append(record)

        if not records:
            st.error("No valid session data found in the selected folder.")
            return None

        df = pd.DataFrame(records)

        # Light post-validation
        # Ensure expected columns exist; fill if missing
        expected_cols = [
            "overall_quality_score", "semantic_similarity", "originality_score",
            "bias_recognition_count", "strategy_count", "transfer_count",
        ]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = np.nan

        st.success(f"✅ Loaded {len(df)} stage responses from {df['session_id'].nunique()} sessions")
        return df

    except Exception as e:
        st.error(f"Failed to load simulation dataset: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=10)
def get_available_simulations() -> List[str]:
    """
    List available simulation dataset folders (sorted most recent first).
    Assumes datasets live under exports/simulated_datasets/.
    """
    try:
        base = EXPORTS_DIR if EXPORTS_DIR.name == "simulated_datasets" else (EXPORTS_DIR / "simulated_datasets")
        if not base.exists():
            return []
        folders = [f.name for f in base.iterdir() if f.is_dir() and not f.name.startswith(".")]
        return sorted(folders, reverse=True)
    except Exception as e:
        st.error(f"Failed to list simulation folders: {e}")
        return []

# ================================
# DESCRIPTIVES & FACTORIAL SUMMARIES
# ================================

def compute_descriptives(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute sample characteristics used on multiple tabs.

    Returns a dict for quick JSON export and UI metrics.
    """
    try:
        return {
            "total_sessions": int(df["session_id"].nunique()),
            "total_responses": int(len(df)),
            "expertise_distribution": df["user_expertise"].value_counts(dropna=False).to_dict(),
            "ai_assistance_distribution": df["ai_assistance_enabled"].value_counts(dropna=False).to_dict(),
            "bias_type_distribution": df["bias_type"].value_counts(dropna=False).to_dict(),
            "domain_distribution": df["domain"].value_counts(dropna=False).to_dict(),
            "session_quality_distribution": df["session_quality"].value_counts(dropna=False).to_dict() if "session_quality" in df else {},
            "average_response_length": safe_mean(df["word_count"]) if "word_count" in df else 0.0,
            "average_session_time": safe_mean(df["total_session_time_minutes"]) if "total_session_time_minutes" in df else 0.0,
            "guidance_usage_rate": safe_mean(df["guidance_requested"]) if "guidance_requested" in df else 0.0,
            "low_effort_rate": safe_mean(df["low_effort_flag"]) if "low_effort_flag" in df else 0.0,
            "high_similarity_risk_rate": safe_mean(df["high_similarity_risk"]) if "high_similarity_risk" in df else 0.0,
        }
    except Exception as e:
        st.error(f"Error calculating descriptives: {e}")
        return {}


def summarize_factorial(df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate key metrics over 2×2×3 factors (expertise × AI × bias)."""
    try:
        d = df.copy()
        if "ai_assistance_enabled" in d.columns:
            d["ai_assistance_enabled"] = d["ai_assistance_enabled"].astype(bool)
        if "user_expertise" in d.columns:
            d["user_expertise"] = pd.Categorical(d["user_expertise"],
                                                categories=["novice", "expert"],
                                                ordered=True)
        if "bias_type" in d.columns:
            d["bias_type"] = pd.Categorical(d["bias_type"],
                                            categories=["confirmation", "anchoring", "availability"],
                                            ordered=True)

        agg = (
            d.groupby(["user_expertise", "ai_assistance_enabled", "bias_type"]).agg(
                overall_quality_score=("overall_quality_score", "mean"),
                semantic_similarity=("semantic_similarity", "mean"),
                originality_score=("originality_score", "mean"),
                bias_recognition_count=("bias_recognition_count", "mean"),
                strategy_count=("strategy_count", "mean"),
                guidance_requested=("guidance_requested", "mean"),
                n=("session_id", "count"),
            )
        ).reset_index()

        # Ensure stable, readable order in tables/plots
        agg = agg.sort_values(["bias_type", "user_expertise", "ai_assistance_enabled"],
                            ascending=[True, True, False])
        return {"table": agg}

    except Exception as e: 
        st.error(f"Error in factorial summary: {e}")
        return {"table": pd.DataFrame()}

# ================================
# STATISTICAL TESTING (RQ1–RQ3)
# ================================

def ttest_independent(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """Welch t-test returning (t, p).

    Notes:
      - Assumes independent samples and approximate normality of means
      - We use SciPy’s Welch test (equal_var=False) to relax homogeneity
    """
    try:
        from scipy import stats 
    except Exception:
        if not st.session_state.get("_scipy_warned", False):
            st.warning("Statistical tests require SciPy. Install with `pip install scipy` for p-values/effect sizes.")
            st.session_state["_scipy_warned"] = True
        return 0.0, 1.0
    from typing import Any

    x2, y2 = to_numeric_series(x), to_numeric_series(y)
    if not len(x2) or not len(y2):
        return 0.0, 1.0

    res: Any = stats.ttest_ind(x2, y2, equal_var=False)
    t = float(getattr(res, "statistic", (res[0] if isinstance(res, (tuple, list, np.ndarray)) else 0.0)))
    p = float(getattr(res, "pvalue",    (res[1] if isinstance(res, (tuple, list, np.ndarray)) else 1.0)))
    return t, p

def holm_bonferroni(pvals: Dict[str, float]) -> Dict[str, float]:
    """
    Return Holm–Bonferroni adjusted p-values as a dict keyed like input.
    p_adj[i] = max_{j≤i} min(1, (m−j+1) * p_sorted[j]).
    """
    items = [(k, float(v)) for k, v in pvals.items()]
    m = len(items)
    items_sorted = sorted(items, key=lambda kv: kv[1])
    adj_tmp: Dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items_sorted, start=1):
        factor = m - i + 1
        p_adj = min(1.0, factor * p)
        running_max = max(running_max, p_adj)
        adj_tmp[k] = running_max
    # map back to original order
    return {k: adj_tmp[k] for k, _ in items}

def analyze_rq1(df: pd.DataFrame) -> Dict[str, Any]:
    """
    RQ1 — AI vs No‑AI: Reasoning quality under cognitive bias conditions.
    Primary: overall_quality_score. Secondary: semantic_similarity, bias_recognition_count.
    Welch t-tests + Cohen's d. Holm–Bonferroni across metrics.
    """
    try:
        ai = df[df["ai_assistance_enabled"] == True]
        no_ai = df[df["ai_assistance_enabled"] == False]

        out: Dict[str, Any] = {"n_ai": int(len(ai)), "n_no_ai": int(len(no_ai))}
        metrics = [m for m in ["overall_quality_score", "semantic_similarity", "bias_recognition_count"] if m in df.columns]

        # raw tests
        pvals: Dict[str, float] = {}
        tmp: Dict[str, Dict[str, Any]] = {}
        for metric in metrics:
            t, p = ttest_independent(ai[metric], no_ai[metric])
            d = cohens_d(ai[metric], no_ai[metric])
            tmp[metric] = {
                "mean_ai": safe_mean(ai[metric]),
                "mean_no_ai": safe_mean(no_ai[metric]),
                "diff": float(safe_mean(ai[metric]) - safe_mean(no_ai[metric])),
                "t_stat": t,
                "p_value": p,
                "effect_size_d": d,
            }
            pvals[metric] = p

        # Holm–Bonferroni adjust
        adj = holm_bonferroni(pvals) if len(pvals) > 1 else {k: v for k, v in pvals.items()}

        # Finalise output with significance flags (raw and adjusted)
        for metric, res in tmp.items():
            # Coerce to float so Pylance knows the types are comparable
            p_raw: float = float(res.get("p_value") or 1.0)
            p_adj: float = float(adj.get(metric, p_raw))

            # Replace with the coerced values to keep downstream UI consistent
            res["p_value"] = p_raw
            res["p_value_adj"] = p_adj
            res["significant"] = bool(p_raw < STATISTICAL_SIGNIFICANCE_THRESHOLD)
            res["significant_adj"] = bool(p_adj < STATISTICAL_SIGNIFICANCE_THRESHOLD)

            out[metric] = res

        return out
    except Exception as e:
        return {"error": f"RQ1 evaluation failed: {e}"}

def analyze_rq2(df: pd.DataFrame) -> Dict[str, Any]:
    """RQ2 — Distinguishing parroting from authentic reasoning.

    Computes a Mimicry Index = z(semantic_similarity) − z(originality_score)
    Higher values suggest greater surface-level mimicry
    
    Reports:
    1. Mimicry risk threshold (mean + k·SD)
    2. AI vs No-AI group means and risk rates
    3. Overall similarity–originality correlation (expected negative if metrics diverge)
    """
    try:
        data = df[["semantic_similarity", "originality_score", "ai_assistance_enabled"]].copy()
        # One-pass coercion keeps indices aligned
        data["semantic_similarity"] = pd.to_numeric(data["semantic_similarity"], errors="coerce")
        data["originality_score"]   = pd.to_numeric(data["originality_score"],   errors="coerce")

        def z(x: pd.Series) -> pd.Series:
            x_clean = x.dropna()
            mu = float(x_clean.mean()) if len(x_clean) else 0.0
            sd = float(x_clean.std(ddof=1)) if len(x_clean) > 1 else 1.0
            if sd <= 0:
                sd = 1.0
            return (x - mu) / sd  # preserve index; NaNs remain NaN

        data["z_sim"] = z(data["semantic_similarity"])
        data["z_org"] = z(data["originality_score"])
        data["mimicry_index"] = data["z_sim"] - data["z_org"]

        # Threshold decision: > +k SD above mean as risk
        mu = float(np.nanmean(data["mimicry_index"]))
        sd = float(np.nanstd(data["mimicry_index"]))
        thr = mu + MIMICRY_SD_MULTIPLIER * sd
        data["mimicry_risk"] = data["mimicry_index"] > thr

        grp = (
        data.groupby("ai_assistance_enabled")
            .agg(
                mean_mimicry_index=("mimicry_index", "mean"),
                risk_rate=("mimicry_risk", "mean"),
                n=("mimicry_index", "count"),
            )
            .reindex([True, False])
            .reset_index()
        )

        pair = data[["semantic_similarity", "originality_score"]].dropna()
        corr = (
            float(np.corrcoef(pair["semantic_similarity"], pair["originality_score"])[0, 1])
            if len(pair) > 1 else 0.0
        )

        return {
            "threshold": thr,
            "threshold_sd_multiplier": MIMICRY_SD_MULTIPLIER,
            "overall_correlation_sim_vs_org": corr,
            "group_summary": grp,
            "detail_sample": data[["semantic_similarity", "originality_score", "mimicry_index", "mimicry_risk", "ai_assistance_enabled"]],
        }
    except Exception as e:
        return {"error": f"RQ2 evaluation failed: {e}"}

def analyze_rq3(df: pd.DataFrame) -> Dict[str, Any]:
    """RQ3 — Cross‑domain transfer learning with/without AI support.

    We model Δtransfer = transfer_count(stage3) − transfer_count(stage2) per session,
    compare AI vs No‑AI (Welch t‑test, Cohen’s d), and provide domain‑level splits.
    """
    try:
        pivot = (
            df[df["stage_number"].isin([2, 3])]
            .pivot_table(
                index=["session_id", "ai_assistance_enabled", "domain"],
                columns="stage_number",
                values="transfer_count",
                aggfunc="mean",
            )
            .reset_index()
            .rename(columns={2: "stage2", 3: "stage3"})
        )
        if pivot.empty:
            return {"error": "Insufficient stage 2/3 data for transfer analysis"}

        pivot["delta_transfer"] = pd.to_numeric(pivot["stage3"], errors="coerce") - pd.to_numeric(pivot["stage2"], errors="coerce")

        ai = pivot[pivot["ai_assistance_enabled"] == True]["delta_transfer"]
        no_ai = pivot[pivot["ai_assistance_enabled"] == False]["delta_transfer"]

        t, p = ttest_independent(ai, no_ai)
        d = cohens_d(ai, no_ai)

        by_domain = (
            pivot.groupby(["ai_assistance_enabled", "domain"]).agg(
                mean_delta=("delta_transfer", "mean"), n=("delta_transfer", "count")
            ).reset_index()
        )
        by_domain = by_domain.sort_values(["ai_assistance_enabled", "domain"],
                                  ascending=[False, True])

        return {
            "n_sessions": int(len(pivot)),
            "delta_mean_ai": float(pd.to_numeric(ai, errors="coerce").mean()) if len(ai) else 0.0,
            "delta_mean_no_ai": float(pd.to_numeric(no_ai, errors="coerce").mean()) if len(no_ai) else 0.0,
            "t_stat": t,
            "p_value": p,
            "significant": bool(p < STATISTICAL_SIGNIFICANCE_THRESHOLD),
            "effect_size_d": d,
            "by_domain": by_domain,
            "detail": pivot[["session_id", "ai_assistance_enabled", "domain", "stage2", "stage3", "delta_transfer"]],
        }
    except Exception as e:
        return {"error": f"RQ3 evaluation failed: {e}"}

def analyze_research_questions(df: pd.DataFrame) -> Dict[str, Any]:
    """Run statistical tests for RQ1–RQ3 using cleaned dataset values (type‑safe)."""
    return {
        "RQ1": analyze_rq1(df),
        "RQ2": analyze_rq2(df),
        "RQ3": analyze_rq3(df),
    }

# ================================
# VISUALISATIONS (PLOTLY)
# ================================

def fig_factorial(df: pd.DataFrame):
    """2×2×3 factorial bar panels (RQ labels on titles)."""
    if not PLOTTING_AVAILABLE or go is None or make_subplots is None:
        return None
    try:
        agg = summarize_factorial(df)["table"]
        if agg.empty:
            return None
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Overall Quality (RQ1)",
                "Originality (RQ2 context)",
                "Bias Recognition (RQ1)",
                "Guidance Usage",
            ),
        )
        agg["cond"] = agg["user_expertise"].astype(str) + " | AI=" + agg["ai_assistance_enabled"].astype(str)
        for i, metric in enumerate(["overall_quality_score", "originality_score", "bias_recognition_count", "guidance_requested"], start=1):
            row, col = (1 if i <= 2 else 2), (1 if i in [1, 3] else 2)
            for b in agg["bias_type"].unique():
                sub = agg[agg["bias_type"] == b]
                fig.add_trace(
                    go.Bar(
                        x=sub["cond"],
                        y=sub[metric],
                        name=str(b),
                        showlegend=(i == 1),
                    ),
                    row=row,
                    col=col,
                )
        fig.update_layout(
            height=640,
            title_text="Factorial Results across Bias Types",
            barmode="group",
            legend_traceorder="grouped",
        )
        return fig
    except Exception as e: 
        st.error(f"Error building factorial figure: {e}")
        return None

def fig_parroting_scatter(df: pd.DataFrame):
    """Similarity vs Originality scatter, coloured by AI (RQ2)."""
    if not PLOTTING_AVAILABLE or px is None:
        return None
    try:
        d = df[["semantic_similarity", "originality_score", "ai_assistance_enabled", "bias_type"]].copy()
        d = d.dropna()
        return px.scatter(
            d,
            x="semantic_similarity",
            y="originality_score",
            color="ai_assistance_enabled",
            symbol="bias_type",
            opacity=0.7,
            title="RQ2: Similarity vs Originality (colour = AI)",
        )
    except Exception as e:
        st.error(f"Error building parroting scatter: {e}")
        return None

def fig_transfer_lines(df: pd.DataFrame):
    """Transfer count by Stage (2→3), split by AI and domain (RQ3)."""
    if not PLOTTING_AVAILABLE or px is None:
        return None
    try:
        d = df[df["stage_number"].isin([2, 3])][["stage_number", "transfer_count", "ai_assistance_enabled", "domain"]].copy()
        d = d.dropna()
        return px.line(
            d,
            x="stage_number",
            y="transfer_count",
            color="ai_assistance_enabled",
            line_dash="domain",
            markers=True,
            title="RQ3: Transfer Count by Stage (2→3)",
        )
    except Exception as e: 
        st.error(f"Error building transfer lines: {e}")
        return None

# ================================
# STREAMLIT TABS (UI)
# ================================

def tab_dataset_generation() -> None:
    """Tab 0 — Dataset selection and generation controls."""
    st.header("📊 Simulated Dataset Generation & Management")
    sims = get_available_simulations()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📁 Available Datasets")

        r1, r2 = st.columns([1, 3])
        with r1:
            if st.button("🔄 Refresh datasets", use_container_width=True):
                try:
                    st.cache_data.clear()
                finally:
                    st.toast("Datasets list refreshed.", icon="✅")
                    st.rerun()

        sims = get_available_simulations()

        if sims:
            selected = st.selectbox("Select simulation dataset:", sims, key="dataset_selector")
            if selected:
                df = load_simulation_dataset(selected)
                if df is not None:
                    st.session_state["current_dataset"] = df
                    st.session_state["current_sim_name"] = selected

                    stats = compute_descriptives(df)
                    a, b, c, d = st.columns(4)
                    a.metric("Sessions", stats.get("total_sessions", 0))
                    b.metric("Responses", stats.get("total_responses", 0))
                    c.metric("Avg Words", f"{stats.get('average_response_length', 0):.1f}")
                    d.metric("Quality OK", f"{(1 - stats.get('low_effort_rate', 0)) * 100:.1f}%")

                    st.write("**Experimental Conditions Distribution:**")
                    d1, d2, d3 = st.columns(3)
                    d1.write("*Expertise:* ")
                    d1.json(stats.get("expertise_distribution", {}))
                    d2.write("*AI Assistance:* ")
                    d2.json(stats.get("ai_assistance_distribution", {}))
                    d3.write("*Bias Types:* ")
                    d3.json(stats.get("bias_type_distribution", {}))
        else:
            st.info("No simulation datasets found. Generate one below.")

    with c2:
        st.subheader("🧪 Generate New Dataset")

        if SIMULATION_AVAILABLE and SimulatedUserGenerator is not None:
            st.write("**2×2×3 Factorial Design:**")
            st.write("• Expertise: Novice, Expert")
            st.write("• AI Assistance: Enabled, Disabled")
            st.write("• Bias Type: Confirmation, Anchoring, Availability")

            with st.form("gen_controls", clear_on_submit=False):
                replicates = st.slider("Replicates per condition", 1, 10, 3)

                # Live size preview: 2 expertise × 2 AI × 3 biases = 12 sessions per replicate; 4 stages each
                sessions = 12 * int(replicates)
                responses = sessions * 4
                st.caption(f"Planned size: {sessions} sessions • {responses} responses (4 stages).")

                use_seed = st.checkbox("Use random seed", value=True)
                seed_val = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
                model_name = st.text_input("Model (Ollama id)", value="llama3.2:instruct")
                target_words = st.slider("Target response length (words)", 60, 400, 180, 10)

                submitted = st.form_submit_button("🚀 Generate Simulated Dataset", type="primary")

            if submitted:
                prog = st.progress(0)
                status = st.empty()

                def cb(cur: int, tot: int, sid: str) -> None:
                    p = float(cur) / float(max(tot, 1))
                    prog.progress(p)
                    status.text(f"Generating: {sid} ({cur}/{tot})")

                try:
                    with st.spinner("Initializing simulation generator..."):
                        # Resolve a usable scenarios.csv location
                        candidate_paths = [
                            "/mnt/data/scenarios.csv",  # Chat upload path
                            str(DATA_DIR / "scenarios.csv"),  # project data folder (config-driven)
                            "data/scenarios.csv",
                            "scenarios.csv",
                        ]
                        scenarios_path = None
                        for cand in candidate_paths:
                            try:
                                if Path(cand).exists():
                                    scenarios_path = cand
                                    break
                            except Exception:
                                pass
                        if scenarios_path is None:
                            scenarios_path = str(DATA_DIR / "scenarios.csv")

                        gen = SimulatedUserGenerator(
                            scenarios_csv_path=scenarios_path,
                            replicates_per_condition=int(replicates),
                            seed=int(seed_val) if use_seed else None,
                            model_name=model_name.strip(),
                            target_words=int(target_words),
                        )

                    res = gen.generate_full_dataset(progress_callback=cb)  # auto dataset_#
                    if res.get("success"):
                        st.success(f"✅ Generated {res['summary']['generation_metadata']['total_generated']} sessions!")
                        st.info("🔄 Refresh the page to see the new dataset in the dropdown.")
                        with st.expander("📋 Generation Summary"):
                            st.json(res.get("summary", {}))
                    else:
                        st.error(f"❌ Generation failed: {res.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Generation error: {e}")
        else:
            st.error("Simulation generator not available")

def tab_rq1_stats() -> None:
    """Tab 1 — RQ1: AI vs No‑AI statistical comparison."""
    st.header("📈 RQ1 – AI vs No‑AI: Reasoning Quality under Bias Conditions")
    if "current_dataset" not in st.session_state:
        st.warning("⚠️ Please select a dataset in Tab 0 first.")
        return
    df = st.session_state["current_dataset"]

    with st.spinner("Running RQ1 analyses..."):
        rq1 = analyze_rq1(df)

    if "error" in rq1:
        st.error(rq1["error"]) 
        return

    m1, m2, m3 = st.columns(3)
    q = rq1.get("overall_quality_score", {})
    s = rq1.get("semantic_similarity", {})
    b = rq1.get("bias_recognition_count", {})
    m1.metric("Quality Δ (AI−NoAI)", f"{q.get('diff', 0):.3f}")
    m2.metric("Similarity Δ (AI−NoAI)", f"{s.get('diff', 0):.3f}")
    m3.metric("Bias Rec Δ (AI−NoAI)", f"{b.get('diff', 0):.3f}")

    def show_test_block(title: str, res: Dict[str, Any]) -> None:
        st.subheader(title)
        if res:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("AI Mean", f"{res.get('mean_ai', 0):.3f}")
            c2.metric("No‑AI Mean", f"{res.get('mean_no_ai', 0):.3f}")
            c3.metric("p (raw)", f"{res.get('p_value', 1):.4f}")
            c4.metric("p (Holm)", f"{res.get('p_value_adj', res.get('p_value', 1)):.4f}")
            c5.metric("Cohen d", f"{res.get('effect_size_d', 0):.2f}")
            if res.get("significant_adj", False):
                st.success("✅ Significant after Holm–Bonferroni (α=0.05)")
            elif res.get("significant", False):
                st.warning("⚠️ Significant (raw) but not after Holm–Bonferroni")
            else:
                st.info("ℹ️ Not statistically significant (α=0.05)")

    show_test_block("Overall Quality Score", q)
    show_test_block("Semantic Similarity (Dependency Signal)", s)
    show_test_block("Bias Recognition Count", b)

    if PLOTTING_AVAILABLE:
        fig = fig_factorial(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def tab_rq2_parroting() -> None:
    """Tab 2 — RQ2: Parroting vs Reasoning analysis."""
    st.header("🧠 RQ2 – Parroting vs Reasoning: Can our metrics tell them apart?")
    if "current_dataset" not in st.session_state:
        st.warning("⚠️ Please select a dataset in Tab 0 first.")
        return
    df = st.session_state["current_dataset"]

    with st.spinner("Computing mimicry indices and correlations..."):
        rq2 = analyze_rq2(df)

    if "error" in rq2:
        st.error(rq2["error"]) 
        return

    c1, c2 = st.columns(2)
    corr_val = rq2.get("overall_correlation_sim_vs_org", 0.0)
    corr_display = "N/A" if (corr_val is None or (isinstance(corr_val, float) and np.isnan(corr_val))) else f"{corr_val:.3f}"
    thr_val = rq2.get("threshold", None)
    thr_display = "N/A" if (thr_val is None or (isinstance(thr_val, float) and np.isnan(thr_val))) else f"{thr_val:.2f}"
    c1.metric("Similarity↔Originality correlation", corr_display)
    c2.metric("Mimicry threshold (z-sim−z-org)", thr_display)

    grp = rq2.get("group_summary")
    if isinstance(grp, pd.DataFrame) and not grp.empty:
        st.subheader("AI vs No-AI Mimicry Index")
        grp_sorted = grp.sort_values("ai_assistance_enabled", ascending=False)  
        st.dataframe(grp.rename(columns={
            "ai_assistance_enabled": "AI Enabled",
            "mean_mimicry_index": "Mean Mimicry Index",
            "risk_rate": "Risk Proportion",
        }))
    elif grp is not None:
        st.subheader("AI vs No-AI Mimicry Index")
        st.json(grp)

    if PLOTTING_AVAILABLE and px is not None:
        sc = fig_parroting_scatter(df)
        if sc:
            st.plotly_chart(sc, use_container_width=True)

        # Risk bar plot
        if isinstance(grp, pd.DataFrame) and not grp.empty and px is not None:
            df_bar = grp.copy()
            label_map = {True: "AI On", False: "No AI"}
            df_bar["AI Enabled"] = df_bar["ai_assistance_enabled"].map(label_map).fillna("Unknown")
            df_bar["AI Enabled"] = pd.Categorical(df_bar["AI Enabled"],
                                      categories=["AI On", "No AI"],
                                      ordered=True)
            df_bar["risk_rate"] = pd.to_numeric(df_bar["risk_rate"], errors="coerce").clip(0, 1)

            bar = px.bar(
                df_bar,
                x="AI Enabled",
                y="risk_rate",
                text="risk_rate",
                title="RQ2: Proportion Above Mimicry Risk Threshold",
            )
            # Percentage axis + labels
            bar.update_traces(texttemplate="%{text:.0%}", textposition="outside")
            bar.update_layout(
                yaxis_tickformat=".0%",
                yaxis_title="Mimicry Risk (Proportion)",
                xaxis_title=None,
                bargap=0.25,
                legend_traceorder="grouped",  # harmless now; useful if you add a 'color=' split later
                barmode="group",
            )
            # Cleaner hover
            bar.update_traces(hovertemplate="Condition=%{x}<br>Risk=%{y:.1%}<extra></extra>")

            st.plotly_chart(bar, use_container_width=True)

def tab_rq3_transfer() -> None:
    """Tab 3 — RQ3: Transfer learning analysis."""
    st.header("🔁 RQ3 – Cross‑Domain Transfer Learning with/without AI")
    if "current_dataset" not in st.session_state:
        st.warning("⚠️ Please select a dataset in Tab 0 first.")
        return
    df = st.session_state["current_dataset"]

    with st.spinner("Evaluating transfer deltas (Stage 2→3)..."):
        rq3 = analyze_rq3(df)

    if "error" in rq3:
        st.error(rq3["error"]) 
        return

    a, b, c = st.columns(3)
    a.metric("ΔTransfer (AI)", f"{rq3.get('delta_mean_ai', 0):.3f}")
    b.metric("ΔTransfer (No‑AI)", f"{rq3.get('delta_mean_no_ai', 0):.3f}")
    c.metric("p‑value", f"{rq3.get('p_value', 1):.4f}")
    if rq3.get("significant"):
        st.success("✅ Statistically significant (α=0.05)")
    else:
        st.info("ℹ️ Not statistically significant (α=0.05)")

    if isinstance(rq3.get("by_domain"), pd.DataFrame):
        st.subheader("Domain‑level ΔTransfer (Exploratory)")
        st.dataframe(rq3["by_domain"].rename(columns={
            "ai_assistance_enabled": "AI Enabled",
            "mean_delta": "Mean ΔTransfer",
            "n": "N",
        }))

    if PLOTTING_AVAILABLE:
        ln = fig_transfer_lines(df)
        if ln:
            st.plotly_chart(ln, use_container_width=True)


def tab_exports() -> None:
    """Tab 4 — Research exports (CSV/JSON + summary JSON)."""
    st.header("📤 Research Exports")
    if "current_dataset" not in st.session_state:
        st.warning("⚠️ Please select a dataset in Tab 0 first.")
        return
    df = st.session_state["current_dataset"]
    sim = st.session_state.get("current_sim_name", "dataset")

    st.subheader("📊 Dataset Exports")
    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download Full Dataset (CSV)", csv_bytes, file_name=f"clarusai_{sim}.csv", mime="text/csv")

        json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button("📋 Download Full Dataset (JSON)", json_bytes, file_name=f"clarusai_{sim}.json", mime="application/json")

    with c2:
        st.write("**Statistical Summary (RQ‑aligned)**")
        stats = compute_descriptives(df)
        rq = analyze_research_questions(df)
        fact = summarize_factorial(df)["table"]
        summary = {
            "dataset": {
                "name": sim,
                "export_timestamp": datetime.now().isoformat(),
                "n_sessions": int(stats.get("total_sessions", 0)),
                "n_responses": int(stats.get("total_responses", 0)),
                "design": "2×2×3 (expertise × AI × bias)",
            },
            "descriptives": stats,
            "factorial_table_head": fact.head(50).to_dict(orient="records") if isinstance(fact, pd.DataFrame) else [],
            "RQ": rq,
        }
        st.download_button(
            "📈 Download Statistical Summary (JSON)",
            json.dumps(summary, indent=2, default=str).encode("utf-8"),
            file_name=f"clarusai_summary_{sim}.json",
            mime="application/json",
        )


def tab_results_summary() -> None:
    """Tab 5 — Viva‑ready consolidated RQ summary."""
    st.header("📋 Results Summary Panel (RQ1–RQ3)")
    if "current_dataset" not in st.session_state:
        st.warning("⚠️ Please select a dataset in Tab 0 first.")
        return
    df = st.session_state["current_dataset"]

    stats = compute_descriptives(df)
    rq = analyze_research_questions(df)

    st.subheader("Sample Characteristics")
    st.write(f"- Total sessions: {stats.get('total_sessions', 0)}")
    st.write(f"- Total responses: {stats.get('total_responses', 0)}")
    st.write(f"- Mean response length: {stats.get('average_response_length', 0):.1f} words")

    st.subheader("Experimental Conditions")
    exp = stats.get("expertise_distribution", {})
    aid = stats.get("ai_assistance_distribution", {})
    st.write(f"- Novice: {exp.get('novice', 0)}, Expert: {exp.get('expert', 0)}")
    st.write(f"- AI‑assisted: {aid.get(True, 0)}, Unassisted: {aid.get(False, 0)}")

    st.subheader("RQ Findings (high‑level)")
    r1, r2, r3 = st.columns(3)
    rq1 = rq.get("RQ1", {})
    q = rq1.get("overall_quality_score", {})
    r1.metric("RQ1 Quality Δ (AI−NoAI)", f"{q.get('diff', 0):.3f}")

    rq2 = rq.get("RQ2", {})
    _c = rq2.get("overall_correlation_sim_vs_org", 0.0)
    _c_disp = "N/A" if (_c is None or (isinstance(_c, float) and np.isnan(_c))) else f"{_c:.3f}"
    r2.metric("RQ2 corr(sim, org)", _c_disp)

    rq3 = rq.get("RQ3", {})
    r3.metric("RQ3 p‑value", f"{rq3.get('p_value', 1):.4f}")

    with st.expander("Details: RQ1"):
        st.json({k: v for k, v in rq1.items() if isinstance(v, dict)})
    with st.expander("Details: RQ2"):
        if isinstance(rq2.get("group_summary"), pd.DataFrame):
            st.dataframe(rq2["group_summary"])
        else:
            st.json({k: v for k, v in rq2.items() if k != "detail_sample"})
    with st.expander("Details: RQ3"):
        if isinstance(rq3.get("by_domain"), pd.DataFrame):
            st.dataframe(rq3["by_domain"])
        else:
            st.json({k: v for k, v in rq3.items() if k != "detail"})

# ================================
# APP ENTRY POINT
# ================================

def main() -> None:
    """Entry point for the Streamlit dashboard.

    Tabs:
      0. Dataset Generation (load/generate datasets)
      1. RQ1 – Statistical Analysis (AI vs No‑AI)
      2. RQ2 – Parroting Detection
      3. RQ3 – Transfer Learning
      4. Research Exports (CSV/JSON + RQ summary JSON)
      5. Results Summary (viva‑ready overview)
    """
    try:
        st.set_page_config(
            page_title="ClārusAI Research Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        load_css()

        st.markdown("# 📊 ClārusAI Research Dashboard")
        st.markdown("*UCL Master's Dissertation: Beyond Surface Learning*")
        st.markdown("---")

        if not PLOTTING_AVAILABLE:
            st.warning("⚠️ Some visualization features unavailable. Try: pip install plotly")
        if not SIMULATION_AVAILABLE or SimulatedUserGenerator is None:
            st.info("ℹ️ Simulation generator module not found – dataset loading still works.")

        tabs = st.tabs([
            "📊 Dataset Generation",  # 0
            "📈 RQ1 – Statistical Analysis",  # 1
            "🧠 RQ2 – Parroting Detection",  # 2
            "🔁 RQ3 – Transfer Learning",  # 3
            "📤 Research Exports",  # 4
            "📋 Results Summary",  # 5
        ])

        with tabs[0]:
            tab_dataset_generation()
        with tabs[1]:
            tab_rq1_stats()
        with tabs[2]:
            tab_rq2_parroting()
        with tabs[3]:
            tab_rq3_transfer()
        with tabs[4]:
            tab_exports()
        with tabs[5]:
            tab_results_summary()

        render_academic_footer()

    except Exception as e: 
        st.error("❌ Critical dashboard error occurred")
        st.error(f"Error: {e}")
        if getattr(config, "DEBUG", False):
            st.exception(e)
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
