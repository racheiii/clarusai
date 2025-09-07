"""
ClārusAI: Simulated User Generator (LLM‑only, no scoring)

Simulated User Generator (LLM‑only, no scoring).

Generates 2×2×3 factorial session data (UserExpertise × AI_Assistance × BiasType)
with 4-stage responses. Scoring is deferred to the dashboard. Writes per‑session
JSON files under exports/simulated_datasets/dataset_N and a summary file.
"""

from __future__ import annotations

# ================================
# STANDARD LIBRARY IMPORTS
# ================================
import os
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ================================
# THIRD-PARTY IMPORTS
# ================================
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================================
# PROJECT IMPORTS 
# ================================
try:
    from src.llm_feedback import generate_stage_feedback 
except Exception: 
    generate_stage_feedback = None  

from src.models import BiasType, UserExpertise

# ================================
# CONSTANTS & PATHS 
# ================================
EXPORTS_DIR: Path = Path("exports")
OUTPUT_DIR: Path = EXPORTS_DIR / "simulated_datasets"

# Factorial design: 2×2×3 × replicates
DEFAULT_REPLICATES_PER_CONDITION: int = 3
STAGE_COUNT: int = 4

# ================================
# UTILS
# ================================
def convert_numpy(obj: Any) -> Any:
    """Convert NumPy scalars to native Python types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj

def _next_available_name(base_dir: Path, prefix: str = "dataset") -> Path:
    """Return base_dir/prefix_N where N is the next available integer."""
    base_dir.mkdir(parents=True, exist_ok=True)
    nums: List[int] = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith(f"{prefix}_"):
            try:
                nums.append(int(p.name.split("_", 1)[1]))
            except Exception:
                pass
    n = max(nums) + 1 if nums else 1
    return base_dir / f"{prefix}_{n}"

def check_ollama_available() -> bool:
    """True if `ollama list` responds; False otherwise."""
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        return res.returncode == 0
    except Exception:
        return False

def validate_experimental_completeness(items: List[Dict[str, Any]]) -> bool:
    """Ensure all 12 condition codes (2×2×3) are present at least once."""
    conds: set[str] = set()
    for it in items:
        cc = it.get("condition") or (
            it.get("experimental_metadata", {}).get("condition_code")
            if isinstance(it, dict)
            else None
        )
        if isinstance(cc, str) and cc:
            conds.add(cc)
    expected = 12
    if len(conds) < expected:
        raise ValueError(
            f"❌ Incomplete dataset: {len(conds)}/{expected} conditions present"
        )
    print(f"✅ All {expected} experimental conditions present")
    return True

# ===== Persona primers =====
PERSONA_PRIMERS = {
    "novice": """
You are a novice professional (≈2–3 years) operating in a high-stakes context.

Behavioural targets:
- Analytical depth (~6/10): Identify 2–3 key factors; avoid long causal chains.
- Terminology specificity (5/10): Prefer broadly understood terms; minimal jargon.
- Argument structure (5/10): Short bullets/paragraphs; may anchor on first idea.
- Bias susceptibility (7/10): Prone to anchoring/availability/confirmation; may accept suggestions verbatim.
- AI reliance (8/10): Consult AI early; limited challenge to its output.
- Ambiguity tolerance (6/10): Discomfort with uncertainty; prefers clear next steps.
- Metacognitive markers (6/10): Include one brief bias self-check (e.g., “I may be anchoring because …”).

Style rules:
- Keep answers concise (bullets or short paragraphs).
- Include exactly ONE explicit “bias check” sentence.
- Calibrate confidence with hedges (“likely”, “uncertain”, “provisionally”).
- If an 'AI hint' is provided, refer to it directly unless it conflicts with obvious facts.
- Include exactly ONE explicit line starting with 'Bias check:'.
""".strip(),

    "expert": """
You are a senior professional (15+ years) operating in a high-stakes context.

Behavioural targets:
- Analytical depth (~9/10): Compare competing hypotheses; weigh trade-offs/risks.
- Terminology specificity (9/10): Use precise terms when useful; briefly define uncommon ones.
- Argument structure (9/10): Numbered reasoning → inference → decision.
- Bias susceptibility (4/10): Actively counter anchoring/availability/confirmation; challenge weak suggestions.
- AI reliance (5/10): Use AI selectively for cross-check; preserve original reasoning.
- Ambiguity tolerance (8/10): State residual uncertainty and contingency plans.
- Metacognitive markers (8/10): Include a calibration line and one explicit counter-bias move.

Style rules:
- Structure as “Assessment: … / Reasoning: … / Plan: …” (adapt labels if needed).
- Include exactly ONE counter-bias step (e.g., alternative interpretation).
- Provide a justified confidence statement (“moderate confidence because …”).
- If an 'AI hint' is provided, cross-check it; accept only with justification or offer an alternative.
- Include exactly ONE explicit line starting with 'Counter-bias:' and a confidence line starting with 'Confidence:'.
""".strip(),
}

# ================================
# GENERATOR
# ================================
class SimulatedUserGenerator:
    """
    Generates realistic simulated user sessions for experimental research.

    Architecture (Option B):
    - LLM-only responses (no template fallback).
    - Scoring deferred to dashboard (on-load), using per-stage `ideal_answer`.
    """

    def __init__(
        self,
        scenarios_csv_path: str = "data/scenarios.csv",
        replicates_per_condition: int = DEFAULT_REPLICATES_PER_CONDITION,
        seed: int | None = None,
        model_name: str = "llama3.2",
        target_words: int | None = None,
    ) -> None:
        self.scenarios_csv_path = scenarios_csv_path
        self.replicates_per_condition = int(replicates_per_condition)
        self.model_name = model_name
        self.target_words = int(target_words) if target_words else None

        # Deterministic RNG for scenario sampling
        self._seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        self.scenarios_df: Optional[pd.DataFrame] = None
        self.generated_sessions: List[Dict[str, Any]] = []
        self.generation_log: List[Dict[str, Any]] = []
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # ----- Scenarios -----
    def load_scenarios(self) -> bool:
        """Load the scenarios CSV (with fallbacks) and validate required columns"""
        try:
            # 1) Path fallback: try common locations if the provided path doesn't exist
            candidate_paths = [
                self.scenarios_csv_path,
                "/mnt/data/scenarios.csv",   # Chat upload path (preferred if present)
                "data/scenarios.csv",
                "scenarios.csv",
            ]
            csv_path = None
            for cand in candidate_paths:
                if cand and Path(cand).exists():
                    csv_path = cand
                    break

            if not csv_path:
                print(f"❌ Scenarios file not found. Tried: {candidate_paths}")
                return False

            df = pd.read_csv(csv_path)

            # 2) Column schema check (fail fast with a clear error)
            required_cols = {
                "scenario_id", "domain", "scenario_text",
                "primary_prompt", "follow_up_1", "follow_up_2", "follow_up_3",
                "ideal_primary_answer", "ideal_answer_1", "ideal_answer_2", "ideal_answer_3",
                "bias_type",
            }
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                return False

            # 3) Bias types present check (your three core biases)
            required_biases = {"Confirmation", "Anchoring", "Availability"}
            available = set(df["bias_type"].dropna().astype(str).unique().tolist())
            missing_biases = required_biases - available
            if missing_biases:
                print(f"❌ Missing bias types in CSV: {sorted(missing_biases)}")
                return False

            self.scenarios_df = df
            print(f"✅ Loaded {len(df)} scenarios from {csv_path} | biases: {sorted(available)}")
            return True

        except Exception as e:
            print(f"❌ Failed to load scenarios: {e}")
            return False

    def get_scenario_for_bias(self, bias_type: BiasType) -> Optional[pd.Series]:
        """Sample a random scenario row for the given bias type"""
        if self.scenarios_df is None:
            return None
        csv_bias = {
            BiasType.CONFIRMATION_BIAS: "Confirmation",
            BiasType.ANCHORING_BIAS: "Anchoring",
            BiasType.AVAILABILITY_HEURISTIC: "Availability",
        }.get(bias_type)
        if not csv_bias:
            return None
        sub = self.scenarios_df[self.scenarios_df["bias_type"] == csv_bias]
        if sub.empty:
            print(f"⚠️ No scenarios for bias type: {csv_bias}")
            return None
        # use seeded RNG for reproducibility
        return sub.sample(
            n=1,
            random_state=int(self._rng.integers(0, 2**31 - 1))
        ).iloc[0]

    def _build_persona_prompt(
        self,
        expertise_key: str,
        domain: str,
        scenario_text: str,
        stage_name: str,
        stage_prompt: str,
        ai_assistance: bool,
        length_hint: str = "",
        ai_hint: str = "",
    ) -> str:
        """Compose a domain-agnostic persona prompt with behavioural constraints."""
        persona_block = PERSONA_PRIMERS[expertise_key]

        domain_adapter = (
            f"Domain context: {domain}.\n"
            "- Use domain-appropriate language sparingly; define uncommon terms briefly.\n"
            "- Do not invent facts outside the scenario; if information is missing, state the assumption.\n"
        )

        ai_note = (
            "AI assistance is AVAILABLE. You may consult it, but: state explicitly when you relied on it; "
            "challenge any suggestion that conflicts with your reasoning; and justify acceptance or rejection."
            if ai_assistance else
            "No AI assistance is available; rely solely on your own reasoning."
        )

        output_rules = (
            "Output requirements:\n"
            "- Follow the persona constraints above (they encode behaviour, not domain knowledge).\n"
            "- Include exactly one bias/self-check sentence (novice) or one explicit counter-bias step (expert).\n"
            "- Be concise and structured; do not restate these instructions.\n"
        )

        return (
            f"{persona_block}\n\n"
            f"{domain_adapter}\n"
            f"Scenario:\n{scenario_text}\n\n"
            f"Task ({stage_name}):\n{stage_prompt}\n\n"
            + (f"{ai_note}\n{ai_hint}\n\n" if ai_assistance and ai_hint else f"{ai_note}\n\n")
            + f"{output_rules}"
            + f"{length_hint}"
        )

    def _make_ai_hint(
        self,
        scenario: Dict[str, Any],
        stage: int,
        hint_tokens: Tuple[int, int] = (6, 12),   # dial 1: informativeness
        mask_ratio: float = 0.20                    # dial 2: noise (0.0 – 0.5)
    ) -> str:
        """Derive a lightweight, noisy hint from the ideal answer for this stage."""
        ideal_fields = ["ideal_primary_answer", "ideal_answer_1", "ideal_answer_2", "ideal_answer_3"]
        ideal = (scenario.get(ideal_fields[stage], "") or "").strip()
        if not ideal:
            return ""

        import re, random
        rnd = random.Random(int(self._rng.integers(0, 2**31 - 1)))

        # Extract alphanumeric-ish tokens, keep content words
        toks = re.findall(r"[A-Za-z][A-Za-z\-]+", ideal)
        if len(toks) < 6:
            return ""

        # Deduplicate while preserving order to keep some coherence
        seen = set()
        content = []
        STOP = {"the","a","an","and","or","but","to","of","for","in","on","at","by","with","from","that","this","these","those","is","are","be"}
        for t in toks:
            tl = t.lower()
            if tl in STOP or tl in seen:
                continue
            seen.add(tl)
            content.append(t)

        if not content:
            content = toks[:]  # fallback to original tokens

        # Shuffle to avoid verbatim leakage, then keep a slice sized by hint_tokens
        rnd.shuffle(content)
        lo, hi = hint_tokens
        k = max(lo, min(hi, max(6, len(content)//2)))
        kept = content[:k]

        # Apply masking noise to a subset of tokens
        def mask_token(w: str) -> str:
            if len(w) <= 3:
                return w
            # mask inner characters to keep a readable silhouette
            inner = max(1, int(len(w) * 0.6))
            return w[0] + "•" * inner + w[-1]

        if 0.0 < mask_ratio <= 0.6:
            num_mask = int(len(kept) * mask_ratio)
            idxs = list(range(len(kept)))
            rnd.shuffle(idxs)
            for i in idxs[:num_mask]:
                kept[i] = mask_token(kept[i])

        hint = " ".join(kept)
        return f"AI hint (noisy): {hint}"

   
    # ----- Response generation -----
    def generate_llama3_response(
        self,
        scenario: Dict[str, Any],
        stage: int,
        expertise: UserExpertise,
        ai_assistance: bool,
    ) -> Tuple[str, float]:
        """Generate a stage response via Ollama.

        Returns:
            (response_text, response_time_seconds)
        """
        from src.llm_utils import generate_ollama_response

        stage_prompts = ["primary_prompt", "follow_up_1", "follow_up_2", "follow_up_3"]
        stage_names = ["Primary Analysis", "Cognitive Factors", "Mitigation Strategies", "Transfer Learning"]

        prompt_field = stage_prompts[stage]
        stage_name = stage_names[stage]
        domain = scenario.get("domain", "unspecified")
        scenario_text = scenario.get("scenario_text", "")
        stage_prompt = scenario.get(prompt_field, "")

        expertise_key = "novice" if expertise == UserExpertise.NOVICE else "expert"

        length_hint = ""
        if self.target_words:
            length_hint = f"\n\nAim for approximately {self.target_words} words (±20%)."

        ai_hint = self._make_ai_hint(scenario, stage) if ai_assistance else ""

        full_prompt = self._build_persona_prompt(
            expertise_key=expertise_key,
            domain=str(domain),
            scenario_text=scenario_text,
            stage_name=stage_name,
            stage_prompt=stage_prompt,
            ai_assistance=ai_assistance,
            length_hint=length_hint,
            ai_hint=ai_hint
        )

        # Stage 3 (index 2)
        if stage == 2:
            full_prompt += (
                "\n\nPlease include one line starting with 'Transfer:' "
                "that generalises your mitigation to another high-stakes domain."
            )

        # Stage 4 (index 3)
        if stage == 3:
            full_prompt += (
                "\n\nTransfer requirement:\n"
                "- Generalise the core reasoning and bias-mitigation principles to at least ONE other high-stakes domain "
                "using neutral language and minimal jargon."
            )

        try:
            # Use the model set on the generator (dashboard-controlled)
            response = generate_ollama_response(full_prompt, model=self.model_name)
        except Exception:
            response = f"⚠️ Ollama generation failed at {stage_name}."

        base_time = 45 if expertise == UserExpertise.EXPERT else 30
        response_time = base_time * (stage + 1) * 0.85
        return response, response_time

    # ----- Behavioural controls -----
    def simulate_guidance_usage(
        self, ai_assistance: bool, expertise: UserExpertise, stage: int
    ) -> bool:
        """Return True if guidance is requested at this stage (stochastic)"""
        if not ai_assistance:
            return False
        stage_complexity = {0: 0.3, 1: 0.6, 2: 0.8, 3: 0.5}
        weight = stage_complexity.get(stage, 0.5)
        base = 0.8 if expertise == UserExpertise.NOVICE else 0.3
        prob = max(0.0, min(1.0, base * weight))
        return float(self._rng.random()) < prob

    def _assess_session_quality(self, stages: List[Dict[str, Any]]) -> str:
        """Heuristic quality label from word counts and elapsed time"""
        avg_words = sum(r["word_count"] for r in stages) / max(len(stages), 1)
        total_time_min = stages[-1]["cumulative_time_seconds"] / 60.0
        if avg_words >= 25 and total_time_min >= 3:
            return "high"
        if avg_words >= 15 and total_time_min >= 2:
            return "moderate"
        return "low"

    def create_session_data(
        self,
        session_id: str,
        expertise: UserExpertise,
        ai_assistance: bool,
        bias_type: BiasType,
    ) -> Optional[Dict[str, Any]]:
        """Build a 4‑stage session record (no scoring)"""
        row = self.get_scenario_for_bias(bias_type)
        if row is None:
            return None
        scenario = row.to_dict()

        session_start = datetime.now()
        stage_responses: List[Dict[str, Any]] = []
        llm_feedback_per_stage: List[Optional[str]] = []
        cumulative_time = 0.0

        ideal_fields = ["ideal_primary_answer", "ideal_answer_1", "ideal_answer_2", "ideal_answer_3"]

        for stage in range(STAGE_COUNT):
            # LLM‑only generation
            response_text, response_time = self.generate_llama3_response(
                scenario, stage, expertise, ai_assistance
            )
            cumulative_time += response_time

            # Optional tutor feedback
            tutor_feedback: Optional[str] = None
            if ai_assistance and (generate_stage_feedback is not None):
                try:
                    tutor_feedback = generate_stage_feedback(scenario, stage, response_text)
                except Exception:
                    tutor_feedback = "⚠️ Feedback generation failed."

            guidance_used = self.simulate_guidance_usage(ai_assistance, expertise, stage)
            ideal_answer = scenario.get(ideal_fields[stage], "")

            stage_responses.append(
                {
                    "llm_feedback": tutor_feedback,
                    "stage_number": stage,
                    "stage_name": ["Primary Analysis", "Cognitive Factors", "Mitigation Strategies", "Transfer Learning"][stage],
                    "response_text": response_text,
                    "response_time_seconds": response_time,
                    "cumulative_time_seconds": cumulative_time,
                    "guidance_requested": bool(guidance_used),
                    "word_count": len(response_text.split()),
                    "character_count": len(response_text),
                    "quality_level": ("high" if len(response_text.split()) >= 40
                    else "moderate" if len(response_text.split()) >= 20
                    else "low"),
                    # crucial for dashboard scoring:
                    "ideal_answer": ideal_answer,
                    # scores intentionally omitted in Option B
                    "timestamp": (session_start + timedelta(seconds=cumulative_time)).isoformat(),
                }
            )
            llm_feedback_per_stage.append(tutor_feedback)

        session_data: Dict[str, Any] = {
            "session_metadata": {
                "session_id": session_id,
                "is_simulated": True,
                "simulation_version": "2.0",
                "generated_timestamp": datetime.now().isoformat(),
                "user_expertise": expertise.value,
                "ai_assistance_enabled": ai_assistance,
                "bias_type": scenario["bias_type"],
                "domain": scenario["domain"],
                "scenario_id": scenario["scenario_id"],
                "total_session_time_minutes": cumulative_time / 60.0,
                "total_stages_completed": STAGE_COUNT,
                "total_guidance_requests": sum(1 for r in stage_responses if r["guidance_requested"]),
                "total_word_count": sum(r["word_count"] for r in stage_responses),
                "session_quality": self._assess_session_quality(stage_responses),
            },
            "scenario_data": scenario,  # used by dashboard scorer
            "stage_responses": stage_responses,
            "llm_feedback_per_stage": llm_feedback_per_stage,
            "experimental_metadata": {
                "condition_code": f"{expertise.value}_{ai_assistance}_{bias_type.value}",
                "factorial_design": "2x2x3",
                "bias_revelation_timing": "post_completion",
                "data_collection_method": "simulated_4stage_progression",
            },
        }
        return session_data

    # ----- Generate full dataset -----
    def generate_full_dataset(
        self,
        progress_callback=None,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a 2×2×3 dataset (12 × replicates).
        Returns:
            {"success": bool, "summary": {...}, "output_directory": str}
        """
        if not self.load_scenarios():
            return {"success": False, "error": "Failed to load scenarios"}
        if not check_ollama_available():
            return {"success": False, "error": "Ollama is not available. Start it and ensure your model is installed."}

        total_sessions = 2 * 2 * 3 * self.replicates_per_condition
        print("🚀 Starting simulated dataset generation…")
        print(f"📊 Target: {total_sessions} sessions")

        # Decide output directory
        if dataset_name:
            desired = OUTPUT_DIR / dataset_name
            output_dir = desired if not desired.exists() else _next_available_name(OUTPUT_DIR, dataset_name)
        else:
            output_dir = _next_available_name(OUTPUT_DIR, "dataset")
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_sessions: List[Dict[str, Any]] = []
        failed_sessions: List[Dict[str, Any]] = []
        session_counter = 0

        for expertise in [UserExpertise.NOVICE, UserExpertise.EXPERT]:
            for ai_assistance in [False, True]:
                for bias_type in [BiasType.CONFIRMATION_BIAS, BiasType.ANCHORING_BIAS, BiasType.AVAILABILITY_HEURISTIC]:
                    for replicate in range(1, self.replicates_per_condition + 1):
                        session_counter += 1
                        session_id = f"SIM_{expertise.value}_{ai_assistance}_{bias_type.value}_{replicate:02d}"

                        if progress_callback:
                            try:
                                progress_callback(session_counter, total_sessions, session_id)
                            except Exception:
                                pass  # UI callbacks must not break generation

                        try:
                            session_data = self.create_session_data(session_id, expertise, ai_assistance, bias_type)
                            if session_data is None:
                                failed_sessions.append({"session_id": session_id, "error": "No scenario available"})
                                print(f"❌ Failed: {session_id} - No scenario available")
                                continue

                            fp = output_dir / f"{session_id}.json"
                            with open(fp, "w", encoding="utf-8") as f:
                                json.dump(session_data, f, indent=2, ensure_ascii=False, default=convert_numpy)

                            generated_sessions.append(
                                {
                                    "session_id": session_id,
                                    "file_path": str(fp),
                                    "condition": session_data["experimental_metadata"]["condition_code"],
                                    "quality": session_data["session_metadata"]["session_quality"],
                                }
                            )
                            print(f"✅ Generated: {session_id}")
                        except Exception as e:
                            failed_sessions.append({"session_id": session_id, "error": str(e)})
                            print(f"❌ Failed: {session_id} - {e}")

        # Ensure 12 conditions present
        try:
            validate_experimental_completeness(generated_sessions)
        except Exception as e:
            summary = {
                "generation_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "simulation_version": "2.0",
                    "output_directory": str(output_dir),
                    "total_attempted": total_sessions,
                    "total_generated": len(generated_sessions),
                    "total_failed": len(failed_sessions),
                    "model_name": self.model_name,
                    "seed": self._seed,
                    "target_words": self.target_words,
                },
                "generated_sessions": generated_sessions,
                "failed_sessions": failed_sessions,
                "error": str(e),
            }
            with open(output_dir / "generation_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            return {"success": False, "error": str(e), "output_directory": str(output_dir)}

        # Success summary
        summary = {
            "generation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "simulation_version": "2.0",
                "output_directory": str(output_dir),
                "total_attempted": total_sessions,
                "total_generated": len(generated_sessions),
                "total_failed": len(failed_sessions),
                "model_name": self.model_name,
                "seed": self._seed,
                "target_words": self.target_words,
            },
            "experimental_design": {
                "factors": ["user_expertise", "ai_assistance", "bias_type"],
                "levels": [2, 2, 3],
                "replicates_per_condition": self.replicates_per_condition,
                "theoretical_total": total_sessions,
            },
            "generated_sessions": generated_sessions,
            "failed_sessions": failed_sessions,
            "quality_distribution": self._analyze_quality_distribution(generated_sessions),
        }
        with open(output_dir / "generation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📁 Saved to: {output_dir}")
        print(f"✅ Successfully generated: {len(generated_sessions)}/{total_sessions} sessions")
        return {"success": True, "summary": summary, "output_directory": str(output_dir)}

    # ----- Diagnostics -----
    def _analyze_quality_distribution(self, sessions: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"high": 0, "moderate": 0, "low": 0}
        for s in sessions:
            q = s.get("quality", "unknown")
            if q in counts:
                counts[q] += 1
        return counts

# ================================
# STANDALONE EXECUTION
# ================================
def main() -> None:
    gen = SimulatedUserGenerator()

    def progress(cur: int, total: int, sid: str) -> None:
        print(f"Progress: {cur}/{total} ({(cur/total)*100:.1f}%) - {sid}")

    res = gen.generate_full_dataset(progress_callback=progress)  # auto dataset_#
    if res.get("success"):
        s = res["summary"]
        print("\n🎉 Simulation Generation Complete!")
        print(f"📊 Generated: {s['generation_metadata']['total_generated']} sessions")
        print(f"📁 Location: {s['generation_metadata']['output_directory']}")
        print(f"🔗 Quality Distribution: {s['quality_distribution']}")
    else:
        print(f"❌ Generation failed: {res.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
