"""
ClārusAI: Enhanced Simulated User Generator
UCL Master's Dissertation: "Building AI Literacy Through Simulation"

sim_user_generator.py - Generates controlled synthetic user sessions for 2×2×3 factorial research

PURPOSE:
- Create reproducible experimental data for hypothesis testing
- Eliminate human subject variability while maintaining realistic response patterns
- Generate statistical power for ANOVA analysis of AI-assisted learning effectiveness

DESIGN:
- 2×2×3 Factorial: UserExpertise × AI_Assistance × BiasType  
- 3 replicates per condition = 36 total sessions
- 4-stage progressive responses per session
- Realistic variation in response quality and timing
"""

import os
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np

# Import project modules
from src.llm_feedback import generate_stage_feedback  # 🔄 Injected for tutor feedback

from src.models import UserExpertise, BiasType, Domain
from src.scoring_engine import calculate_comprehensive_scores
from config import EXPORTS_DIR

# Safety fallback for domain mapping
try:
    from src.models import Domain
    DOMAIN_MAPPING_AVAILABLE = True
except ImportError:
    DOMAIN_MAPPING_AVAILABLE = False
    print("⚠️ Domain mapping not available - using string values")

def convert_numpy(obj: Any) -> Any:
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.floating,)): 
        return float(obj)
    if isinstance(obj, (np.integer,)):   
        return int(obj)
    return str(obj)
    
# =============================
# CONFIGURATION
# =============================

# Experimental design parameters
NUM_REPLICATES_PER_CONDITION = 3  # 2×2×3×3 = 36 sessions total
STAGE_COUNT = 4
OUTPUT_DIR = Path(EXPORTS_DIR) / "simulated_datasets"
LOG_FILE = Path(EXPORTS_DIR) / "simulation_log.json"

# Response generation parameters
RESPONSE_VARIATION_FACTOR = 0.3  # Controls response diversity
MIN_WORDS_NOVICE = 15
MAX_WORDS_NOVICE = 40
MIN_WORDS_EXPERT = 25
MAX_WORDS_EXPERT = 80

# =============================
# ENHANCED RESPONSE TEMPLATES
# =============================

RESPONSE_TEMPLATES = {
    "novice": {
        "openings": [
            "I think the main issue here is",
            "From what I can see,",
            "My initial reaction is that",
            "Based on my experience,",
            "It seems to me that",
            "I would approach this by"
        ],
        "reasoning_patterns": [
            "following my gut instinct",
            "looking at the obvious patterns",
            "using what worked before",
            "going with my first impression",
            "applying common sense",
            "focusing on the key details"
        ],
        "conclusions": [
            "so I believe this is the right approach.",
            "which leads me to think we should proceed.",
            "therefore my recommendation would be to act quickly.",
            "so I'm confident in this assessment.",
            "which is why I think this is the best option.",
            "so this seems like the most logical choice."
        ]
    },
    "expert": {
        "openings": [
            "Drawing from established protocols,",
            "Based on systematic analysis,",
            "Considering multiple factors,",
            "From a strategic perspective,",
            "Taking into account best practices,",
            "Using a structured approach,"
        ],
        "reasoning_patterns": [
            "examining both primary and secondary indicators",
            "cross-referencing multiple data sources",
            "applying systematic decision-making frameworks",
            "considering potential alternative explanations",
            "evaluating risks and benefits comprehensively",
            "incorporating lessons learned from similar cases"
        ],
        "conclusions": [
            "leading to a measured recommendation for action.",
            "supporting a cautious but decisive approach.",
            "indicating the need for further verification before proceeding.",
            "suggesting a multi-staged implementation strategy.",
            "recommending continued monitoring of key indicators.",
            "supporting the proposed course of action with reservations noted."
        ]
    }
}

STAGE_SPECIFIC_CONTENT = {
    0: {  # Primary Analysis
        "novice": "The situation appears straightforward based on the information presented.",
        "expert": "Initial assessment requires careful evaluation of all available evidence and context."
    },
    1: {  # Cognitive Factors  
        "novice": "I'm relying on my experience and what feels right in this situation.",
        "expert": "Several cognitive factors may be influencing the decision-making process here."
    },
    2: {  # Mitigation Strategies
        "novice": "To avoid problems, I would double-check the important details.",
        "expert": "Implementing systematic verification processes and seeking independent validation would be advisable."
    },
    3: {  # Transfer Learning
        "novice": "This approach could probably work in other similar situations too.",
        "expert": "The principles identified here have broader applicability across multiple professional contexts."
    }
}

# =============================
# SIMULATED USER GENERATOR CLASS
# =============================

class SimulatedUserGenerator:
    """
    Generates realistic simulated user sessions for experimental research.
    
    Academic Purpose: Creates controlled experimental data while maintaining
    realistic variation patterns for valid statistical analysis.
    """
    
    def __init__(self, scenarios_csv_path: str = "data/scenarios.csv"):
        """Initialize generator with scenario database."""
        self.scenarios_df = None
        self.scenarios_csv_path = scenarios_csv_path
        self.generated_sessions = []
        self.generation_log = []
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_scenarios(self) -> bool:
        """Load and validate scenarios database."""
        try:
            if not os.path.exists(self.scenarios_csv_path):
                print(f"❌ Scenarios file not found: {self.scenarios_csv_path}")
                return False
            
            self.scenarios_df = pd.read_csv(self.scenarios_csv_path)
            print(f"✅ Loaded {len(self.scenarios_df)} scenarios")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load scenarios: {e}")
            return False
    
    def get_scenario_for_bias(self, bias_type: BiasType) -> Optional[pd.Series]:
        """Get random scenario matching the specified bias type."""
        if self.scenarios_df is None:
            return None
        
        # FIXED: Use correct CSV format mapping
        csv_bias_name = {
            BiasType.CONFIRMATION_BIAS: "Confirmation",
            BiasType.ANCHORING_BIAS: "Anchoring", 
            BiasType.AVAILABILITY_HEURISTIC: "Availability"
        }.get(bias_type)
        
        if not csv_bias_name:
            return None
        
        matching_scenarios = self.scenarios_df[
            self.scenarios_df['bias_type'] == csv_bias_name
        ]
        
        if matching_scenarios.empty:
            print(f"⚠️ No scenarios found for bias type: {csv_bias_name}")
            return None
        
        return matching_scenarios.sample(n=1).iloc[0]
    
    def generate_realistic_response(self, expertise: UserExpertise, stage: int, 
                                  scenario_context: str, ai_assistance: bool) -> Tuple[str, float]:
        """
        Generate realistic response with appropriate variation.
        
        Returns:
            Tuple[str, float]: (response_text, simulated_response_time_seconds)
        """
        expertise_key = expertise.value  # "novice" or "expert"
        templates = RESPONSE_TEMPLATES[expertise_key]
        
        # Generate response components
        opening = random.choice(templates["openings"])
        reasoning = random.choice(templates["reasoning_patterns"])
        stage_content = STAGE_SPECIFIC_CONTENT[stage][expertise_key]
        conclusion = random.choice(templates["conclusions"])
        
        # Add AI assistance influence for AI-assisted users
        ai_influence = ""
        if ai_assistance and random.random() < 0.4:  # 40% chance of AI influence
            ai_influence = " Following systematic analysis principles, "
        
        # Construct response
        response = f"{opening} {stage_content} {ai_influence}{reasoning}, {conclusion}"
        
        # Add realistic variation
        if random.random() < 0.3:  # 30% chance of additional detail
            extra_detail = self._generate_extra_detail(expertise, stage)
            response += f" {extra_detail}"
        
        # Simulate realistic response timing
        base_time = 45 if expertise == UserExpertise.EXPERT else 30
        variation = random.uniform(0.7, 1.5)
        response_time = base_time * variation * (stage + 1) * 0.8  # Stages get slightly faster
        
        return response, response_time
    
    def _generate_extra_detail(self, expertise: UserExpertise, stage: int) -> str:
        """Generate additional detail based on expertise and stage."""
        if expertise == UserExpertise.EXPERT:
            details = [
                "Additionally, past cases have shown similar patterns.",
                "Cross-referencing with established protocols supports this approach.",
                "Multiple factors need to be considered before final implementation.",
                "Risk assessment indicates this is within acceptable parameters."
            ]
        else:
            details = [
                "This feels like the right direction to take.",
                "I've seen something similar work before.",
                "The key seems to be acting decisively here.",
                "Trust and experience guide this decision."
            ]
        
        return random.choice(details)
    
    def simulate_guidance_usage(self, ai_assistance: bool, expertise: UserExpertise, 
                               stage: int) -> bool:
        """Simulate realistic AI guidance usage patterns."""
        if not ai_assistance:
            return False
        
        # Realistic usage patterns based on expertise and stage
        if expertise == UserExpertise.NOVICE:
            # Novices use guidance more frequently, especially early stages
            usage_probability = 0.7 if stage < 2 else 0.5
        else:
            # Experts use guidance more selectively
            usage_probability = 0.3 if stage < 2 else 0.4
        
        return random.random() < usage_probability
    
    def create_session_data(self, session_id: str, expertise: UserExpertise, 
                          ai_assistance: bool, bias_type: BiasType) -> Optional[Dict]:
        """Create complete simulated session data."""
        
        # Get scenario
        scenario_row = self.get_scenario_for_bias(bias_type)
        if scenario_row is None:
            return None
        
        scenario_dict = scenario_row.to_dict()
        
        # Create experimental session metadata
        session_start_time = datetime.now()
        
        llm_feedback_per_stage = []

        # Generate responses for all 4 stages
        stage_responses = []
        cumulative_time = 0
        
        for stage in range(STAGE_COUNT):
            # Generate response
            response_text, response_time = self.generate_realistic_response(
                expertise, stage, scenario_dict['scenario_text'], ai_assistance
            )
            
            cumulative_time += response_time
            
            
            # 🔄 Tutor Feedback (only for AI-assisted sessions)
            tutor_feedback = None
            if ai_assistance:
                try:
                    tutor_feedback = generate_stage_feedback(scenario_dict, stage, response_text)
                except Exception as e:
                    tutor_feedback = "⚠️ Feedback generation failed."
            
            # Simulate guidance usage
            guidance_used = self.simulate_guidance_usage(ai_assistance, expertise, stage)
            
            # Calculate scores using existing scoring engine
            ideal_answer_fields = [
                'ideal_primary_answer', 'ideal_answer_1', 
                'ideal_answer_2', 'ideal_answer_3'
            ]
            ideal_answer = scenario_dict.get(ideal_answer_fields[stage], "")
            
            try:
                scores = calculate_comprehensive_scores(
                    response=response_text,
                    ideal_answer=ideal_answer,
                    scenario=scenario_dict
                )
            except Exception as e:
                print(f"⚠️ Scoring failed for session {session_id}, stage {stage}: {e}")
                scores = {"error": str(e)}
            
            stage_data = {
                "llm_feedback": tutor_feedback,
                "stage_number": stage,
                "stage_name": ["Primary Analysis", "Cognitive Factors", 
                              "Mitigation Strategies", "Transfer Learning"][stage],
                "response_text": response_text,
                "response_time_seconds": response_time,
                "cumulative_time_seconds": cumulative_time,
                "guidance_requested": bool(guidance_used),
                "word_count": len(response_text.split()),
                "character_count": len(response_text),
                "quality_level": "high" if len(response_text.split()) >= 40 else "mid" if len(response_text.split()) >= 20 else "low",
                "scores": scores,
                "timestamp": (session_start_time + timedelta(seconds=cumulative_time)).isoformat()
            }
            
            stage_responses.append(stage_data)
            llm_feedback_per_stage.append(tutor_feedback)
        
        # Compile complete session data
        session_data = {
            "session_metadata": {
                "session_id": session_id,
                "is_simulated": True,
                "simulation_version": "2.0",
                "generated_timestamp": datetime.now().isoformat(),
                
                # Experimental condition
                "user_expertise": expertise.value,
                "ai_assistance_enabled": ai_assistance,
                "bias_type": scenario_dict['bias_type'],
                "domain": scenario_dict['domain'],
                "scenario_id": scenario_dict['scenario_id'],
                
                # Session analytics
                "total_session_time_minutes": cumulative_time / 60,
                "total_stages_completed": STAGE_COUNT,
                "total_guidance_requests": sum(1 for r in stage_responses if r["guidance_requested"]),
                "total_word_count": sum(r["word_count"] for r in stage_responses),
                "session_quality": self._assess_session_quality(stage_responses)
            },
            
            "scenario_data": scenario_dict,
            "stage_responses": stage_responses,
            "llm_feedback_per_stage": llm_feedback_per_stage,
            
            "experimental_metadata": {
                "condition_code": f"{expertise.value}_{ai_assistance}_{bias_type.value}",
                "factorial_design": "2x2x3",
                "bias_revelation_timing": "post_completion",
                "data_collection_method": "simulated_4stage_progression"
            }
        }
        
        return session_data
    
    def _assess_session_quality(self, stage_responses: List[Dict]) -> str:
        """Assess simulated session quality for research validation."""
        avg_words = sum(r["word_count"] for r in stage_responses) / len(stage_responses)
        total_time = stage_responses[-1]["cumulative_time_seconds"] / 60  # minutes
        
        if avg_words >= 25 and total_time >= 3:
            return "high"
        elif avg_words >= 15 and total_time >= 2:
            return "moderate"
        else:
            return "low"
    
    def generate_full_dataset(self, progress_callback=None) -> Dict[str, Any]:
        """
        Generate complete simulated dataset for 2×2×3 factorial design.
        
        Args:
            progress_callback: Optional function to call with progress updates
        
        Returns:
            Dict with generation results and metadata
        """
        if not self.load_scenarios():
            return {"success": False, "error": "Failed to load scenarios"}
        
        print("🚀 Starting simulated dataset generation...")
        print(f"📊 Target: {2 * 2 * 3 * NUM_REPLICATES_PER_CONDITION} sessions")
        
        generated_sessions = []
        failed_sessions = []
        session_counter = 0
        total_sessions = 2 * 2 * 3 * NUM_REPLICATES_PER_CONDITION
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_DIR / f"simulation_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sessions for all conditions
        for expertise in [UserExpertise.NOVICE, UserExpertise.EXPERT]:
            for ai_assistance in [False, True]:
                for bias_type in [BiasType.CONFIRMATION_BIAS, BiasType.ANCHORING_BIAS, BiasType.AVAILABILITY_HEURISTIC]:
                    for replicate in range(1, NUM_REPLICATES_PER_CONDITION + 1):
                        
                        session_counter += 1
                        session_id = f"SIM_{expertise.value}_{ai_assistance}_{bias_type.value}_{replicate:02d}"
                        
                        # Progress callback
                        if progress_callback:
                            progress_callback(session_counter, total_sessions, session_id)
                        
                        try:
                            # Generate session data
                            session_data = self.create_session_data(
                                session_id, expertise, ai_assistance, bias_type
                            )
                            
                            if session_data:
                                # Save session file
                                session_file = output_dir / f"{session_id}.json"
                                with open(session_file, 'w', encoding='utf-8') as f:
                                    json.dump(session_data, f, indent=2, ensure_ascii=False, default=convert_numpy)

                                
                                generated_sessions.append({
                                    "session_id": session_id,
                                    "file_path": str(session_file),
                                    "condition": session_data["experimental_metadata"]["condition_code"],
                                    "quality": session_data["session_metadata"]["session_quality"]
                                })
                                
                                print(f"✅ Generated: {session_id}")
                            else:
                                failed_sessions.append({"session_id": session_id, "error": "No scenario available"})
                                print(f"❌ Failed: {session_id} - No scenario available")
                                
                        except Exception as e:
                            failed_sessions.append({"session_id": session_id, "error": str(e)})
                            print(f"❌ Failed: {session_id} - {e}")
        
        # Generate summary and log
        generation_summary = {
            "generation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "simulation_version": "2.0",
                "output_directory": str(output_dir),
                "total_attempted": total_sessions,
                "total_generated": len(generated_sessions),
                "total_failed": len(failed_sessions)
            },
            "experimental_design": {
                "factors": ["user_expertise", "ai_assistance", "bias_type"],
                "levels": [2, 2, 3],
                "replicates_per_condition": NUM_REPLICATES_PER_CONDITION,
                "theoretical_total": total_sessions
            },
            "generated_sessions": generated_sessions,
            "failed_sessions": failed_sessions,
            "quality_distribution": self._analyze_quality_distribution(generated_sessions)
        }
        
        # Save generation log
        log_file = output_dir / "generation_summary.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(generation_summary, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Saved to: {output_dir}")
        print(f"✅ Successfully generated: {len(generated_sessions)}/{total_sessions} sessions")
        
        return {
            "success": True,
            "summary": generation_summary,
            "output_directory": str(output_dir)
        }
    
    def _analyze_quality_distribution(self, sessions: List[Dict]) -> Dict[str, int]:
        """Analyze quality distribution of generated sessions."""
        quality_counts = {"high": 0, "moderate": 0, "low": 0}
        for session in sessions:
            quality = session.get("quality", "unknown")
            if quality in quality_counts:
                quality_counts[quality] += 1
        return quality_counts

# =============================
# STANDALONE EXECUTION
# =============================

def main():
    """Main execution function for standalone running."""
    generator = SimulatedUserGenerator()
    
    def progress_callback(current: int, total: int, session_id: str):
        print(f"Progress: {current}/{total} ({current/total*100:.1f}%) - {session_id}")
    
    result = generator.generate_full_dataset(progress_callback=progress_callback)
    
    if result["success"]:
        print("\n🎉 Simulation Generation Complete!")
        summary = result["summary"]
        print(f"📊 Generated: {summary['generation_metadata']['total_generated']} sessions")
        print(f"📁 Location: {summary['generation_metadata']['output_directory']}")
        print(f"🔗 Quality Distribution: {summary['quality_distribution']}")
    else:
        print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()