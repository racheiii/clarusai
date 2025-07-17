# ClārusAI Project Context
UCL Master's Dissertation: "Building AI Literacy Through Simulation"

## Research Framework
- 2×2×3 factorial design (Expertise × AI Assistance × Bias Type)
- 6-dimensional automated scoring system
- 72 simulated responses for statistical validation
- High-stakes domains: Military, Medical, Emergency

## Technical Stack
- Streamlit frontend, Python backend
- Claude API integration
- NLP scoring algorithms
- Statistical analysis capabilities

## File Structure
ClarusAI/

├── Home.py                              # Streamlit main entry point & page routing

├── requirements.txt                    # Python dependencies (streamlit, anthropic, pandas, etc.)

├── config.py                          # API keys, scoring thresholds, experiment parameters

├── README.md                          # Project overview, setup instructions, research context

├── .env.example                       # Environment variables template (API keys)

├── .gitignore                         # Git exclusions (data/, .env, pycache)

│

├── assets/

│   └── styles/

│       ├── main.css                   # Global Streamlit styling & layout

│       ├── research.css               # Dashboard-specific professional styling

│       └── components.css             # Reusable UI element styles

│

├── components/

│   └── ui_components.py               # All reusable UI widgets (bias cards, scoring displays, charts)

│

├── data/

│   ├── scenarios.csv                  # Your 6 cognitive bias scenarios (existing file)

│   └── responses/                     # Generated response datasets & logs

│       ├── raw_responses.json         # Unprocessed Claude API outputs

│       ├── scored_responses.csv       # Processed with 6-dimensional scores

│       └── session_logs.json         # User interaction timestamps & metadata

│

├── src/

│   ├── models.py                      # Data classes (Scenario, Response, Persona, ExperimentCondition)
|
│   ├── scoring_engine.py              # 6-dimensional assessment algorithms (similarity, originality, bias recognition, etc.)

│   ├── research_pipeline.py           # Automated 72-response generation & live statistical validation

│   └── visualizations.py             # Chart generation for scoring displays & research analytics

│

├── pages/

│   ├── 01_Scenarios.py             # User training interface with scenario selection & AI assistance toggle

│   ├── 02_Results.py              # Personal analytics, session summary, research contribution consent

│   ├── 03_Dashboard.py            # Research automation control centre & hypothesis testing
│

└── exports/

    ├── statistical_reports/           # ANOVA results, correlation matrices, effect sizes

    ├── raw_datasets/                  # CSV exports for SPSS/R analysis

    └── research_outputs/              # Formatted results for dissertation appendicesprojec

## Key Requirements
- Professional academic presentation
- Research automation capabilities  
- Real-time statistical validation
- Algorithm transparency
- Dissertation-ready outputs

Core Reference Files (attach to all chats):

config.py - Configuration settings and constants
data/scenarios.csv - Your scenario database
assets/styles/main.css - Styling reference
requirements.txt - Dependencies context

Purpose 
Designed to address the central research question: 
"Does LLM-assisted training in high-stakes decision contexts develop authentic AI literacy and internalized bias mitigation skills, or does it encourage algorithmic dependence and performative compliance?"
Aligned to the following goals:
Assess AI Literacy Development: Measure whether users learn to critically evaluate, appropriately utilize, and maintain autonomy when using AI assistance
Evaluate Learning Authenticity: Distinguish between genuine conceptual understanding and algorithmic mimicry through semantic depth, originality, and transfer measures
Measure Cognitive Autonomy: Determine whether AI assistance enhances or undermines independent reasoning capabilities in high-stakes contexts
Validate Automated Assessment: Implement and test computational frameworks for reliably measuring authentic learning vs. surface-level compliance
Demonstrate Practical Applications: Provide evidence-based insights for designing AI literacy training systems in professional high-stakes environments
Rather than embedding an open-ended chat interface, the system provides conditional, domain-specific LLM-generated guidance at each reasoning stage. This controlled interaction allows for measurement of user reliance, critical engagement, and learning outcomes, while preserving experimental control over bias exposure

Primary Research Goal
To investigate whether AI-generated feedback in high-stakes decision-making contexts supports genuine AI literacy and conceptual understanding, or merely encourages superficial compliance and algorithmic dependence in cognitive bias mitigation.
This project explores: Do users develop authentic AI literacy—understanding when and how to effectively use AI assistance—or do they simply mimic AI outputs without meaningful cognitive engagement?
System Goals
The system is designed to evaluate:
AI Literacy Development: Whether users learn to critically evaluate and appropriately utilize AI assistance
Cognitive Depth vs. Surface Learning: The difference between genuine understanding and algorithmic mimicry
Adaptive Decision-Making: Whether AI assistance enhances or undermines independent reasoning capabilities
Automated Assessment Validity: The reliability of computational methods to measure authentic learning vs. performative compliance
Transfer and Generalisation: Whether AI-supported learning transfers to novel high-stakes contexts
Research Questions
Bias Recognition & AI Literacy: Can users accurately identify cognitive biases in complex scenarios while demonstrating understanding of when AI assistance is appropriate?
Conceptual Understanding vs. Algorithmic Dependence: Do users show genuine comprehension of bias mitigation principles, or do they exhibit patterns of AI-dependent reasoning?
Strategic Application: Can users propose reasoned mitigation strategies that demonstrate both domain knowledge and appropriate AI literacy?
Transfer Learning: Are users able to apply bias mitigation understanding to unfamiliar contexts without relying on AI assistance?
AI Assistance Impact: Does exposure to LLM feedback enhance authentic learning and AI literacy, or does it promote algorithmic dependence and shallow reasoning?
Automated Assessment Reliability: Can computational scoring frameworks reliably distinguish between genuine AI literacy/understanding and surface-level algorithmic compliance?
