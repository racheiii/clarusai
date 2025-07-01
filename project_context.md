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

├── app.py                              # Streamlit main entry point & page routing

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

│   ├── claude_service.py              # Claude API integration & persona-based response generation

│   ├── scoring_engine.py              # 6-dimensional assessment algorithms (similarity, originality, bias recognition, etc.)

│   ├── research_pipeline.py           # Automated 72-response generation & live statistical validation

│   └── visualizations.py             # Chart generation for scoring displays & research analytics

│

├── pages/

│   ├── 01_Home.py              # Research introduction, project overview, pathway selection

│   ├── 02_Scenarios.py             # User training interface with scenario selection & AI assistance toggle

│   ├── 03_Assessment.py           # Live 6-dimensional scoring demonstration & educational feedback

│   ├── 04_Results.py              # Personal analytics, session summary, research contribution consent

│   ├── 05_Dashboard.py            # Research automation control centre & hypothesis testing

│   └── 06_Methodology.py          # Algorithm transparency, validation metrics, reproducibility documentation

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