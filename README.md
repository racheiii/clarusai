# ClārusAI

**Dissertation:** 
Beyond Surface Learning: Evaluating AI Literacy Through LLM-Supported Cognitive Bias Simulation  
**Author:** 
Rachel Seah Yan Ting
**Supervising Professor:** 
Graham Roberts
**Institution:** 
University College London (UCL) — MSc Computer Science  

---

## Overview

ClārusAI is a Streamlit-based research platform developed for the MSc summer dissertation project.  
It evaluates whether LLM-assisted training fosters genuine reasoning skills or encourages algorithmic dependence in high-stakes decision-making contexts.

The system delivers structured cognitive bias scenarios in military, medical, and emergency domains, using a 2×2×3 factorial design:

- **User Expertise:** Novice / Expert  
- **AI Assistance:** Enabled / Disabled  
- **Bias Type:** Confirmation / Anchoring / Availability  

Dataset generation is fully automated using simulated user personas.  

A live demonstration mode is included for illustration but is not used in the main dataset or analysis.

---

## Modes of Operation

1. **Dataset Generation Mode**  
   - Runs all factorial combinations using LLaMA 3.2 via Ollama.  
   - Produces raw JSON sessions (no scoring at this stage).  

2. **Dashboard Mode**  
   - Loads datasets and computes scores on demand using a 6-dimension rubric:  
     1. Semantic Similarity  
     2. Bias Recognition  
     3. Depth of Understanding  
     4. Mitigation Strategy Quality  
     5. Domain Transferability  
     6. Metacognitive Awareness  
   - Runs statistical analyses aligned to the research questions (RQ1–RQ3) and exports results.  

3. **Live Demo Mode**  
   - 4-stage interactive interface for human users.  
   - For demonstration purposes only; excluded from dataset generation and analysis.

---

## Installation

1. **Clone the repository**
```bash
git clone < xxx >
cd clarusai
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **(Optional) Configure environment variables**  
   Copy `.env.example` to `.env` and adjust if required.  
   Defaults in `config.py` will run without modification.

---

## Running the Application

### Option A — Full App
```bash
streamlit run Home.py
```
- From the homepage:
  - **"Start Training"** launches the demo interface (for illustration only).  
  - **"Access Research Dashboard"** opens the dataset generation and analysis tools.

### Option B — Directly Open the Dashboard
```bash
streamlit run pages/02_Dashboard.py
```
- Tab 0: Generate a new simulated dataset or load an existing one under `exports/simulated_datasets/`.  
- Tabs 1–3: View RQ1–RQ3 analyses.  
- Tab 4: Export processed datasets (CSV/JSON).  
- Tab 5: View results summary.

### Option C — Batch Dataset Generation from CLI
```bash
python -m src.research_pipeline
```
- Creates a folder under `exports/simulated_datasets/` containing JSON session files.
- Load this folder in the Dashboard to compute scores and run analyses.

---
## Research Questions & Dashboard Mapping

The analyses in the Dashboard directly address the three core research questions:

**RQ1 — AI vs. Non-AI Performance**  
How does access to LLM assistance affect overall response quality in cognitive bias decision-making scenarios?  
→ Addressed in **Tab 1**: AI vs. Non-AI Performance.

**RQ2 — Parroting vs. Reasoning**  
To what extent do LLM-assisted responses exhibit parroting behaviour (high similarity to ideal answers but low originality) compared to unassisted responses?  
→ Addressed in **Tab 2**: Parroting Detection.

**RQ3 — Transfer Learning Across Domains**  
How effectively do participants (or simulated personas) apply bias mitigation strategies learned in one domain (e.g., medical) to a different domain (e.g., military or emergency response)?  
→ Addressed in **Tab 3**: Transfer Learning.
---

## Marker Instructions

To replicate the core results presented in the dissertation:

1. **Launch the Dashboard** (`streamlit run Home.py` → "Access Research Dashboard")  
2. **Generate Dataset** in Tab 0 (or load an existing one under `exports/simulated_datasets/`).  
3. **Review Analyses** in Tabs 1–3 for AI vs Non-AI performance, parroting detection, and transfer learning.  
4. **Export Data** in Tab 4 for verification.  
5. **Review Summary** in Tab 5 for a concise overview.  

The **Live Demo** is for demonstration only and does not influence the results.

---

## Research Notes

- Bias type is hidden until scenario completion to maintain bias-blind methodology.  
- Scoring is performed only in Dashboard mode for a clean separation between data generation and evaluation.  
- All datasets are synthetic, avoiding the need for human subjects in the main study.  
- Outputs are reproducible — every dataset run is timestamped.