# 🏥 Simulated Healthcare Claims Analytics

> **An end-to-end healthcare analytics pipeline exploring cost, utilization, quality metrics, and AI-powered predictions on a synthetic 10,000-member claims dataset.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualizations-11557c)
![License](https://img.shields.io/badge/License-MIT-green)
![Author](https://img.shields.io/badge/Author-Ritvik%20Raj%20Padige-navy)
![Date](https://img.shields.io/badge/Date-November%202025-blue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Analytics Modules](#-analytics-modules)
- [ML Models](#-ml-models)
- [Visualizations](#-visualizations)
- [Deliverables](#-deliverables)
- [Limitations & Future Work](#-limitations--future-work)
- [Author](#-author)

---

## 🔍 Overview

This project simulates a realistic healthcare claims environment to analyze healthcare spending efficiency, utilization patterns, and quality-of-care indicators. It was designed to demonstrate how **AI and data analytics** can solve real-world challenges in:

- 💰 Payer cost management and risk stratification
- 🏨 Hospital utilization and care coordination
- ✅ Quality improvement and HEDIS compliance
- 🤖 Predictive modeling for high-cost claimant identification

Because real claims data is protected under HIPAA, a **fully synthetic dataset** of 10,000 members was generated using statistically realistic distributions that mirror actual payer data.

---

## 📊 Key Findings

| Metric | Result | Benchmark |
|---|---|---|
| Total Simulated Spend | **$75.1M** | 10,000 members, 1 year |
| Average PMPM | **$626** | Industry avg ~$550–750 |
| PPO Plan PMPM | **$738** | Highest-cost plan |
| HDHP Plan PMPM | **$551** | Lowest-cost plan |
| Top 10% Cost Concentration | **42% of spend** | Industry ~60–70% |
| ER Visits per 1,000 | **193** | Benchmark: 150 |
| 30-Day Readmission Rate | **10.2%** | CMS target: <12% ✅ |
| Diabetic A1c Monitoring | **71.1%** | HEDIS target: ≥80% ⚠️ |
| Mammography Screening | **67.3%** | HEDIS target: ≥72% ⚠️ |
| High-Cost Classifier AUC | **0.616** | Moderate discrimination |

---

## 📁 Project Structure

```
healthcare-claims-analytics/
│
├── healthcare_claims_analytics.py   # Main analytics pipeline (all 6 modules)
│
├── outputs/                         # Generated charts and data exports
│   ├── 01_cost_analysis.png
│   ├── 02_utilization_analysis.png
│   ├── 03_quality_metrics.png
│   ├── 04_ml_models.png
│   └── summary_by_plan.csv
│
├── docs/                            # Project documents
│   ├── proposal.docx                # Research proposal
│   ├── report.docx                  # Full analytical report
│   └── presentation.pptx           # Executive slide deck
│
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Data Simulation** | NumPy, Pandas |
| **Machine Learning** | scikit-learn (GradientBoosting) |
| **Visualization** | Matplotlib |
| **Documents** | docx (Node.js), pptxgenjs |
| **Data Export** | CSV (Pandas) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/healthcare-claims-analytics.git
cd healthcare-claims-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full analytics pipeline
python healthcare_claims_analytics.py
```

### requirements.txt

```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
scikit-learn>=1.3
```

### Expected Output

```
=======================================================
  HEALTHCARE CLAIMS ANALYTICS — SIMULATION START
=======================================================

[1/6] Simulating 10,000 member records...
    ✓ Dataset shape: (10000, 22)
    ✓ Total simulated spend: $75,138,922

[2/6] Running Cost Analysis...
    ✓ Top 10% of members drive 42.0% of total cost
    ✓ Chart saved: 01_cost_analysis.png

[3/6] Running Utilization Analysis...
    ✓ ER visits/1,000: 193.4
    ✓ Chart saved: 02_utilization_analysis.png

[4/6] Running Quality Metrics Analysis...
    ✓ A1c monitoring rate: 71.1%
    ✓ Chart saved: 03_quality_metrics.png

[5/6] Building AI / ML Predictive Models...
    ✓ Regression — MAE: $5,501 | R²: 0.041
    ✓ Classifier — AUC-ROC: 0.616
    ✓ Chart saved: 04_ml_models.png

[6/6] Generating Summary Statistics Table...
    ✓ Summary saved: summary_by_plan.csv

=======================================================
  ALL ANALYSES COMPLETE
=======================================================
```

---

## 🔬 Analytics Modules

### Module 1 — Data Simulation
Generates a synthetic dataset of 10,000 members with:
- **Demographics:** Age (18–85, normal distribution), gender, region
- **Chronic conditions:** Diabetes, hypertension, COPD, heart disease, depression, obesity — all age-adjusted probabilities
- **Plan types:** HMO (30%), PPO (35%), HDHP (25%), EPO (10%)
- **Cost:** Lognormal base cost × chronic condition loading × age factor × plan factor
- **Utilization:** Poisson-distributed ER visits, inpatient admits, outpatient visits, and Rx fills
- **Quality flags:** HEDIS-inspired — A1c checks, mammography, 30-day readmissions, preventive visits

### Module 2 — Cost Analysis
- Per-Member-Per-Month (PMPM) by plan type and age band
- Pareto analysis: top X% of members → Y% of total spend
- Cost by chronic condition burden (0–6 conditions)
- High-cost member flagging (top 10th percentile)

### Module 3 — Utilization Analysis
- ER visits per 1,000 members by plan
- Inpatient admissions per 1,000 members by plan
- Utilization rates segmented by chronic condition count
- Rx fill distribution across the population

### Module 4 — Quality Metrics
- **Diabetic A1c Monitoring Rate** (HEDIS: HbA1c Testing)
- **Mammography Screening Rate** (HEDIS: Breast Cancer Screening)
- **30-Day Hospital Readmission Rate** by plan type
- **Composite Quality Score** per chronic burden cohort

### Module 5 — AI & ML Modeling
- **GBM Regressor** to predict log-transformed annual cost (MAE, R²)
- **GBM Classifier** to flag high-cost members (AUC-ROC, feature importance)
- ROC curve visualization
- Feature importance ranking for clinical interpretability

### Module 6 — Summary Reporting
- Aggregated summary statistics table by plan type (CSV export)
- All charts auto-saved to `/outputs/`

---

## 🤖 ML Models

### Cost Prediction (Regression)
```
Algorithm:    Gradient Boosting Regressor
Target:       log(total_annual_cost)
Train/Test:   80% / 20%  (n=8,000 / n=2,000)
MAE:          $5,501
R²:           0.041
Top Features: chronic_count, er_visits, rx_fills, ip_admits, age
```

### High-Cost Classifier (Binary Classification)
```
Algorithm:    Gradient Boosting Classifier
Target:       high_cost_flag (top 10th percentile = 1)
Train/Test:   80% / 20%
AUC-ROC:      0.616
Top Features: chronic_count, er_visits, rx_fills, ip_admits
```

> **Note:** The modest R² and AUC reflect the intentional randomness in the simulation. Real-world models incorporating ICD-10 diagnoses, DRG codes, and lab values typically achieve AUC of 0.75–0.85.

---

## 📈 Visualizations

Four dashboard charts are auto-generated when you run the pipeline:

| File | Contents |
|---|---|
| `01_cost_analysis.png` | PMPM by plan, cost distribution, Pareto curve, cost by chronic burden & age band |
| `02_utilization_analysis.png` | ER/IP per 1,000 by plan, utilization by chronic burden, Rx distribution |
| `03_quality_metrics.png` | Quality compliance rates, readmissions by plan, composite quality score |
| `04_ml_models.png` | Feature importance (regression & classifier), ROC curve |

---

## 📦 Deliverables

This project includes four complete deliverables:

| Deliverable | Description |
|---|---|
| `healthcare_claims_analytics.py` | Full Python pipeline — simulation → EDA → ML → charts |
| `proposal.docx` | 8-section research proposal with methodology and timeline |
| `report.docx` | Full analytical report with embedded charts and recommendations |
| `presentation.pptx` | 9-slide executive deck with all findings and strategic recommendations |

---

## ⚠️ Limitations & Future Work

**Current Limitations:**
- Based entirely on synthetic data — does not replicate real adjudication logic, ICD-10 hierarchies, or provider variation
- ML model performance is bounded by the simulated feature set
- Specialty pharmacy and behavioral health are not explicitly modeled

**Planned Extensions:**
- [ ] Multi-year temporal modeling for trend analysis
- [ ] Social Determinants of Health (SDOH) integration
- [ ] Provider-level network analysis (high-value vs. low-value)
- [ ] NLP on clinical notes for richer risk stratification
- [ ] Interactive Streamlit/Dash dashboard
- [ ] SHAP explainability integration

---

## 🎯 Skills Demonstrated

`Python` `Data Simulation` `Exploratory Data Analysis` `Healthcare Analytics`  
`Machine Learning` `Gradient Boosting` `Classification` `Regression`  
`Feature Engineering` `HEDIS Metrics` `Claims Analytics` `Data Visualization`  
`Risk Stratification` `Care Management Analytics` `scikit-learn` `Matplotlib` `Pandas`

---

## 👤 Author

**Ritvik Raj Padige**  
Student | Healthcare Analytics & AI  
📧 rpadige@yahoo.com  
🐙 [GitHub](https://github.com/ritvikrajpadige)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> *This project was developed in November 2025 by Ritvik Raj Padige to deepen expertise in applying AI and analytics to solve real-world healthcare challenges. All data is fully synthetic and does not represent any real individuals or organizations.*
