# Diagnosis Prediction Model for Pediatric Appendicitis using Artificial Intelligence Transformer Model

## Overview
This repository provides an AI-based clinical decision support system for diagnosing acute appendicitis in pediatric patients. It implements a custom tabular Transformer architecture and compares its diagnostic efficacy against baseline machine learning models, including Decision Trees, Gradient Boosting, and XGBoost.

## Repository Structure
```
├── 00_Dataset Integration/      # Raw dataset combination scripts
├── 01_Data Audit/               # Data quality and missing value audits
├── 02_Target Leak Control/      # Target leakage prevention and validation
├── 03_Preprocessing Pipeline/   # Automated scaling, encoding, and train/test splits
├── 04_Transformer Model/        # Tabular Transformer architecture and training
├── 05_Decision Trees/           # Baseline Decision Tree implementation
├── 06_Gradient_Boosting/        # Gradient Boosting model implementation
├── 07_XGBoost/                  # XGBoost model implementation
├── 08_GUI Application/          # Tkinter graphical user interface and backend
│   ├── gui/                     # UI layout and navigation modules
│   ├── saved_models/            # Centralized model artifacts (.pkl, .pt)
│   ├── backend_predictor.py     # Unified inference and threshold engine
│   ├── predictor_gui.py         # Main GUI execution script
│   └── run_tests.py             # Automated backend testing suite
├── Dataset 1/                   # Primary clinical and imaging dataset
├── Dataset 2/                   # Regensburg pediatric appendicitis dataset
├── model_comparison.py          # Comparative benchmarking script
└── requirements.txt             # Master dependency list
```

## System Architecture & Features
- **Multi-Model Consensus Engine**: Integrates four distinct algorithm families with individually optimized classification thresholds.
- **Missing Lab Robustness**: Automatically flags missing laboratory telemetry while maintaining model inference stability.
- **Centralized Artifact Management**: All production weights and preprocessors reside within a single directory (`saved_models/`) for deployment portability.
- **Interactive GUI**: Navigation-based desktop application designed for clinical review and educational demonstration.

## Input Features (38-Feature Contract)
- **Demographics**: Age, Weight, Height, BMI, Sex
- **Clinical Symptoms**: Lower Right Abdominal Pain, Migratory Pain, Loss of Appetite, Nausea, Coughing Pain, Dysuria, Peritonitis, Severity
- **Physical Examination**: Contralateral Rebound Tenderness, Ipsilateral Rebound Tenderness, Psoas Sign, Neutrophilia, Ketones in Urine, RBC in Urine, WBC in Urine
- **Laboratory Metrics**: WBC Count, RBC Count, Hemoglobin, RDW, Segmented Neutrophils, Thrombocyte Count, CRP, Neutrophil Percentage, plus 8 missing indicator flags.

## Model Performance & Validation Metrics
| Model | Accuracy | Precision | Sensitivity (Recall) | Specificity | PPV | NPV |
|-------|----------|-----------|----------------------|-------------|-----|-----|
| **Decision Tree** | 81.41% | 85.56% | 82.80% | 79.37% | 85.56% | 75.76% |
| **Gradient Boosting** | 89.74% | 90.53% | 92.47% | 85.71% | 90.53% | 88.52% |
| **XGBoost** | 88.46% | 88.66% | 92.47% | 82.54% | 88.66% | 88.14% |
| **Transformer** | 89.10% | 89.58% | 92.47% | 84.13% | 89.58% | 88.33% |

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Balmaaa/DiagnosisPredictionForPediatricAppendicitis.git
   cd DiagnosisPredictionForPediatricAppendicitis
   ```

2. Install dependencies from the root directory:
   ```bash
   pip install -r requirements.txt
   ```

### Running the GUI Application
```bash
cd "08_GUI Application"
python predictor_gui.py
```

### Running Automated Tests
```bash
cd "08_GUI Application"
python run_tests.py
```

## Clinical Interpretation
- **Sensitivity (Recall)**: Measures the proportion of actual appendicitis cases correctly identified, prioritizing high-risk detection to prevent perforation.
- **Specificity**: Measures the proportion of non-appendicitis cases correctly identified to minimize unnecessary interventions.
- **Positive Predictive Value (PPV) / Negative Predictive Value (NPV)**: Evaluates diagnostic reliability under clinical probability distributions.

## Ethical Considerations
This system functions as a clinical decision support tool designed to assist, not replace, professional medical evaluation and clinical judgment.

## License
This project is developed for academic thesis submission and research purposes.