# 🎓 EduPlace AI — College Placement Analytics & AI Decision Support System

**Course**: Business Analytics and Visualization (BAV) • Course Code: `202046719`  
**Department**: B.Tech in Artificial Intelligence & Data Science (Semester VII)  
**Institution**: CVM University, Faculty of Engineering & Technology  

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.9+ installed. In your terminal:

```bash
# Clone or unzip into a folder, then enter the directory
cd "BAV project"

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
BAV project/
├── app.py                               # Main Streamlit web application (5 core modules)
├── deep_learning_model.py               # PyTorch ANN model definition & training script
├── requirements.txt                     # Python package dependencies
├── README.md                            # Project setup and documentation
├── PROJECT_PRESENTATION_AND_VIVA_GUIDE.md # Complete presentation script, viva Q&A, and ChatGPT prompt
├── college_placement_analytics.ipynb   # Exploratory Jupyter notebook
├── data/
│   ├── college_placement_data.csv       # Full cohort dataset (2,500 engineering students)
│   └── sample_roster_upload.csv        # Sample student roster for testing Module 5
└── model/
    ├── model_meta.json                  # Model metadata, scaler parameters, test metrics
    ├── placement_classifier_ann.pth     # PyTorch weights for Placement Classifier (91.73% Acc)
    ├── salary_regressor_ann.pth         # PyTorch weights for Salary Regressor (0.45 LPA MAE)
    └── scaler.pkl                       # Scikit-learn StandardScaler object
```

---

## 🏛️ Platform Architecture (5 Core Modules)

1. **📌 Executive Overview**: Macro recruitment KPIs, stream-wise conversion (AI & DS, CS, IT, EE, ME, CE), salary distribution, and recruitment tiering.
2. **📊 Exploratory & Statistical Analytics**:
   - *Sub-tab 1: Visual EDA* — Feature correlations, backlog penalties, and practical internship ROI.
   - *Sub-tab 2: Hypothesis Testing & ANOVA* — Welch's t-test ($t = 14.05, p < 0.0001$, Cohen's $d = 0.66$) and One-Way ANOVA across disciplines ($F = 32.42, p < 0.0001$) with Tukey HSD.
   - *Sub-tab 3: Regression & Six Sigma* — OLS Multiple Linear Regression ($R^2 = 0.772$, VIF $< 2.5$) and Six Sigma DMAIC framework ($2.18\sigma \to 4.0\sigma$).
3. **🧠 Deep Learning Architecture & Diagnostics**: PyTorch ANN learning curves, ROC-AUC curve ($0.9469$), confusion matrix, and permutation feature importance.
4. **🔮 Live Student Placement Predictor**: Real-time slider inference for placement likelihood & expected CTC with personalized 4-step prescriptive recommendations.
5. **📁 Batch Evaluation & Risk Triage**: Upload batch CSV rosters (try `data/sample_roster_upload.csv`) to automatically classify students into High, Moderate, and Low risk cohorts with downloadable reports.

---

## 🔄 Retraining the Models
To re-train the PyTorch neural networks from scratch:
```bash
python deep_learning_model.py
```
This updates the weights and evaluation figures in the `model/` folder automatically.

---

## 🎤 Presentation & Viva Preparation
Refer to `PROJECT_PRESENTATION_AND_VIVA_GUIDE.md` for:
- Step-by-step timed presentation script.
- Top faculty viva defense questions and mathematical answers.
- Ready-to-copy Master ChatGPT Prompt for mock viva drills.
