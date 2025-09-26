## Telco Customer Churn — Streamlit App

Live at https://telco-churn-predictionn.streamlit.app/

Interactive Streamlit dashboard to explore, clean, and model Telco Customer Churn data. The app loads a CSV (upload your own or use the bundled dataset), builds a robust ML pipeline, and shows metrics and explainability visuals.

### Features
- Raw, Clean, Insights, and Scenario tabs for a simple end-to-end flow
- CSV upload or fallback to `telco_churn.csv`
- Sidebar controls:
  - Test size
  - Random seed
  - RandomForest number of trees
- Robust sklearn Pipeline:
  - Numeric imputation (median)
  - Categorical imputation (most frequent) + OneHotEncoder with unknown handling
- Metrics and charts:
  - Classification report
  - Confusion matrix heatmap
  - ROC curve with AUC
  - Top feature importances (key drivers)

### Project Structure
```
telco-customer-churn/
├─ app.py                # Streamlit app entry point
├─ telco_churn.csv       # Sample dataset (Kaggle Telco Customer Churn)
├─ README.md             # This file
└─ requirements.txt      # (Recommended) Python dependencies
```

### Requirements
- Python 3.9+ (3.11 recommended)
- Packages: streamlit, pandas, seaborn, matplotlib, scikit-learn

Create a `requirements.txt` (if not present):
```
streamlit
pandas
seaborn
matplotlib
scikit-learn
```

### Run Locally
1) Create and activate a virtual environment (recommended):
```
python3 -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3) Start the app:
```
streamlit run /Users/guzalmaksud/Documents/telco-customer-churn/app.py
```
The terminal will show a local URL (e.g., http://localhost:8501).

### Data Input (CSV)
- Format: CSV (.csv), UTF-8, includes a header row
- Required column: `Churn` with values `Yes`/`No`
- Optional: `customerID` (display only)
- Other columns: numeric and categorical features from the Telco dataset (unseen categories are handled)

If you upload nothing, the app uses `telco_churn.csv` in the repo.

### How It Works
1) Raw tab: Preview the dataset
2) Clean tab: Basic type fixes, target mapping, and preview before encoding
3) Insights tab:
   - Splits data using your sidebar settings
   - Builds an sklearn Pipeline with imputers and OneHotEncoder
   - Trains a RandomForestClassifier
   - Displays classification report, confusion matrix, ROC curve (AUC), and top feature importances
4) Scenario tab: Simple “what-if” slider to simulate a churn reduction effect

### Tuning
- Test size (%): larger is more stable metrics, less training data
- Random state: controls reproducibility of split and model randomness
- RandomForest trees: more trees often improve stability/accuracy up to a point
