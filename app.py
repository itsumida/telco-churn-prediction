import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Mini Data Platform", layout="wide")

# ---------------- RAW LAYER ----------------
st.title("Tiny Data Platform: Raw → Clean → Insights")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Telco CSV", type=["csv"]) 
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
n_estimators = st.sidebar.slider("RandomForest n_estimators", min_value=50, max_value=300, value=100, step=50)

tabs = st.tabs(["Raw", "Clean", "Insights", "Scenario"])

@st.cache_data
def load_data(file_bytes: bytes | None):
    # Load from uploaded file if provided, otherwise use local CSV
    if file_bytes is not None:
        return pd.read_csv(file_bytes)
    return pd.read_csv("telco_churn.csv")

with tabs[0]:
    st.header("Raw Data Layer")
    try:
        df_raw = load_data(uploaded_file)
        st.write("Raw dataset preview:")
        st.dataframe(df_raw.head())
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# ---------------- CLEAN LAYER ----------------
with tabs[1]:
    st.header("Cleaned Data Layer")
    # Work on a copy
    df = df_raw.copy()

    # Keep ID for display only; drop it from features later
    if "customerID" in df.columns:
        customer_ids = df["customerID"].copy()
    else:
        customer_ids = None

    # Coerce numeric columns (notably 'TotalCharges')
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Target to 0/1
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("Int64")

    # Drop ID so it doesn't get one-hot encoded
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"]) 

    # Show a simple preview of cleaned schema (without heavy transformations)
    st.write("Cleaned dataset preview (pre-encoding/imputation):")
    st.dataframe(df.head())

# ---------------- INSIGHTS LAYER ----------------
with tabs[2]:
    st.header("Insights Layer: Churn Prediction")

    # Basic validations
    if "Churn" not in df.columns:
        st.error("Column 'Churn' not found in the dataset.")
        st.stop()

    # Split features/target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    # Identify column types for preprocessing
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state))

    clf = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100.0, random_state=int(random_state), stratify=y
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Try proba for ROC; if not available, skip
    y_proba = None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]

    st.subheader("Model Performance")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(pd.DataFrame(cm, index=[0, 1], columns=[0, 1]), annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    # ROC curve and AUC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"ROC AUC = {auc_val:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    # Feature importance (approximate using permutation not added here) – use model importances if available
    if hasattr(clf.named_steps["model"], "feature_importances_"):
        # Build feature names after OneHotEncoder
        ohe = clf.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist() if categorical_features else []
        feature_names = numeric_features + cat_feature_names
        importances = clf.named_steps["model"].feature_importances_
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
        st.subheader("Key Drivers of Churn")
        fig_fi, ax_fi = plt.subplots()
        sns.barplot(x=fi.values, y=fi.index, ax=ax_fi)
        ax_fi.set_xlabel("Importance")
        ax_fi.set_ylabel("Feature")
        st.pyplot(fig_fi)

# ---------------- BUSINESS SCENARIO ----------------
with tabs[3]:
    st.header("Business Scenario: What If Analysis")
    st.caption(
        "Simple illustrative slider: assume a discount for month-to-month customers reduces churn proportionally."
    )
    contract_discount = st.slider(
        "Simulate discount % for Month-to-Month customers", 0, 50, 10
    )

    # Reuse base churn from available data if present
    try:
        base_churn = y.mean()
    except Exception:
        # Fallback estimate if not computed yet
        base_churn = 0.5

    # toy assumption: each 10% discount gives ~5% relative reduction in churn
    new_churn = base_churn * (1 - (contract_discount / 100) * 0.5)

    st.write(f"Base churn rate: {base_churn:.2f}")
    st.write(f"Simulated churn rate after {contract_discount}% discount: {new_churn:.2f}")
