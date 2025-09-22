import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.set_page_config(page_title="Mini Data Platform", layout="wide")

# ---------------- RAW LAYER ----------------
st.title("Tiny Data Platform: Raw → Clean → Insights")
st.header("Raw Data Layer")

@st.cache_data
def load_data():
    # Kaggle "Telco Customer Churn" CSV in the app folder
    df = pd.read_csv("telco_churn.csv")
    return df

df_raw = load_data()

st.write("Raw dataset preview:")
st.dataframe(df_raw.head())

# ---------------- CLEAN LAYER ----------------
st.header("Cleaned Data Layer")

# Work on a copy
df = df_raw.copy()

# Keep ID for display only; drop it from features later
if "customerID" in df.columns:
    customer_ids = df["customerID"].copy()
else:
    customer_ids = None

# Coerce numeric columns BEFORE dropping NA so 'TotalCharges' becomes numeric
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Basic cleaning
# (You can replace with smarter imputations later)
df = df.dropna()

# Target to 0/1
if "Churn" in df.columns:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

# Drop ID so it doesn't get one-hot encoded
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# One-hot encode *only* categoricals
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Preview (optionally add ID back just for the table)
preview = df.copy()
if customer_ids is not None and len(preview) == len(customer_ids.loc[preview.index]):
    preview = preview.copy()
    preview.insert(0, "customerID", customer_ids.loc[preview.index])

st.write("After cleaning (features ready for modeling):")
st.dataframe(preview.head())

# ---------------- INSIGHTS LAYER ----------------
st.header("Insights Layer: Churn Prediction")

# Split features/target
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.text("Model Performance:")
st.text(classification_report(y_test, y_pred))

# Feature importance
st.subheader("Key Drivers of Churn")
feat_importances = pd.Series(model.feature_importances_, index=X.columns)\
                    .sort_values(ascending=False).head(10)

fig, ax = plt.subplots()
sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=ax)
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
st.pyplot(fig)

# ---------------- BUSINESS SCENARIO ----------------
st.header("Business Scenario: What If Analysis")

st.caption(
    "Simple illustrative slider: assume a discount for month-to-month customers reduces churn proportionally."
)
contract_discount = st.slider(
    "Simulate discount % for Month-to-Month customers", 0, 50, 10
)

base_churn = y.mean()
# toy assumption: each 10% discount gives ~5% relative reduction in churn
new_churn = base_churn * (1 - (contract_discount / 100) * 0.5)

st.write(f"Base churn rate: {base_churn:.2f}")
st.write(f"Simulated churn rate after {contract_discount}% discount: {new_churn:.2f}")
