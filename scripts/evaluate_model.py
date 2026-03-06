import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from src.config import TEST_FILE, BEST_MODEL_FILE

# Load data
df = pd.read_parquet(TEST_FILE)

X_test = df.drop(columns=["y"])
y_test = df["y"]

# Load model
model = joblib.load(BEST_MODEL_FILE)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))