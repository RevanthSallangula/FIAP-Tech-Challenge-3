import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

from src.config import TEST_FILE, BEST_MODEL_FILE, REPORTS_DIR


# ---------------------------
# Load test data
# ---------------------------

df = pd.read_parquet(TEST_FILE)

X_test = df.drop(columns=["y"])
y_test = df["y"]

# ---------------------------
# Load trained model
# ---------------------------

model = joblib.load(BEST_MODEL_FILE)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ---------------------------
# Confusion Matrix
# ---------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png")
plt.show()

# ---------------------------
# ROC Curve
# ---------------------------

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(f"{REPORTS_DIR}/roc_curve.png")
plt.show()

# ---------------------------
# Precision Recall Curve
# ---------------------------

precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.savefig(f"{REPORTS_DIR}/precision_recall_curve.png")
plt.show()

# ---------------------------
# Probability Distribution
# ---------------------------

plt.figure()
plt.hist(y_prob, bins=30)

plt.title("Prediction Probability Distribution")
plt.xlabel("Probability of Subscription")
plt.ylabel("Frequency")

plt.savefig(f"{REPORTS_DIR}/probability_distribution.png")
plt.show()

# ---------------------------
# Feature Importance
# ---------------------------

try:
    importances = model.lgbm_model.feature_importances_

    plt.figure()
    plt.bar(range(len(importances)), importances)

    plt.title("LightGBM Feature Importance")

    plt.savefig(f"{REPORTS_DIR}/feature_importance.png")
    plt.show()

except:
    print("Feature importance not available.")



