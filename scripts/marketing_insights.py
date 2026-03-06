import pandas as pd
import matplotlib.pyplot as plt
import os

from src.config import TRAIN_FILE, REPORTS_DIR

# Create reports folder if missing
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_parquet(TRAIN_FILE)

# -----------------------------
# 1. Subscription Rate by Job
# -----------------------------

plt.figure(figsize=(10,6))
df.groupby("job")["y"].mean().sort_values().plot(kind="barh")
plt.title("Subscription Rate by Profession")
plt.xlabel("Subscription Probability")
plt.ylabel("Profession")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/success_by_profession.png")
plt.close()

# -----------------------------
# 2. Subscription by Age Group
# -----------------------------

df["age_group"] = pd.cut(
    df["age"],
    bins=[18,30,40,50,60,100],
    labels=["18-30","30-40","40-50","50-60","60+"]
)

plt.figure(figsize=(8,5))
df.groupby("age_group")["y"].mean().plot(kind="bar")
plt.title("Subscription Rate by Age Group")
plt.ylabel("Subscription Probability")
plt.xlabel("Age Group")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/success_by_age.png")
plt.close()

# -----------------------------
# 3. Campaign Attempts vs Success
# -----------------------------

plt.figure(figsize=(8,5))
df.groupby("campaign")["y"].mean().plot()
plt.title("Subscription Rate vs Number of Campaign Contacts")
plt.xlabel("Number of Contacts")
plt.ylabel("Subscription Probability")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/campaign_effectiveness.png")
plt.close()

# -----------------------------
# 4. Previous Campaign Outcome
# -----------------------------

plt.figure(figsize=(8,5))
df.groupby("poutcome")["y"].mean().plot(kind="bar")
plt.title("Subscription Rate by Previous Campaign Outcome")
plt.ylabel("Subscription Probability")
plt.xlabel("Previous Outcome")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/previous_campaign_effect.png")
plt.close()

# -----------------------------
# 5. Contact Method Effectiveness
# -----------------------------

plt.figure(figsize=(8,5))
df.groupby("contact")["y"].mean().plot(kind="bar")
plt.title("Subscription Rate by Contact Method")
plt.ylabel("Subscription Probability")
plt.xlabel("Contact Method")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/contact_method_effect.png")
plt.close()

# -----------------------------
# 6. Balance Distribution
# -----------------------------

plt.figure(figsize=(8,5))
plt.hist(df["balance"])
plt.title("Distribution of Customer Account Balance")
plt.xlabel("Balance")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/balance_distribution.png")
plt.close()

# -----------------------------
# 7. Subscription by Loan Status
# -----------------------------

plt.figure(figsize=(8,5))
df.groupby("loan")["y"].mean().plot(kind="bar")
plt.title("Subscription Rate by Personal Loan Status")
plt.ylabel("Subscription Probability")
plt.xlabel("Loan Status")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/loan_vs_subscription.png")
plt.close()

print("Marketing insight graphs saved in /reports folder.")