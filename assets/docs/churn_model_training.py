import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                              precision_score, recall_score, f1_score, accuracy_score)

np.random.seed(42)
N = 6000

age_bands = ["18-24","25-34","35-44","45-54","55-64","65+"]
contract_types = ["Month-to-month","One-year","Two-year"]
payment_methods = ["Direct debit","Credit card (tokenised)","Bank transfer"]

df = pd.DataFrame({
    "customer_id": [f"NT-{i:05d}" for i in range(N)],
    "age_band": np.random.choice(age_bands, N, p=[0.14,0.22,0.22,0.18,0.14,0.10]),
    "tenure_months": np.random.gamma(4, 8, N).clip(1, 96).astype(int),
    "contract_type": np.random.choice(contract_types, N, p=[0.55,0.30,0.15]),
    "monthly_charge": np.round(np.random.normal(38, 12, N).clip(12, 95), 2),
    "monthly_minutes": np.random.gamma(3, 120, N).clip(0, 3000).astype(int),
    "monthly_data_gb": np.round(np.random.gamma(2, 6, N).clip(0, 100), 1),
    "support_tickets_6m": np.random.poisson(0.8, N).clip(0, 12),
    "payment_method": np.random.choice(payment_methods, N, p=[0.45,0.40,0.15]),
})

# Underlying churn propensity — driven by genuine behavioural signals
logit = (
    -1.8
    - 0.035 * df["tenure_months"]
    + 0.55 * (df["contract_type"] == "Month-to-month").astype(int)
    + 0.30 * df["support_tickets_6m"]
    + 0.012 * df["monthly_charge"]
    - 0.004 * df["monthly_minutes"] / 10
)

# Intentional mild age-band skew in the DATA GENERATION PROCESS (not the model logic),
# simulating a real-world scenario where an under-35 cohort has historically had
# noisier support-ticket logging, producing a proxy effect the fairness test should catch.
skew = df["age_band"].isin(["18-24","25-34"]).astype(int) * 0.28
logit = logit + skew

prob = 1 / (1 + np.exp(-logit))
df["churned"] = np.random.binomial(1, prob)

df.to_csv("/home/claude/churn-model/novatel_churn_dataset.csv", index=False)

# ---- Train/test split ----
feature_cols = ["age_band","tenure_months","contract_type","monthly_charge",
                 "monthly_minutes","monthly_data_gb","support_tickets_6m","payment_method"]
X = df[feature_cols]
y = df["churned"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.25, random_state=42, stratify=y
)

categorical = ["age_band","contract_type","payment_method"]
numeric = ["tenure_months","monthly_charge","monthly_minutes","monthly_data_gb","support_tickets_6m"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", StandardScaler(), numeric),
])

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("=== Model Performance ===")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 score:  {f1:.3f}")
print(f"ROC AUC:   {auc:.3f}")
print()
print(classification_report(y_test, y_pred, target_names=["Retained","Churned"]))

# ---- Confusion matrix plot ----
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor="#0b1220")
ax.set_facecolor("#0b1220")
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_xticklabels(["Retained","Churned"], color="#e8eefc")
ax.set_yticks([0,1]); ax.set_yticklabels(["Retained","Churned"], color="#e8eefc")
ax.set_xlabel("Predicted", color="#a9b7d6")
ax.set_ylabel("Actual", color="#a9b7d6")
ax.set_title("Confusion Matrix — Churn Model (test set)", color="#e8eefc", fontsize=12)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("/home/claude/churn-model/confusion_matrix.png", dpi=150, facecolor="#0b1220")
plt.close()

# ---- Feature importance (coefficients) ----
ohe = model.named_steps["prep"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(categorical)
all_names = list(cat_names) + numeric
coefs = model.named_steps["clf"].coef_[0]
imp = pd.Series(coefs, index=all_names).sort_values()

fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0b1220")
ax.set_facecolor("#0b1220")
colors = ["#fb7185" if v < 0 else "#67e8f9" for v in imp.values]
ax.barh(imp.index, imp.values, color=colors)
ax.axvline(0, color="#a9b7d6", linewidth=0.8)
ax.set_title("Feature Coefficients — Churn Model", color="#e8eefc", fontsize=12)
ax.tick_params(colors="#a9b7d6", labelsize=8)
for spine in ax.spines.values():
    spine.set_color("#334155")
plt.tight_layout()
plt.savefig("/home/claude/churn-model/feature_importance.png", dpi=150, facecolor="#0b1220")
plt.close()

# ---- Fairness check: positive prediction (churn-flag) rate by age band ----
test_df = df.loc[idx_test].copy()
test_df["predicted"] = y_pred

flag_rates = test_df.groupby("age_band")["predicted"].mean().reindex(age_bands)
overall_rate = test_df["predicted"].mean()

print("\n=== Fairness Check: Churn-flag rate by age band ===")
print(flag_rates)
print(f"\nOverall flag rate: {overall_rate:.3f}")

min_rate = flag_rates.min()
max_rate = flag_rates.max()
disparate_impact_ratio = min_rate / max_rate
print(f"Disparate impact ratio (min/max group rate): {disparate_impact_ratio:.3f}")
print("(Four-fifths / 80% rule threshold commonly used as a screening flag: ratio < 0.80 warrants investigation)")

fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="#0b1220")
ax.set_facecolor("#0b1220")
colors2 = ["#fb7185" if r > overall_rate * 1.15 else "#67e8f9" for r in flag_rates.values]
bars = ax.bar(flag_rates.index, flag_rates.values, color=colors2)
ax.axhline(overall_rate, color="#fbbf24", linestyle="--", linewidth=1.2, label=f"Overall rate ({overall_rate:.2f})")
ax.set_title("Churn-Flag Rate by Age Band (Fairness Check)", color="#e8eefc", fontsize=12)
ax.set_ylabel("Predicted churn rate", color="#a9b7d6")
ax.tick_params(colors="#a9b7d6")
for spine in ax.spines.values():
    spine.set_color("#334155")
ax.legend(facecolor="#0b1220", edgecolor="#334155", labelcolor="#e8eefc")
plt.tight_layout()
plt.savefig("/home/claude/churn-model/fairness_by_age_band.png", dpi=150, facecolor="#0b1220")
plt.close()

# Save summary results to a text file for the model card
with open("/home/claude/churn-model/results_summary.txt", "w") as f:
    f.write("=== Model Performance ===\n")
    f.write(f"Accuracy:  {acc:.3f}\n")
    f.write(f"Precision: {prec:.3f}\n")
    f.write(f"Recall:    {rec:.3f}\n")
    f.write(f"F1 score:  {f1:.3f}\n")
    f.write(f"ROC AUC:   {auc:.3f}\n\n")
    f.write("=== Fairness Check: Churn-flag rate by age band ===\n")
    f.write(flag_rates.to_string())
    f.write(f"\n\nOverall flag rate: {overall_rate:.3f}\n")
    f.write(f"Disparate impact ratio (min/max): {disparate_impact_ratio:.3f}\n")

print("\nSaved: confusion_matrix.png, feature_importance.png, fairness_by_age_band.png, results_summary.txt, novatel_churn_dataset.csv")
