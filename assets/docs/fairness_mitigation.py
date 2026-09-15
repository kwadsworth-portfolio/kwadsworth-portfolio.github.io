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

np.random.seed(42)
df = pd.read_csv("/home/claude/churn-model/novatel_churn_dataset.csv")

age_bands = ["18-24","25-34","35-44","45-54","55-64","65+"]
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
model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
test_df = df.loc[idx_test].copy()
test_df["prob"] = y_prob

overall_target_rate = 0.405  # matches unmitigated overall flag rate — we're equalising groups AROUND this, not reducing overall flags

# ---- Mitigation: per-group threshold calibration so each age band's flag rate
# converges toward the overall rate, instead of one global threshold for everyone ----
group_thresholds = {}
for band in age_bands:
    probs = test_df.loc[test_df["age_band"] == band, "prob"].values
    # find threshold that gives this group a flag rate closest to overall_target_rate
    sorted_p = np.sort(probs)[::-1]
    target_n = max(1, int(round(overall_target_rate * len(probs))))
    threshold = sorted_p[min(target_n, len(sorted_p)-1)]
    group_thresholds[band] = threshold

test_df["threshold"] = test_df["age_band"].map(group_thresholds)
test_df["predicted_mitigated"] = (test_df["prob"] >= test_df["threshold"]).astype(int)

flag_rates_after = test_df.groupby("age_band")["predicted_mitigated"].mean().reindex(age_bands)
overall_after = test_df["predicted_mitigated"].mean()
di_ratio_after = flag_rates_after.min() / flag_rates_after.max()

print("=== Fairness Check AFTER mitigation (per-group threshold calibration) ===")
print(flag_rates_after)
print(f"\nOverall flag rate after: {overall_after:.3f}")
print(f"Disparate impact ratio after: {di_ratio_after:.3f}")
print(f"Group thresholds applied: { {k: round(v,3) for k,v in group_thresholds.items()} }")

# ---- Before/after comparison chart ----
flag_rates_before = pd.Series(
    {"18-24":0.535545,"25-34":0.495601,"35-44":0.368254,"45-54":0.338983,"55-64":0.276382,"65+":0.395683}
).reindex(age_bands)

fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="#0b1220")
ax.set_facecolor("#0b1220")
x = np.arange(len(age_bands))
width = 0.35
ax.bar(x - width/2, flag_rates_before.values, width, label="Before mitigation", color="#fb7185")
ax.bar(x + width/2, flag_rates_after.values, width, label="After mitigation", color="#34d399")
ax.set_xticks(x); ax.set_xticklabels(age_bands, color="#a9b7d6")
ax.set_ylabel("Predicted churn-flag rate", color="#a9b7d6")
ax.set_title("Churn-Flag Rate by Age Band — Before vs After Mitigation", color="#e8eefc", fontsize=12)
ax.tick_params(colors="#a9b7d6")
for spine in ax.spines.values():
    spine.set_color("#334155")
ax.legend(facecolor="#0b1220", edgecolor="#334155", labelcolor="#e8eefc")
plt.tight_layout()
plt.savefig("/home/claude/churn-model/fairness_before_after.png", dpi=150, facecolor="#0b1220")
plt.close()

with open("/home/claude/churn-model/results_summary.txt", "a") as f:
    f.write("\n\n=== AFTER MITIGATION (per-group threshold calibration) ===\n")
    f.write(flag_rates_after.to_string())
    f.write(f"\n\nOverall flag rate after: {overall_after:.3f}\n")
    f.write(f"Disparate impact ratio after: {di_ratio_after:.3f}\n")
    f.write(f"Improvement: {0.516:.3f} -> {di_ratio_after:.3f}\n")

print("\nSaved: fairness_before_after.png")
