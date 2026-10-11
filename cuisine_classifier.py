from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # saves plots to files (works everywhere)


warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# SETTINGS (feel free to change these)
# ----------------------------------------------------------------------
DATA_FILE = "Dataset_.csv"     # put the CSV in the same folder as this script
MIN_SAMPLES_PER_CUISINE = 100  # cuisines rarer than this are grouped as "Other"
TEST_SIZE = 0.2
RANDOM_STATE = 42
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# STEP 1: LOAD DATA
# ----------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)
df = pd.read_csv(DATA_FILE)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nMissing values per column (only columns with missing data):")
missing = df.isna().sum()
print(missing[missing > 0])


# ----------------------------------------------------------------------
# STEP 2: PREPROCESSING
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

# 2a. Missing values: rows with no cuisine can't be used for training
df = df.dropna(subset=["Cuisines"]).copy()
print(f"After dropping rows with no cuisine: {len(df)} rows")

# 2b. Many restaurants list several cuisines, e.g. "North Indian, Chinese".
#     To keep this a simple single-label problem we use the FIRST cuisine
#     listed as the restaurant's main cuisine.
df["Primary Cuisine"] = df["Cuisines"].str.split(",").str[0].str.strip()
print(f"Distinct primary cuisines: {df['Primary Cuisine'].nunique()}")

# 2c. Rare cuisines (only a handful of restaurants) can't be learned
#     reliably, so we group them into "Other".
counts = df["Primary Cuisine"].value_counts()
keep = counts[counts >= MIN_SAMPLES_PER_CUISINE].index
df["Target"] = df["Primary Cuisine"].where(
    df["Primary Cuisine"].isin(keep), "Other")
print(f"Classes used for the model ({df['Target'].nunique()}):")
print(df["Target"].value_counts().to_string())

# 2d. Choose input features
numeric_features = ["Country Code", "Longitude", "Latitude",
                    "Average Cost for two", "Price range",
                    "Aggregate rating", "Votes"]
categorical_features = ["City", "Has Table booking", "Has Online delivery",
                        "Is delivering now"]
# names like "Pizza Hut" or "Sweet Bakers" are strong hints
text_feature = "Restaurant Name"

# Fill any leftover missing values (safety net)
for col in numeric_features:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_features + [text_feature]:
    df[col] = df[col].fillna("Unknown")

X = df[numeric_features + categorical_features + [text_feature]]
y = df["Target"]

# 2e. Encoding: done inside a ColumnTransformer so it is fitted ONLY on the
#     training data (this avoids "data leakage").
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=3), text_feature),
])


# ----------------------------------------------------------------------
# STEP 3: TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Splitting data (80% train / 20% test)")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)   # stratify keeps the cuisine proportions the same in both sets
print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")


# ----------------------------------------------------------------------
# STEP 4: TRAIN MODELS
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Training models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    # class_weight="balanced" gives rare cuisines more importance
    "Random Forest (balanced)": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="macro", zero_division=0)
    results[name] = {"pipeline": pipe, "pred": pred,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Macro Precision": p, "Macro Recall": r, "Macro F1": f1}
    print(f"  trained: {name}")


# ----------------------------------------------------------------------
# STEP 5: EVALUATE
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Model comparison (on the TEST set)")
print("=" * 60)
summary = pd.DataFrame({k: {m: v[m] for m in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]}
                        for k, v in results.items()}).T.round(3)
print(summary.to_string())
print("\nAccuracy = overall % correct.")
print("Macro scores average over all cuisines equally, so they show how")
print("well the model does on SMALL cuisines too (accuracy hides this).")

# Pick the best model by Macro F1 (fair across big and small cuisines)
best_name = summary["Macro F1"].idxmax()
best = results[best_name]
y_pred = best["pred"]
print(f"\nBest model: {best_name}")

summary.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")


# ----------------------------------------------------------------------
# STEP 6: PER-CUISINE ANALYSIS (challenges & bias)
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"STEP 6: Per-cuisine performance ({best_name})")
print("=" * 60)
report = classification_report(
    y_test, y_pred, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report).T.round(3)
print(classification_report(y_test, y_pred, zero_division=0))
report_df.to_csv(f"{OUTPUT_DIR}/per_cuisine_report.csv")

# Baseline: how good is "always guess the most common cuisine"?
majority = y_train.value_counts().idxmax()
baseline_acc = (y_test == majority).mean()
print(f"Baseline accuracy (always predict '{majority}'): {baseline_acc:.3f}")
print(f"Model accuracy: {best['Accuracy']:.3f}")

# Bias check: do big cuisines score better than small ones?
cuisine_rows = report_df.drop(["accuracy", "macro avg", "weighted avg"])
print("\nBest 3 cuisines by F1:")
print(cuisine_rows.sort_values("f1-score", ascending=False).head(3)
      [["precision", "recall", "f1-score", "support"]])
print("\nWorst 3 cuisines by F1:")
print(cuisine_rows.sort_values("f1-score").head(3)
      [["precision", "recall", "f1-score", "support"]])
corr = cuisine_rows["support"].corr(cuisine_rows["f1-score"])
print(f"\nCorrelation between cuisine size and F1-score: {corr:.2f}")
print("(a high positive number means the model is biased toward popular cuisines)")


# ----------------------------------------------------------------------
# STEP 7: PLOTS
# ----------------------------------------------------------------------
print("\nSaving plots to the 'outputs' folder...")
sns.set_theme(style="whitegrid")

# Plot 1: class distribution
plt.figure(figsize=(9, 6))
order = y.value_counts().index
sns.countplot(y=y, order=order, color="steelblue")
plt.title("Number of restaurants per cuisine (class imbalance)")
plt.xlabel("Restaurants")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_class_distribution.png", dpi=120)
plt.close()

# Plot 2: F1 per cuisine
plt.figure(figsize=(9, 6))
f1_sorted = cuisine_rows["f1-score"].sort_values()
plt.barh(f1_sorted.index, f1_sorted.values, color="seagreen")
plt.title(f"F1-score per cuisine - {best_name}")
plt.xlabel("F1-score (higher is better)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_f1_per_cuisine.png", dpi=120)
plt.close()

# Plot 3: confusion matrix (rows = true cuisine, normalised to 100%)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
plt.figure(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion matrix (each row sums to 1)")
plt.xlabel("Predicted cuisine")
plt.ylabel("True cuisine")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_confusion_matrix.png", dpi=120)
plt.close()

# Plot 4: model comparison
summary[["Accuracy", "Macro F1"]].plot(kind="bar", figsize=(8, 5), rot=15)
plt.title("Model comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_model_comparison.png", dpi=120)
plt.close()


# ----------------------------------------------------------------------
# STEP 8: MOST CONFUSED PAIRS
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: Which cuisines get confused most often?")
print("=" * 60)
cm_raw = confusion_matrix(y_test, y_pred, labels=labels)
pairs = [(labels[i], labels[j], cm_raw[i, j]) for i in range(len(labels))
         for j in range(len(labels)) if i != j and cm_raw[i, j] > 0]
for true_c, pred_c, n in sorted(pairs, key=lambda t: -t[2])[:8]:
    print(f"  True '{true_c}' predicted as '{pred_c}': {n} times")

print("\nDone! Check the 'outputs' folder for CSV tables and PNG charts.")
