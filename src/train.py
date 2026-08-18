import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score

DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_detection_pipeline.joblib"
ARTIFACT_DIR = "artifacts"

os.makedirs("models", exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
if "Class" not in df.columns:
    raise ValueError("Dataset must contain a 'Class' target column.")

df = df.drop_duplicates().reset_index(drop=True)
X = df.drop(columns=["Class"])
y = df["Class"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

features = X.columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), features)
])

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

pred = pipeline.predict(X_test)
prob = pipeline.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, prob)),
    "test_rows": int(len(y_test)),
    "fraud_test_rows": int(y_test.sum())
}

with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

joblib.dump(pipeline, MODEL_PATH)

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_matrix.png"), dpi=160)
plt.close()

fpr, tpr, _ = roc_curve(y_test, prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC={metrics['roc_auc']:.4f})")
plt.plot([0, 1], [0, 1], "--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACT_DIR, "roc_curve.png"), dpi=160)
plt.close()

importance = pd.DataFrame({
    "feature": features,
    "importance": pipeline.named_steps["model"].feature_importances_
}).sort_values("importance", ascending=False)
importance.to_csv(os.path.join(ARTIFACT_DIR, "feature_importance.csv"), index=False)

print(json.dumps(metrics, indent=2))
print(classification_report(y_test, pred, digits=4))
print("Model saved:", MODEL_PATH)
