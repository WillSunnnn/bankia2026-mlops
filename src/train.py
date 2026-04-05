import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

# ── Chargement et preprocessing ──────────────────────────────────────────────
df = pd.read_csv("Loan_Data.csv")
df = df.drop(columns=["customer_id"])

df["debt_to_income"] = df["total_debt_outstanding"] / (df["income"] + 1)
df["loan_to_income"] = df["loan_amt_outstanding"] / (df["income"] + 1)

X = df.drop(columns=["default"])
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Fonction MLflow ───────────────────────────────────────────────────────────
def train_and_log(experiment_name, model, params, run_name):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"\n=== {run_name} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return model, metrics

# ── Sélection automatique du meilleur modèle ─────────────────────────────────
tous_les_modeles = {}

params1 = {"max_depth": 5, "min_samples_split": 20, "class_weight": "balanced"}
model, metrics = train_and_log("DecisionTree_LoanDefault", DecisionTreeClassifier(**params1, random_state=42), params1, "DT_depth5")
tous_les_modeles["DT_depth5"] = (model, metrics)

params2 = {"max_depth": 10, "min_samples_split": 50, "class_weight": "balanced"}
model, metrics = train_and_log("DecisionTree_LoanDefault", DecisionTreeClassifier(**params2, random_state=42), params2, "DT_depth10")
tous_les_modeles["DT_depth10"] = (model, metrics)

params3 = {"C": 0.1, "max_iter": 1000, "class_weight": "balanced"}
model, metrics = train_and_log("LogisticRegression_LoanDefault", LogisticRegression(**params3, random_state=42), params3, "LR_C0.1")
tous_les_modeles["LR_C0.1"] = (model, metrics)

params4 = {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}
model, metrics = train_and_log("LogisticRegression_LoanDefault", LogisticRegression(**params4, random_state=42), params4, "LR_C1.0")
tous_les_modeles["LR_C1.0"] = (model, metrics)

params5 = {"n_estimators": 100, "max_depth": 8, "class_weight": "balanced"}
model, metrics = train_and_log("RandomForest_LoanDefault", RandomForestClassifier(**params5, random_state=42), params5, "RF_100trees")
tous_les_modeles["RF_100trees"] = (model, metrics)

params6 = {"n_estimators": 200, "max_depth": 12, "class_weight": "balanced"}
model, metrics = train_and_log("RandomForest_LoanDefault", RandomForestClassifier(**params6, random_state=42), params6, "RF_200trees")
tous_les_modeles["RF_200trees"] = (model, metrics)

# Sélection basée sur le ROC-AUC
best_name = max(tous_les_modeles, key=lambda k: tous_les_modeles[k][1]["roc_auc"])
best_model = tous_les_modeles[best_name][0]

print(f"\n🏆 Meilleur modèle : {best_name}")
print(f"   ROC-AUC : {tous_les_modeles[best_name][1]['roc_auc']:.4f}")

# ── Sauvegarde ────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("\n✅ Meilleur modèle sauvegardé dans models/")