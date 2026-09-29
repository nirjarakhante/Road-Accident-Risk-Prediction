"""
train_model.py
--------------
Trains and evaluates 5 machine learning models for road accident risk prediction:
- Logistic Regression (Linear baseline)
- Decision Tree (Interpretable tree baseline)
- K-Nearest Neighbors (Instance-based baseline)
- Random Forest (Bagging ensemble)
- XGBoost (Gradient boosting with L1/L2 regularization)

Addresses class imbalance using SMOTE on training data.
Prevents target leakage by excluding accident counts and severities from features.
Saves model artifacts, scalers, and explainability metadata.
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "PctJunction", "PctCrossing", "PctTrafficSignal",
    "PctRushHour", "PctWeekend",
    "AvgVisibility", "AvgWindSpeed", "AvgPrecipitation", "PctBadWeather",
]
TARGET_COLUMN = "Risk_Level"


def load_data(path: Path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y, df


def build_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=15,
            max_features="sqrt", random_state=42, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            min_child_weight=8, subsample=0.75, colsample_bytree=0.75,
            reg_alpha=1.0, reg_lambda=2.0, eval_metric="mlogloss", random_state=42
        ),
    }


def run(input_path: Path, models_dir: Path):
    X, y, df = load_data(input_path)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance training set with SMOTE (High-risk cells are the minority class)
    print("Class distribution before SMOTE:", np.bincount(y_train))
    k_neighbors = max(1, min(5, np.min(np.bincount(y_train)) - 1))
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print("Class distribution after SMOTE:", np.bincount(y_train_bal))

    models = build_models()
    results = {}
    best_model_name, best_f1, best_model = None, -1.0, None

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        test_preds = model.predict(X_test_scaled)
        test_f1 = f1_score(y_test, test_preds, average="macro")

        train_preds = model.predict(X_train_bal)
        train_f1 = f1_score(y_train_bal, train_preds, average="macro")
        gap = train_f1 - test_f1

        report = classification_report(
            y_test, test_preds, target_names=label_encoder.classes_, output_dict=True
        )

        results[name] = {
            "macro_f1": float(test_f1),
            "train_f1": float(train_f1),
            "generalization_gap": float(gap),
            "report": report
        }

        print(f"\n=================== {name} ===================")
        print(f"Train F1: {train_f1:.4f} | Test Macro F1: {test_f1:.4f} | Generalization Gap: {gap:.4f}")
        print(classification_report(y_test, test_preds, target_names=label_encoder.classes_))

        if test_f1 > best_f1:
            best_f1, best_model_name, best_model = test_f1, name, model

    print(f"\n>>> Best Model: {best_model_name} (Macro F1 = {best_f1:.4f}) <<<")

    # Confusion matrix for best model
    best_preds = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, best_preds)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(label_encoder.classes_)
    print(cm)

    # Save model artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "risk_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    joblib.dump(FEATURE_COLUMNS, models_dir / "feature_columns.pkl")

    # Save summary metrics
    with open(models_dir / "results_summary.json", "w") as f:
        json.dump(
            {name: {"macro_f1": r["macro_f1"], "train_f1": r["train_f1"], "gap": r["generalization_gap"]}
             for name, r in results.items()},
            f, indent=2
        )

    # Extract feature importances if available
    importances = {}
    if hasattr(best_model, "feature_importances_"):
        raw_imp = best_model.feature_importances_
        norm_imp = raw_imp / np.sum(raw_imp)
        for col, imp in zip(FEATURE_COLUMNS, norm_imp):
            importances[col] = round(float(imp), 4)
    elif hasattr(best_model, "coef_"):
        coef_mean = np.mean(np.abs(best_model.coef_), axis=0)
        norm_imp = coef_mean / np.sum(coef_mean)
        for col, imp in zip(FEATURE_COLUMNS, norm_imp):
            importances[col] = round(float(imp), 4)

    with open(models_dir / "feature_importance.json", "w") as f:
        json.dump(importances, f, indent=2)

    print(f"\nAll models and pipeline artifacts safely stored in {models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/grid_risk.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    run(Path(args.input), Path(args.models_dir))
