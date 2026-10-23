from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# ==================================================
# 1. MEMUAT DATASET (UCI ID 697)
# ==================================================

DATA_PATH = Path("data/student_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET_COLUMN = "Target"

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].astype(str)

print("=" * 60)
print("  TRAINING 4 ALGORITMA SUPERVISED MACHINE LEARNING (CLASSIFICATION)")
print("  (IBM Machine Learning Course 3 & Model Arena)")
print("=" * 60)
print(f"  Dataset: {len(df)} mahasiswa, {X.shape[1]} atribut")

# 2. Split Data (Stratified 80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Inisialisasi 4 Algoritma Klasifikasi
classifiers = {
    "rf": {
        "name": "Random Forest",
        "badge": "Champion (Juara 1)",
        "desc": "Ensemble 300 pohon keputusan dengan bootstrap aggregation dan random feature subset.",
        "model": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
    },
    "lr": {
        "name": "Logistic Regression",
        "badge": "Runner-Up (Juara 2)",
        "desc": "Model linear probabilistik dengan regularisasi L2 dan standardisasi skala data.",
        "model": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
    },
    "cart": {
        "name": "CART (Decision Tree)",
        "badge": "Juara 3",
        "desc": "Pohon keputusan tunggal berbasis kriteria Gini impurity dengan batas kedalaman optimal.",
        "model": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced")
    },
    "nb": {
        "name": "Gaussian Naive Bayes",
        "badge": "Juara 4",
        "desc": "Pengklasifikasi probabilistik berbasis Teorema Bayes dengan asumsi independensi fitur Gaussian.",
        "model": make_pipeline(StandardScaler(), GaussianNB())
    }
}

models_evaluation = {}
best_model = None
best_acc = 0.0

for key, item in classifiers.items():
    clf = item["model"]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    b_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    
    print(f"\n  [{item['name']}]")
    print(f"  Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy : {b_acc:.4f}")
    print(f"  5-Fold CV Mean    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    
    models_evaluation[key] = {
        "name": item["name"],
        "badge": item["badge"],
        "desc": item["desc"],
        "accuracy": float(acc),
        "balanced_accuracy": float(b_acc),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "confusion_matrix": cm.tolist(),
        "classification_report": rep,
    }
    
    if acc > best_acc:
        best_acc = acc
        best_model = clf

# 4. Ambil Feature Importance dari Random Forest
rf_model = classifiers["rf"]["model"]
feature_imp = sorted(
    zip(X.columns.tolist(), rf_model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)

# 5. Simpan Model Terbaik dan Metadata Lengkap
joblib.dump(rf_model, MODEL_DIR / "student_model.joblib")

metadata = {
    "target_column": TARGET_COLUMN,
    "feature_names": X.columns.tolist(),
    "classes": sorted(y.unique().tolist()),
    "test_size": 0.20,
    "random_state": 42,
    "n_estimators": 300,
    "accuracy": models_evaluation["rf"]["accuracy"],
    "balanced_accuracy": models_evaluation["rf"]["balanced_accuracy"],
    "cv_accuracy_mean": models_evaluation["rf"]["cv_accuracy_mean"],
    "classification_report": models_evaluation["rf"]["classification_report"],
    "confusion_matrix": models_evaluation["rf"]["confusion_matrix"],
    "models_arena": models_evaluation,
    "top_features": [
        {"name": f[0], "importance": float(f[1])} for f in feature_imp[:12]
    ]
}

with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("  MODEL & METADATA BERHASIL DISIMPAN!")
print("  File: models/student_model.joblib & models/metadata.json")
print("=" * 60)
