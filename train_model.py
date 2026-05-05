"""
train_model.py
--------------
Run this once to train and save your model.

Usage:
    python train_model.py

Requirements:
    pip install opencv-python numpy scikit-learn
"""

import os
import pickle
import numpy as np
from modules.feature_extractor import extract_features
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score)
from sklearn.pipeline import Pipeline


# ================================================================
#  COLLECT DATA
# ================================================================

X, y = [], []

REAL_PATH = "dataset/real images db"
AI_PATH   = "dataset/AI images"
EXTS      = (".jpg", ".jpeg", ".png", ".webp")


def load_folder(folder, label, name):
    count, errors = 0, 0
    for file in os.listdir(folder):
        if not file.lower().endswith(EXTS):
            continue
        full_path = os.path.join(folder, file)
        try:
            feats = extract_features(full_path)
            if len(feats) != 22:
                raise ValueError(f"Expected 22 features, got {len(feats)}")
            X.append(feats)
            y.append(label)
            count += 1
            if count % 50 == 0:
                print(f"  [{name}] {count} images processed...")
        except Exception as e:
            errors += 1
            print(f"  ERROR {file}: {e}")
    print(f"  [{name}] Done — {count} loaded, {errors} errors\n")


print("=" * 50)
print("  STEP 1: Extracting features")
print("=" * 50)

print("\n[REAL images]")
load_folder(REAL_PATH, label=0, name="REAL")

print("[AI images]")
load_folder(AI_PATH, label=1, name="AI")

X = np.array(X)
y = np.array(y)

print(f"Total samples : {len(X)}")
print(f"  Real (0)    : {int(np.sum(y == 0))}")
print(f"  AI   (1)    : {int(np.sum(y == 1))}")

if len(X) < 10:
    print("\nERROR: Not enough samples to train. "
          "Check your dataset paths and image files.")
    exit(1)


# ================================================================
#  TRAIN / TEST SPLIT
# ================================================================

print("\n" + "=" * 50)
print("  STEP 2: Splitting data (80/20)")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # keep class ratio in both splits
)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")


# ================================================================
#  MODEL — Pipeline(scaler + GradientBoosting)
#
#  Why GradientBoosting over RandomForest?
#    - Learns feature interactions better on small datasets
#    - More resistant to the low-sample overfitting that was
#      causing your 85% stuck score
#  Why Pipeline?
#    - Scaler is fitted ONLY on train data, applied to test
#    - Saved together so app.py just calls pipeline.predict()
# ================================================================

print("\n" + "=" * 50)
print("  STEP 3: Training model")
print("=" * 50)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    ))
])

# Cross-validation on training set (5-fold)
print("\nCross-validation (5-fold) on training data...")
cv_scores = cross_val_score(pipeline, X_train, y_train,
                            cv=5, scoring="accuracy")
print(f"  CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Final fit on full training set
pipeline.fit(X_train, y_train)


# ================================================================
#  EVALUATE ON HELD-OUT TEST SET
# ================================================================

print("\n" + "=" * 50)
print("  STEP 4: Evaluation on test set")
print("=" * 50)

y_pred = pipeline.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy : {acc:.3f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred,
                            target_names=["Real", "AI Generated"]))

print("Confusion matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred)
print(f"  Real  → predicted Real: {cm[0][0]}  |  predicted AI: {cm[0][1]}")
print(f"  AI    → predicted Real: {cm[1][0]}  |  predicted AI: {cm[1][1]}")

# Feature importance (from the classifier inside the pipeline)
clf       = pipeline.named_steps["clf"]
feat_names = [
    "DCT mean", "DCT std", "DCT block std", "DCT p95",
    "FFT mean", "FFT std", "FFT max", "FFT p99", "FFT HF ratio",
    "PRNU std", "PRNU mean abs", "PRNU uniformity",
    "LBP std", "LBP max", "LBP entropy",
    "Edge density", "Gradient mean", "Gradient std",
    "Saturation std", "Saturation mean", "RG correlation",
    "JPEG blocking"
]
importances = clf.feature_importances_
top_idx     = np.argsort(importances)[::-1][:8]
print("\nTop 8 most important features:")
for i in top_idx:
    print(f"  {feat_names[i]:<22} {importances[i]:.4f}")


# ================================================================
#  SAVE
# ================================================================

print("\n" + "=" * 50)
print("  STEP 5: Saving model")
print("=" * 50)

os.makedirs("model", exist_ok=True)
model_path = os.path.join("model", "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"\n  Model saved → {model_path}")
print(f"  Test accuracy : {acc:.1%}")
print(f"  CV accuracy   : {cv_scores.mean():.1%}")
print("\n  Done! Now run:  python app.py")