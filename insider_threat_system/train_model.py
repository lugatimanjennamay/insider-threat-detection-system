import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "insider_threat_clean_dataset.csv"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

os.makedirs(
    ARTIFACT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print("\nDataset successfully loaded.")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 3. TARGET AND FEATURES
# ============================================================

TARGET = "is_malicious"

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

print("\nOriginal class distribution:")
print(y.value_counts())


# ============================================================
# 4. COLUMN TYPES
# ============================================================

categorical_columns = (
    X.select_dtypes(include="object")
    .columns
    .tolist()
)

numeric_columns = (
    X.select_dtypes(exclude="object")
    .columns
    .tolist()
)

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumeric columns:")
print(numeric_columns)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining distribution:")
print(y_train.value_counts())

print("\nTesting distribution:")
print(y_test.value_counts())


# ============================================================
# 6. FIT / VALIDATION SPLIT
# ============================================================

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ],
    remainder="drop"
)


# ============================================================
# 8. PROCESS FIT AND VALIDATION DATA
# ============================================================

X_fit_processed = preprocessor.fit_transform(
    X_fit
)

X_val_processed = preprocessor.transform(
    X_val
)


# ============================================================
# 9. SMOTE ON FIT DATA ONLY
# ============================================================

smote = SMOTE(
    random_state=42
)

X_fit_balanced, y_fit_balanced = smote.fit_resample(
    X_fit_processed,
    y_fit
)

print("\nBefore SMOTE:")
print(y_fit.value_counts())

print("\nAfter SMOTE:")
print(
    pd.Series(
        y_fit_balanced
    ).value_counts()
)


# ============================================================
# 10. TEMP MODEL FOR THRESHOLD
# ============================================================

validation_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

validation_model.fit(
    X_fit_balanced,
    y_fit_balanced
)


# ============================================================
# 11. VALIDATION PROBABILITIES
# ============================================================

validation_probabilities = (
    validation_model
    .predict_proba(
        X_val_processed
    )[:, 1]
)


# ============================================================
# 12. BEST THRESHOLD
# ============================================================

best_threshold = 0.50
best_f1 = -1
best_precision = 0
best_recall = 0

for threshold in np.arange(
    0.20,
    0.701,
    0.025
):

    validation_predictions = (
        validation_probabilities
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = float(threshold)
        best_precision = precision
        best_recall = recall


print("\nBEST THRESHOLD:")
print(round(best_threshold, 3))

print(
    "Validation Precision:",
    round(best_precision * 100, 2),
    "%"
)

print(
    "Validation Recall:",
    round(best_recall * 100, 2),
    "%"
)

print(
    "Validation F1:",
    round(best_f1 * 100, 2),
    "%"
)


# ============================================================
# 13. PROCESS FULL TRAINING AND TEST SET
# ============================================================

X_train_processed = preprocessor.transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# ============================================================
# 14. SMOTE ON FULL TRAINING DATA
# ============================================================

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nFull training BEFORE SMOTE:")
print(y_train.value_counts())

print("\nFull training AFTER SMOTE:")
print(
    pd.Series(
        y_train_balanced
    ).value_counts()
)


# ============================================================
# 15. FINAL RANDOM FOREST
# ============================================================

model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

model_rf.fit(
    X_train_balanced,
    y_train_balanced
)


# ============================================================
# 16. FINAL TEST PREDICTIONS
# ============================================================

test_probabilities = (
    model_rf
    .predict_proba(
        X_test_processed
    )[:, 1]
)

test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


# ============================================================
# 17. FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nFINAL TEST RESULTS")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall:", round(recall * 100, 2), "%")
print("F1 Score:", round(f1 * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Normal",
            "Malicious"
        ],
        zero_division=0
    )
)


# ============================================================
# 18. SAVE INPUT METADATA
# ============================================================

metadata = {
    "columns": X.columns.tolist(),
    "categorical_columns": categorical_columns,
    "numeric_columns": numeric_columns,
    "categorical_options": {},
    "numeric_min": {},
    "numeric_max": {},
    "numeric_default": {}
}

for column in categorical_columns:

    metadata[
        "categorical_options"
    ][column] = sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

for column in numeric_columns:

    metadata[
        "numeric_min"
    ][column] = float(
        df[column].min()
    )

    metadata[
        "numeric_max"
    ][column] = float(
        df[column].max()
    )

    metadata[
        "numeric_default"
    ][column] = float(
        df[column].median()
    )


# ============================================================
# 19. SAVE METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "threshold": float(best_threshold),
    "confusion_matrix": cm.tolist()
}


# ============================================================
# 20. SAVE MODEL FILES
# ============================================================

joblib.dump(
    model_rf,
    os.path.join(
        ARTIFACT_DIR,
        "random_forest_model.joblib"
    ),
    compress=("xz", 3)
)

joblib.dump(
    preprocessor,
    os.path.join(
        ARTIFACT_DIR,
        "preprocessor.joblib"
    )
)

joblib.dump(
    best_threshold,
    os.path.join(
        ARTIFACT_DIR,
        "best_threshold.joblib"
    )
)

joblib.dump(
    metadata,
    os.path.join(
        ARTIFACT_DIR,
        "input_metadata.joblib"
    )
)

with open(
    os.path.join(
        ARTIFACT_DIR,
        "metrics.json"
    ),
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print("\nMODEL SAVED SUCCESSFULLY")
print("Check the artifacts folder.")