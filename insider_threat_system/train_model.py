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
from imblearn.under_sampling import RandomUnderSampler


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "insider_threat_clean_dataset.csv"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)

os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Rows and columns:", df.shape)

TARGET = "is_malicious"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 3. IDENTIFY COLUMN TYPES
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


# ============================================================
# 4. PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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


def make_preprocessor():
    return ColumnTransformer(
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
# 7. PREPROCESS FITTING / VALIDATION DATA
# ============================================================

validation_preprocessor = make_preprocessor()

X_fit_processed = validation_preprocessor.fit_transform(
    X_fit
)

X_val_processed = validation_preprocessor.transform(
    X_val
)

feature_names = (
    validation_preprocessor
    .get_feature_names_out()
)

print(
    "\nTotal encoded features:",
    len(feature_names)
)


# ============================================================
# 8. PRELIMINARY RANDOM FOREST FOR FEATURE RANKING
# ============================================================

feature_ranker = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

feature_ranker.fit(
    X_fit_processed,
    y_fit
)

feature_importances = (
    feature_ranker.feature_importances_
)

ranked_indices = np.argsort(
    feature_importances
)[::-1]


# ============================================================
# 9. VALIDATION-BASED FEATURE SELECTION
# ============================================================

candidate_feature_counts = [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100
]

candidate_feature_counts = [
    k
    for k in candidate_feature_counts
    if k <= X_fit_processed.shape[1]
]

smote_validation = SMOTE(
    random_state=42
)

selection_results = []

best_feature_count = None
best_threshold = 0.50
best_f1 = -1
best_precision = 0
best_recall = 0


for k in candidate_feature_counts:

    print(
        f"\nTesting Top {k} features..."
    )

    top_k_indices = ranked_indices[:k]

    X_fit_k = X_fit_processed[
        :,
        top_k_indices
    ]

    X_val_k = X_val_processed[
        :,
        top_k_indices
    ]

    X_fit_balanced, y_fit_balanced = (
        smote_validation.fit_resample(
            X_fit_k,
            y_fit
        )
    )

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

    val_probabilities = (
        validation_model
        .predict_proba(X_val_k)[:, 1]
    )

    local_best_threshold = 0.50
    local_best_f1 = -1
    local_best_precision = 0
    local_best_recall = 0

    for threshold in np.arange(
        0.20,
        0.701,
        0.025
    ):

        val_predictions = (
            val_probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            val_predictions,
            zero_division=0
        )

        if f1 > local_best_f1:

            local_best_f1 = f1
            local_best_threshold = float(
                threshold
            )
            local_best_precision = precision
            local_best_recall = recall

    selection_results.append({
        "Top_K_Features": k,
        "Best_Threshold":
            local_best_threshold,
        "Validation_Precision":
            local_best_precision,
        "Validation_Recall":
            local_best_recall,
        "Validation_F1":
            local_best_f1
    })

    if local_best_f1 > best_f1:

        best_feature_count = k
        best_threshold = (
            local_best_threshold
        )
        best_f1 = local_best_f1
        best_precision = (
            local_best_precision
        )
        best_recall = (
            local_best_recall
        )


selection_df = pd.DataFrame(
    selection_results
)

print("\nFEATURE-SELECTION RESULTS")
print(selection_df.to_string(index=False))

print(
    "\nBEST FEATURE COUNT:",
    best_feature_count
)

print(
    "BEST THRESHOLD:",
    round(best_threshold, 3)
)

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
# 10. REFIT PREPROCESSOR ON FULL TRAINING SET
# ============================================================

preprocessor = make_preprocessor()

X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)

full_feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    "\nFinal encoded feature count:",
    len(full_feature_names)
)


# ============================================================
# 11. RERANK FEATURES USING FULL TRAINING DATA
# ============================================================

final_feature_ranker = (
    RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
)

final_feature_ranker.fit(
    X_train_processed,
    y_train
)

final_importances = (
    final_feature_ranker.feature_importances_
)

final_ranked_indices = np.argsort(
    final_importances
)[::-1]

final_selected_indices = (
    final_ranked_indices[
        :best_feature_count
    ]
)

final_selected_features = (
    full_feature_names[
        final_selected_indices
    ]
)


# IMPORTANT:
# Do NOT sort the indices.
# Keep exactly the same feature order
# used by the Colab notebook.

X_train_selected = (
    X_train_processed[
        :,
        final_selected_indices
    ]
)

X_test_selected = (
    X_test_processed[
        :,
        final_selected_indices
    ]
)


print(
    "\nFINAL SELECTED FEATURE COUNT:",
    len(final_selected_features)
)

print("\nFINAL SELECTED FEATURES:")

for i, feature in enumerate(
    final_selected_features,
    start=1
):
    print(
        f"{i}. {feature}"
    )


# ============================================================
# 12. APPLY SMOTE TO FULL TRAINING DATA
# ============================================================

print(
    "\nTRAINING DISTRIBUTION BEFORE SMOTE:"
)
print(y_train.value_counts())

smote_final = SMOTE(
    random_state=42
)

X_train_balanced, y_train_balanced = (
    smote_final.fit_resample(
        X_train_selected,
        y_train
    )
)

print(
    "\nTRAINING DISTRIBUTION AFTER SMOTE:"
)

print(
    pd.Series(
        y_train_balanced
    ).value_counts()
)


# ============================================================
# 13. TRAIN FINAL RANDOM FOREST
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
# 14. TEST PREDICTIONS
# ============================================================

test_probabilities = (
    model_rf.predict_proba(
        X_test_selected
    )[:, 1]
)

test_predictions = (
    test_probabilities
    >= best_threshold
).astype(int)


# ============================================================
# 15. FINAL TEST METRICS
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

cm_normalized = confusion_matrix(
    y_test,
    test_predictions,
    normalize="true"
)


print("\nFINAL TEST RESULTS")

print(
    "Selected feature count:",
    best_feature_count
)

print(
    "Chosen threshold:",
    round(best_threshold, 3)
)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1 * 100, 2),
    "%"
)

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
# 16. BALANCED DIAGNOSTIC TEST
# ============================================================

undersampler = RandomUnderSampler(
    random_state=42
)

X_test_balanced, y_test_balanced = (
    undersampler.fit_resample(
        X_test_selected,
        y_test
    )
)

balanced_probabilities = (
    model_rf.predict_proba(
        X_test_balanced
    )[:, 1]
)

balanced_predictions = (
    balanced_probabilities
    >= best_threshold
).astype(int)

balanced_cm = confusion_matrix(
    y_test_balanced,
    balanced_predictions
)

balanced_accuracy = accuracy_score(
    y_test_balanced,
    balanced_predictions
)

balanced_precision = precision_score(
    y_test_balanced,
    balanced_predictions,
    zero_division=0
)

balanced_recall = recall_score(
    y_test_balanced,
    balanced_predictions,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test_balanced,
    balanced_predictions,
    zero_division=0
)


# ============================================================
# 17. INPUT METADATA
# ============================================================

metadata = {
    "columns": X.columns.tolist(),

    "categorical_columns":
        categorical_columns,

    "numeric_columns":
        numeric_columns,

    "categorical_options": {},

    "numeric_min": {},

    "numeric_max": {},

    "numeric_default": {},

    "selected_feature_count":
        int(best_feature_count),

    "selected_feature_names":
        final_selected_features.tolist()
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
# 18. METRICS FILE
# ============================================================

metrics = {

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1":
        float(f1),

    "threshold":
        float(best_threshold),

    "selected_feature_count":
        int(best_feature_count),

    "selected_feature_names":
        final_selected_features.tolist(),

    "confusion_matrix":
        cm.tolist(),

    "normalized_confusion_matrix":
        cm_normalized.tolist(),

    "balanced_accuracy":
        float(balanced_accuracy),

    "balanced_precision":
        float(balanced_precision),

    "balanced_recall":
        float(balanced_recall),

    "balanced_f1":
        float(balanced_f1),

    "balanced_confusion_matrix":
        balanced_cm.tolist()
}


# ============================================================
# 19. SAVE ARTIFACTS
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
    final_selected_indices,
    os.path.join(
        ARTIFACT_DIR,
        "selected_feature_indices.joblib"
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


print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")

print(
    "Selected features:",
    best_feature_count
)

print(
    "Threshold:",
    round(best_threshold, 3)
)

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1:",
    round(f1 * 100, 2),
    "%"
)

print("\nArtifacts saved:")
print("- random_forest_model.joblib")
print("- preprocessor.joblib")
print("- best_threshold.joblib")
print("- selected_feature_indices.joblib")
print("- input_metadata.joblib")
print("- metrics.json")
