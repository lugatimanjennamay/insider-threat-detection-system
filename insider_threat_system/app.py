import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR,
    "artifacts"
)


# ============================================================
# 2. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Insider Threat Detection System",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# 3. LOAD SAVED MODEL FILES
# ============================================================

@st.cache_resource
def load_system():

    model = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "random_forest_model.joblib"
        )
    )

    preprocessor = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "preprocessor.joblib"
        )
    )

    threshold = joblib.load(
        os.path.join(
            ARTIFACT_DIR,
            "best_threshold.joblib"
        )
    )

    metadata = joblib.load(
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
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(file)

    return (
        model,
        preprocessor,
        threshold,
        metadata,
        metrics
    )


(
    model,
    preprocessor,
    best_threshold,
    metadata,
    metrics
) = load_system()


# ============================================================
# 4. HEADER
# ============================================================

st.title(
    "🛡️ Insider Threat Detection System"
)

st.write(
    """
    This system uses Random Forest and SMOTE to classify
    employee activity as Normal or potentially Malicious.
    """
)


# ============================================================
# 5. TABS
# ============================================================

prediction_tab, performance_tab, about_tab = st.tabs(
    [
        "🔍 Analyze Activity",
        "📊 Model Performance",
        "ℹ️ About"
    ]
)


# ============================================================
# 6. PREDICTION TAB
# ============================================================

with prediction_tab:

    st.header(
        "Employee Activity Input"
    )

    st.write(
        """
        Enter the employee profile and activity information
        below, then click **Analyze Behavior**.
        """
    )

    user_input = {}

    left_column, right_column = st.columns(2)

    columns = metadata["columns"]

    half = (len(columns) + 1) // 2

    left_fields = columns[:half]
    right_fields = columns[half:]


    # ========================================================
    # BINARY FIELDS
    #
    # These fields are shown as Yes / No in the GUI.
    #
    # Internally:
    # No  = 0
    # Yes = 1
    # ========================================================

    binary_fields = [
        "is_contractor",
        "has_foreign_citizenship",
        "has_criminal_record",
        "has_medical_history",
        "burned_from_other",
        "is_abroad",
        "late_exit_flag",
        "entry_during_weekend"
    ]


    # ========================================================
    # CUSTOM LABELS
    # Makes the GUI easier to understand.
    # ========================================================

    custom_labels = {
        "employee_department": "Employee Department",
        "employee_campus": "Employee Campus",
        "employee_position": "Employee Position",
        "employee_seniority_years": "Employee Seniority Years",
        "is_contractor": "Is Contractor?",
        "employee_classification": "Employee Classification",
        "has_foreign_citizenship": "Has Foreign Citizenship?",
        "has_criminal_record": "Has Criminal Record?",
        "has_medical_history": "Has Medical History?",
        "employee_origin_country": "Employee Origin Country",
        "total_printed_pages": "Total Printed Pages",
        "num_printed_pages_off_hours": "Printed Pages During Off-Hours",
        "total_files_burned": "Total Files Burned",
        "burned_from_other": "Files Burned From Other Source?",
        "is_abroad": "Is Employee Abroad?",
        "trip_day_number": "Trip Day Number",
        "hostility_country_level": "Hostility Country Level",
        "num_entries": "Number of Entries",
        "num_unique_campus": "Number of Unique Campuses",
        "late_exit_flag": "Late Exit?",
        "entry_during_weekend": "Entry During Weekend?"
    }


    # ========================================================
    # INPUT FIELD FUNCTION
    # ========================================================

    def create_input(column):

        label = custom_labels.get(
            column,
            column.replace("_", " ").title()
        )


        # ----------------------------------------------------
        # CATEGORICAL FIELDS
        # ----------------------------------------------------

        if column in metadata["categorical_columns"]:

            return st.selectbox(
                label,
                metadata[
                    "categorical_options"
                ][column],
                key=column
            )


        # ----------------------------------------------------
        # BINARY YES / NO FIELDS
        # ----------------------------------------------------

        if column in binary_fields:

            yes_no = st.selectbox(
                label,
                ["No", "Yes"],
                key=column
            )

            return 1 if yes_no == "Yes" else 0


        # ----------------------------------------------------
        # NUMERIC VALUES
        # ----------------------------------------------------

        minimum = metadata[
            "numeric_min"
        ][column]

        maximum = metadata[
            "numeric_max"
        ][column]

        default = metadata[
            "numeric_default"
        ][column]


        # ----------------------------------------------------
        # WHOLE NUMBER INPUT
        #
        # All remaining numeric fields in this dataset represent
        # years, counts, days, classifications, or levels.
        # Therefore they are displayed without unnecessary .00.
        # ----------------------------------------------------

        return st.number_input(
            label,
            min_value=int(minimum),
            max_value=int(maximum),
            value=int(round(default)),
            step=1,
            key=column
        )


    # ========================================================
    # LEFT COLUMN
    # ========================================================

    with left_column:

        st.subheader(
            "Employee Information"
        )

        for column in left_fields:

            user_input[column] = create_input(
                column
            )


    # ========================================================
    # RIGHT COLUMN
    # ========================================================

    with right_column:

        st.subheader(
            "Activity Information"
        )

        for column in right_fields:

            user_input[column] = create_input(
                column
            )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    st.divider()

    if st.button(
        "🔍 Analyze Behavior",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # CREATE ONE-ROW DATAFRAME FROM USER INPUT
        # ----------------------------------------------------

        input_dataframe = pd.DataFrame(
            [
                {
                    column: user_input[column]

                    for column
                    in metadata["columns"]
                }
            ]
        )


        # ----------------------------------------------------
        # PREPROCESS INPUT
        #
        # Uses the same preprocessing used during training.
        # ----------------------------------------------------

        processed_input = preprocessor.transform(
            input_dataframe
        )


        # ----------------------------------------------------
        # GET MALICIOUS PROBABILITY
        # ----------------------------------------------------

        malicious_probability = (
            model.predict_proba(
                processed_input
            )[0, 1]
        )


        # ----------------------------------------------------
        # APPLY SELECTED THRESHOLD
        # ----------------------------------------------------

        prediction = int(
            malicious_probability
            >= best_threshold
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        st.divider()

        st.header(
            "Prediction Result"
        )

        (
            result_column,
            probability_column,
            threshold_column
        ) = st.columns(3)


        with result_column:

            st.metric(
                "Classification",
                "MALICIOUS"
                if prediction == 1
                else "NORMAL"
            )


        with probability_column:

            st.metric(
                "Malicious Probability",
                f"{malicious_probability * 100:.2f}%"
            )


        with threshold_column:

            st.metric(
                "Decision Threshold",
                f"{best_threshold:.3f}"
            )


        if prediction == 1:

            st.error(
                "⚠️ Potential malicious behavior detected."
            )

        else:

            st.success(
                "✅ Employee activity classified as normal."
            )


        # ----------------------------------------------------
        # RESULT EXPLANATION
        # ----------------------------------------------------

        with st.expander(
            "How was this result determined?"
        ):

            st.write(
                "The Random Forest estimated a malicious "
                "probability of:"
            )

            st.write(
                f"**{malicious_probability * 100:.2f}%**"
            )

            st.write(
                "The selected decision threshold is:"
            )

            st.write(
                f"**{best_threshold * 100:.2f}%**"
            )


            if prediction == 1:

                st.write(
                    """
                    The malicious probability is equal to or
                    greater than the decision threshold.
                    Therefore, the activity is classified as
                    **Malicious**.
                    """
                )

            else:

                st.write(
                    """
                    The malicious probability is below the
                    decision threshold. Therefore, the
                    activity is classified as **Normal**.
                    """
                )


# ============================================================
# 7. PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.header(
        "Model Performance"
    )


    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )


    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )


    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )


    col4.metric(
        "F1 Score",
        f"{metrics['f1'] * 100:.2f}%"
    )


    st.write(
        "**Selected Threshold:**",
        round(
            metrics["threshold"],
            3
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "Confusion Matrix"
    )

    cm = np.array(
        metrics["confusion_matrix"]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual Normal",
            "Actual Malicious"
        ],
        columns=[
            "Predicted Normal",
            "Predicted Malicious"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )


    # ========================================================
    # NORMALIZED CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "Normalized Confusion Matrix"
    )

    row_totals = cm.sum(
        axis=1,
        keepdims=True
    )

    normalized_cm = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(
            cm,
            dtype=float
        ),
        where=row_totals != 0
    )

    normalized_df = pd.DataFrame(
        normalized_cm,
        index=[
            "Actual Normal",
            "Actual Malicious"
        ],
        columns=[
            "Predicted Normal",
            "Predicted Malicious"
        ]
    )

    st.dataframe(
        normalized_df.style.format(
            "{:.2%}"
        ),
        use_container_width=True
    )


    # ========================================================
    # PERFORMANCE EXPLANATION
    # ========================================================

    with st.expander(
        "What do these metrics mean?"
    ):

        st.write(
            """
            **Accuracy** – percentage of all predictions that
            were correct.

            **Precision** – among all activities predicted as
            Malicious, the percentage that were actually
            Malicious.

            **Recall** – among all actual Malicious activities,
            the percentage detected by the model.

            **F1 Score** – balance between Precision and Recall.
            """
        )


# ============================================================
# 8. ABOUT TAB
# ============================================================

with about_tab:

    st.header(
        "How the System Works"
    )

    st.markdown(
        """
### System Process

1. **Dataset Loading**  
   Employee profile and activity information are loaded.

2. **Train-Test Split**  
   The dataset is divided into training and testing data.

3. **Validation Split**  
   A portion of the training set is reserved for threshold
   selection.

4. **Preprocessing**  
   Missing values are handled and categorical variables are
   converted using one-hot encoding.

5. **SMOTE**  
   SMOTE is applied only to training data to balance the
   Normal and Malicious classes.

6. **Random Forest Training**  
   Multiple decision trees learn patterns from employee
   profile and behavioral activity information.

7. **Threshold Selection**  
   Different probability thresholds are evaluated using
   validation data. The threshold with the best F1 score is
   selected.

8. **Final Evaluation**  
   The trained model is evaluated using the untouched test
   dataset.

9. **Streamlit GUI**  
   The user enters employee and activity information.

10. **Prediction**  
    The system calculates the malicious probability and
    compares it against the selected decision threshold to
    classify the activity as Normal or Malicious.
        """
    )


    st.info(
        """
        Binary fields are displayed as **Yes / No** for easier
        use.

        Internally:

        **Yes = 1**  
        **No = 0**

        The machine-learning model still receives the same
        numerical values used during training.
        """
    )
