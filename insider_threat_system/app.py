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
    This system uses Random Forest and SMOTE
    to classify employee activity as Normal
    or potentially Malicious.
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

    user_input = {}

    left_column, right_column = st.columns(2)

    columns = metadata["columns"]

    half = (len(columns) + 1) // 2

    left_fields = columns[:half]
    right_fields = columns[half:]


    # ========================================================
    # INPUT FIELD FUNCTION
    # ========================================================

    def create_input(column):

        label = (
            column
            .replace("_", " ")
            .title()
        )

        # --------------------------------------------
        # CATEGORICAL FIELDS
        # --------------------------------------------

        if column in metadata["categorical_columns"]:

            return st.selectbox(
                label,
                metadata[
                    "categorical_options"
                ][column],
                key=column
            )

        # --------------------------------------------
        # NUMERIC INFORMATION
        # --------------------------------------------

        minimum = metadata[
            "numeric_min"
        ][column]

        maximum = metadata[
            "numeric_max"
        ][column]

        default = metadata[
            "numeric_default"
        ][column]

        # --------------------------------------------
        # BINARY 0 / 1 FIELDS
        #
        # GUI:
        # No / Yes
        #
        # Model:
        # No  = 0
        # Yes = 1
        # --------------------------------------------

        if minimum >= 0 and maximum <= 1:

            yes_no = st.selectbox(
                label,
                ["No", "Yes"],
                key=column
            )

            if yes_no == "Yes":
                return 1

            return 0

        # --------------------------------------------
        # OTHER NUMERIC FIELDS
        # --------------------------------------------

        return st.number_input(
            label,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(default),
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

        # --------------------------------------------
        # CREATE INPUT DATAFRAME
        # --------------------------------------------

        input_dataframe = pd.DataFrame(
            [
                {
                    column: user_input[column]

                    for column
                    in metadata["columns"]
                }
            ]
        )

        # --------------------------------------------
        # PREPROCESS INPUT
        # --------------------------------------------

        processed_input = preprocessor.transform(
            input_dataframe
        )

        # --------------------------------------------
        # MALICIOUS PROBABILITY
        # --------------------------------------------

        malicious_probability = (
            model.predict_proba(
                processed_input
            )[0, 1]
        )

        # --------------------------------------------
        # APPLY THRESHOLD
        # --------------------------------------------

        prediction = int(
            malicious_probability
            >= best_threshold
        )

        # --------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------

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


        # --------------------------------------------
        # EXPLANATION
        # --------------------------------------------

        with st.expander(
            "How was this result determined?"
        ):

            st.write(
                "Malicious probability:"
            )

            st.write(
                f"{malicious_probability * 100:.2f}%"
            )

            st.write(
                "Decision threshold:"
            )

            st.write(
                f"{best_threshold * 100:.2f}%"
            )

            if prediction == 1:

                st.write(
                    "The probability is equal to or above "
                    "the decision threshold, so the activity "
                    "is classified as Malicious."
                )

            else:

                st.write(
                    "The probability is below the decision "
                    "threshold, so the activity is classified "
                    "as Normal."
                )


# ============================================================
# 7. PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.header(
        "Model Performance"
    )

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

1. The dataset is loaded.

2. The data is split into training and testing sets.

3. The training data is further divided into fitting and
   validation data.

4. Missing values are handled and categorical values are
   converted using one-hot encoding.

5. SMOTE is applied only to the training data to balance
   Normal and Malicious samples.

6. Random Forest learns patterns from employee profile and
   behavioral activity data.

7. Validation data is used to select the best decision
   threshold.

8. The final trained Random Forest is evaluated using the
   untouched test dataset.

9. Streamlit loads the saved preprocessing system,
   trained Random Forest model, and selected threshold.

10. The user enters employee activity information and
    receives a Normal or Malicious prediction.
        """
    )

    st.info(
        """
        Binary fields are shown as Yes or No for easier use.

        Internally:
        Yes = 1
        No = 0

        This keeps the Streamlit interface user-friendly while
        remaining compatible with the trained machine-learning model.
        """
    )
