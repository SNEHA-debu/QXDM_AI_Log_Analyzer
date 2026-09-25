
import os
import json
import joblib
import pandas as pd

from .parser import parse_log_line


# Automatically locate the project directory
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.dirname(SRC_PATH)
MODEL_PATH = os.path.join(PROJECT_PATH, "models")


# --------------------------------------------------
# Load saved ML/NLP artifacts
# --------------------------------------------------

vectorizer = joblib.load(
    os.path.join(MODEL_PATH, "tfidf_vectorizer.joblib")
)

nlp_model = joblib.load(
    os.path.join(MODEL_PATH, "nlp_classifier.joblib")
)

root_cause_model = joblib.load(
    os.path.join(MODEL_PATH, "root_cause_model.joblib")
)

with open(
    os.path.join(MODEL_PATH, "feature_metadata.json"),
    "r"
) as f:
    metadata = json.load(f)

feature_columns = metadata["feature_columns"]


# --------------------------------------------------
# UE analysis function
# --------------------------------------------------

def analyze_ue(log_data):

    records = []

    # Parse raw log text
    for line in log_data.strip().splitlines():

        parsed = parse_log_line(line)

        if parsed is not None:
            records.append(parsed)

    if not records:
        return []

    df = pd.DataFrame(records)

    # Convert numeric fields
    df["ue_id"] = pd.to_numeric(
        df["ue_id"],
        errors="coerce"
    )

    df["count"] = pd.to_numeric(
        df["count"],
        errors="coerce"
    )

    # Detect failure-related events
    failure_keywords = [
        "failure",
        "reject",
        "timeout",
        "error"
    ]

    pattern = "|".join(failure_keywords)

    df["is_failure"] = df["event"].str.contains(
        pattern,
        case=False,
        na=False
    )

    # --------------------------------------------------
    # NLP event classification
    # --------------------------------------------------

    text_features = vectorizer.transform(
        df["event"]
    )

    df["nlp_predicted_category"] = (
        nlp_model.predict(text_features)
    )

    results = []

    # --------------------------------------------------
    # Create features per UE
    # --------------------------------------------------

    for ue_id, ue_df in df.groupby("ue_id"):

        total_events = len(ue_df)

        total_failures = int(
            ue_df["is_failure"].sum()
        )

        failure_rate = (
            total_failures / total_events
            if total_events > 0
            else 0
        )

        authentication_failures = int(
            ue_df["event"]
            .str.contains(
                "Authentication Failure",
                case=False,
                na=False
            )
            .sum()
        )

        handover_failures = int(
            ue_df["event"]
            .str.contains(
                "Handover Failure",
                case=False,
                na=False
            )
            .sum()
        )

        retransmission_values = ue_df.loc[
            ue_df["event"].str.contains(
                "Retransmission",
                case=False,
                na=False
            ),
            "count"
        ].dropna()

        total_retransmissions = float(
            retransmission_values.sum()
        )

        avg_retransmission = (
            float(retransmission_values.mean())
            if len(retransmission_values) > 0
            else 0.0
        )

        # NLP category counts
        category_counts = (
            ue_df["nlp_predicted_category"]
            .value_counts()
        )

        feature_data = {
            "total_events": total_events,
            "total_failures": total_failures,
            "avg_retransmission": avg_retransmission,
            "failure_rate": failure_rate,
            "authentication_failures": authentication_failures,
            "handover_failures": handover_failures,
            "total_retransmissions": total_retransmissions,
            "nlp_authentication_count": int(
                category_counts.get("Authentication", 0)
            ),
            "nlp_connection_count": int(
                category_counts.get("Connection", 0)
            ),
            "nlp_mobility_count": int(
                category_counts.get("Mobility", 0)
            ),
            "nlp_registration_count": int(
                category_counts.get("Registration", 0)
            ),
            "nlp_retransmission_count": int(
                category_counts.get("Retransmission", 0)
            )
        }

        # Exact training feature order
        model_input = pd.DataFrame(
            [feature_data]
        )[feature_columns]

        prediction = root_cause_model.predict(
            model_input
        )[0]

        result = {
            "ue_id": int(ue_id),
            **feature_data,
            "predicted_root_cause": prediction
        }

        results.append(result)

    return results
