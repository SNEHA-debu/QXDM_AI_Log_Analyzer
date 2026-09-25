
import os
import sys

# Project root
PROJECT_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Make project root importable
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

from src.inference import analyze_ue


def test_authentication_failure_prediction():

    log_path = os.path.join(
        PROJECT_PATH,
        "data",
        "sample_qxdm.log"
    )

    with open(log_path, "r", encoding="utf-8") as f:
        log_data = f.read()

    results = analyze_ue(log_data)

    assert len(results) == 1

    result = results[0]

    assert result["ue_id"] == 2001
    assert result["total_events"] == 8
    assert result["total_failures"] == 6

    assert result["authentication_failures"] == 4
    assert result["handover_failures"] == 1

    assert result["total_retransmissions"] == 27.0
    assert result["avg_retransmission"] == 13.5

    assert (
        result["predicted_root_cause"]
        == "Authentication Failure"
    )
