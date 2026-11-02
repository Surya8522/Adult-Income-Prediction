
import json
from pathlib import Path
from datetime import datetime, timezone

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DRIFT_DIR = BASE_DIR / "outputs" / "drift"
MONITORING_DIR = BASE_DIR / "outputs" / "monitoring"
RETRAINING_DIR = BASE_DIR / "outputs" / "retraining"

OUTPUT_DIR = MONITORING_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "lab12_monitor_trigger_report.json"

INPUT_REPORTS = {
    "data_drift": DRIFT_DIR / "data_drift_summary.json",
    "prediction_drift": DRIFT_DIR / "prediction_drift_report.json",
    "concept_drift": DRIFT_DIR / "concept_drift_report.json",
    "performance": MONITORING_DIR / "performance_monitoring_report.json",
    "performance_trend": MONITORING_DIR / "performance_trend_report.json",
    "api_performance": MONITORING_DIR / "api_performance_report.json",
    "model_comparison": RETRAINING_DIR / "model_comparison_v1_v2.json"
}


# --------------------------------------------------
# LOAD JSON REPORT
# --------------------------------------------------

def load_report(path):

    if not path.exists():
        print("[WARNING] Report not found:", path.name)
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError) as error:
        print("[WARNING] Could not read", path.name, ":", error)
        return None


# --------------------------------------------------
# FIND DRIFT SIGNALS IN REPORTS
# --------------------------------------------------

def find_drift_signals(data, path=""):

    signals = []

    if isinstance(data, dict):

        for key, value in data.items():

            current_path = f"{path}.{key}" if path else key
            key_lower = key.lower()

            if isinstance(value, bool):
                if "drift" in key_lower and value:
                    signals.append(current_path)

            elif isinstance(value, str):

                text = value.upper()

                if (
                    "DRIFT" in key_lower
                    and text in ["TRUE", "DETECTED", "YES"]
                ):
                    signals.append(current_path)

                if (
                    "STATUS" in key_lower
                    and "DEGRADATION" in text
                ):
                    signals.append(current_path)

            elif isinstance(value, (dict, list)):
                signals.extend(
                    find_drift_signals(value, current_path)
                )

    elif isinstance(data, list):

        for index, value in enumerate(data):
            signals.extend(
                find_drift_signals(value, f"{path}[{index}]")
            )

    return signals


# --------------------------------------------------
# MAIN MONITOR TRIGGER
# --------------------------------------------------

def main():

    print("=" * 65)
    print("LAB 12 - AUTOMATED MONITORING TRIGGER")
    print("=" * 65)

    reports = {}

    for name, path in INPUT_REPORTS.items():

        reports[name] = load_report(path)

    # Track monitoring signals
    trigger_reasons = []
    warnings = []

    # --------------------------------------------------
    # 1. DATA DRIFT
    # --------------------------------------------------

    data_drift = reports["data_drift"]

    if data_drift is not None:

        signals = find_drift_signals(data_drift)

        if signals:
            trigger_reasons.append(
                "Data drift detected in baseline monitoring data."
            )

            print("\nData drift: DETECTED")

        else:
            print("\nData drift: No explicit drift flag found.")

    # --------------------------------------------------
    # 2. PREDICTION DRIFT
    # --------------------------------------------------

    prediction_drift = reports["prediction_drift"]

    if prediction_drift is not None:

        signals = find_drift_signals(prediction_drift)

        if signals:
            trigger_reasons.append(
                "Prediction drift detected."
            )

            print("Prediction drift: DETECTED")

        else:
            print("Prediction drift: No explicit drift flag found.")

    # --------------------------------------------------
    # 3. CONCEPT DRIFT
    # --------------------------------------------------

    concept_drift = reports["concept_drift"]

    if concept_drift is not None:

        signals = find_drift_signals(concept_drift)

        if signals:
            trigger_reasons.append(
                "Concept drift or related performance degradation detected."
            )

            print("Concept drift: DETECTED")

        else:
            print("Concept drift: No explicit drift flag found.")

    # --------------------------------------------------
    # 4. MODEL PERFORMANCE
    # --------------------------------------------------

    performance = reports["performance"]

    if performance is not None:

        status = str(
            performance.get("monitoring_status", "")
        ).upper()

        if "DEGRADATION" in status:
            trigger_reasons.append(
                "Model performance degradation detected."
            )

            print("Model performance: DEGRADATION DETECTED")

        else:
            print("Model performance:", status or "Status unavailable")

    # --------------------------------------------------
    # 5. PERFORMANCE TREND
    # --------------------------------------------------

    performance_trend = reports["performance_trend"]

    if performance_trend is not None:

        trend_status = str(
            performance_trend.get("trend_status", "")
        ).upper()

        print(
            "Performance trend:",
            trend_status or "Status unavailable"
        )

        if "DEGRADATION" in trend_status:
            warnings.append(
                "Performance trend indicates degradation."
            )

    # --------------------------------------------------
    # 6. API PERFORMANCE
    # --------------------------------------------------

    api_performance = reports["api_performance"]

    if api_performance is not None:

        api_status = str(
            api_performance.get("monitoring_status", "")
        ).upper()

        failed_requests = int(
            api_performance.get("failed_requests", 0)
        )

        print("\nAPI performance:", api_status)
        print("API failed requests:", failed_requests)

        if failed_requests > 0 or api_status != "HEALTHY":
            warnings.append(
                "API performance issue detected. "
                "Investigate service health separately."
            )

    # --------------------------------------------------
    # 7. MODEL COMPARISON
    # --------------------------------------------------

    comparison = reports["model_comparison"]

    if comparison is not None:

        eligible = comparison.get(
            "eligible_for_staging", False
        )

        print("\nV2 eligible for Staging:", eligible)

        if eligible:
            print("V2 Staging candidate is available.")

    # --------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------

    retraining_required = len(trigger_reasons) > 0

    if retraining_required:
        trigger_status = "RETRAINING_REQUIRED"
    else:
        trigger_status = "NO_RETRAINING_REQUIRED"

    report = {
        "project": "Adult Income Prediction",
        "monitoring_type": "Automated Monitoring Trigger",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "trigger_status": trigger_status,
        "retraining_required": retraining_required,
        "trigger_reasons": trigger_reasons,
        "warnings": warnings,
        "v2_eligible_for_staging": (
            comparison.get("eligible_for_staging", False)
            if comparison else False
        ),
        "production_model_changed": False,
        "automatic_retraining_executed": False,
        "automatic_production_promotion": False
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("AUTOMATED MONITORING DECISION")
    print("=" * 65)

    print("Trigger status:", trigger_status)
    print("Retraining required:", retraining_required)

    print("\nReasons:")

    for reason in trigger_reasons:
        print("-", reason)

    if not trigger_reasons:
        print("No retraining triggers detected.")

    print("\nWarnings:")

    for warning in warnings:
        print("-", warning)

    if not warnings:
        print("No API or trend warnings.")

    print("\nReport saved:", REPORT_PATH)

    print("\nProduction model changed: NO")
    print("Automatic retraining executed: NO")
    print("Automatic Production promotion: NO")

    print("\n[SUCCESS] Monitoring trigger analysis completed.")


if __name__ == "__main__":
    main()
