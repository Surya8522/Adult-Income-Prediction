
import json
import time
import statistics
import urllib.request
import urllib.error

from pathlib import Path
from datetime import datetime, timezone

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

API_URL = "http://localhost:5000"

HEALTH_URL = API_URL + "/health"
PREDICT_URL = API_URL + "/predict"

OUTPUT_DIR = BASE_DIR / "outputs" / "monitoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "api_performance_report.json"

TOTAL_REQUESTS = 30

# Adult Income sample input
SAMPLE_DATA = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education.num": 13,
    "marital.status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital.gain": 2174,
    "capital.loss": 0,
    "hours.per.week": 40,
    "native.country": "United-States"
}


# --------------------------------------------------
# API REQUEST FUNCTION
# --------------------------------------------------

def send_request(url, payload=None):

    data = None

    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET"
    )

    with urllib.request.urlopen(request, timeout=15) as response:

        response_data = response.read().decode("utf-8")

        return response.status, response_data


# --------------------------------------------------
# MAIN MONITORING
# --------------------------------------------------

def main():

    print("=" * 65)
    print("LAB 12 - API PERFORMANCE MONITORING")
    print("=" * 65)

    # Check API health
    print("\nChecking API health...")

    try:

        health_status, health_response = send_request(
            HEALTH_URL
        )

        print("Health HTTP status:", health_status)
        print("Health response:", health_response)

        if health_status != 200:
            raise RuntimeError("API health check failed.")

    except Exception as error:

        print("\n[ERROR] API is not accessible.")
        print("Details:", str(error))
        print("\nMake sure the Docker container is running.")
        print("Expected API:", API_URL)

        raise SystemExit(1)

    # Collect performance measurements
    latencies = []
    successful_requests = 0
    failed_requests = 0
    errors = []

    monitoring_start = time.perf_counter()

    print("\nSending prediction requests...")

    for index in range(TOTAL_REQUESTS):

        start = time.perf_counter()

        try:

            status, response = send_request(
                PREDICT_URL,
                SAMPLE_DATA
            )

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            if 200 <= status < 300:

                successful_requests += 1
                latencies.append(elapsed)

                print(
                    f"Request {index + 1}: "
                    f"SUCCESS | {elapsed:.2f} ms"
                )

            else:

                failed_requests += 1

                errors.append({
                    "request": index + 1,
                    "status": status,
                    "message": response
                })

        except Exception as error:

            failed_requests += 1

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            errors.append({
                "request": index + 1,
                "message": str(error)
            })

            print(
                f"Request {index + 1}: FAILED | "
                f"{str(error)}"
            )

    total_duration = (
        time.perf_counter() - monitoring_start
    )

    # Calculate metrics
    if latencies:

        average_latency = statistics.mean(latencies)
        minimum_latency = min(latencies)
        maximum_latency = max(latencies)

        if len(latencies) >= 2:
            p95_latency = statistics.quantiles(
                latencies,
                n=100
            )[94]
        else:
            p95_latency = latencies[0]

    else:

        average_latency = 0
        minimum_latency = 0
        maximum_latency = 0
        p95_latency = 0

    throughput = (
        successful_requests / total_duration
        if total_duration > 0 else 0
    )

    success_rate = (
        successful_requests / TOTAL_REQUESTS
    ) * 100

    error_rate = (
        failed_requests / TOTAL_REQUESTS
    ) * 100

    # Monitoring status
    if failed_requests == 0:
        monitoring_status = "HEALTHY"
    elif successful_requests > 0:
        monitoring_status = "DEGRADED"
    else:
        monitoring_status = "FAILED"

    # Report
    report = {
        "project": "Adult Income Prediction",
        "monitoring_type": "API Performance Monitoring",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "api_url": API_URL,
        "total_requests": TOTAL_REQUESTS,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "success_rate_percent": round(success_rate, 2),
        "error_rate_percent": round(error_rate, 2),
        "average_latency_ms": round(average_latency, 2),
        "minimum_latency_ms": round(minimum_latency, 2),
        "maximum_latency_ms": round(maximum_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "total_duration_seconds": round(total_duration, 3),
        "throughput_requests_per_second": round(
            throughput, 2
        ),
        "monitoring_status": monitoring_status,
        "errors": errors
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

    # Display results
    print("\n" + "=" * 65)
    print("API PERFORMANCE RESULTS")
    print("=" * 65)

    print("Total requests:", TOTAL_REQUESTS)
    print("Successful requests:", successful_requests)
    print("Failed requests:", failed_requests)

    print("Success rate:", f"{success_rate:.2f}%")
    print("Error rate:", f"{error_rate:.2f}%")

    print("Average latency:", f"{average_latency:.2f} ms")
    print("Minimum latency:", f"{minimum_latency:.2f} ms")
    print("Maximum latency:", f"{maximum_latency:.2f} ms")
    print("P95 latency:", f"{p95_latency:.2f} ms")

    print("Throughput:", f"{throughput:.2f} requests/second")

    print("\nMonitoring status:", monitoring_status)

    print("\nReport saved:", REPORT_PATH)

    print("\n[SUCCESS] API performance monitoring completed.")


if __name__ == "__main__":
    main()
