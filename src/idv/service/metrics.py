from prometheus_client import Counter, Histogram

SUBMITTED = Counter("idv_submitted_total", "Verifications accepted by the API")
REJECTED_UPLOADS = Counter("idv_rejected_uploads_total", "Uploads refused", ["reason"])
DECIDED = Counter(
    "idv_decided_total", "Verifications finished by the worker", ["status", "decision"]
)
# Server-side POST receipt to decision written. This is the latency we report as P95.
LATENCY = Histogram(
    "idv_end_to_end_seconds",
    "POST receipt to decision written",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30),
)
PIPELINE_SECONDS = Histogram("idv_pipeline_seconds", "Model pipeline time per verification")
