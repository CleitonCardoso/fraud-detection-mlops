"""Prometheus custom metrics for the fraud detection API."""
from prometheus_client import Counter, Gauge, Histogram

prediction_latency = Histogram(
    "fraud_prediction_latency_seconds",
    "Time spent running a fraud prediction",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

fraud_score = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud probability scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

request_counter = Counter(
    "fraud_api_requests_total",
    "Total API requests",
    labelnames=["endpoint", "status"],
)

agent_latency = Histogram(
    "fraud_agent_latency_seconds",
    "Time spent on a full agent ReAct cycle",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

drift_psi_gauge = Gauge(
    "fraud_drift_psi",
    "Latest PSI score per feature",
    labelnames=["feature"],
)

model_auc_gauge = Gauge(
    "fraud_model_auc",
    "AUC of the current production model",
)
