"""
mlflow_utils.py
-----------------
MLflow here is used for ONE thing only: experiment tracking / model
registry (which run had which params/metrics, which artifact is "current").

It is deliberately NOT used for model serving. `src/api.py` (your FastAPI
service) stays the one and only thing that answers real-time /triage
requests. This keeps the two concerns cleanly separated:

    MLflow tracking server  -> "what did I train, when, with what results"
    Your FastAPI service    -> "classify this ticket right now"

Wrapped behind try/except so `train.py` still runs fine if mlflow isn't
installed -- tracking is a nice-to-have for the dissertation's experiment
comparison table, not a hard dependency of the pipeline.
"""

import os

MLFLOW_AVAILABLE = True
try:
    import mlflow
except ImportError:
    MLFLOW_AVAILABLE = False


def get_tracking_uri():
    """Defaults to a local ./mlruns folder (zero setup, fully offline).
    Override with the MLFLOW_TRACKING_URI env var once you're running a
    real tracking server (see strategy doc / README for hosting options)."""
    return os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")


class TrackingRun:
    """Context manager that no-ops cleanly if mlflow isn't installed, so
    calling code never needs an `if MLFLOW_AVAILABLE` check of its own."""

    def __init__(self, experiment_name="ticket-triage", run_name=None):
        self.enabled = MLFLOW_AVAILABLE
        self.run_name = run_name
        if self.enabled:
            mlflow.set_tracking_uri(get_tracking_uri())
            mlflow.set_experiment(experiment_name)

    def __enter__(self):
        if self.enabled:
            mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            mlflow.end_run()

    def log_params(self, params: dict):
        if self.enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        if self.enabled:
            mlflow.log_metrics(metrics)

    def log_artifact(self, path: str):
        if self.enabled:
            mlflow.log_artifact(path)

    def log_sklearn_model(self, model, artifact_path: str, registered_model_name=None):
        if self.enabled:
            
