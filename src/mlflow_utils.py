"""
mlflow_utils.py
---------------
Dummy implementation for Streamlit Cloud.
"""


class TrackingRun:

    def __init__(self, experiment_name="ticket-triage", run_name=None):
        self.experiment_name = experiment_name
        self.run_name = run_name

    def __enter__(self):
        print(
            f"Tracking disabled. Run={self.run_name}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_params(self, params):
        pass

    def log_metrics(self, metrics):
        pass

    def log_artifact(self, path):
        pass

    def log_sklearn_model(
        self,
        model,
        artifact_path,
        registered_model_name=None
    ):
        pass
