"""
priority_classifier.py
------------------------
Predicts ticket `priority` from the embedded text. Kept as an independent
head (not nested under category/resolver group) since priority mostly
depends on urgency language and impact scope, which cuts across categories.
Feeds directly into "SLA-aware triaging" (Objective) via sla_lookup.py logic
embedded in the orchestrator.
"""

import joblib
from src.models.stage1_classifier import build_model


class PriorityClassifier:
    def __init__(self, model_type="logreg"):
        self.model = build_model(model_type=model_type)
        self.model_type = model_type
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_with_confidence(self, X):
        probs = self.model.predict_proba(X)
        idx = probs.argmax(axis=1)
        labels = self.classes_[idx]
        confidences = probs[range(len(idx)), idx]
        return labels, confidences

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
