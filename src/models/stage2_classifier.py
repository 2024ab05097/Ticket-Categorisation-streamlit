"""
stage2_classifier.py
---------------------
Stage-2 of the hierarchical multi-stage classification (Objective #2):
predicts `resolver_group` (team assignment), conditioned on the Stage-1
predicted category. This mirrors the architecture diagram's
"Stage-1 Request Classification -> Stage-2 Team Assignment" flow.

Implementation: one classifier PER category ("mixture of experts" style)
rather than a single flat classifier over all resolver groups. This directly
targets:
    - Gap-3 / poor routing decisions: each expert only has to discriminate
      among the small set of resolver groups valid for its category, which
      is exactly the "overlapping resolver groups cause confusion" failure
      mode called out in the literature review (paper #23, TaDaa).
    - Priority is trained as a separate, independent head off the same
      embeddings, since priority is largely orthogonal to which team
      resolves the ticket.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class Stage2TeamAssignmentClassifier:
    """A dict-of-classifiers keyed by Stage-1 category."""

    def __init__(self):
        self.experts = {}   # category -> LogisticRegression
        self.classes_by_category = {}

    def fit(self, X, categories, resolver_groups):
        """
        X: feature matrix (n_samples, n_features)
        categories: array-like Stage-1 labels (ground truth or predicted, per row)
        resolver_groups: array-like target labels, per row
        """
        import numpy as np
        categories = np.asarray(categories)
        resolver_groups = np.asarray(resolver_groups)

        for cat in np.unique(categories):
            mask = categories == cat
            X_cat = X[mask]
            y_cat = resolver_groups[mask]
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            if len(set(y_cat)) < 2:
                # Only one resolver group ever seen for this category in training data
                self.experts[cat] = ("constant", y_cat[0])
                continue
            clf.fit(X_cat, y_cat)
            self.experts[cat] = ("model", clf)
        return self

    def predict_with_confidence(self, X, categories):
        """Returns (predicted_resolver_group, confidence) per row, routed through
        the expert selected by that row's Stage-1 category."""
        import numpy as np
        categories = np.asarray(categories)
        labels = np.empty(len(categories), dtype=object)
        confidences = np.zeros(len(categories))

        for cat in np.unique(categories):
            mask = categories == cat
            entry = self.experts.get(cat)
            if entry is None:
                # unseen category at inference time -- flag for human review downstream
                labels[mask] = "UNKNOWN_RESOLVER_GROUP"
                confidences[mask] = 0.0
                continue
            kind, payload = entry
            if kind == "constant":
                labels[mask] = payload
                confidences[mask] = 1.0
            else:
                probs = payload.predict_proba(X[mask])
                idx = probs.argmax(axis=1)
                labels[mask] = payload.classes_[idx]
                confidences[mask] = probs[range(len(idx)), idx]
        return labels, confidences

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
