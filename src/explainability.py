"""
src/explainability.py
---------------------

Explainability module for the Agentic ITSM Ticket Triaging framework.

Supports:
1. Lightweight Linear Explanation (default)
2. LIME
3. SHAP

The lightweight explainer is used by default because it is extremely
fast for TF-IDF + Logistic Regression and provides interpretable
feature contributions.

LIME and SHAP are optional and loaded only if installed.
"""

from __future__ import annotations

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


###############################################################
# Base Explainer
###############################################################

class BaseExplainer:

    def explain(self, x_row, predicted_class_index):
        raise NotImplementedError


###############################################################
# Linear Explainer
###############################################################

class LinearWeightExplainer(BaseExplainer):
    """
    Fast explainer for linear models.

    Contribution =
        TF-IDF value × model coefficient
    """

    def __init__(
            self,
            model,
            feature_names,
            top_k=10):

        if not hasattr(model, "coef_"):
            raise TypeError(
                "LinearWeightExplainer requires "
                "a linear sklearn model."
            )

        self.model = model
        self.feature_names = np.asarray(feature_names)
        self.top_k = top_k

    def explain(
            self,
            x_row,
            predicted_class_index):

        if hasattr(x_row, "toarray"):
            x = x_row.toarray().ravel()
        else:
            x = np.asarray(x_row).ravel()

        coef = self.model.coef_[predicted_class_index]

        contribution = coef * x

        indices = np.argsort(
            np.abs(contribution)
        )[::-1][:self.top_k]

        explanation = []

        for idx in indices:

            if contribution[idx] == 0:
                continue

            explanation.append({

                "feature": str(self.feature_names[idx]),

                "contribution": float(contribution[idx]),

                "importance": float(abs(contribution[idx]))

            })

        return explanation


###############################################################
# LIME
###############################################################

class LIMEExplainer(BaseExplainer):

    def __init__(
            self,
            model,
            X_train,
            feature_names,
            class_names):

        try:

            from lime.lime_tabular import (
                LimeTabularExplainer
            )

        except ImportError:

            raise ImportError(
                "Install lime first:\n"
                "pip install lime"
            )

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        self.model = model

        self.explainer = LimeTabularExplainer(

            training_data=X_train,

            feature_names=feature_names,

            class_names=class_names,

            mode="classification",

            discretize_continuous=True

        )

    def explain(
            self,
            x_row,
            predicted_class_index):

        if hasattr(x_row, "toarray"):
            sample = x_row.toarray()[0]
        else:
            sample = np.asarray(x_row).ravel()

        exp = self.explainer.explain_instance(

            sample,

            self.model.predict_proba,

            num_features=10,

            labels=[predicted_class_index]

        )

        output = []

        for feature, score in exp.as_list(
                label=predicted_class_index):

            output.append({

                "feature": feature,

                "contribution": float(score),

                "importance": float(abs(score))

            })

        return output


###############################################################
# SHAP Explainer
###############################################################

    ###############################################################
# SHAP Explainer
###############################################################

class SHAPExplainer(BaseExplainer):

    def __init__(
            self,
            model,
            X_train):

        try:
            import shap
        except ImportError:
            raise ImportError(
                "Install SHAP first:\n"
                "pip install shap"
            )

        self.model = model
        self.shap = shap

        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        # Use a small background sample for speed
        background = X_train[:100]

        self.explainer = shap.Explainer(
            self.model.predict_proba,
            background
        )

    def explain(
            self,
            x_row,
            predicted_class_index):

        if hasattr(x_row, "toarray"):
            sample = x_row.toarray()
        else:
            sample = np.asarray(x_row).reshape(1, -1)

        values = self.explainer(sample)

        shap_values = values.values[0][:, predicted_class_index]

        explanation = []

        for i, score in enumerate(shap_values):

            if abs(score) < 1e-6:
                continue

            explanation.append({

                "feature": str(values.feature_names[i])
                if values.feature_names
                else f"feature_{i}",

                "contribution": float(score),

                "importance": float(abs(score))

            })

        explanation.sort(
            key=lambda x: x["importance"],
            reverse=True
        )

        return explanation[:10]


###############################################################
# Factory
###############################################################

def get_explainer(
        backend,
        model,
        feature_names,
        X_train=None,
        class_names=None,
        top_k=10):

    backend = backend.lower()

    if backend == "linear":

        return LinearWeightExplainer(
            model=model,
            feature_names=feature_names,
            top_k=top_k
        )

    elif backend == "lime":

        if X_train is None:
            raise ValueError(
                "X_train required for LIME"
            )

        return LIMEExplainer(
            model=model,
            X_train=X_train,
            feature_names=feature_names,
            class_names=class_names
        )

    elif backend == "shap":

        if X_train is None:
            raise ValueError(
                "X_train required for SHAP"
            )

        return SHAPExplainer(
            model=model,
            X_train=X_train
        )

    else:

        raise ValueError(
            f"Unknown backend: {backend}"
        )


###############################################################
# Benchmark Utility
###############################################################

def benchmark_explainability(
        explainer,
        x_row,
        predicted_class_index):

    start = time.perf_counter()

    explanation = explainer.explain(
        x_row,
        predicted_class_index
    )

    end = time.perf_counter()

    return {

        "backend": explainer.__class__.__name__,

        "latency_ms": round(
            (end - start) * 1000,
            2
        ),

        "num_features": len(explanation),

        "explanation": explanation

    }


###############################################################
# Convenience wrapper
###############################################################

def explain_prediction(
        explainer,
        x_row,
        predicted_class_index):

    return benchmark_explainability(
        explainer,
        x_row,
        predicted_class_index
        )
