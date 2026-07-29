"""
explainability.py
-------------------
Implements Objective: "Integrate LIME/SHAP explainability mechanisms" and
"Incorporate explainability for routing decisions".

Design note: SHAP/LIME are wrapped behind try/except so the rest of the
pipeline runs even before those packages are installed on the training box.
The fallback (`LinearWeightExplainer`) is a legitimate lightweight
explainer for the TF-IDF + LogisticRegression baseline (it just surfaces
the top contributing terms by |coefficient * tfidf_value|), and doubles as
a discussion point for Gap-6 ("High computational cost of explainability")
-- you can report a latency comparison between this and full SHAP/LIME in
the dissertation's evaluation chapter.
"""

import numpy as np


class LinearWeightExplainer:
    """Cheap, always-available explainer for linear models over sparse TF-IDF features."""

    def __init__(self, model, feature_names, top_k=8):
        self.model = model          # sklearn LogisticRegression
        self.feature_names = np.array(feature_names)
        self.top_k = top_k

    def explain(self, x_row, predicted_class_index):
        """x_row: single sparse row (1, n_features). Returns list of (term, contribution)."""
        if not hasattr(self.model, "coef_"):
            raise TypeError(
                "LinearWeightExplainer only works on linear models with a .coef_ "
                "attribute (e.g. LogisticRegression). For a nonlinear model like "
                "MLPClassifier, use SHAPExplainer (shap.KernelExplainer) or "
                "LIMEExplainer instead -- see get_explainer(backend='shap'/'lime')."
            )
        coef = self.model.coef_[predicted_class_index]
        x_dense = np.asarray(x_row.todense()).ravel() if hasattr(x_row, "todense") else np.asarray(x_row).ravel()
        contributions = coef * x_dense
        top_idx = np.argsort(np.abs(contributions))[::-1][: self.top_k]
        return [
            {"term": self.feature_names[i], "contribution": float(contributions[i])}
            for i in top_idx if contributions[i] != 0
        ]


class SHAPExplainer:
    """Wraps shap.LinearExplainer / TreeExplainer depending on model type."""

    def __init__(self, model, background_data):
        try:
            import shap
        except ImportError as e:
            raise ImportError("pip install shap to use SHAPExplainer") from e
        self._shap = shap
        if hasattr(model, "coef_"):
            self.explainer = shap.LinearExplainer(model, background_data)
        elif hasattr(model, "estimators_") or hasattr(model, "get_booster"):
            self.explainer = shap.TreeExplainer(model)
        else:
            # Model-agnostic fallback -- covers MLPClassifier and anything else
            # without a fast closed-form explainer. Slower (samples the model
            # many times per explanation), which is exactly the "high
            # computational cost of explainability" tradeoff (Gap-6) to report
            # in your evaluation chapter when you use the ANN option.
            self.explainer = shap.KernelExplainer(model.predict_proba, background_data)

    def explain(self, x_row, top_k=8):
        shap_values = self.explainer.shap_values(x_row)
        vals = np.asarray(shap_values).ravel()
        top_idx = np.argsort(np.abs(vals))[::-1][:top_k]
        return top_idx, vals[top_idx]


class LIMEExplainer:
    """Wraps lime.lime_text for model-agnostic local explanations."""

    def __init__(self, predict_proba_fn, class_names):
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError as e:
            raise ImportError("pip install lime to use LIMEExplainer") from e
        self.explainer = LimeTextExplainer(class_names=class_names)
        self.predict_proba_fn = predict_proba_fn

    def explain(self, raw_text, num_features=8):
        exp = self.explainer.explain_instance(raw_text, self.predict_proba_fn, num_features=num_features)
        return exp.as_list()


def get_explainer(model, feature_names=None, backend="linear", **kwargs):
    if backend == "linear":
        return LinearWeightExplainer(model, feature_names, **kwargs)
    if backend == "shap":
        return SHAPExplainer(model, **kwargs)
    if backend == "lime":
        return LIMEExplainer(**kwargs)
    raise ValueError(f"Unknown explainability backend: {backend}")
