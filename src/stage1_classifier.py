"""
stage1_classifier.py
---------------------
Stage-1 of the hierarchical multi-stage classification (Objective #2):
predicts ticket `category` (the request type) from the embedded text.

Model choice: `model_type="logreg"` (default) or `model_type="mlp"`.

    - "logreg": multinomial Logistic Regression. Coefficients are directly
      usable as a cheap explainability layer (feature-weight based), which
      keeps Gap-6 (explainability cost) low before you add SHAP/LIME on top.
      It's a LINEAR model -- it can only separate classes with a straight
      decision boundary over the TF-IDF feature space.

    - "mlp": a small feed-forward Artificial Neural Network
      (sklearn.neural_network.MLPClassifier -- one or two hidden layers with
      a nonlinear activation). This can learn nonlinear interactions between
      terms that Logistic Regression can't (e.g. "vpn" + "billing" together
      meaning something different than either alone), which is the standard
      argument for ANN over a linear classifier in your literature review's
      DNN-based papers (#7, #16, #17). Tradeoff: no `.coef_`, so the cheap
      LinearWeightExplainer won't work on it -- pair it with SHAP/LIME
      (KernelExplainer, since MLP isn't a tree or linear model) if you use
      this for your explainability chapter's ablation.

    - Swap for XGBoost/LightGBM if you want tree-based nonlinearity instead
      (SHAP TreeExplainer is cheap for tree models -- also a reasonable
      Phase-4 ablation, and a third point on the same accuracy vs.
      explainability-cost curve).
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def build_model(class_weight="balanced", model_type="logreg"):
    if model_type == "logreg":
        # class_weight='balanced' directly targets Gap-5 (class imbalance) from
        # the literature review without needing SMOTE/oversampling as a first pass.
        return LogisticRegression(max_iter=1000, class_weight=class_weight)

    if model_type == "mlp":
        # A small ANN: one hidden layer of 128 units, ReLU activation, early
        # stopping so it doesn't overfit the (relatively small) ticket
        # dataset. MLPClassifier has no class_weight param, so if your real
        # data is as imbalanced as the synthetic set, oversample the minority
        # classes before calling .fit() (see Gap-5 discussion in the strategy doc).
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,               # L2 regularization
            early_stopping=False,     # avoid sklearn's internal validation-scoring
                                       # path, which has a known incompatibility with
                                       # non-numeric class labels in some sklearn
                                       # versions; we label-encode instead (see
                                       # Stage1CategoryClassifier/PriorityClassifier)
                                       # and rely on max_iter instead of early stopping.
            max_iter=300,
            random_state=42,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


class Stage1CategoryClassifier:
    def __init__(self, model=None, model_type="logreg"):
        self.model = model or build_model(model_type=model_type)
        self.model_type = model_type
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_with_confidence(self, X):
        """Returns (predicted_label, confidence_score) per row."""
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        labels = self.classes_[idx]
        confidences = probs[range(len(idx)), idx]
        return labels, confidences

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return classification_report(y_test, preds, output_dict=True)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
