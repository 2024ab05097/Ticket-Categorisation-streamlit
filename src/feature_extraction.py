"""
feature_extraction.py
----------------------
Section 4 / Phase 4 calls for "Transformer-based feature extraction (BERT/GUSE)".

For local development (no GPU / no internet egress from the training box) we
default to a TF-IDF vectorizer, which needs zero downloads and is a fully
legitimate baseline to report against in the dissertation's "Evaluation and
Benchmarking" phase. `TransformerEmbedder` is a drop-in replacement: flip
`EMBEDDING_BACKEND=transformer` once you have sentence-transformers installed
and model-download access on the training environment (e.g. Infosys internal
model registry / approved HF mirror).

Both embedders expose the same .fit_transform()/.transform() interface so
nothing else in the pipeline (classifiers, explainability, orchestrator)
needs to know which backend is active.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEmbedder(ABC):
    @abstractmethod
    def fit_transform(self, texts): ...

    @abstractmethod
    def transform(self, texts): ...

    def get_feature_names(self):
        """Optional: only meaningful for sparse/linear-explainable backends like TF-IDF."""
        return None


class TfidfEmbedder(BaseEmbedder):
    """Baseline embedder. Also the backend LIME/SHAP explanations are cheapest against,
    which matters for Gap-6 ('High computational cost of explainability') in the synopsis."""

    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()


class TransformerEmbedder(BaseEmbedder):
    """BERT/GUSE-style dense embeddings via sentence-transformers.

    Requires: pip install sentence-transformers
    Recommended models to benchmark per the synopsis's "Models under consideration":
        - "bert-base-uncased" (mean-pooled) as the vanilla BERT baseline
        - "sentence-transformers/all-mpnet-base-v2" as a GUSE-equivalent dense encoder
        - a fine-tuned "Ticket-BERT" once you have enough labeled internal data
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. Run "
                "`pip install sentence-transformers` to use TransformerEmbedder, "
                "or use TfidfEmbedder for local development."
            ) from e
        self.model = SentenceTransformer(model_name)

    def fit_transform(self, texts):
        return np.array(self.model.encode(list(texts), show_progress_bar=False))

    def transform(self, texts):
        return np.array(self.model.encode(list(texts), show_progress_bar=False))


def get_embedder(backend: str = "tfidf", **kwargs) -> BaseEmbedder:
    if backend == "tfidf":
        return TfidfEmbedder(**kwargs)
    if backend == "transformer":
        return TransformerEmbedder(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend}")
