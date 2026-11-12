"""
Pluggable text-similarity backends.

Two built in, both `similarity(a, b) -> float in [0, 1]`:

    tfidf  scikit-learn TF-IDF cosine over word 1-2 grams. Default; no model.
    spacy  cosine of spaCy content-word vectors (needs the `spacy` extra and
           a model with vectors, e.g. `python -m spacy download en_core_web_md`).

`get_backend("auto")` uses spaCy when the model is installed, else TF-IDF.
Any object with `name`, `suggested_threshold` and `similarity(a, b)` can be
passed to `MatchingEngine(backend=...)` instead.

The two backends do not share a scale: averaged word vectors cluster in the
high 0.8s while TF-IDF cosine over two documents rarely passes 0.3. That is
why thresholds are per backend and why `match_score` calibrates each
source onto 0-1 before weighting it (see `ScoringConfig.calibration`).
"""

from __future__ import annotations

import logging
import math
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)

__all__ = ["SimilarityBackend", "SpacyBackend", "TfidfBackend", "get_backend", "cosine"]

DEFAULT_SPACY_MODEL = "en_core_web_md"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine of two vectors, clamped to [0, 1]."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return max(0.0, dot / norms) if norms else 0.0


class SimilarityBackend:
    """Interface every backend implements."""

    name: str = "base"
    detail: str = ""
    #: A sensible starting threshold on THIS backend's scale.
    suggested_threshold: float = 0.6

    def similarity(self, text_a: str, text_b: str) -> float:
        raise NotImplementedError


class SpacyBackend(SimilarityBackend):
    """
    Cosine similarity between spaCy document vectors.

    The document vector is averaged over content words only -- no stopwords,
    punctuation or numbers. Averaging everything let "the/and/with/for"
    dominate both vectors, so unrelated resumes all landed inside a narrow
    band; filtering widens it enough for a threshold to be meaningful.
    """

    name = "spacy"
    suggested_threshold = 0.85

    def __init__(self, model_name: str = DEFAULT_SPACY_MODEL):
        import spacy  # optional dependency

        self._model_name = model_name
        try:
            self._nlp = spacy.load(model_name, exclude=["parser", "ner"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
                f"Run: python -m spacy download {model_name}"
            ) from exc

        if not self._nlp.vocab.vectors_length:
            raise RuntimeError(
                f"spaCy model '{model_name}' ships no word vectors, so similarity "
                f"would be meaningless. Use a model such as en_core_web_md."
            )
        self.detail = f"{model_name} ({self._nlp.vocab.vectors_length}d vectors, content words)"
        self._lock = threading.Lock()

    def _content_vector(self, text: str):
        """Mean vector over content words; None when the text carries none."""
        import numpy as np

        with self._lock:  # spaCy pipelines are not thread-safe
            doc = self._nlp(text)

        vectors = [
            token.vector
            for token in doc
            if token.has_vector
            and token.is_alpha
            and not token.is_stop
            and not token.is_punct
            and not token.is_space
        ]
        if not vectors:
            return doc.vector if doc.has_vector else None
        return np.mean(vectors, axis=0)

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0

        import numpy as np

        vector_a = self._content_vector(text_a)
        vector_b = self._content_vector(text_b)
        if vector_a is None or vector_b is None:
            return 0.0

        norms = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        if norms == 0.0:
            return 0.0
        return max(0.0, float(np.dot(vector_a, vector_b) / norms))


class TfidfBackend(SimilarityBackend):
    """
    TF-IDF over word 1- and 2-grams, cosine between the two texts.

    Only the pair being compared is vectorised, so each call is independent
    of whatever else has been scored: a candidate gets the same score alone
    or inside a batch.
    """

    name = "tfidf"
    detail = "scikit-learn TfidfVectorizer, word 1-2 grams"
    suggested_threshold = 0.10

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform([text_a, text_b])
        except ValueError:
            # Both texts were entirely stop words / punctuation.
            return 0.0
        return max(0.0, float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]))


@lru_cache(maxsize=None)
def get_backend(name: str = "auto", spacy_model: str = DEFAULT_SPACY_MODEL) -> SimilarityBackend:
    """
    A cached backend by name: "auto", "spacy" or "tfidf".

    Cached because loading a spaCy model is the expensive part.
    """
    requested = (name or "auto").strip().lower()
    if requested == "tfidf":
        return TfidfBackend()
    if requested == "spacy":
        return SpacyBackend(spacy_model)
    if requested != "auto":
        raise ValueError(f"Unknown similarity backend '{requested}'. Use auto, spacy, or tfidf.")
    try:
        return SpacyBackend(spacy_model)
    except (ImportError, RuntimeError) as exc:
        logger.info("spaCy backend unavailable (%s); using TF-IDF similarity.", exc)
        return TfidfBackend()
