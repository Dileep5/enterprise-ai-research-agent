"""Cross-encoder reranking."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        # Raw logits (no activation) so the evidence threshold is model-native.
        scores = self.model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]
