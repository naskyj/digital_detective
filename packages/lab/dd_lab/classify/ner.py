"""NER voter.

Uses GLiNER when installed (zero-shot, small, ONNX-friendly) to spot PERSON /
LOCATION / ORG entities that map to USER / LOCATION_INFO / GROUP_DETAIL. Falls
back to a lightweight heuristic so the ensemble always has three voters.

The model is loaded lazily and reused; if it can't load, `ner_vote` simply
returns None and the ensemble carries on with regex + keyword.
"""

from __future__ import annotations

import logging
import re

from dd_core.enums import ArtifactCategory as C

log = logging.getLogger("dd_lab.classify.ner")

_LABELS = {"person": C.USER, "location": C.LOCATION_INFO, "organization": C.GROUP_DETAIL}

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")
_PROPER_NAME = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")


class _NerModel:
    def __init__(self) -> None:
        self._model = None
        self._tried = False

    def _load(self):
        if self._tried:
            return self._model
        self._tried = True
        try:
            from gliner import GLiNER

            self._model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
            log.info("GLiNER loaded for NER voting")
        except Exception:
            log.info("GLiNER unavailable; NER voter uses heuristics only")
            self._model = None
        return self._model

    def vote(self, text: str) -> tuple[C, float] | None:
        text = text[:512].strip()
        if not text:
            return None
        model = self._load()
        if model is not None:
            try:
                ents = model.predict_entities(text, list(_LABELS), threshold=0.5)
                if ents:
                    best = max(ents, key=lambda e: e["score"])
                    return _LABELS[best["label"]], min(0.9, 0.5 + best["score"] / 2)
            except Exception:
                log.debug("GLiNER prediction failed", exc_info=True)
        return _heuristic(text)


def _heuristic(text: str) -> tuple[C, float] | None:
    if _EMAIL.search(text):
        return C.USER, 0.55
    if _PROPER_NAME.search(text) and len(text) < 60:
        return C.USER, 0.4
    return None


_MODEL = _NerModel()


def ner_vote(key: str, value: str) -> tuple[C, float] | None:
    return _MODEL.vote(value)
