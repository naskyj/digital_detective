"""Pretrained zero-shot classifier — a stopgap 4th voter until the fine-tuned
model from the training track is ready.

Uses a Hugging Face zero-shot-classification pipeline (an NLI model) with the
artifact categories as candidate labels. No training required; slower and less
accurate than a fine-tuned model, so it is opt-in via the `zeroshot` extra and
disabled by default. Same call signature as `OnnxCategoryModel`, so the ensemble
treats them interchangeably.

    uv sync --all-packages --extra chromium --extra zeroshot
    # first use downloads the model (~500 MB) to the HF cache
"""

from __future__ import annotations

import logging

from dd_core.enums import ArtifactCategory as C

log = logging.getLogger("dd_lab.classify.zeroshot")

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

# natural-language hypotheses read better to an NLI model than the enum names
_LABELS: dict[str, C] = {
    "a chat message or conversation": C.CHAT_CONTENT,
    "a password, API key, token or credential": C.CREDENTIAL,
    "payment card or billing information": C.PAYMENT_INFO,
    "a geographic location, address or coordinates": C.LOCATION_INFO,
    "a username, display name, email or user account": C.USER,
    "a chat group, server, channel or team": C.GROUP_DETAIL,
    "a video game detail": C.GAME_DETAIL,
    "a network address, URL, host or connection detail": C.NETWORK_DETAIL,
    "operating system, device or software detail": C.SYSTEM_DETAIL,
    "a date or timestamp": C.TIMESTAMP,
    "nothing sensitive or noteworthy": C.GENERAL,
}


class ZeroShotVoter:
    def __init__(self, model: str = DEFAULT_MODEL, min_score: float = 0.55):
        self._model_name = model
        self._min = min_score
        self._pipe = None
        self._load()

    def _load(self) -> None:
        try:
            from transformers import pipeline

            self._pipe = pipeline("zero-shot-classification", model=self._model_name)
            log.info("zero-shot voter ready (%s)", self._model_name)
        except Exception:
            log.info("zero-shot voter unavailable; ensemble uses 3 voters")
            self._pipe = None

    @property
    def available(self) -> bool:
        return self._pipe is not None

    def __call__(self, key: str, value: str) -> tuple[C, float] | None:
        if not self._pipe:
            return None
        text = f"{key.split('.')[-1]}: {value}"[:400]
        try:
            out = self._pipe(text, list(_LABELS), multi_label=False)
            label, score = out["labels"][0], out["scores"][0]
        except Exception:
            log.debug("zero-shot inference failed", exc_info=True)
            return None
        cat = _LABELS.get(label, C.GENERAL)
        if cat is C.GENERAL or score < self._min:
            return None
        return cat, float(score)


def load_zeroshot_voter(enabled: bool, model: str = DEFAULT_MODEL):
    if not enabled:
        return None
    voter = ZeroShotVoter(model)
    return voter if voter.available else None
