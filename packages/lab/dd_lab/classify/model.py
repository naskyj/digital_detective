"""Optional trained transformer voter (ONNX Runtime).

Populated by the model training track (PLAN.md section 5). Until a model exists
at the configured path this loads nothing and the ensemble runs on three voters.
The expected artifact is a directory containing `model.onnx`, `tokenizer.json`
and `labels.json`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dd_core.enums import ArtifactCategory as C

log = logging.getLogger("dd_lab.classify.model")


class OnnxCategoryModel:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self._session = None
        self._tokenizer = None
        self._labels: list[str] = []
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(
                str(self.dir / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
            self._labels = json.loads((self.dir / "labels.json").read_text("utf-8"))
            log.info("ONNX category model loaded from %s (%d labels)", self.dir, len(self._labels))
        except Exception:
            log.info("ONNX category model not available at %s; ensemble uses 3 voters", self.dir)
            self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None

    def __call__(self, key: str, value: str) -> tuple[C, float] | None:
        if not self.available:
            return None
        try:
            import numpy as np

            enc = self._tokenizer.encode(f"{key} = {value}"[:512])
            ids = np.array([enc.ids], dtype="int64")
            mask = np.array([enc.attention_mask], dtype="int64")
            logits = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
            probs = _softmax(logits)
            idx = int(probs.argmax())
            return C(self._labels[idx]), float(probs[idx])
        except Exception:
            log.debug("ONNX inference failed", exc_info=True)
            return None


def _softmax(x):
    import numpy as np

    e = np.exp(x - x.max())
    return e / e.sum()


def load_model_voter(model_dir: str | Path | None):
    if not model_dir:
        return None
    model = OnnxCategoryModel(model_dir)
    return model if model.available else None
