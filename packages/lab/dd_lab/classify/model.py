"""Two optional trained transformer voters (ONNX Runtime) - different
slots in ensemble.py's Classifier, different jobs:

- `OnnxCategoryModel` / `load_model_voter` - predicts a CATEGORY
  (CREDENTIAL, CHAT_CONTENT, ...), feeds `Classifier`'s `model_voter` slot.
  Populated by the model training track (PLAN.md section 5). Currently
  unconfigured/unused in production - until a model exists at the
  configured path this loads nothing and category assignment runs on the
  other voters.
- `OnnxRelevanceModel` / `load_relevance_voter` - predicts binary
  relevant/not_relevant directly, feeds `Classifier`'s separate
  `relevance_voter` slot. Trained by the standalone `train_relevance_model.py`
  project; see that class's own docstring for the text-format contract.

Both expect a directory containing `model.onnx`, `tokenizer.json` and
`labels.json` - only the label set inside `labels.json` differs (the
real category list vs. `["not_relevant", "relevant"]`).
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


def _softmax_rows(x):
    """Row-wise softmax for a 2D (batch, num_labels) logits array - _softmax
    above only handles a single row (used by both __call__ methods)."""
    import numpy as np

    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_model_voter(model_dir: str | Path | None):
    if not model_dir:
        return None
    model = OnnxCategoryModel(model_dir)
    return model if model.available else None


class OnnxRelevanceModel:
    """The binary relevant/not_relevant voter (ensemble.py's `relevance_voter`
    slot) - a DIFFERENT model/slot from `OnnxCategoryModel` above, which
    predicts a category. This one predicts relevance directly: trained by
    `train_relevance_model.py` (a standalone project, see its own README)
    on this project's real labelled export, `labels.json` here is
    `["not_relevant", "relevant"]`, not the category list.

    Text format matters: training built each example as
    `f"[{source_type}] {key} = {value}"` (every one of the 8 source CSVs
    used that convention - confirmed in that project's README). `__call__`
    below must match it exactly, or every prediction is running on
    out-of-distribution input the model never saw the shape of during
    training - a real, easy-to-miss mismatch, since `OnnxCategoryModel`
    above doesn't take `source_type` at all and would silently produce a
    different string if reused as-is.

    Also worth knowing: training data only covered 8 of this project's 12
    SourceType values (no memory_process/memory_network/memory_artifact/
    app_data rows existed in the labelled export) - predictions for those
    four are somewhat out-of-distribution, not a wiring bug, just a real
    gap in what the model has actually seen.
    """

    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self._session = None
        self._tokenizer = None
        self._relevant_idx: int | None = None
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._session = ort.InferenceSession(
                str(self.dir / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            self._tokenizer = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
            # Needed for batch() below - encode_batch() only produces a
            # uniform-length (and therefore stackable-into-one-array) batch
            # if padding is explicitly enabled; __call__ above never
            # batches so never needed this. pad_id=0 confirmed against the
            # real deployed tokenizer ([PAD] token id, standard for a
            # BERT-family vocab); truncation is already baked into this
            # tokenizer.json from training (max_length=256), confirmed by
            # inspecting tok.truncation directly - no separate
            # enable_truncation() call needed here.
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
            labels = json.loads((self.dir / "labels.json").read_text("utf-8"))
            self._relevant_idx = labels.index("relevant")
            log.info("ONNX relevance model loaded from %s", self.dir)
        except Exception:
            log.info("ONNX relevance model not available at %s; relevance uses rules only", self.dir)
            self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None

    def __call__(self, key: str, value: str, source_type: str | None) -> float | None:
        if not self.available:
            return None
        try:
            import numpy as np

            text = f"[{source_type}] {key} = {value}" if source_type else f"{key} = {value}"
            enc = self._tokenizer.encode(text[:512])
            ids = np.array([enc.ids], dtype="int64")
            mask = np.array([enc.attention_mask], dtype="int64")
            logits = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0][0]
            probs = _softmax(logits)
            return float(probs[self._relevant_idx])
        except Exception:
            log.debug("ONNX relevance inference failed", exc_info=True)
            return None

    # Real chunk size, not a guess: measured directly against the deployed
    # model on real hardware (an 8-core x86_64 Mac, CPU-only) - 32 was the
    # sweet spot among 16/32/64/128 (bigger batches did NOT keep improving
    # ms/row past this point, and cost more peak memory for no return).
    # Re-benchmark if this model or the deployment hardware changes a lot.
    _BATCH_SIZE = 32

    def batch(self, items: list[tuple[str, str, str | None]]) -> list[float | None]:
        """Batched form of __call__ - real, measured win on CPU (roughly
        1.4x throughput over one-row-at-a-time on the hardware this was
        tuned against; batching amortizes per-call tokenize/dispatch
        overhead, but the dominant cost on CPU is the matmuls themselves,
        which don't get cheaper just from batching - don't expect
        multiples). `items` is (key, value, source_type) tuples, same
        shape __call__ takes positionally. Returns one float|None per item,
        same order, same length - `None` for every item if the whole batch
        call fails (mirrors __call__'s per-item graceful-degradation
        philosophy, just at batch granularity: a batch failing shouldn't
        be worse than that batch's rows individually falling back to
        rules-only relevance).
        """
        if not self.available or not items:
            return [None] * len(items)
        try:
            import numpy as np

            texts = [
                f"[{st}] {k} = {v}"[:512] if st else f"{k} = {v}"[:512] for k, v, st in items
            ]
            out: list[float | None] = []
            # Sub-chunk internally so callers can pass arbitrarily large
            # lists (e.g. a whole classify_warehouse batch) without every
            # caller needing to know/manage the tuned chunk size itself.
            for start in range(0, len(texts), self._BATCH_SIZE):
                chunk = texts[start : start + self._BATCH_SIZE]
                encs = self._tokenizer.encode_batch(chunk)
                ids = np.array([e.ids for e in encs], dtype="int64")
                mask = np.array([e.attention_mask for e in encs], dtype="int64")
                logits = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
                probs = _softmax_rows(logits)
                out.extend(float(p[self._relevant_idx]) for p in probs)
            return out
        except Exception:
            log.debug("ONNX relevance batch inference failed", exc_info=True)
            return [None] * len(items)


def load_relevance_voter(model_dir: str | Path | None):
    if not model_dir:
        return None
    model = OnnxRelevanceModel(model_dir)
    return model if model.available else None
