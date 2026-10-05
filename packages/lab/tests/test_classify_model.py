"""OnnxRelevanceModel.batch()'s ordering contract.

batch() sorts its rows by token length before running them, so that a chunk
of 32 pads to something near its members' real length instead of to the
longest row in an arbitrary mix (measured 2.3x on real memory_string rows -
see the method's own comment). Every caller, including
Classifier.classify_batch, then zips the returned scores straight back onto
the rows it passed in - so "same order, same length as `items`" is a hard
contract, and a sort that forgot to invert its permutation would silently
attach every score to the wrong artifact. That is the failure this file
exists to catch: not a crash, not an exception, just quietly wrong
relevance on every row of a 12-hour pass.

Fakes rather than the real model on purpose: the deployed relevance model
is a 268 MB ONNX file living outside the repo (a user-configured
`relevance_model_dir`), so a test that needed it would simply not run.
"""

from __future__ import annotations

import math
import re

import pytest

# Ships with the `classify` extra (onnxruntime). Skip rather than error at
# collection when it is absent - CI runs without that extra.
np = pytest.importorskip("numpy")

from dd_lab.classify.model import OnnxRelevanceModel  # noqa: E402


class _Enc:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1] * len(ids)

    def pad_to(self, width: int) -> None:
        gap = width - len(self.ids)
        self.ids = self.ids + [0] * gap
        self.attention_mask = self.attention_mask + [0] * gap


class _FakeTokenizer:
    """Encodes a text to `int(text)` repeated `int(text)` times - so a row's
    token length AND its token values both encode which row it is, and any
    permutation bug is visible in the output rather than hidden behind
    interchangeable rows."""

    def __init__(self, *, pads: bool) -> None:
        self.pads = pads

    @staticmethod
    def _n(text: str) -> int:
        # batch() wraps the value in `f"{k} = {v}"`; pull the number back out.
        return int(re.search(r"\d+", text).group(0))

    def encode_batch(self, texts: list[str]) -> list[_Enc]:
        encs = [_Enc([self._n(t)] * self._n(t)) for t in texts]
        if self.pads and encs:
            width = max(len(e.ids) for e in encs)
            for e in encs:
                e.pad_to(width)
        return encs


class _FakeSession:
    """Returns a 'relevant' logit of ids[0]/100 for each row - i.e. the score
    is a direct function of which row it is."""

    def __init__(self) -> None:
        self.widths: list[int] = []

    def run(self, _out, feed):
        ids = feed["input_ids"]
        self.widths.append(int(ids.shape[1]))
        return [np.array([[0.0, float(row[0]) / 100.0] for row in ids])]


def _model(pads: bool = True) -> OnnxRelevanceModel:
    m = object.__new__(OnnxRelevanceModel)
    m._session = _FakeSession()
    m._tokenizer = _FakeTokenizer(pads=pads)
    m._len_tokenizer = _FakeTokenizer(pads=False)
    m._relevant_idx = 1
    return m


def _items(lengths: list[int]) -> list[tuple[str, str, str | None]]:
    # batch() builds its text as f"[{st}] {k} = {v}"; the fake tokenizer
    # parses that whole string as an int, so put the number in and leave
    # the rest empty.
    return [("", str(n), None) for n in lengths]


def _expected(lengths: list[int]) -> list[float]:
    # batch() softmaxes the logits before returning, and the fake session's
    # logits are [0, n/100] - so the score is sigmoid(n/100), still strictly
    # monotonic in n and therefore still a unique fingerprint per row.
    return [1.0 / (1.0 + math.exp(-n / 100.0)) for n in lengths]


def test_batch_returns_scores_in_the_callers_order_not_sorted_order():
    # Deliberately hostile order: descending, so a missing inverse
    # permutation returns exactly the reverse of the right answer.
    lengths = [40, 35, 30, 25, 20, 15, 10, 5]
    m = _model()
    assert m.batch(_items(lengths)) == pytest.approx(_expected(lengths))


def test_batch_order_survives_chunk_boundaries():
    """The permutation has to be inverted across the whole call, not within
    each _BATCH_SIZE chunk - so this spans several chunks and interleaves
    short and long rows."""
    lengths = [(97 * i) % 200 + 1 for i in range(100)]  # > 3 chunks, shuffled
    m = _model()
    assert m.batch(_items(lengths)) == pytest.approx(_expected(lengths))


def test_batch_actually_buckets_by_length():
    """The point of the sort: each chunk pads to its own members' length,
    not to the longest row in the whole call. Without bucketing, one 200-
    token row would widen every chunk it landed in."""
    lengths = [1] * 32 + [200] + [1] * 32  # 65 rows -> chunks of 32/32/1
    m = _model()
    m.batch(_items(lengths))
    widths = m._session.widths
    # The long row sits alone in the last chunk; the other two stay 1 token
    # wide. Unsorted it would land mid-call and drag 31 one-token rows up
    # to 200 with it - 6,400 token-slots of transformer instead of 200.
    assert widths == [1, 1, 200], widths


def test_batch_is_length_and_order_stable_for_duplicate_rows():
    """Equal sort keys must not reorder anything observable - duplicates are
    the case where a naive `sorted` on (key, index) vs `sorted` on key alone
    would diverge."""
    lengths = [7, 7, 7, 3, 3, 11]
    m = _model()
    out = m.batch(_items(lengths))
    assert len(out) == len(lengths)
    assert out == pytest.approx(_expected(lengths))


def test_batch_of_one_and_empty_batch():
    m = _model()
    assert m.batch([]) == []
    assert m.batch(_items([4])) == pytest.approx(_expected([4]))


def test_batch_failure_degrades_to_none_per_item_not_an_exception():
    """Pre-existing contract, pinned here because the sort added a second
    place (the length pass) that can now raise: a failing batch must return
    one None per item so those rows fall back to rules-only relevance,
    exactly as __call__ does."""
    m = _model()

    class _Boom:
        def encode_batch(self, texts):
            raise RuntimeError("tokenizer exploded")

    m._len_tokenizer = _Boom()
    assert m.batch(_items([1, 2, 3])) == [None, None, None]
