"""Two things this ensemble decides, on purpose kept separate: `category`
(what kind of thing is this - CREDENTIAL, CHAT_CONTENT, ... - a display/
grouping label) and `relevant`/`relevance_score` (does this matter to the
investigation - the PRIMARY output this pipeline exists to produce, since
it's what report generation, timeline construction, and event correlation
actually key off of).

`category` is a weighted vote among regex/keyword/NER/entropy/an optional
category-predicting model (`model_voter`) - unchanged from the original
design: each voter proposes a category and a strength in [0, 1], strengths
for the same category are combined, the top category wins, ties break by
CATEGORY_PRIORITY, confidence reflects both the winning score and how much
the voters agreed.

`relevant`/`relevance_score` are their OWN combination - regex + keyword +
NER (their combined evidence across every category, not just the winner)
plus an optional trained binary relevant/not_relevant model (`relevance_voter`,
a DIFFERENT slot from `model_voter` - it predicts relevance directly, not a
category). Entropy deliberately never feeds this - see `_W_ENTROPY`'s own
comment for why it's too weak a signal to trust for the primary decision.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from dd_core.enums import ArtifactCategory as C
from dd_core.enums import SourceType, ValueType
from dd_core.schema import ClassificationResult
from dd_lab.classify.entropy import entropy_vote
from dd_lab.classify.gates import boosted_relevance, is_structurally_empty
from dd_lab.classify.infra_gazetteer import is_infra_noise
from dd_lab.classify.ner import ner_vote
from dd_lab.classify.rules import CATEGORY_PRIORITY, content_field_hits, keyword_hits, regex_hits
from dd_lab.memory.extract import memory_derived_hits

CLASSIFIER_VERSION = "ensemble-0.3.0"

_PRIORITY_INDEX = {c: i for i, c in enumerate(CATEGORY_PRIORITY)}

# How much each voter counts toward the combined score.
_W_REGEX = 1.0
_W_KEYWORD = 0.85
_W_NER = 0.7
_W_MODEL = 1.1
# Weakest voter, deliberately - a value-shape heuristic with no key-name or
# keyword context at all (see entropy.py's own docstring). Its job is to
# break ties in favour of/against GENERAL when nothing else has a strong
# opinion, not to override a real signal from another voter. CATEGORY-ONLY:
# never contributes to relevance - too weak a signal to trust for the
# decision that now drives reports/timeline/correlation (a deliberate
# choice, not an oversight - confirmed with the project owner).
_W_ENTROPY = 0.5

# relevance_score combines rule_evidence (regex+keyword+NER, combined
# across every category - see `rule_evidence_total` in Classifier.classify())
# and the relevance model's own P(relevant) via noisy-OR - see the formula
# itself in classify() for why (a plain weighted average was tried first
# and turned out to be a real bug: it dilutes a confident model opinion
# toward 0 whenever rule_evidence is 0, defeating the model's ability to
# flag something the rules missed entirely - caught by a unit test before
# this ever shipped). `relevant` is this score crossing a threshold -
# tunable once real data on false positive/negative rates comes in from
# actual case reclassification.
_RELEVANCE_THRESHOLD = 0.5

# Model veto. Noisy-OR above can only ever RAISE relevance - which turned
# out to be a real, measured problem on the first full run against the real
# case (675,463 artifacts): rule_evidence saturates at 1.0 easily (regex 1.0
# + keyword 0.85, both capped), and 1 - (1-1.0)*(1-x) == 1.0 for ANY x, so a
# confident model "not relevant" was arithmetically incapable of lowering
# the score. 156,697 of the 213,016 artifacts flagged relevant (74%) were
# flagged despite the trained model scoring them under 0.01 - measured, not
# estimated. Real examples: `cache.content_type = image/png`, `text/css`,
# and minified webpack JS bundles, all sitting at relevance_score 1.0.
#
# So below this threshold the model is treated as confidently negative and
# its own score stands instead of the noisy-OR result (no invented cap
# constant - the model's probability IS the answer in that case). Observed
# noise scored ~5e-6 to ~5e-3, far below 0.05; genuinely borderline rows
# (~0.3-0.5) are nowhere near it and are never vetoed.
_MODEL_VETO_THRESHOLD = 0.05

# The categories where a false negative is a serious forensic error - a
# missed credential is not the same kind of mistake as a retained CSS header.
# These are exactly the top 3 of rules.py's own CATEGORY_PRIORITY ranking,
# i.e. this project's existing definition of "most sensitive".
#
# These used to be IMMUNE from the veto: a confident model negative could not
# suppress them. That was a policy, defended on principle, and measuring it
# showed what the principle cost. Immunity rescued 921 artifacts from the
# veto, of which 18 were genuinely relevant and 903 were not - about 50 false
# positives retained per real finding kept, and it took deployment precision
# from 0.619 down to 0.427 (paper §7.9).
#
# The policy is now implemented differently rather than abandoned. The veto
# applies to every category, so the REPORT no longer carries those 903; but a
# sensitive-category artifact the veto suppressed is not discarded either - it
# is routed to a dedicated lane of the examiner review queue
# (Warehouse.review_queue(lane="suppressed_sensitive")), where a human decides.
# Nothing is lost (R3), the 18 remain one click away, and the automated output
# stops asserting 903 things that are not true. Where a judgement about
# sensitive evidence is genuinely uncertain, this system's answer is to put it
# in front of a person, not to assert it.
SENSITIVE_CATEGORIES = frozenset({C.CREDENTIAL, C.PAYMENT_INFO, C.LOCATION_INFO})

#: Categories exempt from the veto. EMPTY by design - see above. Kept as a
#: named seam rather than deleted so scripts/ablation.py can restore the
#: historical behaviour and the paper can keep reporting what it cost; a
#: measurement that can no longer be reproduced is not a measurement.
_VETO_EXEMPT_CATEGORIES: frozenset = frozenset()

# Per-source-type voter routing (classifier_backend_spec.md's "PER-SOURCE
# ROUTING" finding): on the collaborator's own back-test, structured
# key/value sources already reach ~97% recall / ~82% precision on schema-
# key + value-shape (regex/keyword) alone - adding the fuzzy voters there
# (NER, model, entropy) gained 0 recall and only cost precision, since a
# rogue vote on an already-precise structured record has nothing to rescue
# and only adds a chance of a false positive. So for these source types,
# NER/model/entropy are skipped outright, not just down-weighted - see
# Classifier.classify() below. Content-dependent sources (cache/memory/
# app_data) keep the full voter set, since that's where real recall was
# actually missing. MEMORY_PROCESS is structured key=value process
# metadata (PID/name/CreateTime, ...), not free text, so it's grouped with
# the structured tier even though it's a memory-derived source type.
_STRUCTURED_SOURCES = frozenset(
    {
        SourceType.LOCAL_STORAGE,
        SourceType.SESSION_STORAGE,
        SourceType.INDEXEDDB,
        SourceType.COOKIES,
        SourceType.PREFERENCES,
        SourceType.LOCAL_STATE,
        SourceType.MEMORY_PROCESS,
    }
)
_CONTENT_SOURCES = frozenset(
    {
        SourceType.CACHE,
        SourceType.MEMORY_STRING,
        SourceType.MEMORY_ARTIFACT,
        SourceType.MEMORY_NETWORK,
        SourceType.APP_DATA,
    }
)


def _combine(scores: dict[C, float]) -> tuple[C, float]:
    """(category, confidence) - the category-assignment vote only.
    `relevant`/`relevance_score` used to be read off this same step (the
    winning category's own raw score, zeroed for GENERAL) - now computed
    entirely separately in Classifier.classify() (see `rule_evidence_total`
    there), so category assignment can keep using entropy/the old
    category-model voter without those two leaking into the relevance
    decision, which deliberately excludes them."""
    if not scores:
        return C.GENERAL, 0.0
    top = max(scores.items(), key=lambda kv: (kv[1], -_PRIORITY_INDEX.get(kv[0], 99)))
    cat, raw = top
    total = sum(scores.values())
    agreement = raw / total if total else 0.0
    confidence = max(0.0, min(1.0, 0.5 * min(raw, 1.0) + 0.5 * agreement))
    return cat, round(confidence, 3)


@dataclass
class _Pending:
    """Everything classify()/classify_batch() can compute for one artifact
    BEFORE the (possibly batched) relevance-model call - every voter has
    already run, rule_evidence is final, category/confidence are final.
    Only relevance_score/relevant/relevance_model_score and the gates that
    depend on them are still outstanding. Exists purely to let
    classify_batch() batch the one genuinely batchable, genuinely
    expensive step (the ONNX relevance model) across many rows while
    every cheap per-row step (regex/keyword/NER/gates) still just runs
    per-row exactly as classify() always has."""

    category: C
    confidence: float
    regex_vote: C | None
    keyword_vote: C | None
    ner_vote: C | None
    model_vote: C | None
    entropy_vote: C | None
    rule_evidence: float
    spans: list[str]
    key: str
    value: str
    source_type: SourceType | None
    deleted: bool


class Classifier:
    """Stateless apart from the lazily-loaded NER/ONNX models."""

    def __init__(self, model_voter=None, relevance_voter=None) -> None:
        self._model_voter = model_voter  # callable(key, value) -> (C, float) | None - category
        # callable(key, value, source_type) -> float | None - P(relevant).
        # A DIFFERENT slot/model from model_voter above.
        self._relevance_voter = relevance_voter

    def classify(
        self,
        key: str,
        value: str,
        value_type: ValueType | str = ValueType.STRING,
        *,
        app: str | None = None,
        source_type: SourceType | str | None = None,
        deleted: bool = False,
    ) -> ClassificationResult:
        prepared = self._prepare(key, value, value_type, app=app, source_type=source_type, deleted=deleted)
        if isinstance(prepared, ClassificationResult):
            return prepared  # empty-body gate already produced a complete result
        relevance_model_score = (
            self._relevance_voter(key, value, prepared.source_type)
            if self._relevance_voter is not None
            else None
        )
        return self._finish(prepared, relevance_model_score)

    def classify_batch(
        self, rows: list[dict], *, reuse_model_scores: bool = False
    ) -> list[ClassificationResult]:
        """Batched form of classify() - same result as calling classify()
        once per row, but the relevance-model call (the one genuinely
        expensive, genuinely batchable step - see model.py::
        OnnxRelevanceModel.batch) runs once across every row that needs
        it instead of once per row. `rows` takes the same fields classify()
        takes as params: `key`/`value` required, `value_type`/`app`/
        `source_type`/`deleted` optional (missing = same defaults as
        classify()). Real, measured perf win on CPU when a relevance model
        is configured (~1.4x throughput on the hardware this was tuned
        against - batching amortizes per-call overhead, but doesn't make
        the underlying matmuls cheaper, so don't expect multiples); no
        behavior or performance difference from calling classify() in a
        loop when no relevance model is configured.

        Used by classify_warehouse (pipeline.py), which already reads
        rows in large batches from the warehouse. classify_stream still
        calls classify() per-artifact - ingest is a generator over
        artifacts as they're parsed from evidence, not a natural batch
        shape to retrofit here.

        `reuse_model_scores=True` reads each row's already-stored
        `relevance_model_score` instead of running the model at all. The
        model is by far the most expensive part of a re-classify (a
        measured ~27 hours for 675k artifacts on CPU, vs ~30 minutes for
        everything else) and its output only depends on the artifact's own
        text, which does not change - so re-deriving relevance after a
        CHANGE TO THE COMBINATION LOGIC (not to the model itself) never
        needs to pay that cost twice. Requires the caller to actually
        supply the column; a row missing it simply scores None (rules
        only) rather than silently re-running inference.
        """
        prepared = [
            self._prepare(
                r["key"],
                r["value"],
                r.get("value_type", ValueType.STRING),
                app=r.get("app"),
                source_type=r.get("source_type"),
                deleted=r.get("deleted", False),
            )
            for r in rows
        ]
        # Rows the empty-body gate already fully resolved skip the
        # relevance-model call entirely, same as classify() does.
        to_score = [(i, p) for i, p in enumerate(prepared) if isinstance(p, _Pending)]
        model_scores: list[float | None] = [None] * len(rows)
        if reuse_model_scores:
            for i, _ in to_score:
                model_scores[i] = rows[i].get("relevance_model_score")
        elif self._relevance_voter is not None and to_score:
            items = [(p.key, p.value, p.source_type) for _, p in to_score]
            # Duck-typed: use a real batched call if the configured voter
            # offers one (OnnxRelevanceModel does), else fall back to
            # calling it once per item - keeps classify_batch() correct
            # (if not fast) for a plain test-fake callable that only
            # implements the classify()-style single-item signature.
            batch_fn = getattr(self._relevance_voter, "batch", None)
            batch_scores = (
                batch_fn(items) if batch_fn is not None
                else [self._relevance_voter(k, v, st) for k, v, st in items]
            )
            for (i, _), score in zip(to_score, batch_scores, strict=True):
                model_scores[i] = score
        return [
            p if isinstance(p, ClassificationResult) else self._finish(p, model_scores[i])
            for i, p in enumerate(prepared)
        ]

    def _prepare(
        self,
        key: str,
        value: str,
        value_type: ValueType | str = ValueType.STRING,
        *,
        app: str | None = None,
        source_type: SourceType | str | None = None,
        deleted: bool = False,
    ) -> ClassificationResult | _Pending:
        """Everything classify()/classify_batch() need except the
        relevance-model call and the final relevance combination/gates -
        see _Pending's own docstring for why this is split out. Returns a
        complete ClassificationResult directly only when the empty-body
        gate fires (nothing left to compute); a _Pending otherwise."""
        # Normalize source_type once - classify_warehouse's re-classify path
        # reads it back out of DuckDB as a plain string, not a SourceType
        # instance (see pipeline.py). An unrecognized/absent value falls
        # through to `None`, which routes as "run every voter" below - the
        # same behavior as every pre-phase-4 caller that never passed
        # source_type at all, so an unknown source type never silently
        # loses recall by being mis-routed into the structured tier.
        try:
            st = SourceType(source_type) if source_type is not None else None
        except ValueError:
            st = None
        structured = st in _STRUCTURED_SOURCES
        # Content-field flattening (below) is gated on explicit membership
        # in _CONTENT_SOURCES, unlike the NER/model/entropy routing above -
        # `source_type=None` (every pre-phase-5 caller) does NOT get it. A
        # brand-new, JSON-parsing-on-every-value feature defaults off for
        # "unspecified", rather than on-by-default like the pre-existing
        # voters that predate this phase.
        content_tier = st in _CONTENT_SOURCES

        # Empty-body pre-gate (classifier_backend_spec.md): runs FIRST,
        # before any voter, and overrides everything - a structurally empty
        # value carries no evidence no matter what ID the key/URL contains.
        if is_structurally_empty(value):
            return ClassificationResult(
                category=C.GENERAL,
                confidence=0.0,
                relevant=False,
                relevance_score=0.0,
                classifier_version=CLASSIFIER_VERSION,
                classified_ts=datetime.now(UTC),
            )

        scores: dict[C, float] = defaultdict(float)
        spans: list[str] = []
        # Relevance-only accumulator: sums regex+keyword+NER weighted hits
        # across EVERY category, not just the eventual winner - `scores`
        # above answers "which category", this answers "how much evidence
        # of ANYTHING mattering did the rule-based voters find at all".
        # Deliberately excludes entropy and the old category-model voter -
        # see the module docstring and _W_ENTROPY's comment for why.
        rule_evidence_total = 0.0

        rx = regex_hits(key, value, app)
        if content_tier:
            # Memory-derived extraction (URL-embedded ids, WebAuthn
            # records, Windows toast content - memory/extract.py) - source-
            # agnostic (a URL's own host names the app), gated the same way
            # as the content-field gazetteer above: content tier only,
            # never for source_type=None. Mechanically a regex/value-shape
            # match, so unioned straight into `rx` and reuses _W_REGEX.
            rx = rx + memory_derived_hits(value)
        regex_vote = _best(rx)
        for hit in rx:
            scores[hit.category] += _W_REGEX * hit.weight
            rule_evidence_total += _W_REGEX * hit.weight
            spans.append(hit.span)

        kw = keyword_hits(key, value, app)
        if content_tier:
            # Content-bearing field-name gazetteer, applied to flattened
            # JSON leaf keys (rules.py::content_field_hits) - the spec's
            # single biggest validated cache-recall fix (+7pts, precision
            # held). Mechanically a keyword hit (a lexicon match on a field
            # name), so it's unioned straight into `kw` rather than tracked
            # as a separate signal - reuses _W_KEYWORD below like every
            # other keyword-mechanism hit.
            kw = kw + content_field_hits(value)
        keyword_vote = _best(kw)
        for hit in kw:
            scores[hit.category] += _W_KEYWORD * hit.weight
            rule_evidence_total += _W_KEYWORD * hit.weight

        # Structured sources: literally don't call these three voters (not
        # just zero-weight them) - see _STRUCTURED_SOURCES' comment above.
        # Skipping outright also means GLiNER/ONNX inference is never
        # invoked for the ~70%+ of rows that are structured, a real perf
        # win alongside the precision one.
        ner_cat = None
        model_cat = None
        entropy_cat = None
        if not structured:
            nv = ner_vote(key, value)
            if nv:
                ner_cat, ner_strength = nv
                scores[ner_cat] += _W_NER * ner_strength
                rule_evidence_total += _W_NER * ner_strength

            if self._model_voter is not None:
                mv = self._model_voter(key, value)
                if mv:
                    model_cat, model_strength = mv
                    scores[model_cat] += _W_MODEL * model_strength

            ev = entropy_vote(key, value)
            if ev:
                entropy_cat, entropy_strength = ev
                scores[entropy_cat] += _W_ENTROPY * entropy_strength

        # Binary relevance model: unlike NER/the category model/entropy
        # above, this runs for EVERY row regardless of the structured/
        # content-tier routing - it was trained across all 8 real source
        # files, structured ones included, and relevance is now the
        # PRIMARY output this pipeline produces, so skipping the ~70% of
        # rows the structured tier represents (the way the category-only
        # voters do, for good reason specific to category assignment)
        # would mean most artifacts never get a real relevance opinion at
        # all. A deliberate difference from the routing above, not an
        # oversight - confirmed with the project owner. The call itself
        # happens in classify()/classify_batch() above, not here - THIS
        # function only prepares rule_evidence and everything else; the
        # relevance-model call is the one step classify_batch() batches
        # across many rows instead of doing here, per-row.

        # A bare timestamp / number with no other signal is TIMESTAMP / GENERAL.
        # `not scores` means nothing else fired at all, so rule_evidence_total
        # is still exactly 0.0 here too - feed it the same way, or a bare
        # timestamp would win TIMESTAMP as its category yet score 0.0
        # relevance (no regex/keyword/NER hit ever touched the accumulator).
        if not scores and str(value_type) == ValueType.TIMESTAMP:
            scores[C.TIMESTAMP] = 0.6
            rule_evidence_total = 0.6

        category, confidence = _combine(scores)
        rule_evidence = round(min(1.0, rule_evidence_total), 3)
        return _Pending(
            category=category,
            confidence=confidence,
            regex_vote=regex_vote,
            keyword_vote=keyword_vote,
            ner_vote=ner_cat,
            model_vote=model_cat,
            entropy_vote=entropy_cat,
            rule_evidence=rule_evidence,
            spans=spans,
            key=key,
            value=value,
            source_type=st,
            deleted=deleted,
        )

    def _finish(self, p: _Pending, relevance_model_score: float | None) -> ClassificationResult:
        """The other half of _prepare() - combines rule_evidence with the
        (possibly batched) relevance-model score, applies the gates that
        depend on the combined result, and builds the final
        ClassificationResult. Called once per row by both classify() and
        classify_batch(), after each has obtained relevance_model_score
        its own way (one call vs. one batched call)."""
        # Relevance: its own combination, not read off category's score.
        # p.rule_evidence is regex+keyword+NER's combined evidence across
        # every category (capped at 1.0); when a relevance model is
        # configured, combine via noisy-OR (1 - (1-a)*(1-b)), NOT a
        # weighted average - either signal being strongly positive is
        # enough on its own to call something relevant (the whole point of
        # the trained model is catching what rules miss entirely; a
        # weighted average would dilute a confident model opinion toward 0
        # whenever rule_evidence is 0 - e.g. rule_evidence=0, model=0.9
        # averaged to ~0.47, BELOW threshold, silently defeating the
        # model's own confident call - a real bug caught by a unit test
        # before this shipped). Two independent moderate signals also
        # correctly reinforce each other this way (0.4 and 0.4 combine to
        # 0.64, more convincing than either alone) - matches this
        # project's existing bias toward recall over precision for
        # forensic evidence. No relevance model configured -> rules alone.
        if relevance_model_score is not None:
            relevance_score = round(1 - (1 - p.rule_evidence) * (1 - relevance_model_score), 3)
        else:
            relevance_score = p.rule_evidence

        category, confidence = p.category, p.confidence

        # Model veto (see _MODEL_VETO_THRESHOLD / _VETO_EXEMPT_CATEGORIES).
        # Keyed on the final category OR the regex vote specifically: regex
        # is the deterministic, highest-trust voter, so a regex hit on a
        # credential/payment/location pattern protects the row even if the
        # combined vote landed on some other category. Keyword/NER are
        # deliberately NOT consulted here - they're the voters that produced
        # the `content_type` -> CHAT_CONTENT style false fires this veto
        # exists to clear, so letting them confer immunity would defeat it.
        # _VETO_EXEMPT_CATEGORIES is empty by default - no category is exempt.
        # See SENSITIVE_CATEGORIES for why the exemption was removed and where
        # those artifacts go instead.
        if (
            relevance_model_score is not None
            and relevance_model_score < _MODEL_VETO_THRESHOLD
            and not ({category, p.regex_vote} & _VETO_EXEMPT_CATEGORIES)
        ):
            relevance_score = min(relevance_score, round(relevance_model_score, 3))
        # ID-in-plumbing carve-out (classifier_backend_spec.md): an id
        # riding along in app plumbing or a telemetry URL isn't evidence -
        # overrides whatever the voters above produced (category AND
        # relevance alike), same override tier as the empty-body gate but
        # necessarily checked after voting since it needs the accumulated
        # result, not just the raw key/value.
        if is_infra_noise(p.key, p.value):
            category, confidence, relevance_score = C.GENERAL, 0.0, 0.0

        relevance_score = boosted_relevance(
            relevance_score, deleted=p.deleted, is_general=category == C.GENERAL
        )
        relevant = relevance_score >= _RELEVANCE_THRESHOLD
        return ClassificationResult(
            category=category,
            confidence=confidence,
            relevant=relevant,
            relevance_score=relevance_score,
            regex_vote=p.regex_vote,
            keyword_vote=p.keyword_vote,
            ner_vote=p.ner_vote,
            model_vote=p.model_vote,
            entropy_vote=p.entropy_vote,
            relevance_model_score=relevance_model_score,
            evidence_spans=p.spans[:5],
            classifier_version=CLASSIFIER_VERSION,
            classified_ts=datetime.now(UTC),
        )


def _best(hits) -> C | None:
    if not hits:
        return None
    return max(hits, key=lambda h: h.weight).category
