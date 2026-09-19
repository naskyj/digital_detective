"""The normalized artifact — the single row shape every parser emits and the
warehouse stores.

A parser's job is to turn a raw source record (a cache entry, a LevelDB
key/value pair, a memory string) into one or more `NormalizedArtifact` rows.
Classification fields are filled in later by the NLP ensemble and are null until
then.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from dd_core.enums import ArtifactCategory, SourceType, ValueType
from dd_core.hashing import sha256_text, stable_id


class ClassificationResult(BaseModel):
    """Output of the NLP ensemble for one artifact.

    `category` answers "what kind of thing is this" (10 forensic categories
    + GENERAL); `relevant`/`relevance_score` answer a genuinely different
    question - "does this matter to the investigation" - as their own
    explicit, first-class fields rather than something every consumer
    re-derives from `category != 'GENERAL'` independently (found scattered,
    inconsistently, across correlate.py/agent tools/the API/the report
    before this existed as a real field - two different informal
    definitions of "relevant" already disagreed: category != GENERAL vs.
    v_high_priority's stricter category whitelist). Computed from the same
    per-voter scores category already uses, not a new/separate ensemble.

    `Classifier.classify()` (ensemble.py) always sets both explicitly. A
    caller that hand-constructs a result (test fixtures, mainly - real
    ingest always goes through the classifier) and doesn't mention either
    field gets a sensible default derived from `category`/`confidence`
    instead of silently landing on `relevant=False` regardless of category -
    real correctness risk, not just a test convenience: every consumer that
    now filters on `relevant` (correlate_entities, search_keyword, the
    artifact browser's own sort order) would have silently excluded any
    hand-built fixture row with a real category. Passing `relevant=`
    explicitly always wins - this only fills in what wasn't said."""

    category: ArtifactCategory
    confidence: float = Field(ge=0.0, le=1.0)
    relevant: bool = False
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    regex_vote: ArtifactCategory | None = None
    ner_vote: ArtifactCategory | None = None
    keyword_vote: ArtifactCategory | None = None
    model_vote: ArtifactCategory | None = None
    entropy_vote: ArtifactCategory | None = None
    evidence_spans: list[str] = Field(default_factory=list)
    classifier_version: str = "unknown"
    classified_ts: datetime | None = None

    @model_validator(mode="after")
    def _default_relevance_from_category(self) -> ClassificationResult:
        is_general = self.category == ArtifactCategory.GENERAL
        if "relevant" not in self.model_fields_set:
            self.relevant = not is_general
        if "relevance_score" not in self.model_fields_set:
            self.relevance_score = 0.0 if (is_general or not self.relevant) else max(self.relevance_score, self.confidence)
        return self


class NormalizedArtifact(BaseModel):
    """One extracted fact, with full provenance back to the evidence file."""

    artifact_id: str
    case_id: str
    app: str

    source_type: SourceType
    source_file: str  # path relative to <case>/evidence/
    source_locator: str  # cache key, "db/store/key", byte offset, etc.

    key: str
    value: str
    value_type: ValueType = ValueType.STRING

    record_ts: datetime | None = None  # timestamp carried by the artifact itself
    extracted_ts: datetime  # when the parser produced this row
    size_bytes: int = 0
    value_sha256: str = ""
    deleted: bool = False  # recovered from a tombstoned/overwritten LevelDB record, not the live value

    classification: ClassificationResult | None = None

    def model_post_init(self, __context: object) -> None:  # noqa: D401
        if not self.value_sha256:
            object.__setattr__(self, "value_sha256", sha256_text(self.value))
        if self.size_bytes == 0:
            object.__setattr__(self, "size_bytes", len(self.value.encode("utf-8")))

    @staticmethod
    def make_id(
        case_id: str, source_type: SourceType, source_file: str, locator: str, key: str
    ) -> str:
        return stable_id(case_id, str(source_type), source_file, locator, key)


class ProcessRef(BaseModel):
    pid: int
    name: str
    path: str | None = None


class TargetApp(BaseModel):
    """An Electron application selected for acquisition."""

    name: str
    user_data_dir: str
    version: str | None = None
    processes: list[ProcessRef] = Field(default_factory=list)
