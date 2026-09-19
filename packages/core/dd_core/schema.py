"""The normalized artifact — the single row shape every parser emits and the
warehouse stores.

A parser's job is to turn a raw source record (a cache entry, a LevelDB
key/value pair, a memory string) into one or more `NormalizedArtifact` rows.
Classification fields are filled in later by the NLP ensemble and are null until
then.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from dd_core.enums import ArtifactCategory, SourceType, ValueType
from dd_core.hashing import sha256_text, stable_id


class ClassificationResult(BaseModel):
    """Output of the NLP ensemble for one artifact."""

    category: ArtifactCategory
    confidence: float = Field(ge=0.0, le=1.0)
    regex_vote: ArtifactCategory | None = None
    ner_vote: ArtifactCategory | None = None
    keyword_vote: ArtifactCategory | None = None
    model_vote: ArtifactCategory | None = None
    entropy_vote: ArtifactCategory | None = None
    evidence_spans: list[str] = Field(default_factory=list)
    classifier_version: str = "unknown"
    classified_ts: datetime | None = None


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
