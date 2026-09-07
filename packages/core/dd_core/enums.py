"""Controlled vocabularies shared across the collector and the lab app."""

from __future__ import annotations

from enum import StrEnum


class SourceType(StrEnum):
    """Where a normalized artifact was extracted from."""

    CACHE = "cache"
    LOCAL_STORAGE = "local_storage"
    SESSION_STORAGE = "session_storage"
    INDEXEDDB = "indexeddb"
    COOKIES = "cookies"
    PREFERENCES = "preferences"
    LOCAL_STATE = "local_state"
    MEMORY_STRING = "memory_string"
    MEMORY_PROCESS = "memory_process"
    MEMORY_NETWORK = "memory_network"
    MEMORY_ARTIFACT = "memory_artifact"


class ValueType(StrEnum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"
    URL = "url"
    TIMESTAMP = "timestamp"
    BINARY = "binary"
    NULL = "null"


class ArtifactCategory(StrEnum):
    """Investigation-relevant categories the NLP ensemble assigns."""

    CHAT_CONTENT = "CHAT_CONTENT"
    GAME_DETAIL = "GAME_DETAIL"
    GROUP_DETAIL = "GROUP_DETAIL"
    PAYMENT_INFO = "PAYMENT_INFO"
    LOCATION_INFO = "LOCATION_INFO"
    TIMESTAMP = "TIMESTAMP"
    SYSTEM_DETAIL = "SYSTEM_DETAIL"
    NETWORK_DETAIL = "NETWORK_DETAIL"
    CREDENTIAL = "CREDENTIAL"
    USER = "USER"
    GENERAL = "GENERAL"


class CustodyEvent(StrEnum):
    """Event types recorded in the hash-chained chain-of-custody log."""

    ACQUISITION_START = "acquisition_start"
    ARTIFACT_ACQUIRED = "artifact_acquired"
    ACQUISITION_COMPLETE = "acquisition_complete"
    MANIFEST_SIGNED = "manifest_signed"
    TRANSFER = "transfer"
    IMPORT = "import"
    INTEGRITY_VERIFIED = "integrity_verified"
    INTEGRITY_FAILED = "integrity_failed"
    ANALYSIS_START = "analysis_start"
    ANALYSIS_STEP = "analysis_step"
    ANALYSIS_COMPLETE = "analysis_complete"
    REPORT_GENERATED = "report_generated"
    NOTE = "note"
