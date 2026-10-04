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
    APP_DATA = "app_data"


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
    """Investigation-relevant categories the NLP ensemble assigns.

    MEDIA, DRAFT, NAVIGATION_ACTIVITY, ACCOUNT_SETTINGS, NOTIFICATION and
    CONTACT_PII were added for the collaborator's classifier_backend_spec.md
    taxonomy (identity/communication/timestamp/contact_pii/media/credential/
    navigation_activity/draft/geolocation/account_settings/notification/
    financial) - added additively rather than replacing the existing set, so
    675k+ already-classified real warehouse rows, the UI, and the report
    builder don't need to change. The spec's identity/communication/
    credential/geolocation/financial/timestamp map onto USER/CHAT_CONTENT/
    CREDENTIAL/LOCATION_INFO/PAYMENT_INFO/TIMESTAMP below, unchanged.
    """

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
    CONTACT_PII = "CONTACT_PII"
    MEDIA = "MEDIA"
    DRAFT = "DRAFT"
    NAVIGATION_ACTIVITY = "NAVIGATION_ACTIVITY"
    ACCOUNT_SETTINGS = "ACCOUNT_SETTINGS"
    NOTIFICATION = "NOTIFICATION"
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
    # An examiner accepting or rejecting an artifact the classifier did not
    # flag. Distinct from ANALYSIS_STEP because it is a human judgement that
    # overrides an automated one, which is exactly the kind of decision the
    # custody log exists to make attributable and non-repudiable.
    REVIEW_DECISION = "review_decision"
    REPORT_GENERATED = "report_generated"
    NOTE = "note"
