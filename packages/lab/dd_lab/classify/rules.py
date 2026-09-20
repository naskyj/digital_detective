"""Regex and keyword rules for artifact classification.

These double as the weak-supervision labelling functions for the model training
track (see PLAN.md section 5), so they are kept declarative and importable on
their own.

Each rule looks at the artifact's *key* (a dotted leaf path such as
`cache.body.user.friends[0].name`) and its *value*. Key-name signals are usually
more reliable than value signals, so the ensemble weights them accordingly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dd_core.enums import ArtifactCategory as C

# Priority for tie-breaking: earlier = wins a tie. Sensitive categories first.
CATEGORY_PRIORITY = [
    C.CREDENTIAL,
    C.PAYMENT_INFO,
    C.LOCATION_INFO,
    C.CHAT_CONTENT,
    C.USER,
    C.CONTACT_PII,
    C.GROUP_DETAIL,
    C.GAME_DETAIL,
    C.MEDIA,
    C.DRAFT,
    C.NAVIGATION_ACTIVITY,
    C.ACCOUNT_SETTINGS,
    C.NOTIFICATION,
    C.NETWORK_DETAIL,
    C.SYSTEM_DETAIL,
    C.TIMESTAMP,
    C.GENERAL,
]


@dataclass(frozen=True)
class RuleHit:
    category: C
    weight: float
    span: str  # the substring that matched, for the report


# --------------------------------------------------------------------- regexes
# (compiled once). `value` patterns are matched case-insensitively unless noted.

_VALUE_PATTERNS: dict[C, list[tuple[re.Pattern, float]]] = {
    C.CREDENTIAL: [
        (re.compile(r"eyJ[A-Za-z0-9_-]{6,}\.eyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"), 0.98),  # JWT
        (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"), 0.99),
        (re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}"), 0.95),  # Slack
        (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"), 0.95),  # GitHub
        (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), 0.9),  # Google API key
        (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0.95),  # AWS
        (re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}"), 0.8),
        # classifier_backend_spec.md's cross-app Chromium field: Local State's
        # os_crypt.encrypted_key is the DPAPI-wrapped master key that
        # decrypts the cookie store/saved secrets - high value for cookie
        # decryption, present across all three apps since each is
        # Electron/WebView2-hosted. Local State is kept whole as one JSON
        # artifact (see parsers/prefs.py), so this matches inside `value`,
        # never `key`.
        (re.compile(r"os_crypt.*?encrypted_key", re.I), 0.85),
    ],
    C.PAYMENT_INFO: [
        (re.compile(r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b"), 0.9),
        (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), 0.6),  # IBAN-ish
        (re.compile(r"\b(?:visa|mastercard|amex|american express|discover|maestro)\b", re.I), 0.5),
    ],
    C.LOCATION_INFO: [
        (re.compile(r"[-+]?\d{1,3}\.\d{4,}\s*,\s*[-+]?\d{1,3}\.\d{4,}"), 0.85),
    ],
    C.NETWORK_DETAIL: [
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0.55),
        (re.compile(r"\b(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}\b", re.I), 0.8),  # MAC
        (re.compile(r"\bwss?://[^\s\"']+"), 0.6),
        (re.compile(r"\bhttps?://[^\s\"']+"), 0.4),
    ],
    C.TIMESTAMP: [
        (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"), 0.8),
        (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), 0.4),
    ],
    C.SYSTEM_DETAIL: [
        (re.compile(r"\b(?:Windows NT|Mac OS X|X11; Linux|Android \d)\b"), 0.7),
        (re.compile(r"[A-Za-z]:\\(?:Users|Windows|Program Files)\\"), 0.75),
        (re.compile(r"%(?:APPDATA|LOCALAPPDATA|USERPROFILE)%"), 0.7),
        # Preferences' electron.media.device_id_salt - a persistent
        # per-install device identifier (spec's "attribution" field); closest
        # existing category to "this install can be re-identified across
        # sessions". Same whole-JSON-blob reasoning as os_crypt above.
        (re.compile(r"device_id_salt", re.I), 0.7),
    ],
    C.USER: [
        (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), 0.7),  # email
    ],
}

# Key-name substrings -> category (weight). Matched against the last 2 path
# segments, lower-cased, punctuation-stripped.
_KEY_KEYWORDS: dict[C, tuple[tuple[str, float], ...]] = {
    C.CREDENTIAL: (
        ("password", 0.9), ("passwd", 0.9), ("secret", 0.85), ("token", 0.75),
        ("apikey", 0.9), ("api_key", 0.9), ("accesskey", 0.85), ("auth", 0.5),
        ("credential", 0.9), ("privatekey", 0.9), ("session_id", 0.5), ("refreshtoken", 0.85),
    ),
    C.PAYMENT_INFO: (
        ("card", 0.7), ("creditcard", 0.9), ("last4", 0.9), ("lastdigits", 0.9),
        ("expmonth", 0.85), ("expyear", 0.85), ("cvv", 0.95), ("cvc", 0.95),
        ("billing", 0.7), ("payment", 0.75), ("stripe", 0.7), ("paypal", 0.7),
        ("invoice", 0.6), ("subscription", 0.45), ("price", 0.35), ("currency", 0.4),
        ("robux", 0.4),
    ),
    C.LOCATION_INFO: (
        ("latitude", 0.9), ("longitude", 0.9), ("coordinates", 0.85), ("geo", 0.6),
        ("location", 0.55), ("gps", 0.85), ("address", 0.65), ("city", 0.6),
        ("country", 0.5), ("postal", 0.7), ("zipcode", 0.7), ("timezone", 0.3),
        ("last_seen", 0.4),
    ),
    C.NETWORK_DETAIL: (
        ("ipaddress", 0.85), ("ip_addr", 0.85), ("hostname", 0.7), ("dns", 0.7),
        ("url", 0.4), ("domain", 0.55), ("endpoint", 0.5), ("proxy", 0.7),
        ("webhook", 0.7), ("useragent", 0.6), ("user_agent", 0.6), ("macaddress", 0.9),
        ("port", 0.4),
    ),
    C.TIMESTAMP: (
        ("timestamp", 0.85), ("created", 0.55), ("updated", 0.5), ("modified", 0.5),
        ("_at", 0.5), ("expires", 0.6), ("lastlogin", 0.6), ("installdate", 0.6),
        ("date", 0.4),
    ),
    C.SYSTEM_DETAIL: (
        ("osversion", 0.85), ("os_version", 0.85), ("platform", 0.6), ("buildnumber", 0.7),
        ("architecture", 0.6), ("cpu", 0.55), ("memory", 0.5), ("device", 0.5),
        ("browser", 0.55), ("screen", 0.45), ("locale", 0.4), ("systeminfo", 0.8),
    ),
    C.USER: (
        ("username", 0.85), ("userid", 0.8), ("user_id", 0.8), ("displayname", 0.8),
        ("display_name", 0.8), ("nickname", 0.75), ("handle", 0.6), ("email", 0.8),
        ("account", 0.5), ("profile", 0.45), ("avatar", 0.5), ("firstname", 0.7),
        ("lastname", 0.7), ("fullname", 0.75),
    ),
    C.CHAT_CONTENT: (
        ("message", 0.8), ("content", 0.5), ("body", 0.45), ("chat", 0.8),
        ("conversation", 0.8), ("thread", 0.55), ("shout", 0.7), ("wallpost", 0.7),
        ("comment", 0.6), ("reply", 0.6), ("dm", 0.6), ("directmessage", 0.85),
        ("text", 0.4),
    ),
    C.GROUP_DETAIL: (
        ("group", 0.7), ("guild", 0.8), ("server", 0.4), ("channel", 0.65),
        ("role", 0.55), ("member", 0.5), ("permission", 0.6), ("invite", 0.6),
        ("team", 0.5), ("workspace", 0.6), ("clan", 0.7),
    ),
    C.GAME_DETAIL: (
        ("gameid", 0.85), ("placeid", 0.9), ("universeid", 0.9), ("gameplay", 0.8),
        ("inventory", 0.7), ("catalog", 0.6), ("experience", 0.5), ("achievement", 0.6),
        ("leaderboard", 0.7), ("avatar", 0.4), ("world", 0.4), ("score", 0.4),
    ),
}

# Value keyword signals (weaker; matched anywhere in a lower-cased value).
_VALUE_KEYWORDS: dict[C, tuple[tuple[str, float], ...]] = {
    C.CREDENTIAL: (("password", 0.35), ("unauthorized", 0.2), ("invalid token", 0.3)),
    C.CHAT_CONTENT: (),
}

# classifier_backend_spec.md's cross-app cookie-NAME allowlist: cookie
# *values* are opaque/encrypted (nothing to pattern-match), so the name
# is the signal instead. parsers/cookies.py sets key="cookie.<host>.<name>",
# so the name is always the final dot-separated segment - anchored at the
# end of `key`, not run through _norm_key_tail's substring-in-tail
# mechanism, because some of these names are short enough (`_ga`) that a
# bare substring search would false-positive on unrelated keys (e.g.
# "organization" contains "ga"). Deliberately excludes the spec's named
# non-evidence cookies (__cf_bm, cf_clearance, __cflb, hmt_id - CDN/
# bot-management, pure config) by simply not listing them.
_COOKIE_CREDENTIAL_NAMES = re.compile(
    r"\.(ROBLOSECURITY|ARID|__dcfduid|__sdcfduid|ssb_instance_id|persona_device_id|"
    r"__stripe_mid|_ga(?:_[A-Z0-9]+)?|__utma|__utmz|_iidt|_vid_t)$",
    re.I,
)

_PUNCT = re.compile(r"[^a-z0-9]+")


def _norm_key_tail(key: str) -> str:
    tail = key.replace("[", ".").replace("]", "")
    segs = [s for s in tail.split(".") if s]
    return _PUNCT.sub("", "".join(segs[-2:]).lower())


def regex_hits(key: str, value: str, app: str | None = None) -> list[RuleHit]:
    hits: list[RuleHit] = []
    sample = value[:4000]
    for cat, patterns in _VALUE_PATTERNS.items():
        for pat, weight in patterns:
            m = pat.search(sample)
            if m:
                hits.append(RuleHit(cat, weight, m.group(0)[:80]))
    m = _COOKIE_CREDENTIAL_NAMES.search(key)
    if m:
        hits.append(RuleHit(C.CREDENTIAL, 0.6, m.group(1)))
    # `app` is accepted here (and by keyword_hits below) so both functions
    # can grow the per-app profile lookup (classifier_backend_spec.md) in a
    # later phase without another signature change; a no-op until then.
    del app
    return hits


def keyword_hits(key: str, value: str, app: str | None = None) -> list[RuleHit]:
    hits: list[RuleHit] = []
    ktail = _norm_key_tail(key)
    for cat, kws in _KEY_KEYWORDS.items():
        best = max((w for sub, w in kws if sub in ktail), default=0.0)
        if best:
            hits.append(RuleHit(cat, best, f"key~{ktail}"))
    vlow = value[:2000].lower()
    for cat, kws in _VALUE_KEYWORDS.items():
        best = max((w for sub, w in kws if sub in vlow), default=0.0)
        if best:
            hits.append(RuleHit(cat, best, "value-keyword"))
    del app
    return hits
