"""Negative infra gazetteer + ID-in-plumbing carve-out
(classifier_backend_spec.md, "PER-SOURCE ROUTING" §3).

An identifier alone is not evidence when it sits inside app plumbing or a
telemetry URL - a Discord snowflake showing up in a `/rtc` housekeeping
route, or a Roblox user id riding along in a 1x1 tracking-pixel query
string, is not the same thing as that id appearing in a message or an
attachment URL. Validated by the collaborator on 444k real memory lines and
the cache export: this carve-out caught ~35 cache false positives and
accounted for ~30% of naive memory-string positives.

Like gates.py, this is not a voter - it's a suppression rule applied after
the ensemble has already voted (see ensemble.py), overriding the result to
GENERAL when it fires. It needs an id-shaped token to be *present* to fire
at all: plumbing text with no id in it is already correctly scored low/
GENERAL by the regular voters (nothing in it looks like a credential/user/
payment keyword), so this module's whole job is narrower than "detect
noise" - it's "stop an id from rescuing text that is otherwise noise."
"""

from __future__ import annotations

import re

# --------------------------------------------------------------------- ID shapes
# Broad on purpose - this only matters once _PLUMBING has already matched,
# so a false positive here just means a plumbing hit that also happens to
# contain a longish number doesn't get carved out for nothing.
_HAS_ID_SHAPE = re.compile(
    r"\b\d{7,20}\b"  # Roblox (7+) / Discord snowflake (17-20) shaped
    r"|\b[UWTCBDGF]0[A-Z0-9]{7,}\b"  # Slack id shape
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",  # UUID
    re.I,
)

# --------------------------------------------------------------------- plumbing
# Spec's own generic list, plus the memory-proven and cache-proven additions.
_PLUMBING = re.compile(
    r"request_time|response_time|etag|sequenceNumber|namespace|masterId|"
    r"ablation_seed|heartbeat|__storage_test__|lastFetched|LeaderReconnected|"
    r"StateTracker|ConnectionEvent|server-options|kingmaker|version\.stable|"
    # memory-proven: SDK/console-log prefixes
    r"\[SOCKET-MANAGER\]|\[DND_V2\]|\[HUDDLE-SDK\]|\[MESSAGE-LIST\]|\[API-Q\]|\[Logger\]|"
    # memory-proven: React/state churn
    r"componentWill\w*|componentDid\w*|REDUX_UPDATE|isLoading|throttleRate|Thunk|"
    # memory-proven: devtools/telemetry
    r"Target\.targetInfoChanged|cleanup telemetry|__analytics__|sentry|breadcrumb|ui\.click|"
    # memory-proven: profile/asset CDNs
    r"/avatars/|/icons/|/emojis/|assetdelivery|collectibles|"
    # memory-proven: housekeeping API routes
    r"api/v9/users|/profile|/skus|published-listings|science|/rtc|\.ping|/apex|experiments|/quests|"
    # cache-proven: telemetry-beacon 1x1 tracking pixels
    r"ecsv2\.\w+\.com/www/e\.png|/e\.png\?|pageHeartbeat|evt=|/pixel|/beacon|rotating-",
    re.I,
)

# --------------------------------------------------------------------- carve-out
# Genuine-content exceptions: even with a plumbing/telemetry match, one of
# these means real evidence, not noise - the single biggest precision fix
# on both memory (id-in-plumbing ~= 30% of naive positives) and cache
# (telemetry pixels carrying the user id) per the spec's own measurement.
_CONTENT_EXCEPTIONS = re.compile(
    r"/attachments/|files-tmb|files-pri|slack-files\.com|/forums/|invite_code|"
    r"invite_guild|message-content|message-username|chat-messages|"
    r'"content"|"text"|/avatars/\d{17,20}|user-settings|birthdate',
    re.I,
)


def is_infra_noise(key: str, value: str) -> bool:
    """True when the artifact carries an id-shaped token AND matches the
    plumbing/telemetry gazetteer, UNLESS it also matches a genuine-content
    exception. Searches key and value together, same as the spec's own
    examples (a plumbing route in the key, an id in the value, or both)."""
    haystack = f"{key} {value}"
    if not _HAS_ID_SHAPE.search(haystack):
        return False
    if not _PLUMBING.search(haystack):
        return False
    return not _CONTENT_EXCEPTIONS.search(haystack)
