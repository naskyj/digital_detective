"""Memory-derived extraction patterns (classifier_backend_spec.md's
"MEMORY-DERIVED ADDITIONS" section) - value-shapes specific to how memory
carries evidence differently from disk: chiefly as CDN URLs, WebAuthn
records, and OS toast notifications, rather than clean store records.

Source-agnostic (a URL's own host identifies the app, a WebAuthn record's
own `rpId` field identifies the relying party), unlike app_profiles.py -
no `app` parameter needed. Category mapping follows the same rule
app_profiles.py's own docstring already established: the spec's per-item
"-> identity"/"-> media"/"-> communication"/"-> credential" labels map onto
this project's categories per phase 1's decision (identity -> USER,
communication -> CHAT_CONTENT, credential -> CREDENTIAL); `notification` is
that phase's own new member, used here for the toast-content rule - the
spec's own rationale for adding NOTIFICATION as a first-class category in
the first place ("OS toasts can hold message content that no app store
record does").

Hits are unioned into ensemble.py's `scores`/`spans` directly (not folded
into regex_hits/keyword_hits like app_profiles.py, since these hits aren't
scoped by `app` and are only ever consulted for the content-tier source
types - see ensemble.py), reusing `_W_REGEX` since every pattern here is a
deterministic URL/JSON-shape match, mechanically identical to any other
regex hit.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from dd_core.enums import ArtifactCategory as C

if TYPE_CHECKING:
    from dd_lab.classify.rules import RuleHit

# --------------------------------------------------------- URL-embedded ID extraction
# Memory holds identity/media chiefly as CDN URLs, not JSON - extract the
# id/filename from the URL itself rather than requiring a clean record.

_DISCORD_AVATAR = re.compile(r"cdn\.discordapp\.com/avatars/(\d{17,20})/")
_DISCORD_ATTACHMENT = re.compile(
    r"discordapp\.(?:com|net)/attachments/(\d{17,20})/(\d{17,20})/([^/?\"]+)"
)
_SLACK_FILE = re.compile(r"files(?:-tmb)?\.slack\.com/.*?(F0[A-Z0-9]{7,})")
_ROBLOX_URL = re.compile(r"roblox\.com/(users|games|communities)/(\d+)")

# --------------------------------------------------------- WebAuthn / passkey record
# Seen verbatim in memory: {"rpId":"discord.com","userId":"...",
# "name":"<email>","displayName":"<username>"}. rpId is the relying-party
# domain (which app the credential is for) - attribution value alongside
# the credential/identity signal itself.
_WEBAUTHN_KEYS = re.compile(r'"rpId"\s*:|"credentialId"\s*:|"userId"\s*:|"name"\s*:|"displayName"\s*:')

# --------------------------------------------------------- Windows toast notification
# The rendered message text surfaced in memory as a Windows toast
# (<toast>...<text>...</text>), including content no app store record
# holds (e.g. a message composed and sent entirely within a session that
# left no disk trace) - why NOTIFICATION is a first-class category.
_TOAST_TEXT = re.compile(r"<text[^>]*>([^<]{3,})</text>", re.I)
_TOAST_LAUNCH = re.compile(r'<toast[^>]*launch="([^"]+)"', re.I)


def memory_derived_hits(value: str) -> list["RuleHit"]:
    # Deferred import - same circular-import reasoning as
    # app_profiles.py::app_profile_hits (rules.py will import this
    # function, so a module-level import back of RuleHit would cycle).
    from dd_lab.classify.rules import RuleHit

    hits: list[RuleHit] = []

    m = _DISCORD_AVATAR.search(value)
    if m:
        hits.append(RuleHit(C.USER, 0.6, m.group(0)[:80]))

    m = _DISCORD_ATTACHMENT.search(value)
    if m:
        hits.append(RuleHit(C.MEDIA, 0.6, m.group(0)[:80]))
        hits.append(RuleHit(C.CHAT_CONTENT, 0.4, f"channel={m.group(1)} message={m.group(2)}"))

    m = _SLACK_FILE.search(value)
    if m:
        hits.append(RuleHit(C.MEDIA, 0.6, m.group(0)[:80]))

    m = _ROBLOX_URL.search(value)
    if m:
        cat = C.USER if m.group(1) == "users" else C.NAVIGATION_ACTIVITY
        hits.append(RuleHit(cat, 0.6, m.group(0)[:80]))

    # Require 2+ of the distinctive WebAuthn key names together, not just
    # one match anywhere - "name"/"userId" alone are common enough
    # elsewhere that a single hit isn't distinctive on its own, but
    # "rpId"+"credentialId" (or any 2 of the 5) together essentially only
    # occurs in a real WebAuthn/passkey record. Deliberately not requiring
    # a full json.loads() parse to succeed - memory carves are often
    # truncated fragments (a cut-off closing brace shouldn't cost the hit
    # when the key names are already unambiguous).
    #
    # Single CREDENTIAL hit only, not also a USER hit for the spec's
    # "identity" half of "credential + identity": a WebAuthn record's own
    # `name`/`displayName` fields are commonly a real email/username, which
    # already fires the ordinary generic USER-category signals (e.g. the
    # email regex in rules.py) independently - adding a second, competing
    # USER hit here would just dilute against that already-present signal
    # and, in the single-winner scoring this ensemble uses, can flip the
    # result to USER even though "a stored credential exists on this
    # device" is the more decisive forensic finding. Weighted like the
    # other high-distinctiveness credential shapes in rules.py (JWT 0.98,
    # Slack token 0.95) since 2+ of these field names together is just as
    # specific a structural signature.
    if len(_WEBAUTHN_KEYS.findall(value)) >= 2:
        hits.append(RuleHit(C.CREDENTIAL, 0.9, "webauthn record"))

    m = _TOAST_TEXT.search(value)
    if m:
        hits.append(RuleHit(C.NOTIFICATION, 0.6, m.group(1)[:80]))
    m = _TOAST_LAUNCH.search(value)
    if m:
        hits.append(RuleHit(C.NOTIFICATION, 0.4, m.group(1)[:80]))

    return hits
